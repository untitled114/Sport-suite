#!/usr/bin/env python3
"""
NBA XL Daily Predictions Generator
===================================
Main prediction script for XL betting system.

Generates picks using validated line shopping strategy.

Validation Performance (Oct 23 - Nov 4, 2024):
- Line Shopping: 54.5% WR, +4.16% ROI
- POINTS: 56.7% WR, +8.27% ROI
- REBOUNDS: 61.2% WR, +16.96% ROI
- High-spread goldmine (≥2.5): 70.6% WR, +34.82% ROI

Part of Phase 5: XL Betting Pipeline (Task 5.3)

Usage:
    python3 generate_xl_predictions.py --date 2025-11-07
    python3 generate_xl_predictions.py --output predictions/xl_picks_20251107.json
    python3 generate_xl_predictions.py --dry-run
"""

import argparse
import json
import os
import sys
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import psycopg2

from nba.betting_xl.line_optimizer import PRODUCTION_CONFIG, LineOptimizer
from nba.betting_xl.risk_filters import RiskFilter
from nba.betting_xl.utils.hit_rate_loader import HitRateCache
from nba.betting_xl.xl_predictor import XLPredictor
from nba.core.drift_service import DriftService
from nba.core.logging_config import add_logging_args, get_logger, setup_logging
from nba.features.extract_live_features_xl import LiveFeatureExtractorXL
from nba.utils.name_normalizer import NameNormalizer

# Logger will be configured in main() - use get_logger for module-level access
logger = get_logger(__name__)

# Database configs
DB_DEFAULT_USER = os.getenv("NBA_DB_USER", os.getenv("DB_USER", "nba_user"))
DB_DEFAULT_PASSWORD = os.getenv("NBA_DB_PASSWORD", os.getenv("DB_PASSWORD"))

DB_INTELLIGENCE = {
    "host": os.getenv("NBA_INT_DB_HOST", "localhost"),
    "port": int(os.getenv("NBA_INT_DB_PORT", 5539)),
    "user": os.getenv("NBA_INT_DB_USER", DB_DEFAULT_USER),
    "password": os.getenv("NBA_INT_DB_PASSWORD", DB_DEFAULT_PASSWORD),
    "database": os.getenv("NBA_INT_DB_NAME", "nba_intelligence"),
}

DB_PLAYERS = {
    "host": os.getenv("NBA_PLAYERS_DB_HOST", "localhost"),
    "port": int(os.getenv("NBA_PLAYERS_DB_PORT", 5536)),
    "user": os.getenv("NBA_PLAYERS_DB_USER", DB_DEFAULT_USER),
    "password": os.getenv("NBA_PLAYERS_DB_PASSWORD", DB_DEFAULT_PASSWORD),
    "database": os.getenv("NBA_PLAYERS_DB_NAME", "nba_players"),
}

# Drift detection configuration
DRIFT_DETECTION_ENABLED = os.getenv("NBA_DRIFT_DETECTION_ENABLED", "true").lower() == "true"
DRIFT_Z_THRESHOLD = float(os.getenv("NBA_DRIFT_Z_THRESHOLD", "3.0"))
DRIFT_BLOCK_ON_HIGH = os.getenv("NBA_DRIFT_BLOCK_ON_HIGH", "false").lower() == "true"


class XLPredictionsGenerator:
    """
    Generates daily XL predictions with line shopping.

    Process:
    1. Query today's props from nba_props_xl (multi-book data)
    2. For each prop:
       - Extract 102 features
       - Run XL model prediction
       - Find softest line via line shopping
       - Calculate edge and confidence
    3. Filter by production thresholds
    4. Output formatted picks
    """

    def __init__(
        self,
        game_date=None,
        as_of_date=None,
        backtest_mode: bool = False,
        predictions_dir: str = None,
        underdog_only: bool = None,
        model_versions: list = None,
        standard_only: bool = False,
    ):
        self.game_date = game_date or datetime.now().strftime("%Y-%m-%d")
        self.game_date_obj = datetime.strptime(self.game_date, "%Y-%m-%d").date()

        # Backtest support - as_of_date limits calibration data
        self.as_of_date = as_of_date  # datetime object or None
        self.backtest_mode = backtest_mode
        # Custom predictions directory for calibrator (backtest saves here, calibrator reads from here)
        self.predictions_dir = predictions_dir
        # Underdog-only mode: only accept props where Underdog is softest
        self.underdog_only = underdog_only
        # Model versions to load: ['xl', 'v3', 'dfs']
        self.model_versions = model_versions or ["xl", "v3"]
        # Standard-only mode: exclude PrizePicks alternate lines (goblin/demon) not in training data
        self.standard_only = standard_only

        # Components
        self.feature_extractor = None
        self.predictors = {}  # {market: {version: XLPredictor}}
        self.line_optimizer = None
        self.risk_filter = RiskFilter()  # Risk assessment for volatility/defense/trend
        self.hit_rate_cache = HitRateCache(self.game_date)
        self.normalizer = NameNormalizer()
        self.player_status: Dict[str, Dict[str, Any]] = {}
        self.bias_adjustments: Dict[str, float] = {}
        self.opp_rank_cache: Dict[str, int] = {}  # (opponent_team, stat_type) -> rank

        # Database connections
        self.conn_intelligence = None
        self.conn_players = None

        # Drift detection
        self.drift_services: Dict[str, DriftService] = {}
        self.drift_status: Dict[str, Dict[str, Any]] = {}
        self._feature_samples: Dict[str, list] = {}  # market -> list of feature dicts

        # Results
        self.picks = []

    def connect_databases(self):
        """Connect to NBA databases"""
        self.conn_intelligence = psycopg2.connect(**DB_INTELLIGENCE)
        self.conn_players = psycopg2.connect(**DB_PLAYERS)
        logger.info("[OK] Connected to databases")

    def load_models(self):
        """Load XL and V3 predictors for enabled markets"""
        logger.info(f"Loading models: {self.model_versions}...")

        enabled_markets = [k for k, v in PRODUCTION_CONFIG.items() if v.get("enabled", False)]

        for market in enabled_markets:
            self.predictors[market] = {}
            for version in self.model_versions:
                try:
                    self.predictors[market][version] = XLPredictor(
                        market,
                        use_3head=False,
                        as_of_date=self.as_of_date,
                        backtest_mode=self.backtest_mode,
                        predictions_dir=self.predictions_dir,
                        model_version=version,
                        enable_dynamic_calibration=True,
                    )
                except (psycopg2.Error, KeyError, TypeError, ValueError, FileNotFoundError) as e:
                    logger.warning(f"Failed to load {market} {version.upper()} model: {e}")

        # Count successfully loaded models
        loaded_counts = {}
        for ver in self.model_versions:
            loaded_counts[ver] = sum(1 for m in self.predictors.values() if ver in m)

        # NOTE: Odds API picks are now handled by standalone generate_odds_api_picks.py
        counts_str = " + ".join(f"{c} {v.upper()}" for v, c in loaded_counts.items())
        logger.info(f"[OK] Loaded {counts_str} models")

    def initialize_components(self):
        """Initialize feature extractor and line optimizer"""
        self.feature_extractor = LiveFeatureExtractorXL()
        self.line_optimizer = LineOptimizer(standard_only=self.standard_only)
        if self.standard_only:
            logger.info(
                "[OK] Initialized feature extractor and line optimizer (STANDARD ONLY - no goblin/demon)"
            )
        else:
            logger.info("[OK] Initialized feature extractor and line optimizer")

    def initialize_drift_detection(self):
        """Initialize drift detection services for enabled markets."""
        if not DRIFT_DETECTION_ENABLED:
            logger.info("[SKIP] Drift detection disabled via NBA_DRIFT_DETECTION_ENABLED")
            return

        enabled_markets = list(self.predictors.keys())
        for market in enabled_markets:
            try:
                self.drift_services[market] = DriftService(
                    market=market,
                    z_threshold=DRIFT_Z_THRESHOLD,
                )
                status = self.drift_services[market].get_status()
                if status["status"] == "ready":
                    logger.info(
                        f"[OK] Drift detection ready for {market} "
                        f"({status['features']} features)"
                    )
                else:
                    logger.warning(
                        f"[WARN] Drift detection not ready for {market}: {status['status']}"
                    )
            except (FileNotFoundError, ValueError, KeyError) as e:
                logger.warning(f"[WARN] Failed to initialize drift detection for {market}: {e}")

    def check_feature_drift(self, market: str, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Check if extracted features show drift from training distribution.

        Args:
            market: Market name (POINTS, REBOUNDS, etc.)
            features: Dict of feature name -> value

        Returns:
            Dict with drift check result
        """
        if market not in self.drift_services:
            return {"status": "disabled", "has_drift": False}

        try:
            result = self.drift_services[market].check_drift(features)
            return result.to_dict()
        except (ValueError, KeyError, TypeError) as e:
            logger.debug(f"Drift check failed for {market}: {e}")
            return {"status": "error", "has_drift": False, "error": str(e)}

    def aggregate_drift_results(self):
        """
        Aggregate drift results from all predictions into summary.

        Called after generate_picks to produce overall drift status.
        Runs batch drift detection on collected feature samples.
        """
        for market in self.predictors.keys():
            if market not in self.drift_services:
                self.drift_status[market] = {"status": "disabled"}
                continue

            service = self.drift_services[market]
            status = service.get_status()

            market_status = {
                "status": status["status"],
                "features_monitored": status.get("features", 0),
                "reference_created": status.get("reference_created"),
                "thresholds": status.get("thresholds", {}),
            }

            # Run batch drift detection if we have feature samples
            samples = self._feature_samples.get(market, [])
            if samples and status["status"] == "ready":
                try:
                    feature_df = pd.DataFrame(samples)
                    drift_result = service.check_batch_drift(feature_df)

                    market_status["batch_check"] = {
                        "samples_checked": len(samples),
                        "features_checked": drift_result.checked_features,
                        "drifted_count": len(drift_result.drifted_features),
                        "drift_percentage": round(drift_result.drift_percentage, 2),
                        "severity": drift_result.severity,
                        "drifted_features": drift_result.drifted_features[:5],  # Top 5
                    }

                    # Log drift warnings
                    if drift_result.severity in ("medium", "high"):
                        logger.warning(
                            f"[DRIFT] {market}: {drift_result.severity.upper()} drift detected "
                            f"({len(drift_result.drifted_features)} features, "
                            f"{drift_result.drift_percentage:.1f}%)"
                        )
                        for feat in drift_result.drifted_features[:3]:
                            details = drift_result.drift_details.get(feat, {})
                            logger.warning(f"  - {feat}: {details.get('reasons', ['unknown'])}")
                    elif drift_result.has_drift:
                        logger.info(
                            f"[DRIFT] {market}: Low drift detected "
                            f"({len(drift_result.drifted_features)} features)"
                        )

                except (ValueError, KeyError, TypeError) as e:
                    logger.debug(f"Batch drift check failed for {market}: {e}")
                    market_status["batch_check"] = {"status": "error", "error": str(e)}
            else:
                market_status["batch_check"] = {"status": "no_samples", "samples": len(samples)}

            self.drift_status[market] = market_status

    def load_opponent_defense_ranks(self):
        """Load opponent defense ranks from cheatsheet_data for display"""
        try:
            cursor = self.conn_intelligence.cursor()

            # Get opponent defense ranks by joining props with cheatsheet data
            cursor.execute(
                """
                SELECT DISTINCT
                    p.opponent_team,
                    p.stat_type,
                    c.opp_rank
                FROM nba_props_xl p
                JOIN cheatsheet_data c ON c.player_name = p.player_name
                    AND c.game_date = p.game_date
                    AND c.stat_type = p.stat_type
                WHERE p.game_date = %s
                  AND p.opponent_team IS NOT NULL
                  AND c.opp_rank IS NOT NULL
                  AND c.platform = 'underdog'
                  AND p.stat_type IN ('POINTS', 'REBOUNDS', 'ASSISTS')
            """,
                (self.game_date,),
            )

            for row in cursor.fetchall():
                opp_team, stat_type, opp_rank = row
                if opp_team and opp_rank:
                    key = f"{opp_team}_{stat_type}"
                    if key not in self.opp_rank_cache:
                        self.opp_rank_cache[key] = int(opp_rank)

            cursor.close()
            logger.info(f"[OK] Loaded {len(self.opp_rank_cache)} opponent defense ranks")
        except (psycopg2.Error, KeyError, TypeError, ValueError) as e:
            logger.warning(f"Could not load opponent defense ranks: {e}")
            # Rollback in case of error to reset transaction state
            try:
                self.conn_intelligence.rollback()
            except (psycopg2.Error, AttributeError):
                pass  # Connection may already be closed or invalid

    def get_opp_rank(self, opponent_team: str, stat_type: str) -> Optional[int]:
        """Get opponent defense rank for a team/stat combo"""
        if not opponent_team:
            return None
        key = f"{opponent_team}_{stat_type}"
        return self.opp_rank_cache.get(key)

    # Minutes/availability guard configuration
    MIN_MINUTES_FLOOR = 25.0
    MAX_DAYS_SINCE_GAME = 10
    # Allow UNKNOWN/DTD to avoid over-filtering when status feed is sparse, but still require minutes/recent game.
    ALLOWED_STATUSES = {"ACTIVE", "PROBABLE", "QUESTIONABLE", "UNKNOWN", "DAY_TO_DAY", "DTD"}

    # Bias adjustment configuration (optional, default OFF)
    ENABLE_BIAS_ADJUSTMENT = False
    BIAS_LOOKBACK_DAYS = 14
    BIAS_MIN_SAMPLES = 30
    BIAS_MAX_ADJ = {
        "POINTS": 1.5,
        "REBOUNDS": 1.0,
    }
    # Residual trimming thresholds
    BIAS_TRIM_LOWER_PCT = 0.10
    BIAS_TRIM_UPPER_PCT = 0.90
    BIAS_MIN_ABS_THRESHOLD = 0.2  # Ignore tiny shifts

    def load_player_status(self):
        """
        Load projected minutes, injury status, and last game date for gating.

        Uses injury_report from intelligence DB (updated daily) instead of
        the stale player_injury_status table.
        """
        # Query 1: Player info from players DB (profile, minutes, last game)
        player_query = """
        WITH last_games AS (
            SELECT
                pp.player_id,
                MAX(pgl.game_date) AS last_game_date
            FROM player_profile pp
            LEFT JOIN player_game_logs pgl ON pp.player_id = pgl.player_id
            GROUP BY pp.player_id
        )
        SELECT
            pp.player_id,
            pp.full_name,
            pmp.projected_mpg AS projected_minutes,
            lg.last_game_date
        FROM player_profile pp
        LEFT JOIN player_minutes_projections pmp ON pp.player_id = pmp.player_id
        LEFT JOIN last_games lg ON pp.player_id = lg.player_id;
        """

        # Query 2: Injury status for the target date from intelligence DB (injury_report)
        # Uses self.game_date (not CURRENT_DATE) to support historical backtesting
        injury_query = """
        SELECT player_id, status AS injury_status
        FROM injury_report
        WHERE report_date = %s;
        """

        df_players = pd.read_sql_query(player_query, self.conn_players)
        df_injuries = pd.read_sql_query(
            injury_query, self.conn_intelligence, params=(self.game_date,)
        )

        # Merge injury status into player data
        df = df_players.merge(df_injuries, on="player_id", how="left")
        df["injury_status"] = df["injury_status"].fillna("UNKNOWN")
        status_map = {}
        for _, row in df.iterrows():
            normalized = self.normalizer.normalize_name(row["full_name"])
            name_key = normalized.strip().lower()

            projected_minutes = (
                float(row["projected_minutes"]) if row["projected_minutes"] is not None else None
            )
            injury_status = str(row["injury_status"] or "UNKNOWN").upper()
            last_game = row["last_game_date"]
            if isinstance(last_game, (datetime, pd.Timestamp)):
                last_game = last_game.date()

            existing = status_map.get(name_key, {})

            # Prefer most recent game date to avoid stale overwrites from dup names
            existing_last_game = existing.get("last_game_date")
            if existing_last_game is None or (
                last_game and existing_last_game and last_game > existing_last_game
            ):
                last_game_chosen = last_game
            else:
                last_game_chosen = existing_last_game

            # Take max projected minutes when duplicates exist (safer floor)
            existing_minutes = existing.get("projected_minutes")
            if existing_minutes is None:
                minutes_chosen = projected_minutes
            elif projected_minutes is None:
                minutes_chosen = existing_minutes
            else:
                minutes_chosen = max(existing_minutes, projected_minutes)

            # Prefer a non-UNKNOWN injury status if available
            existing_status = existing.get("injury_status", "UNKNOWN")
            if existing_status == "UNKNOWN" and injury_status != "UNKNOWN":
                injury_chosen = injury_status
            else:
                injury_chosen = existing_status if existing_status != "UNKNOWN" else injury_status

            status_map[name_key] = {
                "projected_minutes": minutes_chosen,
                "injury_status": injury_chosen,
                "last_game_date": last_game_chosen,
            }

        self.player_status = status_map
        logger.info(f"[OK] Loaded player status for {len(self.player_status)} players")

    def _is_player_recent(self, status: Dict[str, Any]) -> bool:
        """
        Check if a player has played recently enough to avoid DNP risk.
        """
        last_game = status.get("last_game_date")
        if not last_game or not isinstance(last_game, (datetime, date)):
            return False
        delta_days = (self.game_date_obj - last_game).days
        return delta_days <= self.MAX_DAYS_SINCE_GAME

    def _should_skip_player(self, player_name: str) -> Optional[str]:
        """
        Determine if a player should be skipped based on minutes/availability guards.
        Returns None if OK, otherwise a string reason.
        """
        normalized = self.normalizer.normalize_name(player_name).strip().lower()
        status = self.player_status.get(normalized)
        if not status:
            return "no status"

        minutes = status.get("projected_minutes")
        injury_status = status.get("injury_status", "UNKNOWN")

        if minutes is None:
            return "no minutes projection"

        if minutes < self.MIN_MINUTES_FLOOR:
            return f"low minutes ({minutes:.1f})"

        if injury_status not in self.ALLOWED_STATUSES:
            return f"injury status {injury_status}"

        if not self._is_player_recent(status):
            return "stale last game"

        return None

    def compute_bias_adjustments(self) -> Dict[str, float]:
        """
        Compute rolling mean residuals (actual - predicted) over recent days and
        return clamped adjustments per market.
        """
        if not self.ENABLE_BIAS_ADJUSTMENT:
            return {}

        predictions_dir = Path(__file__).parent / "predictions"
        if not predictions_dir.exists():
            return {}

        start_date = self.game_date_obj - timedelta(days=self.BIAS_LOOKBACK_DAYS)
        residuals: Dict[str, list] = {}

        # Collect files within window
        for file in predictions_dir.glob("xl_picks_*.json"):
            try:
                file_date_str = file.stem.replace("xl_picks_", "")
                cleaned = file_date_str.replace("-", "")
                file_date_obj = datetime.strptime(cleaned, "%Y%m%d").date()
            except ValueError:
                continue

            if not (start_date <= file_date_obj < self.game_date_obj):
                continue

            with open(file, "r") as f:
                payload = json.load(f)

            # Fetch actuals once per date
            actuals = self._get_actuals_for_date(file_date_obj)

            for pick in payload.get("picks", []):
                stat_type = pick.get("stat_type")
                player = pick.get("player_name")
                prediction = pick.get("prediction")
                player_actuals = actuals.get(player.strip().lower(), {})
                actual = player_actuals.get(stat_type)
                if stat_type not in PRODUCTION_CONFIG or actual is None or prediction is None:
                    continue
                residuals.setdefault(stat_type, []).append(actual - prediction)

        adjustments = {}
        for market, vals in residuals.items():
            if len(vals) < self.BIAS_MIN_SAMPLES:
                continue
            arr = np.array(vals, dtype=float)
            # Trim outliers using percentiles
            lower = np.percentile(arr, self.BIAS_TRIM_LOWER_PCT * 100)
            upper = np.percentile(arr, self.BIAS_TRIM_UPPER_PCT * 100)
            trimmed = arr[(arr >= lower) & (arr <= upper)]
            if len(trimmed) < self.BIAS_MIN_SAMPLES:
                trimmed = arr  # fall back if trimming removes too much

            mean_resid = float(np.mean(trimmed))
            # Ignore tiny shifts
            if abs(mean_resid) < self.BIAS_MIN_ABS_THRESHOLD:
                continue

            max_adj = self.BIAS_MAX_ADJ.get(market, 1.0)
            adjustments[market] = max(-max_adj, min(max_adj, mean_resid))

        if adjustments:
            logger.info(f"Bias adjustments (last {self.BIAS_LOOKBACK_DAYS}d): {adjustments}")
        else:
            logger.info("Bias adjustments disabled or insufficient samples")

        return adjustments

    def _get_actuals_for_date(self, target_date: date) -> Dict[str, Dict[str, float]]:
        """
        Return a mapping of player_name -> {STAT: value} for the target date.
        """
        query = """
            SELECT
                pp.full_name,
                l.points,
                l.rebounds,
                l.assists,
                l.three_pointers_made
            FROM player_game_logs l
            JOIN player_profile pp ON l.player_id = pp.player_id
            WHERE l.game_date = %s
        """
        df = pd.read_sql_query(query, self.conn_players, params=(target_date,))
        results = {}
        for _, row in df.iterrows():
            name_key = row["full_name"].strip().lower()
            results[name_key] = {
                "POINTS": row["points"],
                "REBOUNDS": row["rebounds"],
                "ASSISTS": row["assists"],
                "THREES": row["three_pointers_made"],
            }
        return results

    def query_todays_props(self) -> pd.DataFrame:
        """
        Query multi-book props for today from nba_props_xl.

        Returns aggregated props with:
        - player_id, player_name, stat_type
        - opponent_team, is_home
        - consensus_line, min_line, max_line, line_spread
        - num_books
        """
        logger.info(f"Querying props for {self.game_date}...")

        enabled_markets = [k for k, v in PRODUCTION_CONFIG.items() if v.get("enabled", False)]
        markets_str = ", ".join([f"'{m}'" for m in enabled_markets])

        # In backtest mode, include inactive props (historical data)
        # In production, only use active props
        is_active_filter = "" if self.backtest_mode else "AND is_active = true"

        # Standard-only mode: exclude PrizePicks alternate lines (goblin/demon)
        # These weren't in training data (added Feb 2026)
        standard_only_filter = ""
        if self.standard_only:
            from nba.betting_xl.line_optimizer import PRIZEPICKS_ALT_BOOKS

            excluded_books = ", ".join([f"'{b}'" for b in PRIZEPICKS_ALT_BOOKS])
            standard_only_filter = f"AND book_name NOT IN ({excluded_books})"

        query = f"""
        WITH latest_props AS (
            SELECT
                player_id,
                player_name,
                game_date,
                stat_type,
                book_name,
                over_line,
                opponent_team,
                is_home,
                fetch_timestamp,
                ROW_NUMBER() OVER (
                    PARTITION BY player_id, stat_type, book_name
                    ORDER BY
                        CASE WHEN opponent_team != '' THEN 0 ELSE 1 END,  -- Prioritize props with opponent
                        fetch_timestamp DESC
                ) as rn
            FROM nba_props_xl
            WHERE game_date = %s
                AND stat_type IN ({markets_str})
                {is_active_filter}
                {standard_only_filter}
                AND over_line IS NOT NULL
        ),
        filtered_props AS (
            SELECT
                player_id, player_name, game_date, stat_type,
                book_name, over_line, opponent_team, is_home
            FROM latest_props
            WHERE rn = 1
        ),
        aggregated AS (
            SELECT
                MIN(player_id) as player_id,  -- Pick any ID (they're all the same player)
                player_name,
                game_date,
                stat_type,
                opponent_team,  -- Group by opponent to match validation
                is_home,  -- Group by is_home to match validation
                AVG(over_line) as consensus_line,
                MIN(over_line) as min_line,
                MAX(over_line) as max_line,
                MAX(over_line) - MIN(over_line) as line_spread,
                COUNT(*) as num_books
            FROM filtered_props
            GROUP BY player_name, game_date, stat_type, opponent_team, is_home
            HAVING COUNT(*) >= 3  -- Require ≥3 books when allowing identical lines
        )
        SELECT * FROM aggregated
        ORDER BY stat_type, player_name;
        """

        df = pd.read_sql_query(query, self.conn_intelligence, params=(self.game_date,))
        logger.info(f"[DATA] Found {len(df)} props for {self.game_date}")

        # CRITICAL DATE VALIDATION: Verify props are for correct date
        if len(df) > 0:
            unique_dates = df["game_date"].unique()
            wrong_dates = [d for d in unique_dates if str(d) != self.game_date]

            if wrong_dates:
                logger.error(f"[ERROR] CRITICAL: Found props for WRONG dates!")
                logger.error(f"   Expected: {self.game_date}")
                logger.error(f"   Found: {wrong_dates}")
                logger.error(f"   Props by date: {df['game_date'].value_counts().to_dict()}")
                raise ValueError(
                    f"Date validation failed: Expected props for {self.game_date}, "
                    f"but found {len(df[df['game_date'] != self.game_date])} props for other dates: {wrong_dates}"
                )

            logger.info(
                f"[OK] Date validation passed: All {len(df)} props are for {self.game_date}"
            )

        # CRITICAL MATCHUP DATA VALIDATION: Verify opponent_team and is_home are populated
        if len(df) > 0:
            missing_opponent = df["opponent_team"].isna().sum()
            missing_is_home = df["is_home"].isna().sum()
            missing_matchup = df["opponent_team"].isna() | df["is_home"].isna()
            missing_count = missing_matchup.sum()
            total_props = len(df)
            coverage_pct = 100.0 * (total_props - missing_count) / total_props

            if missing_count > 0:
                logger.warning(
                    f"[WARN]  Matchup data coverage: {coverage_pct:.1f}% ({total_props - missing_count}/{total_props})"
                )
                logger.warning(f"   Missing opponent_team: {missing_opponent}")
                logger.warning(f"   Missing is_home: {missing_is_home}")

            # Fail if coverage is below threshold (relaxed in backtest mode)
            min_coverage = 50.0 if self.backtest_mode else 95.0
            if coverage_pct < min_coverage:
                logger.error("")
                logger.error("=" * 80)
                logger.error("[ERROR] CRITICAL: Insufficient matchup data coverage!")
                logger.error("=" * 80)
                logger.error(f"Coverage: {coverage_pct:.1f}% (need ≥{min_coverage:.0f}%)")
                logger.error(
                    f"Missing: {missing_count}/{total_props} props without opponent_team or is_home"
                )
                logger.error("")
                logger.error("ACTION REQUIRED:")
                logger.error(
                    f"  Run: python3 enrich_props_with_matchups.py --date {self.game_date}"
                )
                logger.error("")
                logger.error("Without matchup data, the line optimizer cannot find books,")
                logger.error("resulting in 0 actionable picks even if good edges exist.")
                logger.error("=" * 80)
                raise ValueError(
                    f"Matchup data coverage ({coverage_pct:.1f}%) below {min_coverage:.0f}% threshold. "
                    f"Run enrichment script before generating predictions."
                )

            if coverage_pct >= 99.0:
                logger.info(f"[OK] Matchup data coverage: {coverage_pct:.1f}% (excellent)")
            elif coverage_pct >= 90.0:
                logger.info(f"[OK] Matchup data coverage: {coverage_pct:.1f}% (good)")

        # DEBUG: Save all props to file for analysis
        script_dir = os.path.dirname(os.path.abspath(__file__))
        debug_file = os.path.join(
            script_dir, "predictions", f"debug_all_props_{self.game_date}.csv"
        )
        df.to_csv(debug_file, index=False)
        logger.debug("Saved all props to debug file", extra={"file": debug_file, "count": len(df)})

        return df

    def generate_picks(self):
        """Main prediction logic"""
        logger.info("\n" + "=" * 80)
        logger.info("GENERATING XL PREDICTIONS")
        logger.info("=" * 80)

        # Load player status and optional bias adjustments
        self.load_player_status()
        self.bias_adjustments = self.compute_bias_adjustments()

        # Query props
        props_df = self.query_todays_props()

        if len(props_df) == 0:
            logger.warning("[ERROR] No props found for today")
            return

        logger.info("Generating predictions", extra={"prop_count": len(props_df)})

        # Track skips for transparency
        skip_reasons = Counter()
        total_props = len(props_df)

        # Process each prop
        for idx, row in props_df.iterrows():
            if idx % 50 == 0 and idx > 0:
                logger.info(f"   Progress: {idx}/{len(props_df)}")

            stat_type = row["stat_type"]
            player_name = row["player_name"]

            # Skip if no predictors for this market
            if stat_type not in self.predictors or not self.predictors[stat_type]:
                continue

            try:
                # Minutes/availability guard
                skip_reason = self._should_skip_player(player_name)
                if skip_reason:
                    skip_reasons[skip_reason] += 1
                    continue

                # Extract features
                features = self.feature_extractor.extract_features(
                    player_name=player_name,
                    game_date=self.game_date,
                    is_home=row["is_home"],
                    opponent_team=row["opponent_team"],
                    line=row["min_line"],  # Use softest line for features
                    stat_type=stat_type,
                )

                # Guard against None features
                if features is None:
                    logger.warning(f"[WARN]  No features extracted for {player_name} {stat_type}")
                    continue

                # Validate features before prediction
                is_valid, missing_features = self.feature_extractor.validate_features(features)
                if not is_valid:
                    logger.warning(
                        f"[WARN]  Incomplete features for {player_name} {stat_type}: "
                        f"{len(missing_features)} issues - {missing_features[:3]}..."
                    )
                    continue  # Skip this prediction

                # Collect features for batch drift detection
                if stat_type not in self._feature_samples:
                    self._feature_samples[stat_type] = []
                self._feature_samples[stat_type].append(features.copy())

                # Convert numpy types to Python native types for psycopg2 compatibility
                opp_team = str(row["opponent_team"]) if pd.notna(row["opponent_team"]) else None
                home_flag = bool(row["is_home"]) if pd.notna(row["is_home"]) else None

                # ============================================================
                # RUN BOTH XL AND V3 MODELS (separate picks for each)
                # ============================================================
                for model_version, predictor in self.predictors[stat_type].items():
                    pred_result = predictor.predict(
                        features, row["min_line"], player_name=player_name, game_date=self.game_date
                    )

                    if pred_result is None:
                        continue

                    # Apply optional bias adjustment
                    bias_adj = (
                        self.bias_adjustments.get(stat_type, 0.0) if self.bias_adjustments else 0.0
                    )
                    adjusted_prediction = pred_result["predicted_value"] - bias_adj

                    # Find best line via line shopping
                    # Get avg minutes for tier logic (X = starters, Z = bench)
                    avg_minutes = features.get("ema_minutes_L5", 25.0)

                    optimized = self.line_optimizer.optimize_line(
                        player_name=player_name,
                        game_date=self.game_date,
                        stat_type=stat_type,
                        prediction=adjusted_prediction,
                        p_over=pred_result["p_over"],
                        opponent_team=opp_team,
                        is_home=home_flag,
                        underdog_only=self.underdog_only,
                        avg_minutes=avg_minutes,
                    )

                    if optimized is not None:
                        side = optimized.get("direction", "OVER")
                        pick = {
                            "player_name": player_name,
                            "stat_type": stat_type,
                            "side": side,
                            "prediction": adjusted_prediction,
                            "p_over": optimized["p_over"],
                            "confidence": optimized["confidence"],
                            "filter_tier": optimized.get("filter_tier", "unknown"),
                            "avg_minutes": avg_minutes,  # For X/Z tier visibility
                            "consensus_line": optimized["consensus_line"],
                            "consensus_offset": optimized["consensus_offset"],
                            "line_spread": optimized["line_spread"],
                            "num_books": optimized["num_books"],
                            "opponent_team": optimized["opponent_team"],
                            "opp_rank": self.get_opp_rank(optimized["opponent_team"], stat_type),
                            "is_home": optimized["is_home"],
                            "top_3_lines": optimized["top_3_lines"],
                            "line_distribution": optimized.get("line_distribution", []),
                            "best_book": optimized["best_book"],
                            "best_line": optimized["best_line"],
                            "edge": optimized["edge"],
                            "edge_pct": (
                                (optimized["edge"] / optimized["best_line"] * 100)
                                if optimized["best_line"] > 0
                                else 0
                            ),
                            "p_under": optimized.get("p_under"),
                            "expected_wr": optimized.get("expected_wr"),
                            "model_version": model_version,  # 'xl' or 'v3'
                            "reasoning": self._generate_reasoning(
                                pred_result["predicted_value"],
                                optimized["best_line"],
                                optimized["line_spread"],
                                optimized["confidence"],
                                optimized.get("consensus_offset", 0),
                                side=side,
                            ),
                        }

                        hit_rates = self._get_hit_rate(player_name, stat_type)
                        if hit_rates:
                            pick["hit_rates"] = hit_rates
                        else:
                            # Compute hit rates from features if BettingPros data unavailable
                            pick["hit_rates"] = self._compute_hit_rates_from_features(
                                features, stat_type, optimized["best_line"]
                            )

                        # Add player context for Discord display
                        stat_key = stat_type.lower()
                        # Use L20 as proxy for season avg since season_avg feature doesn't exist
                        avg_L5 = float(features.get(f"ema_{stat_key}_L5") or 0)
                        avg_L10 = float(features.get(f"ema_{stat_key}_L10") or 0)
                        avg_L20 = float(features.get(f"ema_{stat_key}_L20") or 0)
                        minutes_L5 = float(features.get("ema_minutes_L5") or 0)
                        minutes_L10 = float(features.get("ema_minutes_L10") or 0)
                        h2h_games = int(features.get("h2h_games") or 0)
                        h2h_avg = float(features.get(f"h2h_avg_{stat_key}") or 0)

                        pick["player_context"] = {
                            "avg_L5": round(avg_L5, 1),
                            "avg_L10": round(avg_L10, 1),
                            "avg_L20": round(avg_L20, 1),  # Use L20 as season proxy
                            "minutes_L5": round(minutes_L5, 1),
                            "minutes_L10": round(minutes_L10, 1),
                            "h2h_games": h2h_games,
                            "h2h_avg": round(h2h_avg, 1),  # Head-to-head avg vs opponent
                            "trend": self._compute_trend(features, stat_key),
                        }

                        # Risk assessment with line softness (POINTS) or strict filtering (REBOUNDS)
                        opp_rank = self.get_opp_rank(optimized["opponent_team"], stat_type)
                        risk = self.risk_filter.assess_risk(
                            player_name=player_name,
                            stat_type=stat_type,
                            features=features,
                            opp_rank=opp_rank,
                            p_over=optimized["p_over"],
                            line=optimized["best_line"],
                            line_spread=optimized["line_spread"],
                            edge_pct=pick["edge_pct"],
                            prediction=adjusted_prediction,
                            consensus_line=optimized["consensus_line"],
                        )

                        # Add risk info to pick
                        pick["risk_level"] = risk.risk_level
                        pick["risk_score"] = round(risk.total_risk_score, 3)
                        pick["risk_flags"] = []
                        if risk.high_volatility:
                            pick["risk_flags"].append("HIGH_VOLATILITY")
                        if risk.elite_defense:
                            pick["risk_flags"].append("ELITE_DEFENSE")
                        if risk.negative_trend:
                            pick["risk_flags"].append("SLUMP")

                        # Add stake sizing info for all markets
                        pick["recommended_stake"] = risk.recommended_stake
                        pick["stake_reason"] = risk.stake_reason
                        pick["line_is_soft"] = risk.line_is_soft
                        pick["line_softness_score"] = round(risk.line_softness_score, 3)

                        # Skip based on market-specific logic
                        if risk.should_skip:
                            skip_reason = (
                                risk.stake_reason if stat_type == "POINTS" else "high_risk"
                            )
                            logger.debug(
                                f"[RISK] Skipping {player_name} {stat_type}: "
                                f"{risk.risk_level} ({skip_reason})"
                            )
                            skip_reasons["high_risk"] += 1
                            continue

                        # Flag risky picks
                        if risk.should_flag:
                            pick["risk_warning"] = self.risk_filter.format_risk_summary(risk)

                        self.picks.append(pick)

            except (psycopg2.Error, KeyError, TypeError, ValueError) as e:
                logger.debug(f"Skipped {player_name} {stat_type}: {e}")
                continue

        # NOTE: Odds API picks are now handled by standalone generate_odds_api_picks.py
        logger.info(f"\n[OK] Generated {len(self.picks)} actionable XL picks")
        if skip_reasons:
            logger.info(
                f"Skipped {sum(skip_reasons.values())}/{total_props} props due to guards: {dict(skip_reasons)}"
            )

        # Log risk filter summary
        risk_stats = self.risk_filter.get_stats()
        if risk_stats["high_risk_skipped"] > 0:
            logger.info(
                f"[RISK] Skipped {risk_stats['high_risk_skipped']} EXTREME risk picks "
                f"({risk_stats['skip_rate']:.1f}% of assessed)"
            )

        # Log flagged (high-risk but kept) picks
        flagged = [p for p in self.picks if p.get("risk_warning")]
        if flagged:
            logger.info(f"[RISK] {len(flagged)} picks flagged as HIGH risk (kept with warning):")
            for p in flagged[:5]:  # Show first 5
                logger.info(f"  - {p['player_name']} {p['stat_type']}: {p.get('risk_warning')}")

    def _generate_reasoning(
        self, prediction, line, line_spread, confidence, consensus_offset, side="OVER"
    ):
        """Generate human-readable reasoning for pick"""
        reasons = []

        if line_spread >= 2.5:
            reasons.append(f"High-spread goldmine ({line_spread:.1f} pts)")

        # Adjust reasoning based on direction
        if side == "UNDER":
            edge = line - prediction
            reasons.append(f"Model predicts {prediction:.1f} vs hardest line {line:.1f} (UNDER)")
        else:
            edge = prediction - line
            reasons.append(f"Model predicts {prediction:.1f} vs softest line {line:.1f}")

        if edge >= 5.0:
            reasons.append(f"Strong edge ({edge:.1f} pts)")

        # Warn if best line is suspiciously far from consensus
        if abs(consensus_offset) >= 1.5:
            line_type = "Hardest" if side == "UNDER" else "Softest"
            reasons.append(
                f"{line_type} line is {abs(consensus_offset):.1f} pts {'below' if consensus_offset < 0 else 'above'} consensus"
            )

        return ". ".join(reasons) + "."

    def _get_hit_rate(self, player_name: str, stat_type: str):
        record = self.hit_rate_cache.get(player_name, stat_type)
        if not record:
            return None

        hit_rates = record.get("hit_rates", {})
        samples = record.get("samples", {})
        enriched = {}

        for window, values in hit_rates.items():
            total = samples.get(window)
            if total is None:
                total = values.get("over", 0) + values.get("under", 0) + values.get("push", 0)

            enriched[window] = {
                "rate": values.get("rate"),
                "over": values.get("over", 0),
                "under": values.get("under", 0),
                "push": values.get("push", 0),
                "total": total,
            }

        return enriched

    def _compute_hit_rates_from_features(self, features: dict, stat_type: str, line: float) -> dict:
        """Get hit rates from BettingPros features or compute from rolling averages."""
        stat_key = stat_type.lower()
        hit_rates = {}

        # First try BettingPros hit rate features (most accurate)
        bp_l5 = features.get("bp_hit_rate_l5")
        bp_l15 = features.get("bp_hit_rate_l15")
        bp_season = features.get("bp_hit_rate_season")

        if bp_l5 is not None and float(bp_l5) > 0:
            hit_rates["last_5"] = {
                "rate": float(bp_l5),
                "total": 5,
            }

        if bp_l15 is not None and float(bp_l15) > 0:
            hit_rates["last_15"] = {
                "rate": float(bp_l15),
                "total": 15,
            }

        if bp_season is not None and float(bp_season) > 0:
            hit_rates["season"] = {
                "rate": float(bp_season),
                "total": 82,  # Approximate
            }

        # If no BettingPros data, estimate from rolling averages
        if not hit_rates and line > 0:
            avg_L5 = float(features.get(f"ema_{stat_key}_L5") or 0)
            avg_L10 = float(features.get(f"ema_{stat_key}_L10") or 0)
            avg_L20 = float(features.get(f"ema_{stat_key}_L20") or 0)

            if avg_L5 > 0:
                margin = (avg_L5 - line) / line
                hit_rates["last_5"] = {
                    "rate": min(0.95, max(0.05, 0.5 + margin * 0.5)),
                    "total": 5,
                    "estimated": True,
                }

            if avg_L10 > 0:
                margin = (avg_L10 - line) / line
                hit_rates["last_10"] = {
                    "rate": min(0.95, max(0.05, 0.5 + margin * 0.5)),
                    "total": 10,
                    "estimated": True,
                }

            if avg_L20 > 0:
                margin = (avg_L20 - line) / line
                hit_rates["season"] = {
                    "rate": min(0.95, max(0.05, 0.5 + margin * 0.5)),
                    "total": 20,
                    "estimated": True,
                }

        return hit_rates if hit_rates else None

    def _compute_trend(self, features: dict, stat_key: str) -> str:
        """Compute trend direction from rolling averages."""
        avg_L3 = features.get(f"ema_{stat_key}_L3", 0)
        avg_L5 = features.get(f"ema_{stat_key}_L5", 0)
        avg_L10 = features.get(f"ema_{stat_key}_L10", 0)

        if avg_L3 <= 0 or avg_L5 <= 0:
            return "STABLE"

        # Compare short-term to medium-term
        short_vs_medium = (avg_L3 - avg_L5) / avg_L5 if avg_L5 > 0 else 0
        medium_vs_long = (avg_L5 - avg_L10) / avg_L10 if avg_L10 > 0 else 0

        if short_vs_medium > 0.08:  # 8% above recent average
            return "HOT"
        elif short_vs_medium < -0.08:
            return "COLD"
        elif short_vs_medium > 0.03 and medium_vs_long > 0.03:
            return "RISING"
        elif short_vs_medium < -0.03 and medium_vs_long < -0.03:
            return "FALLING"
        else:
            return "STABLE"

    def save_picks(self, output_file: str, dry_run: bool = False):
        """Save picks to JSON file"""
        # DEDUPLICATION: Merge XL and V3 picks for same player/stat/side
        # When both models agree, mark as consensus and use best p_over
        # FIX Feb 4: Changed from separate picks to consensus-based merging
        original_count = len(self.picks)

        if original_count > 0:
            # Group by (player_name, stat_type, side) - WITHOUT model_version
            # When both models pick the same, merge into consensus pick
            pick_groups = {}
            for pick in self.picks:
                key = (
                    pick["player_name"],
                    pick["stat_type"],
                    pick.get("side", "OVER"),
                )
                if key not in pick_groups:
                    pick_groups[key] = []
                pick_groups[key].append(pick)

            # Merge picks: consensus flag + best p_over
            merged_picks = []
            consensus_count = 0
            for _key, picks in pick_groups.items():
                if len(picks) == 1:
                    # Single model pick
                    pick = picks[0]
                    pick["consensus"] = False
                    pick["models_agreeing"] = [pick.get("model_version", "xl")]
                    merged_picks.append(pick)
                else:
                    # Multiple models agree - merge into consensus pick
                    consensus_count += 1
                    # Use pick with best p_over
                    best_pick = max(picks, key=lambda p: p.get("p_over", 0))
                    best_pick["consensus"] = True
                    best_pick["models_agreeing"] = sorted(
                        list(set(p.get("model_version", "xl") for p in picks))
                    )
                    # Store both p_over values for reference
                    best_pick["p_over_by_model"] = {
                        p.get("model_version", "xl"): p.get("p_over") for p in picks
                    }
                    # Use model_version of the best pick but note it's consensus
                    merged_picks.append(best_pick)

            self.picks = merged_picks
            duplicates_merged = original_count - len(self.picks)

            if duplicates_merged > 0:
                logger.info(
                    f"[OK] Merged {duplicates_merged} duplicate picks into {consensus_count} consensus picks"
                )
                logger.info(
                    f"   Original: {original_count} picks → Merged: {len(self.picks)} picks "
                    f"({consensus_count} consensus)"
                )

        tier_counts = Counter([p.get("filter_tier", "unknown") for p in self.picks])

        # Identify star tier picks for special display
        from nba.betting_xl.line_optimizer import STAR_PLAYERS

        star_picks = [p for p in self.picks if p.get("filter_tier") == "star_tier"]
        star_player_picks = [p for p in self.picks if p.get("player_name") in STAR_PLAYERS]

        # Count picks by model version and consensus
        consensus_picks = [p for p in self.picks if p.get("consensus", False)]
        xl_only_picks = [
            p for p in self.picks if not p.get("consensus") and p.get("model_version") == "xl"
        ]
        v3_only_picks = [
            p for p in self.picks if not p.get("consensus") and p.get("model_version") == "v3"
        ]

        output = {
            "generated_at": datetime.now().isoformat(),
            "date": self.game_date,
            "strategy": "XL + V3 Line Shopping (Softest Line, Consensus Merged)",
            "markets_enabled": list(self.predictors.keys()),
            "total_picks": len(self.picks),
            "picks": self.picks,
            "summary": {
                "total": len(self.picks),
                "consensus": len(consensus_picks),
                "xl_only": len(xl_only_picks),
                "v3_only": len(v3_only_picks),
                "by_market": {
                    market: len([p for p in self.picks if p["stat_type"] == market])
                    for market in self.predictors.keys()
                },
                "by_model": {
                    "consensus": len(consensus_picks),
                    "xl_only": len(xl_only_picks),
                    "v3_only": len(v3_only_picks),
                },
                "high_confidence": len([p for p in self.picks if p["confidence"] == "HIGH"]),
                "avg_edge": round(np.mean([p["edge"] for p in self.picks]), 2) if self.picks else 0,
                "avg_line_spread": (
                    round(np.mean([p["line_spread"] for p in self.picks]), 2) if self.picks else 0
                ),
                "by_tier": dict(tier_counts),
                # Star player metrics
                "star_tier_picks": len(star_picks),
                "star_player_picks_total": len(star_player_picks),
                "star_players": [p["player_name"] for p in star_player_picks],
            },
            "expected_performance": {
                "POINTS": {"win_rate": 56.7, "roi": 8.27},
                "REBOUNDS": {"win_rate": 61.2, "roi": 16.96},
                "overall_line_shopping": {"win_rate": 54.5, "roi": 4.16},
                "high_spread_goldmine": {"win_rate": 70.6, "roi": 34.82},
                "star_tier": {
                    "win_rate": 52.5,
                    "roi": 0.5,
                    "note": "Lower edge but higher user engagement",
                },
            },
            "drift_status": self.drift_status if self.drift_status else {"status": "disabled"},
        }

        if dry_run:
            logger.info("DRY RUN MODE - No file saved")
            logger.info("Tier breakdown", extra={"tiers": dict(tier_counts)})
            self._print_summary(output)
            return

        # Save to file
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        logger.info(
            "Saved picks to file", extra={"count": len(self.picks), "path": str(output_path)}
        )
        logger.info("Tier breakdown", extra={"tiers": dict(tier_counts)})
        self._print_summary(output)

    def _print_summary(self, output: dict):
        """Log formatted summary using structured logging"""
        # Log summary header
        logger.info(
            "XL Picks Summary",
            extra={
                "date": output["date"],
                "strategy": output["strategy"],
                "total_picks": output["total_picks"],
            },
        )

        # Log market breakdown
        logger.info(
            "Picks by market",
            extra={
                "by_market": output["summary"]["by_market"],
                "high_confidence": output["summary"]["high_confidence"],
                "avg_edge": output["summary"]["avg_edge"],
                "avg_line_spread": output["summary"]["avg_line_spread"],
            },
        )

        # Log tier breakdown
        tier_counts = output["summary"].get("by_tier", {})
        if tier_counts:
            logger.info("Tier breakdown", extra={"tiers": tier_counts})

            # V3 summary
            v3_over = tier_counts.get("V3_ELITE_OVER", 0) + tier_counts.get("V3_STANDARD_OVER", 0)
            v3_under = tier_counts.get("V3_ELITE_UNDER", 0) + tier_counts.get(
                "V3_STANDARD_UNDER", 0
            )
            if v3_over + v3_under > 0:
                logger.info(
                    "V3 direction summary", extra={"v3_over": v3_over, "v3_under": v3_under}
                )

        # Log star players in picks
        star_players = output["summary"].get("star_players", [])
        if star_players:
            logger.info("Star players included", extra={"players": star_players})

        # Log drift detection status
        drift_status = output.get("drift_status", {})
        if drift_status and drift_status.get("status") != "disabled":
            for market, status in drift_status.items():
                if isinstance(status, dict) and status.get("status") == "ready":
                    logger.info(
                        f"Drift detection: {market}",
                        extra={
                            "market": market,
                            "features_monitored": status.get("features_monitored", 0),
                        },
                    )

        # Log top 5 picks
        if output["picks"]:
            unique_picks = []
            seen_keys = set()
            for pick in sorted(output["picks"], key=lambda x: x["edge"], reverse=True):
                key = (pick["player_name"], pick["stat_type"], pick["side"])
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                unique_picks.append(pick)
                if len(unique_picks) == 5:
                    break

            for i, pick in enumerate(unique_picks, 1):
                tier = pick.get("filter_tier", "unknown").upper()

                # Build line distribution info
                line_info = {}
                if "line_distribution" in pick and len(pick["line_distribution"]) > 0:
                    line_dist = pick["line_distribution"]
                    is_under = pick["side"] == "UNDER"

                    if is_under:
                        best = line_dist[-1]
                        worst = line_dist[0]
                    else:
                        best = line_dist[0]
                        worst = line_dist[-1]

                    line_info = {
                        "best_book": best["books"][0],
                        "best_line": best["line"],
                        "best_edge": best["edge"],
                        "worst_book": worst["books"][0] if len(line_dist) >= 2 else None,
                        "worst_line": worst["line"] if len(line_dist) >= 2 else None,
                    }

                logger.info(
                    f"Top pick #{i}",
                    extra={
                        "rank": i,
                        "player": pick["player_name"],
                        "stat_type": pick["stat_type"],
                        "side": pick["side"],
                        "prediction": round(pick["prediction"], 1),
                        "consensus_line": round(pick["consensus_line"], 1),
                        "line_spread": round(pick["line_spread"], 1),
                        "tier": tier,
                        "confidence": pick["confidence"],
                        "edge": round(pick["edge"], 2),
                        **line_info,
                    },
                )

                # Log warning for large consensus offset
                if abs(pick["consensus_offset"]) >= 1.5:
                    direction = "below" if pick["consensus_offset"] < 0 else "above"
                    logger.warning(
                        "Large consensus offset detected",
                        extra={
                            "player": pick["player_name"],
                            "offset": abs(pick["consensus_offset"]),
                            "direction": direction,
                        },
                    )

    def run(self, output_file: str, dry_run: bool = False):
        """Main execution"""
        try:
            logger.info("\n" + "=" * 80)
            logger.info("NBA XL DAILY PREDICTIONS GENERATOR")
            logger.info("=" * 80)
            logger.info(f"Date: {self.game_date}")
            logger.info(f"Strategy: Line Shopping (Validated 54.5% WR, +4.16% ROI)")
            logger.info("=" * 80 + "\n")

            self.connect_databases()
            self.load_models()
            self.initialize_components()
            self.initialize_drift_detection()
            self.load_opponent_defense_ranks()
            # NOTE: Odds API picks are now standalone (generate_odds_api_picks.py)
            self.generate_picks()
            self.aggregate_drift_results()

            if len(self.picks) == 0:
                logger.warning("\n[ERROR] No actionable picks found")
                logger.info("   Possible reasons:")
                logger.info("   - No props available for today")
                logger.info("   - All props filtered by edge thresholds")
                logger.info("   - Database connectivity issues")
                return

            self.save_picks(output_file, dry_run=dry_run)

            logger.info("\n[OK] XL predictions complete!")

        finally:
            # Cleanup
            if self.feature_extractor:
                self.feature_extractor.close()
            if self.line_optimizer:
                self.line_optimizer.close()
            if self.conn_intelligence:
                self.conn_intelligence.close()
            if self.conn_players:
                self.conn_players.close()


def main():
    parser = argparse.ArgumentParser(description="NBA XL Daily Predictions with Line Shopping")
    parser.add_argument(
        "--date", default=datetime.now().strftime("%Y-%m-%d"), help="Game date (YYYY-MM-DD)"
    )
    parser.add_argument("--output", default=None, help="Output JSON file path")
    parser.add_argument(
        "--dry-run", action="store_true", help="Generate predictions without saving"
    )
    parser.add_argument(
        "--backtest-mode",
        action="store_true",
        help="Enable backtest mode (relaxed freshness checks)",
    )
    parser.add_argument(
        "--as-of-date",
        type=str,
        default=None,
        help="Historical date for backtesting (YYYY-MM-DD). Limits calibration data to before this date.",
    )
    parser.add_argument(
        "--predictions-dir",
        type=str,
        default=None,
        help="Directory for calibrator to read predictions from (default: standard predictions/)",
    )
    parser.add_argument(
        "--underdog-only",
        action="store_true",
        help="Only accept props where Underdog is softest (POINTS: spread>=2.0, REBOUNDS: spread>=1.0). Dec 2025: ~60%% WR",
    )
    add_logging_args(parser)  # Adds --debug and --quiet flags
    args = parser.parse_args()

    # Setup unified logging with JSON structured format
    import logging

    setup_logging(
        "xl_predictions",
        level=logging.DEBUG if args.debug else None,
        quiet=args.quiet,
        console_format="json" if getattr(args, "log_json", False) else "text",
    )
    logger.info("Starting XL predictions", extra={"date": args.date})

    # Star player tier is ALWAYS ENABLED by default in PRODUCTION_CONFIG
    # No CLI flags needed - star_tier.enabled = True in line_optimizer.py
    logger.info("Star player tier enabled (default)")

    # Default output path
    if not args.output:
        predictions_dir = Path(__file__).parent / "predictions"
        args.output = predictions_dir / f"xl_picks_{args.date}.json"

    # Parse as_of_date if provided
    as_of_date = None
    if args.as_of_date:
        as_of_date = datetime.strptime(args.as_of_date, "%Y-%m-%d")
        logger.info("Backtest as_of_date set", extra={"as_of_date": args.as_of_date})

    # Enable backtest_mode automatically if as_of_date is in the past
    backtest_mode = args.backtest_mode
    if as_of_date and as_of_date.date() < datetime.now().date():
        backtest_mode = True
        logger.info("Backtest mode enabled (as_of_date is in the past)")

    # Log underdog-only mode if enabled
    if args.underdog_only:
        logger.info("Underdog-only mode enabled: Only accepting props where Underdog is softest")

    # Run generator
    # FIX Jan 15: Only pass underdog_only if explicitly set to True, otherwise let global config decide
    generator = XLPredictionsGenerator(
        game_date=args.date,
        as_of_date=as_of_date,
        backtest_mode=backtest_mode,
        predictions_dir=args.predictions_dir,
        underdog_only=True if args.underdog_only else None,  # None = use global UNDERDOG_ONLY_MODE
    )
    generator.run(output_file=args.output, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
