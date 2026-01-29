#!/usr/bin/env python3
"""
JSON Calibrator - Calibration System Using REAL Model Predictions
==================================================================

Key Improvement over DynamicCalibrator:
- Uses actual p_over from prediction JSON files (NOT reconstructed from line spread)
- Joins to nba_props_xl.actual_value for ground truth outcomes
- Provides accurate calibration by probability bucket

The DynamicCalibrator was training on FAKE predictions because:
1. It reconstructed probabilities using: sigmoid((consensus_line - over_line) / 2.0)
2. This is NOT the actual model prediction (p_over)
3. Calibration adjustments were based on wrong baseline

This JSONCalibrator fixes that by:
1. Reading actual p_over from prediction JSON files
2. Matching to database actuals via (player_name, game_date, stat_type)
3. Computing real calibration metrics
4. Applying proper bucket-based adjustments

Usage:
    from json_calibrator import JSONCalibrator

    calibrator = JSONCalibrator(market='POINTS', lookback_days=14)
    metrics = calibrator.get_recent_performance()
    adjusted = calibrator.adjust_probability(raw_prob=0.65)

Author: Claude Code
Date: January 2026
"""

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psycopg2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default paths (relative to project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent
PREDICTIONS_DIR = Path(
    os.getenv("NBA_PREDICTIONS_DIR", PROJECT_ROOT / "nba" / "betting_xl" / "predictions")
)
CALIBRATION_LOG_DIR = Path(
    os.getenv("NBA_CALIBRATION_LOG_DIR", PROJECT_ROOT / "nba" / "models" / "calibration_logs")
)

# Database configuration (use environment variables with fallbacks)
DB_INTELLIGENCE = {
    "host": os.getenv("NBA_INT_DB_HOST", "localhost"),
    "port": int(os.getenv("NBA_INT_DB_PORT", 5539)),
    "user": os.getenv(
        "NBA_INT_DB_USER", os.getenv("NBA_DB_USER", os.getenv("DB_USER", "nba_user"))
    ),
    "password": os.getenv(
        "NBA_INT_DB_PASSWORD", os.getenv("NBA_DB_PASSWORD", os.getenv("DB_PASSWORD"))
    ),
    "database": os.getenv("NBA_INT_DB_NAME", "nba_intelligence"),
}


class JSONCalibrator:
    """
    Calibration system that uses REAL model predictions from JSON files.

    Maintains same interface as DynamicCalibrator for drop-in replacement.
    """

    # Configuration constants
    # FIX Jan 2, 2026: Increased MAX_ADJUSTMENT from 0.15 to 0.25 to handle 20%+ overconfidence
    # Root cause: December POINTS had 19.9% bias but calibrator could only correct 15%
    MAX_ADJUSTMENT = 0.25  # Maximum +/- 25% probability adjustment
    # FIX: Lowered MIN_SAMPLES from 50 to 30 to start learning sooner
    MIN_SAMPLES = 30  # Minimum predictions before adjusting
    # FIX Jan 2, 2026: Reduced lookback from 35 to 21 days for faster regime detection
    # Root cause: 35-day lookback was learning from stale November data while December collapsed
    DEFAULT_LOOKBACK = 21  # Default lookback period in days (3 weeks - faster adaptation)
    STALENESS_DAYS = (
        7  # Data is stale if no actuals in this many days (increased for sparse XL picks)
    )
    MIN_SAMPLES_PER_BUCKET = 3  # Minimum samples per calibration bucket (lowered from 5)

    # Market-specific settings (Jan 2, 2026)
    # POINTS is more volatile and showed 2.7x more overconfidence than REBOUNDS
    # UPDATE Jan 2 PM: Lowered thresholds so learner can actually learn (was not meeting thresholds)
    MARKET_CONFIG = {
        "POINTS": {
            "lookback_days": 21,  # Extended from 14 to get more samples
            "max_adjustment": 0.25,  # Full adjustment capacity (was 19.9% overconfident)
            "min_samples": 10,  # Lowered from 15 - XL Tier X picks are sparse
        },
        "REBOUNDS": {
            "lookback_days": 21,  # Standard lookback - more stable market
            "max_adjustment": 0.15,  # Lower cap - only 7.5% overconfident
            "min_samples": 30,  # Standard threshold
        },
        "ASSISTS": {
            "lookback_days": 21,
            "max_adjustment": 0.25,
            "min_samples": 30,
        },
        "THREES": {
            "lookback_days": 21,
            "max_adjustment": 0.25,
            "min_samples": 30,
        },
    }

    # Probability buckets for calibration analysis
    # FIX: Last bucket uses 1.01 to include p=1.0 exactly
    PROB_BUCKETS = [
        (0.50, 0.55),
        (0.55, 0.60),
        (0.60, 0.65),
        (0.65, 0.70),
        (0.70, 0.75),
        (0.75, 0.80),
        (0.80, 0.85),
        (0.85, 0.90),
        (0.90, 1.01),  # Include 1.0 exactly
    ]

    # Breakeven win rate at -110 odds
    BREAKEVEN_RATE = 0.524

    def __init__(
        self,
        market: str,
        lookback_days: int = None,  # Now defaults to market-specific value
        predictions_dir: str = None,
        as_of_date: datetime = None,
        db_only: bool = False,
        model_version: str = "xl",  # 'xl' or 'v3' - filter calibration by model
    ):
        """
        Initialize JSON-based calibrator for a specific market.

        Args:
            market: 'POINTS', 'REBOUNDS', 'ASSISTS', or 'THREES'
            lookback_days: Number of days to look back for performance metrics
                          If None, uses market-specific default from MARKET_CONFIG
            predictions_dir: Directory containing prediction JSON files
            as_of_date: Historical date for backtesting (default: None = use datetime.now())
            db_only: If True, load predictions from database instead of JSON files
                    (useful for backtesting when JSON files don't exist)
            model_version: 'xl' or 'v3' - only use predictions from this model for calibration
        """
        self.market = market.upper()
        self.model_version = model_version.lower() if model_version else "xl"

        # Get market-specific config (or use defaults)
        market_cfg = self.MARKET_CONFIG.get(self.market, {})
        self._market_max_adjustment = market_cfg.get("max_adjustment", self.MAX_ADJUSTMENT)
        self._market_min_samples = market_cfg.get("min_samples", self.MIN_SAMPLES)

        # Use market-specific lookback if not explicitly provided
        if lookback_days is None:
            self.lookback_days = market_cfg.get("lookback_days", self.DEFAULT_LOOKBACK)
        else:
            self.lookback_days = lookback_days

        # BACKTEST SUPPORT: Store as_of_date and db_only mode
        # as_of_date=None means production mode (use datetime.now())
        self.as_of_date = as_of_date
        self.db_only = db_only

        # Reference date for lookback calculations
        # In production: datetime.now()
        # In backtest: the provided as_of_date
        self._reference_date = as_of_date if as_of_date else datetime.now()

        # Set predictions directory
        if predictions_dir is None:
            self.predictions_dir = PREDICTIONS_DIR
        else:
            self.predictions_dir = Path(predictions_dir)

        # Cache management
        self._metrics_cache = None
        self._cache_timestamp = None
        self._cache_ttl_seconds = 300  # 5 minute cache

        # Calibration state
        self._bucket_calibration = {}  # {bucket_label: adjustment_factor}
        self._global_bias = 0.0
        self._bias_correction = 0.0
        self._adjustment_factor = 0.0
        self._is_stale = True
        self._sample_count = 0

        # Adjustment log directory
        self._log_dir = CALIBRATION_LOG_DIR
        self._log_dir.mkdir(parents=True, exist_ok=True)

        mode_str = "db_only" if db_only else "json"
        date_str = as_of_date.strftime("%Y-%m-%d") if as_of_date else "now"
        logger.info(
            f"JSONCalibrator initialized for {self.market} model={self.model_version} "
            f"(lookback={self.lookback_days} days, max_adj={self._market_max_adjustment:.0%}, "
            f"min_samples={self._market_min_samples}, mode={mode_str}, as_of={date_str})"
        )

    def _is_backtest_mode(self) -> bool:
        """Check if running in backtest mode (historical date)."""
        if self.as_of_date is None:
            return False
        # Backtest if as_of_date is in the past
        return self.as_of_date.date() < datetime.now().date()

    def _get_db_connection(self):
        """Get database connection."""
        return psycopg2.connect(**DB_INTELLIGENCE)

    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid."""
        if self._metrics_cache is None or self._cache_timestamp is None:
            return False
        age = (datetime.now() - self._cache_timestamp).total_seconds()
        return age < self._cache_ttl_seconds

    def _return_no_data_metrics(self) -> Dict:
        """Return metrics when no data available."""
        self._is_stale = True
        self._sample_count = 0
        return {
            "status": "no_data",
            "market": self.market,
            "total_predictions": 0,
            "predictions_with_actuals": 0,
            "is_stale": True,
            "lookback_days": self.lookback_days,
            "timestamp": datetime.now().isoformat(),
        }

    def _load_predictions(self) -> List[Dict]:
        """
        Load predictions using appropriate method based on mode.

        In production (db_only=False): Load from JSON files
        In backtest (db_only=True): Load from database
        """
        if self.db_only:
            return self._load_predictions_from_db()
        else:
            return self._load_predictions_from_json()

    def _load_predictions_from_db(self) -> List[Dict]:
        """
        Load predictions from database for backtesting.

        Uses nba_props_xl table with actual_value to reconstruct calibration data.
        This is useful when JSON files don't exist for historical dates.
        """
        predictions = []

        # Use reference date (backtest date) for lookback
        end_date = self._reference_date
        start_date = end_date - timedelta(days=self.lookback_days)

        conn = None
        try:
            conn = self._get_db_connection()
            with conn.cursor() as cur:
                # Query props with actuals for calibration
                # We reconstruct p_over from line spread as a heuristic
                cur.execute(
                    """
                    SELECT DISTINCT ON (player_name, game_date)
                        player_name,
                        game_date,
                        stat_type,
                        over_line,
                        consensus_line,
                        actual_value,
                        line_spread
                    FROM nba_props_xl
                    WHERE stat_type = %s
                      AND game_date >= %s
                      AND game_date < %s
                      AND actual_value IS NOT NULL
                      AND over_line IS NOT NULL
                    ORDER BY player_name, game_date, fetch_timestamp DESC
                """,
                    (self.market, start_date.date(), end_date.date()),
                )

                for row in cur.fetchall():
                    (
                        player_name,
                        game_date,
                        stat_type,
                        over_line,
                        consensus_line,
                        actual_value,
                        line_spread,
                    ) = row

                    # Reconstruct p_over from line spread
                    # This is a heuristic: higher spread = more likely over
                    # Using sigmoid to map spread to probability
                    if line_spread and consensus_line:
                        spread = float(consensus_line) - float(over_line)
                        # sigmoid approximation: spread of 2 points = ~0.6 probability
                        import math

                        p_over = 1 / (1 + math.exp(-spread / 2.0))
                        p_over = max(0.50, min(0.85, p_over))  # Clamp to reasonable range
                    else:
                        p_over = 0.55  # Default if no spread info

                    predictions.append(
                        {
                            "player_name": player_name,
                            "game_date": (
                                game_date.strftime("%Y-%m-%d")
                                if hasattr(game_date, "strftime")
                                else str(game_date)
                            ),
                            "stat_type": stat_type,
                            "p_over": p_over,
                            "prediction": float(over_line) + 1.5,  # Rough estimate
                            "best_line": float(over_line),
                            "actual_value": float(actual_value) if actual_value else None,
                            "hit": float(actual_value) > float(over_line) if actual_value else None,
                        }
                    )

            logger.info(
                f"Loaded {len(predictions)} {self.market} predictions from DB "
                f"({start_date.date()} to {end_date.date()})"
            )

        except Exception as e:
            logger.error(f"Failed to load predictions from DB: {e}")
        finally:
            if conn:
                conn.close()

        return predictions

    def _load_predictions_from_json(self) -> List[Dict]:
        """
        Load all predictions for this market from JSON files in lookback period.

        Returns list of dicts with prediction data including actual p_over.
        """
        predictions = []

        # Use reference date instead of datetime.now() for backtest support
        start_date = self._reference_date - timedelta(days=self.lookback_days)
        end_date = self._reference_date

        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            date_str_compact = current_date.strftime("%Y%m%d")

            # Try multiple filename patterns
            json_files_to_try = [
                self.predictions_dir / f"xl_picks_{date_str}.json",
                self.predictions_dir / f"xl_picks_{date_str_compact}.json",
                self.predictions_dir / f"xl_picks_{date_str}_bias.json",
                self.predictions_dir / f"xl_picks_{date_str}_nobias.json",
            ]

            for json_file in json_files_to_try:
                if json_file.exists():
                    try:
                        with open(json_file) as f:
                            data = json.load(f)

                        # Get date from file or parent structure
                        file_date = data.get("date", date_str)

                        for pick in data.get("picks", []):
                            # Filter by market
                            if pick.get("stat_type") != self.market:
                                continue

                            # Filter by model version (V3 tier = v3 model, others = xl model)
                            # FIX Jan 15, 2026: Use startswith() to match V3_ELITE_*, V3_STANDARD_*, etc.
                            pick_tier = pick.get("filter_tier", "").upper()
                            is_v3_pick = pick_tier.startswith("V3")
                            if self.model_version == "v3":
                                # V3 calibrator only uses V3 tier predictions
                                if not is_v3_pick:
                                    continue
                            else:
                                # XL calibrator uses all non-V3 predictions
                                if is_v3_pick:
                                    continue

                            # Extract fields - THE KEY: using actual p_over!
                            predictions.append(
                                {
                                    "player_name": pick["player_name"],
                                    "game_date": file_date,
                                    "stat_type": pick["stat_type"],
                                    "p_over": float(pick["p_over"]),  # REAL prediction!
                                    "prediction": float(pick.get("prediction", 0)),
                                    "best_line": float(
                                        pick.get("best_line", pick.get("consensus_line", 0))
                                    ),
                                    "confidence": pick.get("confidence", "UNKNOWN"),
                                    "filter_tier": pick.get("filter_tier", "unknown"),
                                    "side": pick.get("side", "OVER"),
                                    "edge": float(pick.get("edge", 0)),
                                    "opponent_team": pick.get("opponent_team"),
                                    "source_file": json_file.name,
                                }
                            )

                        # Found file for this date, move to next
                        break

                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        logger.debug(f"Error parsing {json_file}: {e}")
                        continue

            current_date += timedelta(days=1)

        logger.info(
            f"Loaded {len(predictions)} {self.market} model={self.model_version} predictions "
            f"from {self.lookback_days} days of JSON files"
        )

        return predictions

    def _join_actuals_from_database(self, predictions: List[Dict]) -> List[Dict]:
        """
        Join predictions with actual_value from nba_props_xl.

        Match on (player_name, game_date, stat_type).
        """
        if len(predictions) == 0:
            return []

        # FIX: Safe connection handling to prevent leaks on errors
        conn = None
        cursor = None

        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            # Build unique keys
            player_names = list(set(p["player_name"] for p in predictions))
            game_dates = list(set(p["game_date"] for p in predictions))

            # Query all relevant actuals in batch
            query = """
                SELECT DISTINCT ON (player_name, game_date, stat_type)
                    player_name,
                    game_date::text,
                    stat_type,
                    actual_value
                FROM nba_props_xl
                WHERE player_name = ANY(%s)
                  AND game_date = ANY(%s::date[])
                  AND stat_type = %s
                  AND actual_value IS NOT NULL
                ORDER BY player_name, game_date, stat_type, fetch_timestamp DESC
            """

            cursor.execute(query, (player_names, game_dates, self.market))

            # Build lookup dict
            actuals_map = {}
            for row in cursor.fetchall():
                # Handle date format
                game_date_str = str(row[1])[:10]  # Take YYYY-MM-DD part
                key = (row[0], game_date_str, row[2])
                actuals_map[key] = float(row[3])

            # Enrich predictions with actuals
            enriched = []
            for pred in predictions:
                key = (pred["player_name"], pred["game_date"], pred["stat_type"])
                actual = actuals_map.get(key)

                if actual is not None:
                    line = pred["best_line"]

                    # FIX: Skip pushes (actual == line) - they're not wins or losses
                    if actual == line:
                        continue

                    pred_copy = pred.copy()
                    pred_copy["actual_value"] = actual

                    # Calculate outcome (did OVER hit?)
                    # Note: All picks in this system are OVER, default to OVER if side not specified
                    side = pred.get("side", "OVER")
                    if side == "OVER":
                        pred_copy["outcome"] = 1 if actual > line else 0
                    else:
                        pred_copy["outcome"] = 1 if actual < line else 0

                    enriched.append(pred_copy)

            logger.info(f"Matched {len(enriched)}/{len(predictions)} predictions to actuals")

            return enriched

        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    def _calculate_metrics(self, predictions: List[Dict]) -> Dict:
        """
        Calculate calibration and performance metrics from real predictions.
        """
        if len(predictions) == 0:
            return self._return_no_data_metrics()

        df = pd.DataFrame(predictions)

        # Basic metrics
        total = len(df)
        outcomes = df["outcome"].values
        predicted_probs = df["p_over"].values

        # Win rate
        win_rate = outcomes.mean()

        # Bias (positive = overconfident)
        avg_predicted = predicted_probs.mean()
        bias = avg_predicted - win_rate

        # Brier score (lower is better)
        brier_score = np.mean((predicted_probs - outcomes) ** 2)

        # Calibration error
        calibration_error = np.mean(np.abs(predicted_probs - outcomes))

        # Per-bucket calibration
        bucket_stats = {}
        for bucket_start, bucket_end in self.PROB_BUCKETS:
            mask = (predicted_probs >= bucket_start) & (predicted_probs < bucket_end)
            count = mask.sum()

            if count >= self.MIN_SAMPLES_PER_BUCKET:
                bucket_outcomes = outcomes[mask]
                bucket_probs = predicted_probs[mask]

                actual_rate = bucket_outcomes.mean()
                predicted_rate = bucket_probs.mean()
                bucket_bias = predicted_rate - actual_rate

                label = f"{bucket_start:.2f}-{bucket_end:.2f}"
                bucket_stats[label] = {
                    "count": int(count),
                    "predicted": float(predicted_rate),
                    "actual": float(actual_rate),
                    "bias": float(bucket_bias),
                    "adjustment": float(
                        np.clip(-bucket_bias, -self.MAX_ADJUSTMENT, self.MAX_ADJUSTMENT)
                    ),
                }

        # Staleness check
        # In backtest mode, use reference date instead of now()
        # Also skip staleness in backtest mode since data is historical
        latest_date = df["game_date"].max()
        try:
            reference = self._reference_date
            days_since_latest = (
                reference - datetime.strptime(str(latest_date)[:10], "%Y-%m-%d")
            ).days
        except (ValueError, TypeError, AttributeError):
            days_since_latest = 999

        # Skip staleness check in backtest mode (historical data is expected to be old)
        if self._is_backtest_mode():
            is_stale = False  # Never stale in backtest mode
        else:
            is_stale = days_since_latest > self.STALENESS_DAYS

        # Confidence assessment
        is_overconfident = bias > 0.02
        is_underconfident = bias < -0.02

        # Hit rates by side
        over_mask = predicted_probs > 0.5
        under_mask = predicted_probs <= 0.5
        hit_rate_over = outcomes[over_mask].mean() if over_mask.sum() > 0 else None
        hit_rate_under = (1 - outcomes[under_mask]).mean() if under_mask.sum() > 0 else None

        return {
            "status": "ok",
            "market": self.market,
            "total_predictions": total,
            "predictions_with_actuals": total,
            "win_rate": float(win_rate),
            "hit_rate_over": float(hit_rate_over) if hit_rate_over is not None else None,
            "hit_rate_under": float(hit_rate_under) if hit_rate_under is not None else None,
            "avg_predicted_prob": float(avg_predicted),
            "avg_actual_outcome": float(win_rate),
            "bias": float(bias),
            "calibration_error": float(calibration_error),
            "brier_score": float(brier_score),
            "calibration_by_bucket": bucket_stats,
            "is_overconfident": bool(is_overconfident),
            "is_underconfident": bool(is_underconfident),
            "latest_actual_date": str(latest_date),
            "days_since_latest": days_since_latest,
            "is_stale": bool(is_stale),
            "lookback_days": self.lookback_days,
            "over_predictions_count": int(over_mask.sum()),
            "under_predictions_count": int(under_mask.sum()),
            "timestamp": datetime.now().isoformat(),
        }

    def _update_calibration_state(self, metrics: Dict):
        """Update internal calibration state from metrics."""
        if metrics.get("status") != "ok":
            self._is_stale = True
            self._sample_count = 0
            self._bucket_calibration = {}
            self._global_bias = 0.0
            self._bias_correction = 0.0
            self._adjustment_factor = 0.0
            return

        self._sample_count = metrics["total_predictions"]
        self._is_stale = metrics["is_stale"]
        self._global_bias = metrics["bias"]

        # Store bucket calibration
        self._bucket_calibration = metrics.get("calibration_by_bucket", {})

        # Use market-specific min_samples and max_adjustment
        min_samples = self._market_min_samples
        max_adj = self._market_max_adjustment

        # Calculate adjustment parameters (same as DynamicCalibrator for compatibility)
        if self._sample_count >= min_samples and not self._is_stale:
            # Bias correction (use market-specific max_adjustment)
            self._bias_correction = float(np.clip(-self._global_bias, -max_adj, max_adj))

            # Performance-based adjustment (use market-specific max_adjustment)
            hit_rate = metrics.get("hit_rate_over", 0.5)
            if hit_rate is not None:
                hit_rate_gap = hit_rate - self.BREAKEVEN_RATE
                self._adjustment_factor = float(np.clip(hit_rate_gap * 0.5, -max_adj, max_adj))
            else:
                self._adjustment_factor = 0.0
        else:
            self._bias_correction = 0.0
            self._adjustment_factor = 0.0

        logger.info(
            f"JSONCalibrator state updated: samples={self._sample_count}, "
            f"bias={self._global_bias:.4f}, bias_correction={self._bias_correction:.4f}, "
            f"stale={self._is_stale}"
        )

    def get_recent_performance(self, force_refresh: bool = False) -> Dict:
        """
        Load predictions from JSON files and join with actuals from database.

        This uses REAL p_over values from JSON, not reconstructed!

        Returns dict with performance and calibration metrics.
        """
        # Check cache
        if self._is_cache_valid() and not force_refresh:
            return self._metrics_cache

        try:
            # Step 1: Load predictions (from JSON files or DB depending on mode)
            predictions = self._load_predictions()

            if len(predictions) == 0:
                return self._return_no_data_metrics()

            # Step 2: Get actual values from database
            predictions_with_actuals = self._join_actuals_from_database(predictions)

            if len(predictions_with_actuals) == 0:
                return self._return_no_data_metrics()

            # Step 3: Calculate performance metrics
            metrics = self._calculate_metrics(predictions_with_actuals)

            # Step 4: Update calibration state
            self._update_calibration_state(metrics)

            # Cache results
            self._metrics_cache = metrics
            self._cache_timestamp = datetime.now()

            return metrics

        except Exception as e:
            logger.error(f"Error in get_recent_performance: {e}")
            return self._return_no_data_metrics()

    def adjust_probability(self, raw_prob: float) -> float:
        """
        Apply calibration adjustment to a probability.

        Uses bucket-specific adjustments when available, otherwise global bias.

        Args:
            raw_prob: Calibrated probability from model (0.0 to 1.0)

        Returns:
            Adjusted probability (bounded 0.01 to 0.99)
        """
        # Ensure metrics are loaded
        if self._metrics_cache is None:
            self.get_recent_performance()

        # If stale or insufficient data, return unchanged (use market-specific min_samples)
        if self._is_stale or self._sample_count < self._market_min_samples:
            return float(np.clip(raw_prob, 0.01, 0.99))

        # Try bucket-specific adjustment first
        adjustment = None
        for bucket_label, stats in self._bucket_calibration.items():
            try:
                parts = bucket_label.split("-")
                bucket_start = float(parts[0])
                bucket_end = float(parts[1])

                if bucket_start <= raw_prob < bucket_end:
                    adjustment = stats["adjustment"]
                    break
            except (ValueError, KeyError):
                continue

        # If no bucket found, use global bias correction + performance adjustment
        if adjustment is None:
            # Same logic as DynamicCalibrator for compatibility
            adjustment = self._bias_correction
            confidence_scale = abs(raw_prob - 0.5) * 2
            adjustment += self._adjustment_factor * confidence_scale

        # Bound adjustment (use market-specific max_adjustment)
        adjustment = np.clip(adjustment, -self._market_max_adjustment, self._market_max_adjustment)

        # Apply adjustment
        adjusted = raw_prob + adjustment

        return float(np.clip(adjusted, 0.01, 0.99))

    def apply_adjustment(
        self, raw_prob: float, player_name: str = None, game_date: str = None, line: float = None
    ) -> Dict:
        """
        Apply adjustment and return full audit information.

        Maintains same interface as DynamicCalibrator for drop-in replacement.
        """
        adjusted = self.adjust_probability(raw_prob)
        total_adjustment = adjusted - raw_prob

        # Determine bucket used
        bucket_used = None
        for bucket_label in self._bucket_calibration.keys():
            try:
                parts = bucket_label.split("-")
                bucket_start = float(parts[0])
                bucket_end = float(parts[1])
                if bucket_start <= raw_prob < bucket_end:
                    bucket_used = bucket_label
                    break
            except (ValueError, KeyError):
                continue

        # Determine reason
        if self._is_stale:
            days = self._metrics_cache.get("days_since_latest", "?") if self._metrics_cache else "?"
            reason = f"No adjustment - data stale ({days} days)"
        elif self._sample_count < self._market_min_samples:
            reason = f"No adjustment - insufficient samples ({self._sample_count}/{self._market_min_samples})"
        elif abs(total_adjustment) < 0.001:
            reason = "No significant adjustment needed - model well calibrated"
        elif total_adjustment > 0:
            reason = f"Increased by {total_adjustment:.3f} (model was underconfident)"
        else:
            reason = f"Decreased by {abs(total_adjustment):.3f} (model was overconfident)"

        # Calculate performance adjustment component (for compatibility)
        confidence_scale = abs(raw_prob - 0.5) * 2
        performance_adjustment = self._adjustment_factor * confidence_scale

        result = {
            "raw_prob": float(raw_prob),
            "adjusted_prob": float(adjusted),
            "adjustment_applied": float(total_adjustment),
            "bias_correction": float(self._bias_correction),
            "performance_adjustment": float(performance_adjustment),
            "bucket_used": bucket_used,
            "was_adjusted": bool(abs(total_adjustment) >= 0.001),
            "reason": reason,
            "market": self.market,
            "player_name": player_name,
            "game_date": game_date,
            "line": float(line) if line is not None else None,
            "timestamp": datetime.now().isoformat(),
        }

        # Log adjustment
        self._log_adjustment(result)

        return result

    def _log_adjustment(self, result: Dict):
        """Log adjustment to file for audit trail."""
        log_file = (
            self._log_dir
            / f"{self.market.lower()}_json_adjustments_{datetime.now().strftime('%Y%m%d')}.jsonl"
        )
        try:
            with open(log_file, "a") as f:
                f.write(json.dumps(result) + "\n")
        except Exception as e:
            logger.warning(f"Failed to log adjustment: {e}")

    def get_adjustment_summary(self) -> Dict:
        """Get summary of adjustments applied today."""
        log_file = (
            self._log_dir
            / f"{self.market.lower()}_json_adjustments_{datetime.now().strftime('%Y%m%d')}.jsonl"
        )

        if not log_file.exists():
            return {"adjustments_today": 0, "avg_adjustment": 0.0, "market": self.market}

        adjustments = []
        try:
            with open(log_file, "r") as f:
                for line in f:
                    adjustments.append(json.loads(line))
        except Exception as e:
            logger.warning(f"Failed to read adjustment log: {e}")
            return {
                "adjustments_today": 0,
                "avg_adjustment": 0.0,
                "error": str(e),
                "market": self.market,
            }

        if len(adjustments) == 0:
            return {"adjustments_today": 0, "avg_adjustment": 0.0, "market": self.market}

        actual_adjustments = [a for a in adjustments if a.get("was_adjusted", False)]

        # FIX: Safely extract adjustment values, handling missing keys
        adj_values = [
            a.get("adjustment_applied", 0) for a in adjustments if "adjustment_applied" in a
        ]
        if not adj_values:
            return {
                "adjustments_today": len(adjustments),
                "actual_adjustments_applied": len(actual_adjustments),
                "avg_adjustment": 0.0,
                "max_increase": 0.0,
                "max_decrease": 0.0,
                "market": self.market,
                "warning": "no valid adjustment values found",
            }

        return {
            "adjustments_today": len(adjustments),
            "actual_adjustments_applied": len(actual_adjustments),
            "avg_adjustment": float(np.mean(adj_values)),
            "max_increase": float(max(adj_values)),
            "max_decrease": float(min(adj_values)),
            "market": self.market,
        }


class JSONCalibrationManager:
    """
    Manager for all market calibrators.

    Drop-in replacement for DynamicCalibrationManager.
    """

    MARKETS = ["POINTS", "REBOUNDS", "ASSISTS", "THREES"]

    def __init__(self, lookback_days: int = 14, predictions_dir: str = None):
        """Initialize calibrators for all markets."""
        self.lookback_days = lookback_days
        self.calibrators = {}

        for market in self.MARKETS:
            self.calibrators[market] = JSONCalibrator(market, lookback_days, predictions_dir)

    def get_all_performance(self, force_refresh: bool = False) -> Dict:
        """Get performance metrics for all markets."""
        return {
            market: cal.get_recent_performance(force_refresh)
            for market, cal in self.calibrators.items()
        }

    def adjust(self, market: str, raw_prob: float) -> float:
        """Adjust probability for a specific market."""
        market = market.upper()
        if market not in self.calibrators:
            raise ValueError(f"Unknown market: {market}")
        return self.calibrators[market].adjust_probability(raw_prob)

    def adjust_with_audit(
        self,
        market: str,
        raw_prob: float,
        player_name: str = None,
        game_date: str = None,
        line: float = None,
    ) -> Dict:
        """Adjust probability with full audit trail."""
        market = market.upper()
        if market not in self.calibrators:
            raise ValueError(f"Unknown market: {market}")
        return self.calibrators[market].apply_adjustment(raw_prob, player_name, game_date, line)

    def print_status_report(self):
        """Print status report for all markets."""
        print("\n" + "=" * 80)
        print("JSON CALIBRATION STATUS REPORT (Using REAL Predictions)")
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Lookback: {self.lookback_days} days")
        print("=" * 80)

        for market in self.MARKETS:
            metrics = self.calibrators[market].get_recent_performance()

            print(f"\n{market}:")
            if metrics.get("status") != "ok":
                print(f"  Status: {metrics.get('status', 'error')}")
                continue

            print(f"  Predictions (from JSON): {metrics['total_predictions']}")
            print(f"  Win Rate: {metrics['win_rate']*100:.1f}%")
            print(f"  Avg Predicted: {metrics['avg_predicted_prob']*100:.1f}%")
            print(f"  Bias: {metrics['bias']*100:+.1f}%")
            print(f"  Brier Score: {metrics['brier_score']:.4f}")

            if metrics["is_overconfident"]:
                print(f"  Status: OVERCONFIDENT (will reduce predictions)")
            elif metrics["is_underconfident"]:
                print(f"  Status: UNDERCONFIDENT (will increase predictions)")
            else:
                print(f"  Status: WELL CALIBRATED")

            if metrics["is_stale"]:
                print(f"  WARNING: Data is STALE ({metrics['days_since_latest']} days)")

            # Show bucket calibration
            if metrics.get("calibration_by_bucket"):
                print(f"  Calibration by bucket:")
                for bucket, stats in metrics["calibration_by_bucket"].items():
                    print(
                        f"    {bucket}: pred={stats['predicted']:.2f}, actual={stats['actual']:.2f}, "
                        f"adj={stats['adjustment']:+.3f} (n={stats['count']})"
                    )

        print("\n" + "=" * 80)


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="JSON Calibration System (Using REAL Predictions)")
    parser.add_argument("command", choices=["status", "test", "compare"], help="Command to run")
    parser.add_argument("--market", type=str, default="POINTS", help="Market to operate on")
    parser.add_argument("--lookback", type=int, default=14, help="Lookback period in days")
    parser.add_argument(
        "--prob", type=float, default=0.65, help="Probability to adjust (for test command)"
    )

    args = parser.parse_args()

    if args.command == "status":
        manager = JSONCalibrationManager(args.lookback)
        manager.print_status_report()

    elif args.command == "test":
        print(f"\nTesting JSONCalibrator for {args.market}...")
        cal = JSONCalibrator(args.market, args.lookback)

        metrics = cal.get_recent_performance()
        print("\nPerformance Metrics:")
        print(json.dumps(metrics, indent=2, default=str))

        # Test adjustment
        test_probs = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
        print("\nAdjustment Test:")
        print(f"{'Raw':>8} {'Adjusted':>10} {'Change':>8}")
        print("-" * 30)
        for p in test_probs:
            adj = cal.adjust_probability(p)
            print(f"{p:8.3f} {adj:10.3f} {adj-p:+8.3f}")

    elif args.command == "compare":
        print(f"\nComparing JSONCalibrator vs DynamicCalibrator for {args.market}...")

        # Load both calibrators
        json_cal = JSONCalibrator(args.market, args.lookback)

        try:
            from dynamic_calibrator import DynamicCalibrator

            dyn_cal = DynamicCalibrator(args.market, args.lookback)

            # Get metrics from both
            json_metrics = json_cal.get_recent_performance()
            dyn_metrics = dyn_cal.get_recent_performance()

            print("\n" + "=" * 60)
            print(f"{'Metric':<30} {'JSON':>12} {'Dynamic':>12}")
            print("=" * 60)

            for key in [
                "total_predictions",
                "win_rate",
                "avg_predicted_prob",
                "bias",
                "brier_score",
            ]:
                j_val = json_metrics.get(key, "N/A")
                d_val = dyn_metrics.get(key, "N/A")
                if isinstance(j_val, float):
                    print(f"{key:<30} {j_val:>12.4f} {d_val:>12.4f}")
                else:
                    print(f"{key:<30} {str(j_val):>12} {str(d_val):>12}")

            print("\nKey difference: JSON uses REAL p_over, Dynamic reconstructs from lines!")

        except ImportError:
            print("DynamicCalibrator not available for comparison")
