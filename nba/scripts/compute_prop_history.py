#!/usr/bin/env python3
"""
Prop History Learner - Computes and Maintains prop_performance_history Table
=============================================================================

Follows the same pattern as JSONCalibrator for consistency across the learner system.

Key Features:
- Tracks hit rates per player/stat_type/line_center with Bayesian adjustments
- Calculates context splits (home/away, defense rankings, rest, B2B)
- Computes streaks and recency metrics
- Uses Wilson score for sample quality assessment

Usage:
    # As a module (called from other scripts)
    from compute_prop_history import PropHistoryLearner

    learner = PropHistoryLearner(lookback_days=35)
    learner.update()

    # As CLI
    python3 compute_prop_history.py                    # Full rebuild
    python3 compute_prop_history.py --incremental     # Incremental (7-day lookback for player filter)
    python3 compute_prop_history.py --player "LeBron James"  # Single player

Database: nba_intelligence (port 5539)
Table: prop_performance_history

Author: Claude Code
Date: January 2026
"""

import argparse
import logging
import math
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import psycopg2
from psycopg2.extras import execute_values

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
LOG_DIR = PROJECT_ROOT / "nba" / "scripts" / "logs"

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from nba.config.database import get_intelligence_db_config, get_team_db_config

# Database configuration (centralized)
DB_INTELLIGENCE = get_intelligence_db_config()

DB_TEAM = get_team_db_config()


class PropHistoryLearner:
    """
    Learner system for prop performance history.

    Maintains same interface pattern as JSONCalibrator for consistency.
    """

    # Configuration constants (matching JSONCalibrator style)
    PRIOR_HIT_RATE = 0.50  # Bayesian prior (50% baseline)
    PRIOR_STRENGTH = 10  # Equivalent to 10 observations for shrinkage
    EMA_ALPHA = 0.15  # Exponential decay for recency weighting
    DEFAULT_LOOKBACK = 35  # Default lookback for incremental mode
    MIN_PROPS_FOR_UPDATE = 1  # Minimum recent props to trigger update

    def __init__(self, lookback_days: int = None, player_filter: Optional[str] = None):
        """
        Initialize PropHistoryLearner.

        Args:
            lookback_days: Days to look back for identifying players to update.
                          Set to None for full rebuild (all players).
            player_filter: Single player to process (for debugging)
        """
        self.lookback_days = lookback_days
        self.player_filter = player_filter

        # State
        self._defense_rankings = {}
        self._players_to_update = set()
        self._records_processed = 0
        self._records_upserted = 0

        # Ensure log directory exists
        LOG_DIR.mkdir(parents=True, exist_ok=True)

        mode = "incremental" if lookback_days else "full rebuild"
        logger.info(f"PropHistoryLearner initialized ({mode}, lookback={lookback_days})")

    def _get_intelligence_connection(self):
        """Get connection to intelligence database."""
        return psycopg2.connect(**DB_INTELLIGENCE)

    def _get_team_connection(self):
        """Get connection to team database."""
        return psycopg2.connect(**DB_TEAM)

    def _load_defense_rankings(self) -> Dict[str, int]:
        """
        Get team defensive rankings (1 = best defense, 30 = worst).
        """
        rankings = {}
        conn = None
        try:
            conn = self._get_team_connection()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT team_abbrev,
                           ROW_NUMBER() OVER (ORDER BY defensive_rating ASC) as rank
                    FROM team_season_stats
                    WHERE season = (SELECT MAX(season) FROM team_season_stats)
                """
                )
                for row in cur.fetchall():
                    rankings[row[0]] = row[1]
            logger.info(f"Loaded {len(rankings)} team defense rankings")
        except (psycopg2.Error, KeyError, TypeError, ValueError) as e:
            logger.warning(f"Could not load defense rankings: {e}")
        finally:
            if conn:
                conn.close()

        self._defense_rankings = rankings
        return rankings

    def _get_players_to_update(self, conn) -> Set[Tuple[str, str]]:
        """
        Identify which (player_name, stat_type) combinations need updating.

        In incremental mode, only players with recent props.
        In full mode, all players with any props.
        """
        if self.player_filter:
            # Single player mode - get all stat types for this player
            query = """
                SELECT DISTINCT player_name, stat_type
                FROM nba_props_xl
                WHERE actual_value IS NOT NULL
                  AND player_name = %s
            """
            params = (self.player_filter,)
        elif self.lookback_days:
            # Incremental mode - only players with recent props
            query = """
                SELECT DISTINCT player_name, stat_type
                FROM nba_props_xl
                WHERE actual_value IS NOT NULL
                  AND game_date >= CURRENT_DATE - %s
            """
            params = (self.lookback_days,)
        else:
            # Full rebuild - all players
            query = """
                SELECT DISTINCT player_name, stat_type
                FROM nba_props_xl
                WHERE actual_value IS NOT NULL
            """
            params = None

        players = set()
        with conn.cursor() as cur:
            if params:
                cur.execute(query, params)
            else:
                cur.execute(query)
            for row in cur.fetchall():
                players.add((row[0], row[1]))

        self._players_to_update = players
        logger.info(f"Identified {len(players)} player/stat combinations to update")
        return players

    def _load_full_history_for_players(self, conn, players: Set[Tuple[str, str]]) -> List[dict]:
        """
        Load FULL prop history for the specified players.

        This is critical - even in incremental mode, we load ALL history
        for players being updated to ensure accurate EMA calculations.
        """
        if not players:
            return []

        # Build player/stat filter
        player_stat_list = list(players)

        query = """
            SELECT DISTINCT ON (player_name, stat_type, game_date, over_line)
                player_name,
                stat_type,
                over_line as line,
                actual_value,
                is_home,
                opponent_team,
                game_date
            FROM nba_props_xl
            WHERE actual_value IS NOT NULL
              AND over_line IS NOT NULL
            ORDER BY player_name, stat_type, game_date, over_line, fetch_timestamp DESC
        """

        props = []
        with conn.cursor() as cur:
            cur.execute(query)
            for row in cur.fetchall():
                player_name = row[0]
                stat_type = row[1]

                # Only keep props for players we're updating
                if (player_name, stat_type) not in players:
                    continue

                props.append(
                    {
                        "player_name": player_name,
                        "stat_type": stat_type,
                        "line": float(row[2]),
                        "actual_value": float(row[3]),
                        "is_home": row[4],
                        "opponent_team": row[5],
                        "game_date": row[6],
                        "hit": float(row[3]) > float(row[2]),  # OVER hit
                    }
                )

        logger.info(
            f"Loaded {len(props)} total props (full history for {len(players)} player/stat combos)"
        )
        return props

    @staticmethod
    def _round_to_half(value: float) -> float:
        """Round to nearest 0.5 for line_center."""
        return round(value * 2) / 2

    @staticmethod
    def _wilson_score_lower(hits: int, total: int, z: float = 1.96) -> float:
        """Wilson score confidence interval lower bound."""
        if total == 0:
            return 0.0

        p = hits / total
        denominator = 1 + z**2 / total
        center = p + z**2 / (2 * total)
        spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total)

        return max(0.0, (center - spread) / denominator)

    def _bayesian_hit_rate(self, hits: int, total: int) -> float:
        """Bayesian-adjusted hit rate with shrinkage toward prior."""
        if total == 0:
            return self.PRIOR_HIT_RATE
        return (hits + self.PRIOR_HIT_RATE * self.PRIOR_STRENGTH) / (total + self.PRIOR_STRENGTH)

    def _calculate_bayesian_weight(self, total: int) -> float:
        """Calculate prior weight (1 = all prior, 0 = all observed)."""
        return self.PRIOR_STRENGTH / (total + self.PRIOR_STRENGTH)

    def _calculate_ema_hit_rate(self, props: List[dict], n: int = 20) -> float:
        """Calculate exponentially weighted hit rate over last N props."""
        if not props:
            return self.PRIOR_HIT_RATE

        recent = props[:n]
        if not recent:
            return self.PRIOR_HIT_RATE

        weights = []
        hits = []
        for i, p in enumerate(recent):
            weight = (1 - self.EMA_ALPHA) ** i
            weights.append(weight)
            hits.append(1 if p["hit"] else 0)

        total_weight = sum(weights)
        if total_weight == 0:
            return self.PRIOR_HIT_RATE

        return sum(h * w for h, w in zip(hits, weights)) / total_weight

    @staticmethod
    def _calculate_streaks(props: List[dict]) -> Tuple[int, int, int]:
        """Calculate current streak and max streaks."""
        if not props:
            return 0, 0, 0

        # Current streak (most recent first)
        consecutive = 0
        current_is_over = props[0]["hit"] if props else None

        for p in props:
            if p["hit"] == current_is_over:
                consecutive += 1
            else:
                break

        if not current_is_over:
            consecutive = -consecutive

        # Max streaks (chronological order)
        max_overs = 0
        max_unders = 0
        current_over_streak = 0
        current_under_streak = 0

        for p in reversed(props):
            if p["hit"]:
                current_over_streak += 1
                current_under_streak = 0
                max_overs = max(max_overs, current_over_streak)
            else:
                current_under_streak += 1
                current_over_streak = 0
                max_unders = max(max_unders, current_under_streak)

        return consecutive, max_overs, max_unders

    @staticmethod
    def _calculate_rest_days(props: List[dict]) -> Dict[date, int]:
        """Calculate rest days before each game for a player."""
        if not props:
            return {}

        dates = sorted(set(p["game_date"] for p in props))
        rest_days = {}

        for i, d in enumerate(dates):
            if i == 0:
                rest_days[d] = 3
            else:
                delta = (d - dates[i - 1]).days
                rest_days[d] = delta - 1

        return rest_days

    @staticmethod
    def _get_season_year(game_date: date) -> int:
        """Get the season end year."""
        if game_date.month >= 10:
            return game_date.year + 1
        return game_date.year

    def _compute_player_history(
        self, player_name: str, stat_type: str, props: List[dict], current_date: date
    ) -> List[dict]:
        """Compute prop_performance_history records for one player/stat_type."""
        if not props:
            return []

        # Sort chronologically
        props = sorted(props, key=lambda x: x["game_date"])

        # Calculate rest days
        rest_days = self._calculate_rest_days(props)

        # Enrich props with context
        for p in props:
            p["rest_days"] = rest_days.get(p["game_date"], 2)
            p["is_b2b"] = p["rest_days"] == 0
            p["is_rested"] = p["rest_days"] >= 2

            opp = p.get("opponent_team")
            if opp and opp in self._defense_rankings:
                rank = self._defense_rankings[opp]
                p["vs_top10_def"] = rank <= 10
                p["vs_bottom10_def"] = rank >= 21
            else:
                p["vs_top10_def"] = False
                p["vs_bottom10_def"] = False

        # Group by line_center
        by_line = defaultdict(list)
        for p in props:
            line_center = self._round_to_half(p["line"])
            by_line[line_center].append(p)

        # Calculate season average
        all_values = [p["actual_value"] for p in props]
        season_avg = sum(all_values) / len(all_values) if all_values else 0

        all_lines = sorted(by_line.keys())
        records = []
        season = self._get_season_year(current_date)

        for line_center, line_props in by_line.items():
            # Most recent first for recency calculations
            line_props_recent = sorted(line_props, key=lambda x: x["game_date"], reverse=True)

            # Basic counts
            total = len(line_props)
            hits = sum(1 for p in line_props if p["hit"])

            # L20 and L10
            l20_props = line_props_recent[:20]
            l10_props = line_props_recent[:10]
            l20_count = len(l20_props)
            l10_count = len(l10_props)

            # Hit rates
            hit_rate_all = self._bayesian_hit_rate(hits, total)
            hit_rate_l20 = self._calculate_ema_hit_rate(l20_props, 20)
            hit_rate_l10 = self._calculate_ema_hit_rate(l10_props, 10)

            # Context splits
            home_props = [p for p in line_props if p.get("is_home")]
            away_props = [p for p in line_props if not p.get("is_home")]
            top10_props = [p for p in line_props if p.get("vs_top10_def")]
            bottom10_props = [p for p in line_props if p.get("vs_bottom10_def")]
            rested_props = [p for p in line_props if p.get("is_rested")]
            b2b_props = [p for p in line_props if p.get("is_b2b")]

            n_home = len(home_props)
            n_away = len(away_props)
            n_top10 = len(top10_props)
            n_bottom10 = len(bottom10_props)
            n_rested = len(rested_props)
            n_b2b = len(b2b_props)

            hit_rate_home = (
                self._bayesian_hit_rate(sum(1 for p in home_props if p["hit"]), n_home)
                if n_home > 0
                else None
            )

            hit_rate_away = (
                self._bayesian_hit_rate(sum(1 for p in away_props if p["hit"]), n_away)
                if n_away > 0
                else None
            )

            hit_rate_top10 = (
                self._bayesian_hit_rate(sum(1 for p in top10_props if p["hit"]), n_top10)
                if n_top10 > 0
                else None
            )

            hit_rate_bottom10 = (
                self._bayesian_hit_rate(sum(1 for p in bottom10_props if p["hit"]), n_bottom10)
                if n_bottom10 > 0
                else None
            )

            hit_rate_rested = (
                self._bayesian_hit_rate(sum(1 for p in rested_props if p["hit"]), n_rested)
                if n_rested > 0
                else None
            )

            hit_rate_b2b = (
                self._bayesian_hit_rate(sum(1 for p in b2b_props if p["hit"]), n_b2b)
                if n_b2b > 0
                else None
            )

            # Line positioning
            line_vs_avg = line_center - season_avg
            line_percentile = all_lines.index(line_center) / max(len(all_lines) - 1, 1)

            # Recency
            most_recent = line_props_recent[0]["game_date"] if line_props_recent else None
            days_since_last_prop = (current_date - most_recent).days if most_recent else None

            last_hit_date = None
            for p in line_props_recent:
                if p["hit"]:
                    last_hit_date = p["game_date"]
                    break
            days_since_last_hit = (current_date - last_hit_date).days if last_hit_date else None

            # Streaks
            consecutive, max_overs, max_unders = self._calculate_streaks(line_props_recent)

            # Quality metrics
            sample_quality = self._wilson_score_lower(hits, total)
            bayesian_weight = self._calculate_bayesian_weight(total)

            records.append(
                {
                    "player_name": player_name,
                    "stat_type": stat_type,
                    "line_center": line_center,
                    "total_props": total,
                    "props_l20": l20_count,
                    "props_l10": l10_count,
                    "hit_rate_all": round(hit_rate_all, 4),
                    "hit_rate_l20": round(hit_rate_l20, 4),
                    "hit_rate_l10": round(hit_rate_l10, 4),
                    "hit_rate_home": round(hit_rate_home, 4) if hit_rate_home else None,
                    "hit_rate_away": round(hit_rate_away, 4) if hit_rate_away else None,
                    "hit_rate_vs_top10_def": round(hit_rate_top10, 4) if hit_rate_top10 else None,
                    "hit_rate_vs_bottom10_def": (
                        round(hit_rate_bottom10, 4) if hit_rate_bottom10 else None
                    ),
                    "hit_rate_rested": round(hit_rate_rested, 4) if hit_rate_rested else None,
                    "hit_rate_b2b": round(hit_rate_b2b, 4) if hit_rate_b2b else None,
                    "n_home": n_home,
                    "n_away": n_away,
                    "n_vs_top10_def": n_top10,
                    "n_vs_bottom10_def": n_bottom10,
                    "n_rested": n_rested,
                    "n_b2b": n_b2b,
                    "line_vs_season_avg": round(line_vs_avg, 2),
                    "line_percentile": round(line_percentile, 4),
                    "days_since_last_prop": days_since_last_prop,
                    "days_since_last_hit": days_since_last_hit,
                    "consecutive_overs": consecutive,
                    "max_streak_overs": max_overs,
                    "max_streak_unders": max_unders,
                    "sample_quality_score": round(sample_quality, 4),
                    "bayesian_prior_weight": round(bayesian_weight, 4),
                    "season": season,
                }
            )

        return records

    def _upsert_records(self, conn, records: List[dict]) -> int:
        """Upsert records into prop_performance_history."""
        if not records:
            return 0

        columns = [
            "player_name",
            "stat_type",
            "line_center",
            "total_props",
            "props_l20",
            "props_l10",
            "hit_rate_all",
            "hit_rate_l20",
            "hit_rate_l10",
            "hit_rate_home",
            "hit_rate_away",
            "hit_rate_vs_top10_def",
            "hit_rate_vs_bottom10_def",
            "hit_rate_rested",
            "hit_rate_b2b",
            "n_home",
            "n_away",
            "n_vs_top10_def",
            "n_vs_bottom10_def",
            "n_rested",
            "n_b2b",
            "line_vs_season_avg",
            "line_percentile",
            "days_since_last_prop",
            "days_since_last_hit",
            "consecutive_overs",
            "max_streak_overs",
            "max_streak_unders",
            "sample_quality_score",
            "bayesian_prior_weight",
            "season",
        ]

        values = [tuple(r[col] for col in columns) for r in records]

        update_cols = [
            c for c in columns if c not in ("player_name", "stat_type", "line_center", "season")
        ]
        update_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in update_cols)

        query = f"""
            INSERT INTO prop_performance_history ({', '.join(columns)}, last_updated)
            VALUES %s
            ON CONFLICT (player_name, stat_type, line_center, season)
            DO UPDATE SET {update_clause}, last_updated = CURRENT_TIMESTAMP
        """

        template = "(" + ", ".join(["%s"] * len(columns)) + ", CURRENT_TIMESTAMP)"

        with conn.cursor() as cur:
            execute_values(cur, query, values, template=template)

        conn.commit()
        return len(records)

    def update(self, dry_run: bool = False) -> Dict:
        """
        Main update method - compute and upsert prop history.

        Returns dict with update statistics.
        """
        logger.info("=" * 60)
        logger.info("PROP HISTORY LEARNER - UPDATE")
        logger.info("=" * 60)

        conn = None
        try:
            conn = self._get_intelligence_connection()

            # Load defense rankings
            self._load_defense_rankings()

            # Identify players to update
            players = self._get_players_to_update(conn)

            if not players:
                logger.warning("No players to update")
                return {"status": "no_data", "records": 0}

            # Load FULL history for these players (critical for EMA accuracy)
            props = self._load_full_history_for_players(conn, players)

            if not props:
                logger.warning("No props found")
                return {"status": "no_props", "records": 0}

            # Group by player/stat_type
            by_player_stat = defaultdict(list)
            for p in props:
                key = (p["player_name"], p["stat_type"])
                by_player_stat[key].append(p)

            # Process each player/stat_type
            all_records = []
            current_date = date.today()

            for (player_name, stat_type), player_props in by_player_stat.items():
                records = self._compute_player_history(
                    player_name, stat_type, player_props, current_date
                )
                all_records.extend(records)

            self._records_processed = len(all_records)
            logger.info(f"Computed {len(all_records)} history records")

            # Upsert
            if dry_run:
                logger.info("DRY RUN - not writing to database")
                for r in all_records[:3]:
                    logger.info(
                        f"  Sample: {r['player_name']} {r['stat_type']} @ {r['line_center']}: "
                        f"hit_rate={r['hit_rate_all']:.1%} (n={r['total_props']})"
                    )
            else:
                count = self._upsert_records(conn, all_records)
                self._records_upserted = count
                logger.info(f"Upserted {count} records")

            # Summary
            if all_records:
                avg_total = sum(r["total_props"] for r in all_records) / len(all_records)
                high_quality = sum(1 for r in all_records if r["total_props"] >= 10)
                logger.info(f"Average props per line: {avg_total:.1f}")
                logger.info(f"High-quality records (n>=10): {high_quality}")

            return {
                "status": "success",
                "players_updated": len(players),
                "records_computed": self._records_processed,
                "records_upserted": self._records_upserted,
                "dry_run": dry_run,
            }

        except (psycopg2.Error, KeyError, TypeError, ValueError) as e:
            logger.error(f"Update failed: {e}")
            raise
        finally:
            if conn:
                conn.close()
            logger.info("=" * 60)
            logger.info("COMPLETE")
            logger.info("=" * 60)

    def get_status(self) -> Dict:
        """Get current status of prop_performance_history table."""
        conn = None
        try:
            conn = self._get_intelligence_connection()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        COUNT(*) as total_records,
                        COUNT(DISTINCT player_name) as unique_players,
                        MAX(last_updated) as last_updated,
                        AVG(total_props) as avg_props_per_record
                    FROM prop_performance_history
                """
                )
                row = cur.fetchone()
                return {
                    "total_records": row[0],
                    "unique_players": row[1],
                    "last_updated": row[2].isoformat() if row[2] else None,
                    "avg_props_per_record": round(row[3], 1) if row[3] else 0,
                }
        finally:
            if conn:
                conn.close()


def main():
    parser = argparse.ArgumentParser(description="Prop History Learner")
    parser.add_argument("--player", type=str, help="Process single player")
    parser.add_argument("--days", type=int, default=None, help="Lookback days for incremental mode")
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Incremental update (default 7-day lookback for player filter)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Calculate but do not write to database"
    )
    parser.add_argument("--season", type=int, help="Season year (for logging only)")
    parser.add_argument("--status", action="store_true", help="Show table status and exit")
    args = parser.parse_args()

    # Determine lookback
    lookback = None
    if args.incremental:
        lookback = args.days if args.days else 7  # Default 7 days for incremental
    elif args.days:
        lookback = args.days

    learner = PropHistoryLearner(lookback_days=lookback, player_filter=args.player)

    if args.season:
        logger.info(f"Season: {args.season}")

    if args.status:
        status = learner.get_status()
        print(f"Total records: {status['total_records']}")
        print(f"Unique players: {status['unique_players']}")
        print(f"Last updated: {status['last_updated']}")
        print(f"Avg props/record: {status['avg_props_per_record']}")
        return

    result = learner.update(dry_run=args.dry_run)

    if result["status"] == "success":
        logger.info(
            f"Updated {result['players_updated']} players, " f"{result['records_upserted']} records"
        )


if __name__ == "__main__":
    main()
