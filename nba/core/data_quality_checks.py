#!/usr/bin/env python3
"""
Data Quality Checks for NBA Props ML Pipeline

Great Expectations-style validation for training data quality.
Run before model training to catch data issues early.

Usage:
    python -m nba.core.data_quality_checks --pre-training
    python -m nba.core.data_quality_checks --check-props --date 2026-01-30
    python -m nba.core.data_quality_checks --check-features

Author: Claude Code
"""

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

import psycopg2

from nba.config.database import (
    get_games_db_config,
    get_intelligence_db_config,
    get_players_db_config,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class CheckResult:
    """Result of a data quality check."""

    name: str
    passed: bool
    message: str
    details: dict[str, Any] | None = None

    def __str__(self) -> str:
        status = "\u2705" if self.passed else "\u274c"
        return f"{status} {self.name}: {self.message}"


class DataQualityChecker:
    """
    Data quality validation suite.

    Inspired by Great Expectations but lightweight and focused on our schema.
    """

    def __init__(self):
        self.results: list[CheckResult] = []
        self.conn_players = None
        self.conn_games = None
        self.conn_intel = None

    def connect(self):
        """Establish database connections."""
        self.conn_players = psycopg2.connect(**get_players_db_config())
        self.conn_games = psycopg2.connect(**get_games_db_config())
        self.conn_intel = psycopg2.connect(**get_intelligence_db_config())
        logger.info("Connected to all databases")

    def close(self):
        """Close database connections."""
        for conn in [self.conn_players, self.conn_games, self.conn_intel]:
            if conn:
                conn.close()

    def add_result(self, result: CheckResult):
        """Add a check result."""
        self.results.append(result)
        logger.info(str(result))

    # =========================================================================
    # Player Data Checks
    # =========================================================================

    def check_player_game_logs_freshness(self, max_days: int = 3) -> CheckResult:
        """Verify player game logs are recent."""
        cursor = self.conn_players.cursor()
        cursor.execute(
            """
            SELECT MAX(game_date) as latest, COUNT(*) as total
            FROM player_game_logs
            WHERE game_date >= CURRENT_DATE - INTERVAL '30 days'
        """
        )
        row = cursor.fetchone()
        cursor.close()

        latest_date = row[0]
        recent_count = row[1]

        if latest_date is None:
            return CheckResult(
                name="player_game_logs_freshness",
                passed=False,
                message="No recent game logs found",
            )

        days_old = (date.today() - latest_date).days
        passed = days_old <= max_days

        return CheckResult(
            name="player_game_logs_freshness",
            passed=passed,
            message=f"Latest: {latest_date} ({days_old} days old), {recent_count} logs in L30",
            details={"latest_date": str(latest_date), "days_old": days_old},
        )

    def check_player_rolling_stats_coverage(self, min_coverage: float = 0.8) -> CheckResult:
        """Verify rolling stats exist for active players."""
        cursor = self.conn_players.cursor()
        cursor.execute(
            """
            SELECT
                COUNT(DISTINCT p.player_id) as total_active,
                COUNT(DISTINCT r.player_id) as with_rolling_stats
            FROM player_profile p
            LEFT JOIN player_rolling_stats r ON p.player_id = r.player_id
            WHERE p.team_abbrev IS NOT NULL
        """
        )
        row = cursor.fetchone()
        cursor.close()

        total, with_stats = row
        coverage = with_stats / total if total > 0 else 0
        passed = coverage >= min_coverage

        return CheckResult(
            name="player_rolling_stats_coverage",
            passed=passed,
            message=f"{with_stats}/{total} active players have rolling stats ({coverage:.1%})",
            details={"total": total, "with_stats": with_stats, "coverage": coverage},
        )

    def check_no_null_critical_fields(self) -> CheckResult:
        """Verify no nulls in critical player fields."""
        cursor = self.conn_players.cursor()
        cursor.execute(
            """
            SELECT
                SUM(CASE WHEN points IS NULL THEN 1 ELSE 0 END) as null_points,
                SUM(CASE WHEN minutes IS NULL THEN 1 ELSE 0 END) as null_minutes,
                SUM(CASE WHEN game_date IS NULL THEN 1 ELSE 0 END) as null_dates,
                COUNT(*) as total
            FROM player_game_logs
            WHERE game_date >= '2023-10-01'
        """
        )
        row = cursor.fetchone()
        cursor.close()

        null_points, null_minutes, null_dates, total = row
        total_nulls = null_points + null_minutes + null_dates
        passed = total_nulls == 0

        return CheckResult(
            name="no_null_critical_fields",
            passed=passed,
            message=f"Null counts - points: {null_points}, minutes: {null_minutes}, dates: {null_dates}",
            details={
                "null_points": null_points,
                "null_minutes": null_minutes,
                "total": total,
            },
        )

    # =========================================================================
    # Props Data Checks
    # =========================================================================

    def check_props_freshness(self, max_days: int = 1) -> CheckResult:
        """Verify props data is recent."""
        cursor = self.conn_intel.cursor()
        cursor.execute(
            """
            SELECT MAX(game_date) as latest, COUNT(*) as today_count
            FROM nba_props_xl
            WHERE game_date >= CURRENT_DATE - INTERVAL '1 day'
        """
        )
        row = cursor.fetchone()
        cursor.close()

        latest_date = row[0]
        today_count = row[1]

        if latest_date is None:
            return CheckResult(
                name="props_freshness",
                passed=False,
                message="No props found for today/yesterday",
            )

        days_old = (date.today() - latest_date).days
        passed = days_old <= max_days and today_count > 50

        return CheckResult(
            name="props_freshness",
            passed=passed,
            message=f"Latest: {latest_date}, {today_count} props in last 24h",
            details={"latest_date": str(latest_date), "count": today_count},
        )

    def check_props_multi_book_coverage(self, min_books: int = 3) -> CheckResult:
        """Verify props have multiple book sources."""
        cursor = self.conn_intel.cursor()
        cursor.execute(
            """
            SELECT
                AVG(num_books) as avg_books,
                SUM(CASE WHEN num_books >= %s THEN 1 ELSE 0 END)::float / COUNT(*) as pct_multi
            FROM (
                SELECT player_name, game_date, stat_type, COUNT(DISTINCT book_id) as num_books
                FROM nba_props_xl
                WHERE game_date >= CURRENT_DATE - INTERVAL '7 days'
                GROUP BY player_name, game_date, stat_type
            ) sub
        """,
            (min_books,),
        )
        row = cursor.fetchone()
        cursor.close()

        avg_books = row[0] or 0
        pct_multi = row[1] or 0
        passed = pct_multi >= 0.7 and avg_books >= 3

        return CheckResult(
            name="props_multi_book_coverage",
            passed=passed,
            message=f"Avg {avg_books:.1f} books/prop, {pct_multi:.1%} have {min_books}+ books",
            details={"avg_books": avg_books, "pct_multi_book": pct_multi},
        )

    def check_props_stat_type_distribution(self) -> CheckResult:
        """Verify all stat types are represented."""
        cursor = self.conn_intel.cursor()
        cursor.execute(
            """
            SELECT stat_type, COUNT(*) as cnt
            FROM nba_props_xl
            WHERE game_date >= CURRENT_DATE - INTERVAL '7 days'
            GROUP BY stat_type
            ORDER BY cnt DESC
        """
        )
        rows = cursor.fetchall()
        cursor.close()

        stat_counts = {row[0]: row[1] for row in rows}
        expected = ["POINTS", "REBOUNDS", "ASSISTS", "THREES"]
        missing = [s for s in expected if s not in stat_counts]
        passed = len(missing) == 0

        return CheckResult(
            name="props_stat_type_distribution",
            passed=passed,
            message=f"Stat types: {stat_counts}",
            details={"counts": stat_counts, "missing": missing},
        )

    # =========================================================================
    # Training Data Checks
    # =========================================================================

    def check_training_data_volume(self, min_samples: int = 20000) -> CheckResult:
        """Verify sufficient training data exists."""
        cursor = self.conn_intel.cursor()
        cursor.execute(
            """
            SELECT stat_type, COUNT(*) as cnt
            FROM nba_props_xl
            WHERE actual_value IS NOT NULL
              AND game_date >= '2023-10-01'
            GROUP BY stat_type
        """
        )
        rows = cursor.fetchall()
        cursor.close()

        counts = {row[0]: row[1] for row in rows}
        below_threshold = {k: v for k, v in counts.items() if v < min_samples}
        passed = len(below_threshold) == 0 or all(
            k not in ["POINTS", "REBOUNDS"] for k in below_threshold
        )

        return CheckResult(
            name="training_data_volume",
            passed=passed,
            message=f"Samples with actuals: {counts}",
            details={"counts": counts, "below_threshold": below_threshold},
        )

    def check_home_away_balance(
        self, min_ratio: float = 0.4, max_ratio: float = 0.6
    ) -> CheckResult:
        """Verify home/away distribution is balanced."""
        cursor = self.conn_players.cursor()
        cursor.execute(
            """
            SELECT
                SUM(CASE WHEN is_home = true THEN 1 ELSE 0 END)::float / COUNT(*) as home_pct
            FROM player_game_logs
            WHERE game_date >= '2023-10-01'
              AND is_home IS NOT NULL
        """
        )
        row = cursor.fetchone()
        cursor.close()

        home_pct = row[0] or 0.5
        passed = min_ratio <= home_pct <= max_ratio

        return CheckResult(
            name="home_away_balance",
            passed=passed,
            message=f"Home games: {home_pct:.1%} (expected {min_ratio:.0%}-{max_ratio:.0%})",
            details={"home_pct": home_pct},
        )

    def check_no_future_data_leakage(self) -> CheckResult:
        """Verify no props have future game dates."""
        cursor = self.conn_intel.cursor()
        cursor.execute(
            """
            SELECT COUNT(*) as future_props
            FROM nba_props_xl
            WHERE game_date > CURRENT_DATE + INTERVAL '1 day'
        """
        )
        row = cursor.fetchone()
        cursor.close()

        future_count = row[0]
        passed = future_count == 0

        return CheckResult(
            name="no_future_data_leakage",
            passed=passed,
            message=f"Props with future dates: {future_count}",
            details={"future_count": future_count},
        )

    # =========================================================================
    # Run All Checks
    # =========================================================================

    def run_pre_training_checks(self) -> bool:
        """Run all checks required before training."""
        logger.info("=" * 60)
        logger.info("PRE-TRAINING DATA QUALITY CHECKS")
        logger.info("=" * 60)

        checks = [
            self.check_player_game_logs_freshness,
            self.check_player_rolling_stats_coverage,
            self.check_no_null_critical_fields,
            self.check_training_data_volume,
            self.check_home_away_balance,
            self.check_no_future_data_leakage,
        ]

        for check in checks:
            try:
                result = check()
                self.add_result(result)
            except (psycopg2.Error, KeyError, TypeError, ValueError) as e:
                self.add_result(CheckResult(name=check.__name__, passed=False, message=str(e)))

        return self.summarize()

    def run_daily_checks(self) -> bool:
        """Run daily operational checks."""
        logger.info("=" * 60)
        logger.info("DAILY DATA QUALITY CHECKS")
        logger.info("=" * 60)

        checks = [
            self.check_props_freshness,
            self.check_props_multi_book_coverage,
            self.check_props_stat_type_distribution,
            self.check_player_game_logs_freshness,
        ]

        for check in checks:
            try:
                result = check()
                self.add_result(result)
            except (psycopg2.Error, KeyError, TypeError, ValueError) as e:
                self.add_result(CheckResult(name=check.__name__, passed=False, message=str(e)))

        return self.summarize()

    def summarize(self) -> bool:
        """Print summary and return overall pass/fail."""
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)

        logger.info("=" * 60)
        logger.info(f"SUMMARY: {passed} passed, {failed} failed")
        logger.info("=" * 60)

        if failed > 0:
            logger.error("Data quality checks FAILED")
            for r in self.results:
                if not r.passed:
                    logger.error(f"  - {r.name}: {r.message}")
            return False

        logger.info("All data quality checks PASSED")
        return True


def main():
    parser = argparse.ArgumentParser(description="Data quality checks")
    parser.add_argument("--pre-training", action="store_true", help="Run pre-training checks")
    parser.add_argument("--daily", action="store_true", help="Run daily checks")
    parser.add_argument("--all", action="store_true", help="Run all checks")

    args = parser.parse_args()

    checker = DataQualityChecker()

    try:
        checker.connect()

        if args.pre_training:
            success = checker.run_pre_training_checks()
        elif args.daily:
            success = checker.run_daily_checks()
        else:
            # Default to pre-training
            success = checker.run_pre_training_checks()

        sys.exit(0 if success else 1)

    except (psycopg2.Error, KeyError, TypeError, ValueError) as e:
        logger.error(f"Data quality check failed: {e}")
        sys.exit(1)
    finally:
        checker.close()


if __name__ == "__main__":
    main()
