#!/usr/bin/env python3
"""
Data Freshness Validator
========================
Validates data freshness requirements before generating predictions.
Ensures props, stats, injuries, and models meet production standards.

Part of Phase 9: Production Deployment - Pre-flight Checks
Created: November 7, 2025
"""

import logging
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import psycopg2

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from betting_xl.config.production_policies import DATA_FRESHNESS_POLICY

# Database configuration (override with environment variables as needed)
# Note: Production uses mlb_user for all databases (legacy naming)
DB_DEFAULT_USER = os.getenv("NBA_DB_USER", os.getenv("DB_USER", "mlb_user"))
DB_DEFAULT_PASSWORD = os.getenv("NBA_DB_PASSWORD", os.getenv("DB_PASSWORD"))

DB_CONFIG = {
    "host": os.getenv("NBA_INT_DB_HOST", "localhost"),
    "port": int(os.getenv("NBA_INT_DB_PORT", 5539)),
    "user": os.getenv("NBA_INT_DB_USER", DB_DEFAULT_USER),
    "password": os.getenv("NBA_INT_DB_PASSWORD", DB_DEFAULT_PASSWORD),
    "database": os.getenv("NBA_INT_DB_NAME", "nba_intelligence"),
}

PLAYERS_DB_CONFIG = {
    "host": os.getenv("NBA_PLAYERS_DB_HOST", "localhost"),
    "port": int(os.getenv("NBA_PLAYERS_DB_PORT", 5536)),
    "user": os.getenv("NBA_PLAYERS_DB_USER", DB_DEFAULT_USER),
    "password": os.getenv("NBA_PLAYERS_DB_PASSWORD", DB_DEFAULT_PASSWORD),
    "database": os.getenv("NBA_PLAYERS_DB_NAME", "nba_players"),
}


class DataFreshnessValidator:
    """Validates all data freshness requirements before prediction generation."""

    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.policy = DATA_FRESHNESS_POLICY
        self.failures = []
        self.warnings = []

    def validate_all(self) -> Tuple[bool, Dict[str, any]]:
        """
        Run all data freshness validations.

        Returns:
            Tuple of (success: bool, results: dict)
        """
        self.failures = []
        self.warnings = []

        results = {
            "props_freshness": self._validate_props_freshness(),
            "game_results_freshness": self._validate_game_results_freshness(),
            "injury_reports_freshness": self._validate_injury_reports_freshness(),
            "model_age": self._validate_model_age(),
            "timestamp": datetime.now().isoformat(),
        }

        # Overall success if no critical failures
        success = len(self.failures) == 0

        results["success"] = success
        results["failures"] = self.failures
        results["warnings"] = self.warnings

        return success, results

    def _validate_props_freshness(self) -> Dict[str, any]:
        """Validate that props are from today and meet minimum volume."""
        result = {"status": "pending", "props_count": 0, "props_date": None}

        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()

            # Check props from today
            today = date.today()
            query = """
                SELECT COUNT(*), MIN(game_date), MAX(game_date)
                FROM nba_props_xl
                WHERE game_date = %s
                  AND is_active = true
            """
            cursor.execute(query, (today,))
            props_count, min_date, max_date = cursor.fetchone()

            result["props_count"] = props_count
            result["props_date"] = str(today)
            result["min_date"] = str(min_date) if min_date else None
            result["max_date"] = str(max_date) if max_date else None

            # Check minimum volume
            min_required = self.policy["min_props_required"]
            if props_count < min_required:
                self.failures.append(
                    f"Insufficient props: {props_count} found, {min_required} required"
                )
                result["status"] = "FAILED"
            else:
                result["status"] = "PASSED"
                self.logger.info(f"✅ Props freshness: {props_count} props from {today}")

            cursor.close()
            conn.close()

        except Exception as e:
            self.failures.append(f"Props freshness check failed: {str(e)}")
            result["status"] = "ERROR"
            result["error"] = str(e)

        return result

    def _validate_game_results_freshness(self) -> Dict[str, any]:
        """Validate game results are within acceptable age."""
        result = {"status": "pending", "latest_game_date": None, "age_hours": None}

        try:
            conn = psycopg2.connect(**PLAYERS_DB_CONFIG)
            cursor = conn.cursor()

            # Get latest game date from current season
            # Note: Games in late 2025 are stored as season = 2025 (from game_date year)
            query = """
                SELECT MAX(game_date)
                FROM player_game_logs
                WHERE season >= 2025
            """
            cursor.execute(query)
            latest_game = cursor.fetchone()[0]

            if latest_game:
                result["latest_game_date"] = str(latest_game)

                # Calculate age in hours
                age = datetime.now().date() - latest_game
                age_hours = age.total_seconds() / 3600

                result["age_hours"] = round(age_hours, 1)

                # Check against policy
                max_age = self.policy["max_game_result_age_hours"]
                if age_hours > max_age:
                    self.warnings.append(
                        f"Game results are {age_hours:.1f} hours old (max: {max_age})"
                    )
                    result["status"] = "WARNING"
                else:
                    result["status"] = "PASSED"
                    self.logger.info(
                        f"✅ Game results: Latest from {latest_game} ({age_hours:.1f}h old)"
                    )
            else:
                self.failures.append("No game results found in database")
                result["status"] = "FAILED"

            cursor.close()
            conn.close()

        except Exception as e:
            self.warnings.append(f"Game results check failed: {str(e)}")
            result["status"] = "ERROR"
            result["error"] = str(e)

        return result

    def _validate_injury_reports_freshness(self) -> Dict[str, any]:
        """Validate injury reports are current."""
        result = {"status": "pending", "latest_update": None, "age_hours": None}

        if not self.policy["injuries_required"]:
            result["status"] = "SKIPPED"
            return result

        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()

            # Check latest injury report update
            # NOTE: Using injury_report table (has current data) instead of injuries table (empty)
            # Status values are uppercase in DB: OUT, DAY_TO_DAY, QUESTIONABLE, DOUBTFUL, SUSPENSION
            query = """
                SELECT MAX(updated_at)
                FROM injury_report
                WHERE UPPER(status) IN ('OUT', 'QUESTIONABLE', 'DOUBTFUL', 'DAY_TO_DAY', 'SUSPENSION')
            """
            cursor.execute(query)
            latest_update = cursor.fetchone()[0]

            if latest_update:
                result["latest_update"] = latest_update.isoformat()

                # Calculate age in hours
                age = datetime.now() - latest_update
                age_hours = age.total_seconds() / 3600

                result["age_hours"] = round(age_hours, 1)

                # Check against policy
                max_age = self.policy["max_injury_report_age_hours"]
                if age_hours > max_age:
                    self.warnings.append(
                        f"Injury reports are {age_hours:.1f} hours old (max: {max_age})"
                    )
                    result["status"] = "WARNING"
                else:
                    result["status"] = "PASSED"
                    self.logger.info(f"✅ Injury reports: Updated {age_hours:.1f}h ago")
            else:
                self.warnings.append("No injury reports found in database")
                result["status"] = "WARNING"

            cursor.close()
            conn.close()

        except Exception as e:
            self.warnings.append(f"Injury reports check failed: {str(e)}")
            result["status"] = "ERROR"
            result["error"] = str(e)

        return result

    def _validate_model_age(self) -> Dict[str, any]:
        """Validate XL models are not too old."""
        result = {"status": "pending", "model_age_days": None}

        try:
            # Check model registry for last trained date
            registry_path = (
                Path(__file__).parent.parent.parent / "models" / "saved_xl" / "MODEL_REGISTRY.toml"
            )

            if not registry_path.exists():
                self.warnings.append("MODEL_REGISTRY.toml not found")
                result["status"] = "WARNING"
                return result

            # Read registry to find trained_date
            import toml

            registry = toml.load(registry_path)

            # Get latest trained date from metadata
            trained_date_str = registry.get("metadata", {}).get("updated", "")

            if trained_date_str:
                # Parse date (format: 2025-11-07T02:00:00-05:00)
                trained_date = datetime.fromisoformat(trained_date_str.replace("Z", "+00:00"))
                age = datetime.now() - trained_date.replace(tzinfo=None)
                age_days = age.days

                result["model_age_days"] = age_days
                result["trained_date"] = trained_date_str

                # Check against policy
                max_age = self.policy["max_model_age_days"]
                recommended_retrain = self.policy["recommended_retrain_frequency_days"]

                if age_days > max_age:
                    self.failures.append(
                        f"Models are {age_days} days old (max: {max_age} days). RETRAIN REQUIRED."
                    )
                    result["status"] = "FAILED"
                elif age_days > recommended_retrain:
                    self.warnings.append(
                        f"Models are {age_days} days old (recommended retrain: {recommended_retrain} days)"
                    )
                    result["status"] = "WARNING"
                else:
                    result["status"] = "PASSED"
                    self.logger.info(f"✅ Model age: {age_days} days (trained {trained_date_str})")
            else:
                self.warnings.append("Model trained date not found in registry")
                result["status"] = "WARNING"

        except Exception as e:
            self.warnings.append(f"Model age check failed: {str(e)}")
            result["status"] = "ERROR"
            result["error"] = str(e)

        return result

    def print_report(self, results: Dict[str, any]):
        """Print validation report."""
        print("\n" + "=" * 80)
        print("DATA FRESHNESS VALIDATION REPORT")
        print("=" * 80)

        # Overall status
        if results["success"]:
            print(f"\n✅ VALIDATION PASSED - All checks successful\n")
        else:
            print(f"\n❌ VALIDATION FAILED - {len(self.failures)} critical failure(s)\n")

        # Props freshness
        props = results["props_freshness"]
        print(f"Props Freshness: [{props['status']}]")
        print(f"  - Props Count: {props['props_count']}")
        print(f"  - Props Date: {props['props_date']}")

        # Game results freshness
        games = results["game_results_freshness"]
        print(f"\nGame Results: [{games['status']}]")
        print(f"  - Latest Game: {games.get('latest_game_date', 'N/A')}")
        print(f"  - Age: {games.get('age_hours', 'N/A')} hours")

        # Injury reports
        injuries = results["injury_reports_freshness"]
        print(f"\nInjury Reports: [{injuries['status']}]")
        if injuries.get("age_hours"):
            print(f"  - Age: {injuries['age_hours']} hours")

        # Model age
        models = results["model_age"]
        print(f"\nModel Age: [{models['status']}]")
        if models.get("model_age_days") is not None:
            print(f"  - Age: {models['model_age_days']} days")

        # Failures
        if self.failures:
            print("\n" + "=" * 80)
            print("CRITICAL FAILURES:")
            print("=" * 80)
            for failure in self.failures:
                print(f"  ❌ {failure}")

        # Warnings
        if self.warnings:
            print("\n" + "=" * 80)
            print("WARNINGS:")
            print("=" * 80)
            for warning in self.warnings:
                print(f"  ⚠️  {warning}")

        print("\n" + "=" * 80)
        print(f"Validation completed at: {results['timestamp']}")
        print("=" * 80 + "\n")


def main():
    """Run data freshness validation."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    validator = DataFreshnessValidator()
    success, results = validator.validate_all()

    validator.print_report(results)

    # Exit with appropriate code
    if success:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
