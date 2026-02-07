#!/usr/bin/env python3
"""
NBA XL Models Historical Validator
===================================
Validates XL models on Oct 23 - Nov 4, 2025 historical data.

Generates per-market, per-date analysis with edge tier breakdowns.

CRITICAL: Prevents data leakage by extracting features AS OF historical game date.

Usage:
    python3 nba/betting/validate_xl_models.py --start-date 2025-10-23 --end-date 2025-11-04
"""

import argparse
import json
import pickle
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

from nba.config.database import get_intelligence_db_config, get_players_db_config
from nba.core.logging_config import get_logger, setup_logging
from nba.features.extract_live_features_xl import LiveFeatureExtractorXL

# Initialize logger
logger = get_logger(__name__)

# Database connections
DB_INTELLIGENCE = get_intelligence_db_config()
DB_PLAYERS = get_players_db_config()

MODELS_DIR = Path(__file__).parent.parent / "models" / "saved_xl"
OUTPUT_DIR = Path(__file__).parent

MARKETS = ["points", "rebounds", "assists", "threes"]
MODEL_VERSIONS = ["xl", "v3"]  # Both XL (102 features) and V3 (136 features)


class XLMarketValidator:
    """Validates a single market's XL or V3 model"""

    def __init__(self, market, model_version="xl"):
        self.market = market
        self.market_key = market.upper()
        self.model_version = model_version  # "xl" or "v3"

        # Model components
        self.regressor = None
        self.classifier = None
        self.calibrator = None
        self.imputer = None
        self.scaler = None
        self.features = None

        # Feature extractor
        self.feature_extractor = LiveFeatureExtractorXL()

        self.load_model()

    def load_model(self):
        """Load XL or V3 model components based on model_version"""
        try:
            # Load model based on specified version
            model_prefix = MODELS_DIR / f"{self.market}_{self.model_version}"
            if not (
                model_prefix.parent / f"{self.market}_{self.model_version}_regressor.pkl"
            ).exists():
                logger.error(
                    f"Model not found: {self.market}_{self.model_version}",
                    extra={"market": self.market_key, "version": self.model_version},
                )
                return False
            logger.info(
                f"Loading {self.model_version.upper()} model",
                extra={"market": self.market_key, "version": self.model_version},
            )

            with open(f"{model_prefix}_regressor.pkl", "rb") as f:
                self.regressor = pickle.load(f)
            with open(f"{model_prefix}_classifier.pkl", "rb") as f:
                self.classifier = pickle.load(f)
            with open(f"{model_prefix}_calibrator.pkl", "rb") as f:
                self.calibrator = pickle.load(f)
            with open(f"{model_prefix}_imputer.pkl", "rb") as f:
                self.imputer = pickle.load(f)
            with open(f"{model_prefix}_scaler.pkl", "rb") as f:
                self.scaler = pickle.load(f)
            with open(f"{model_prefix}_features.pkl", "rb") as f:
                self.features = pickle.load(f)

            logger.info(
                "Loaded model successfully",
                extra={"market": self.market_key, "feature_count": len(self.features)},
            )
            return True
        except (psycopg2.Error, KeyError, TypeError, ValueError) as e:
            logger.error("Failed to load model", extra={"market": self.market_key, "error": str(e)})
            return False

    def predict(self, features_dict, line):
        """
        Generate prediction using XL model pipeline

        Args:
            features_dict: Dict of features extracted by LiveFeatureExtractorXL
            line: Market line (float)

        Returns:
            (prediction, prob_over, side, edge)
        """
        try:
            # Build feature vector in model's expected order
            feature_vector = []
            for feat in self.features:
                feature_vector.append(features_dict.get(feat, np.nan))

            X = pd.DataFrame([feature_vector], columns=self.features)

            # Preprocess
            X_imputed = self.imputer.transform(X)
            X_scaled = self.scaler.transform(X_imputed)

            # Stage 1: Regressor predicts stat value
            base_pred = self.regressor.predict(X_scaled)[0]

            # Stage 2: Classifier predicts P(actual > line)
            # IMPORTANT: Replace expected_diff placeholder, don't append (mimics training)
            expected_diff = base_pred - line

            # Convert to DataFrame to allow column replacement
            X_cls_df = pd.DataFrame(X_scaled, columns=self.features)
            X_cls_df["expected_diff"] = expected_diff  # Replaces placeholder at position 101
            X_cls = X_cls_df.values  # Convert back to array with 102 features

            prob_over_raw = self.classifier.predict_proba(X_cls)[0, 1]

            # Stage 3: Calibrate probability
            prob_over = self.calibrator.transform([prob_over_raw])[0]

            # Determine side and edge
            if base_pred > line:
                side = "OVER"
                edge = base_pred - line
            else:
                side = "UNDER"
                edge = line - base_pred

            return base_pred, prob_over, side, abs(edge)

        except (psycopg2.Error, KeyError, TypeError, ValueError) as e:
            logger.warning("Prediction error", extra={"error": str(e)})
            return None, None, None, None

    def validate_props(self, props_df):
        """
        Validate all props for this market

        Args:
            props_df: DataFrame with columns [player_name, game_date, line, actual_result, opponent_team, is_home]

        Returns:
            DataFrame with predictions and results
        """
        results = []

        logger.info(
            "Generating predictions for market",
            extra={"market": self.market_key, "prop_count": len(props_df)},
        )

        for idx, row in props_df.iterrows():
            if idx % 100 == 0:
                logger.debug(
                    "Validation progress",
                    extra={"market": self.market_key, "processed": idx, "total": len(props_df)},
                )

            try:
                # CRITICAL: Extract features AS OF the game date (prevent leakage)
                game_date = pd.to_datetime(row["game_date"])

                features = self.feature_extractor.extract_features(
                    player_name=row["player_name"],
                    game_date=game_date,  # ← Historical date, not today!
                    is_home=row.get("is_home"),
                    opponent_team=row.get("opponent_team"),
                    line=row["line"],
                )

                # Skip if no features extracted
                if features is None:
                    continue

                # Generate prediction
                prediction, prob_over, side, edge = self.predict(features, row["line"])

                if prediction is None:
                    continue

                # Check win
                actual = row["actual_result"]
                if side == "OVER":
                    won = actual > row["line"]
                else:
                    won = actual < row["line"]

                results.append(
                    {
                        "game_date": row["game_date"],
                        "player_name": row["player_name"],
                        "line": row["line"],
                        "actual": actual,
                        "prediction": prediction,
                        "prob_over": prob_over,
                        "side": side,
                        "edge": edge,
                        "won": won,
                    }
                )

            except (psycopg2.Error, KeyError, TypeError, ValueError) as e:
                # Skip props where feature extraction fails
                continue

        return pd.DataFrame(results)


class XLHistoricalValidator:
    """Main validator for all markets - supports both XL and V3 models"""

    def __init__(self, start_date, end_date, model_versions=None):
        self.start_date = start_date
        self.end_date = end_date
        self.conn = None
        self.model_versions = model_versions or ["xl", "v3"]  # Run both by default
        self.market_validators = {}  # {(market, version): validator}

        # Initialize validators for all markets AND all model versions
        for market in MARKETS:
            for version in self.model_versions:
                key = (market.upper(), version)
                self.market_validators[key] = XLMarketValidator(market, model_version=version)

    def connect_db(self):
        """Connect to intelligence and players databases"""
        self.conn_intelligence = psycopg2.connect(**DB_INTELLIGENCE)
        self.conn_players = psycopg2.connect(**DB_PLAYERS)

    def fetch_props(self, stat_type):
        """
        Fetch props with actuals for a specific stat type

        Args:
            stat_type: 'POINTS', 'REBOUNDS', 'ASSISTS', or 'THREES'

        Returns:
            DataFrame with props
        """
        # Fetch from nba_props_xl table (aggregated per player/date/stat)
        query_props = """
        SELECT DISTINCT ON (player_name, game_date)
            player_name,
            game_date,
            consensus_line as line,
            actual_value as actual_result,
            opponent_team,
            is_home
        FROM nba_props_xl
        WHERE game_date BETWEEN %s AND %s
          AND stat_type = %s
          AND actual_value IS NOT NULL
        ORDER BY player_name, game_date, fetch_timestamp DESC
        """

        df = pd.read_sql_query(
            query_props, self.conn_intelligence, params=(self.start_date, self.end_date, stat_type)
        )

        return df[["game_date", "player_name", "line", "actual_result", "opponent_team", "is_home"]]

    def validate_all_markets(self):
        """Run validation on all markets for both XL and V3 models"""
        logger.info(
            "Starting XL/V3 model validation",
            extra={
                "start_date": self.start_date,
                "end_date": self.end_date,
                "versions": self.model_versions,
            },
        )

        self.connect_db()

        all_results = {}  # {(market, version): DataFrame}

        for (market_key, version), validator in self.market_validators.items():
            # Only validate POINTS and REBOUNDS (ASSISTS/THREES disabled)
            if market_key not in ["POINTS", "REBOUNDS"]:
                continue

            logger.info(
                f"Starting {version.upper()} validation",
                extra={"market": market_key, "version": version},
            )

            # Fetch props (same for both versions)
            props_df = self.fetch_props(market_key)
            logger.info(
                "Loaded props with actuals",
                extra={"market": market_key, "version": version, "count": len(props_df)},
            )

            if len(props_df) == 0:
                continue

            # Validate
            results_df = validator.validate_props(props_df)

            if len(results_df) == 0:
                logger.warning(
                    "No predictions generated",
                    extra={"market": market_key, "version": version},
                )
                continue

            # Add model_version column
            results_df["model_version"] = version

            logger.info(
                "Generated predictions",
                extra={"market": market_key, "version": version, "count": len(results_df)},
            )

            # Store results
            all_results[(market_key, version)] = results_df

            # Save to CSV
            date_suffix = f"{self.start_date}_to_{self.end_date}".replace("-", "")
            output_file = OUTPUT_DIR / f"validation_{market_key}_{version}_{date_suffix}.csv"
            results_df.to_csv(output_file, index=False)
            logger.info(
                "Saved validation results",
                extra={"market": market_key, "version": version, "file": str(output_file)},
            )

        self.conn_intelligence.close()
        self.conn_players.close()

        return all_results

    def generate_reports(self, all_results):
        """Generate comprehensive validation reports comparing XL vs V3"""
        logger.info("Generating validation reports")

        report_lines = []
        report_lines.append("# NBA XL vs V3 Models Validation Report")
        report_lines.append(f"**Period:** {self.start_date} to {self.end_date}")
        report_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")

        # XL vs V3 Comparison Summary
        report_lines.append("## XL vs V3 Comparison")
        report_lines.append("")
        report_lines.append("| Model | Market | Bets | Wins | Win Rate | ROI | Avg Edge |")
        report_lines.append("|-------|--------|------|------|----------|-----|----------|")

        for version in self.model_versions:
            for market in ["POINTS", "REBOUNDS"]:
                key = (market, version)
                if key not in all_results:
                    continue
                df = all_results[key]
                total = len(df)
                wins = int(df["won"].sum())
                wr = (wins / total * 100) if total > 0 else 0
                roi = ((wins * 0.91 - (total - wins)) / total * 100) if total > 0 else 0
                avg_edge = df["edge"].mean() if total > 0 else 0
                report_lines.append(
                    f"| {version.upper()} | {market} | {total} | {wins} | {wr:.1f}% | {roi:+.1f}% | {avg_edge:.2f} |"
                )

        report_lines.append("")

        # Overall summary by model version
        report_lines.append("## Overall by Model Version")
        report_lines.append("")

        for version in self.model_versions:
            version_results = {k: v for k, v in all_results.items() if k[1] == version}
            total_bets = sum(len(df) for df in version_results.values())
            total_wins = sum(df["won"].sum() for df in version_results.values())
            overall_wr = (total_wins / total_bets * 100) if total_bets > 0 else 0
            overall_roi = (
                ((total_wins * 0.91 - (total_bets - total_wins)) / total_bets * 100)
                if total_bets > 0
                else 0
            )
            status = "[OK] PROFITABLE" if overall_wr >= 52.4 else "[WARN] UNPROFITABLE"

            report_lines.append(
                f"### {version.upper()} Model ({'102' if version == 'xl' else '136'} features)"
            )
            report_lines.append(f"- **Total Bets:** {total_bets}")
            report_lines.append(f"- **Wins:** {int(total_wins)}")
            report_lines.append(f"- **Win Rate:** {overall_wr:.1f}%")
            report_lines.append(f"- **ROI @ -110:** {overall_roi:+.1f}%")
            report_lines.append(f"- **Status:** {status}")
            report_lines.append("")

        # Per-market detailed analysis (grouped by market, showing both versions)
        for market in ["POINTS", "REBOUNDS"]:
            report_lines.append(f"---")
            report_lines.append(f"## {market} Market - Detailed Analysis")
            report_lines.append("")
            for version in self.model_versions:
                key = (market, version)
                if key in all_results:
                    report_lines.extend(
                        self._generate_market_report(
                            f"{market} ({version.upper()})", all_results[key]
                        )
                    )

        # Daily performance matrix (combined)
        report_lines.extend(self._generate_daily_matrix(all_results))

        # Save report
        date_suffix = f"{self.start_date}_to_{self.end_date}".replace("-", "")
        report_file = OUTPUT_DIR / f"validation_xl_v3_{date_suffix}.md"
        with open(report_file, "w") as f:
            f.write("\n".join(report_lines))

        logger.info("Report saved", extra={"file": str(report_file)})

        # Log key metrics per version
        for version in self.model_versions:
            version_results = {k: v for k, v in all_results.items() if k[1] == version}
            total_bets = sum(len(df) for df in version_results.values())
            total_wins = sum(df["won"].sum() for df in version_results.values())
            overall_wr = (total_wins / total_bets * 100) if total_bets > 0 else 0
            overall_roi = (
                ((total_wins * 0.91 - (total_bets - total_wins)) / total_bets * 100)
                if total_bets > 0
                else 0
            )
            logger.info(
                f"{version.upper()} validation results",
                extra={
                    "version": version,
                    "total_bets": total_bets,
                    "wins": int(total_wins),
                    "losses": int(total_bets - total_wins),
                    "win_rate": round(overall_wr, 1),
                    "roi": round(overall_roi, 1),
                    "profitable": overall_wr >= 52.4,
                },
            )

    def _generate_market_report(self, market_key, df):
        """Generate detailed report for a single market"""
        lines = []

        lines.append(f"## {market_key} Market Analysis")
        lines.append("")

        # Overall stats
        total = len(df)
        wins = df["won"].sum()
        losses = total - wins
        wr = (wins / total * 100) if total > 0 else 0
        roi = ((wins * 0.91 - losses) / total * 100) if total > 0 else 0

        lines.append(
            f"**Overall:** {total} bets | {wins}W-{losses}L | {wr:.1f}% WR | {roi:+.1f}% ROI"
        )
        lines.append("")

        # By date
        lines.append("### Performance by Date")
        lines.append("```")
        lines.append(
            f"{'Date':<12} {'Bets':<6} {'Wins':<6} {'Losses':<8} {'Win Rate':<10} {'ROI':<10}"
        )
        lines.append("-" * 70)

        for date in sorted(df["game_date"].unique()):
            day_df = df[df["game_date"] == date]
            day_total = len(day_df)
            day_wins = day_df["won"].sum()
            day_losses = day_total - day_wins
            day_wr = (day_wins / day_total * 100) if day_total > 0 else 0
            day_roi = ((day_wins * 0.91 - day_losses) / day_total * 100) if day_total > 0 else 0

            lines.append(
                f"{date}  {day_total:<6} {day_wins:<6} {day_losses:<8} {day_wr:>6.1f}%     {day_roi:>+7.1f}%"
            )

        lines.append("```")
        lines.append("")

        # By edge tier
        lines.append("### Performance by Edge Tier")
        lines.append("```")
        lines.append(
            f"{'Edge Tier':<12} {'Bets':<6} {'Wins':<6} {'Win Rate':<10} {'ROI':<10} {'Avg Edge':<10} {'Recommend':<12}"
        )
        lines.append("-" * 80)

        edge_tiers = [
            (7.0, 999, "7.0+"),
            (5.0, 7.0, "5.0-6.9"),
            (3.0, 5.0, "3.0-4.9"),
            (2.0, 3.0, "2.0-2.9"),
            (1.0, 2.0, "1.0-1.9"),
            (0.5, 1.0, "0.5-0.9"),
            (0.0, 0.5, "<0.5"),
        ]

        for min_edge, max_edge, label in edge_tiers:
            tier_df = df[(df["edge"] >= min_edge) & (df["edge"] < max_edge)]
            if len(tier_df) == 0:
                continue

            tier_total = len(tier_df)
            tier_wins = tier_df["won"].sum()
            tier_wr = (tier_wins / tier_total * 100) if tier_total > 0 else 0
            tier_roi = (
                ((tier_wins * 0.91 - (tier_total - tier_wins)) / tier_total * 100)
                if tier_total > 0
                else 0
            )
            avg_edge = tier_df["edge"].mean()

            # Recommendation
            if tier_wr >= 60 and tier_total >= 20:
                recommend = "[OK] ENABLE"
            elif tier_wr >= 55 and tier_total >= 10:
                recommend = "[WARN]  CAUTION"
            else:
                recommend = "[ERROR] DISABLE"

            lines.append(
                f"{label:<12} {tier_total:<6} {tier_wins:<6} {tier_wr:>6.1f}%     {tier_roi:>+7.1f}%   {avg_edge:>6.2f}       {recommend}"
            )

        lines.append("```")
        lines.append("")

        # By direction
        lines.append("### Performance by Direction")
        lines.append("```")
        lines.append(f"{'Direction':<12} {'Bets':<6} {'Wins':<6} {'Win Rate':<10} {'ROI':<10}")
        lines.append("-" * 60)

        for side in ["OVER", "UNDER"]:
            side_df = df[df["side"] == side]
            if len(side_df) == 0:
                continue

            side_total = len(side_df)
            side_wins = side_df["won"].sum()
            side_wr = (side_wins / side_total * 100) if side_total > 0 else 0
            side_roi = (
                ((side_wins * 0.91 - (side_total - side_wins)) / side_total * 100)
                if side_total > 0
                else 0
            )

            lines.append(
                f"{side:<12} {side_total:<6} {side_wins:<6} {side_wr:>6.1f}%     {side_roi:>+7.1f}%"
            )

        lines.append("```")
        lines.append("")

        # Recommendation
        if wr >= 60:
            recommendation = f"[OK] **ENABLE** with min_edge = 3.0+"
        elif wr >= 55:
            recommendation = f"[WARN]  **CAUTION** - More validation needed"
        else:
            recommendation = f"[ERROR] **DISABLE** - Underperforming"

        lines.append(f"**Recommendation:** {recommendation}")
        lines.append("")
        lines.append("---")
        lines.append("")

        return lines

    def _generate_daily_matrix(self, all_results):
        """Generate daily performance matrix comparing XL vs V3"""
        lines = []

        lines.append("## Daily Performance Matrix (XL vs V3)")
        lines.append("")

        # Generate matrix for each version
        for version in self.model_versions:
            lines.append(f"### {version.upper()} Model Daily Breakdown")
            lines.append("```")

            # Header
            header = f"{'Date':<12} |"
            for market_key in ["POINTS", "REBOUNDS"]:
                key = (market_key, version)
                if key in all_results:
                    header += f" {market_key[:8]:<8} |"
            header += " TOTAL"
            lines.append(header)

            lines.append("-" * len(header))

            # Get all unique dates for this version
            all_dates = set()
            for (_market, ver), df in all_results.items():
                if ver == version:
                    all_dates.update(df["game_date"].unique())

            # Per-date rows
            for date in sorted(all_dates):
                row = f"{date}  |"
                total_bets = 0
                total_wins = 0

                for market_key in ["POINTS", "REBOUNDS"]:
                    key = (market_key, version)
                    if key not in all_results:
                        row += f" {'':<8} |"
                        continue

                    df = all_results[key]
                    day_df = df[df["game_date"] == date]

                    if len(day_df) == 0:
                        row += f" {'':<8} |"
                        continue

                    day_total = len(day_df)
                    day_wins = day_df["won"].sum()
                    day_wr = (day_wins / day_total * 100) if day_total > 0 else 0

                    total_bets += day_total
                    total_wins += day_wins

                    row += f" {day_total:>3} {day_wr:>4.0f}% |"

                total_wr = (total_wins / total_bets * 100) if total_bets > 0 else 0
                row += f" {total_wr:>5.1f}%"
                lines.append(row)

            # Totals
            lines.append("-" * len(header))
            total_row = f"{'TOTAL':<12} |"
            grand_total_bets = 0
            grand_total_wins = 0

            for market_key in ["POINTS", "REBOUNDS"]:
                key = (market_key, version)
                if key not in all_results:
                    total_row += f" {'':<8} |"
                    continue

                df = all_results[key]
                market_total = len(df)
                market_wins = df["won"].sum()
                market_wr = (market_wins / market_total * 100) if market_total > 0 else 0

                grand_total_bets += market_total
                grand_total_wins += market_wins

                total_row += f" {market_total:>3} {market_wr:>4.0f}% |"

            grand_wr = (grand_total_wins / grand_total_bets * 100) if grand_total_bets > 0 else 0
            total_row += f" {grand_wr:>5.1f}%"
            lines.append(total_row)

            lines.append("```")
            lines.append("")

        return lines


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate XL and V3 models on historical props")
    parser.add_argument("--start-date", default="2025-10-23", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default="2025-11-04", help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--models",
        default="xl,v3",
        help="Model versions to validate (comma-separated: xl,v3 or just xl or just v3)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    # Parse model versions
    model_versions = [v.strip().lower() for v in args.models.split(",")]
    valid_versions = [v for v in model_versions if v in ["xl", "v3"]]
    if not valid_versions:
        print("Error: No valid model versions specified. Use --models xl,v3")
        sys.exit(1)

    # Setup logging
    import logging

    setup_logging(
        "xl_v3_validation",
        level=logging.DEBUG if args.debug else logging.INFO,
    )

    logger.info(f"Starting validation for models: {', '.join(v.upper() for v in valid_versions)}")

    validator = XLHistoricalValidator(args.start_date, args.end_date, model_versions=valid_versions)
    results = validator.validate_all_markets()
    validator.generate_reports(results)

    logger.info("Validation complete")
