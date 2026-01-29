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
import os
import pickle
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

from nba.features.extract_live_features_xl import LiveFeatureExtractorXL

# Database connections
DB_INTELLIGENCE = {
    "host": "localhost",
    "port": 5539,
    "database": "nba_intelligence",
    "user": os.getenv("DB_USER", "nba_user"),
    "password": os.getenv("DB_PASSWORD"),
}

DB_PLAYERS = {
    "host": "localhost",
    "port": 5536,
    "database": "nba_players",
    "user": os.getenv("DB_USER", "nba_user"),
    "password": os.getenv("DB_PASSWORD"),
}

MODELS_DIR = Path(__file__).parent.parent / "models" / "saved_xl"
OUTPUT_DIR = Path(__file__).parent

MARKETS = ["points", "rebounds", "assists", "threes"]


class XLMarketValidator:
    """Validates a single market's XL model"""

    def __init__(self, market):
        self.market = market
        self.market_key = market.upper()

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
        """Load XL model components"""
        try:
            # Try new naming convention first (*_market_*), then fall back to old (*_xl_*)
            model_prefix = MODELS_DIR / f"{self.market}_market"
            if not (MODELS_DIR / f"{self.market}_market_regressor.pkl").exists():
                model_prefix = MODELS_DIR / f"{self.market}_xl"
                print(f"[INFO] {self.market_key}: Using legacy *_xl_* naming")

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

            print(f"[OK] {self.market_key}: Loaded model ({len(self.features)} features)")
            return True
        except Exception as e:
            print(f"[ERROR] {self.market_key}: Failed to load model: {e}")
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

        except Exception as e:
            print(f"[WARN]  Prediction error: {e}")
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

        print(f"\n🔮 {self.market_key}: Generating predictions for {len(props_df)} props...")

        for idx, row in props_df.iterrows():
            if idx % 100 == 0:
                print(f"   Progress: {idx}/{len(props_df)}")

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

            except Exception as e:
                # Skip props where feature extraction fails
                continue

        return pd.DataFrame(results)


class XLHistoricalValidator:
    """Main validator for all markets"""

    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date
        self.conn = None
        self.market_validators = {}

        # Initialize validators for all markets
        for market in MARKETS:
            self.market_validators[market.upper()] = XLMarketValidator(market)

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
        """Run validation on all markets"""
        print("=" * 100)
        print(f"VALIDATING XL MODELS: {self.start_date} to {self.end_date}")
        print("=" * 100)

        self.connect_db()

        all_results = {}

        for market_key, validator in self.market_validators.items():
            print(f"\n{'='*100}")
            print(f"{market_key} VALIDATION")
            print(f"{'='*100}")

            # Fetch props
            props_df = self.fetch_props(market_key)
            print(f"[DATA] Loaded {len(props_df)} {market_key} props with actuals")

            if len(props_df) == 0:
                continue

            # Validate
            results_df = validator.validate_props(props_df)

            if len(results_df) == 0:
                print(f"[WARN]  No predictions generated for {market_key}")
                continue

            print(f"[OK] Generated {len(results_df)} predictions")

            # Store results
            all_results[market_key] = results_df

            # Save to CSV
            output_file = OUTPUT_DIR / f"validation_{market_key}_oct23_nov4.csv"
            results_df.to_csv(output_file, index=False)
            print(f"💾 Saved to: {output_file}")

        self.conn_intelligence.close()
        self.conn_players.close()

        return all_results

    def generate_reports(self, all_results):
        """Generate comprehensive validation reports"""
        print("\n" + "=" * 100)
        print("GENERATING REPORTS")
        print("=" * 100)

        report_lines = []
        report_lines.append("# NBA XL Models Validation Report")
        report_lines.append(f"**Period:** {self.start_date} to {self.end_date}")
        report_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")

        # Overall summary
        total_bets = sum(len(df) for df in all_results.values())
        total_wins = sum(df["won"].sum() for df in all_results.values())
        overall_wr = (total_wins / total_bets * 100) if total_bets > 0 else 0
        overall_roi = (
            ((total_wins * 0.91 - (total_bets - total_wins)) / total_bets * 100)
            if total_bets > 0
            else 0
        )

        report_lines.append("## Overall Performance")
        report_lines.append(f"- **Total Bets:** {total_bets}")
        report_lines.append(f"- **Wins:** {total_wins}")
        report_lines.append(f"- **Losses:** {total_bets - total_wins}")
        report_lines.append(f"- **Win Rate:** {overall_wr:.1f}%")
        report_lines.append(f"- **ROI @ -110:** {overall_roi:+.1f}%")
        report_lines.append(
            f"- **Status:** {'[OK] PROFITABLE' if overall_wr >= 52.4 else '[ERROR] UNPROFITABLE'}"
        )
        report_lines.append("")

        # Per-market detailed analysis
        for market_key, df in all_results.items():
            report_lines.extend(self._generate_market_report(market_key, df))

        # Daily performance matrix
        report_lines.extend(self._generate_daily_matrix(all_results))

        # Save report
        report_file = OUTPUT_DIR / "validation_summary_oct23_nov4.md"
        with open(report_file, "w") as f:
            f.write("\n".join(report_lines))

        print(f"📄 Report saved to: {report_file}")

        # Print to console
        print("\n" + "\n".join(report_lines))

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
        """Generate daily performance matrix across all markets"""
        lines = []

        lines.append("## Daily Performance Matrix")
        lines.append("")
        lines.append("```")

        # Header
        header = f"{'Date':<12} |"
        for market_key in ["POINTS", "REBOUNDS", "ASSISTS", "THREES"]:
            if market_key in all_results:
                header += f" {market_key[:8]:<8} |"
        header += " TOTAL"

        lines.append(header)

        sep = f"{'':<12} |"
        for market_key in ["POINTS", "REBOUNDS", "ASSISTS", "THREES"]:
            if market_key in all_results:
                sep += f" {'Bets | WR':<8} |"
        sep += " WR"
        lines.append(sep)

        lines.append("-" * len(header))

        # Get all unique dates
        all_dates = set()
        for df in all_results.values():
            all_dates.update(df["game_date"].unique())

        # Per-date rows
        for date in sorted(all_dates):
            row = f"{date}  |"
            total_bets = 0
            total_wins = 0

            for market_key in ["POINTS", "REBOUNDS", "ASSISTS", "THREES"]:
                if market_key not in all_results:
                    row += f" {'':<8} |"
                    continue

                df = all_results[market_key]
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

        for market_key in ["POINTS", "REBOUNDS", "ASSISTS", "THREES"]:
            if market_key not in all_results:
                total_row += f" {'':<8} |"
                continue

            df = all_results[market_key]
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
    parser = argparse.ArgumentParser(description="Validate XL models on historical props")
    parser.add_argument("--start-date", default="2025-10-23", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default="2025-11-04", help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    validator = XLHistoricalValidator(args.start_date, args.end_date)
    results = validator.validate_all_markets()
    validator.generate_reports(results)

    print("\n[OK] Validation complete!")
