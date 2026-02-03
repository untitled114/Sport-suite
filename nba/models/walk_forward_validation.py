#!/usr/bin/env python3
"""
Walk-Forward Validation for NBA Props Models

Implements expanding window walk-forward validation to assess model stability
across different time periods. Essential for betting models where markets evolve.

Walk-Forward Process:
1. Train on data from start to time T
2. Test on data from T to T+window
3. Expand training window, repeat
4. Report metrics per fold to show consistency

Usage:
    python -m nba.models.walk_forward_validation --market POINTS
    python -m nba.models.walk_forward_validation --market REBOUNDS --folds 6
    python -m nba.models.walk_forward_validation --market POINTS --test-months 2

Author: Claude Code
"""

import argparse
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nba.core.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class FoldResult:
    """Results from a single validation fold."""

    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_samples: int
    test_samples: int
    auc: float
    accuracy: float
    win_rate: float
    roi: float
    edge_bets: int
    edge_win_rate: float


@dataclass
class WalkForwardResults:
    """Aggregated walk-forward validation results."""

    market: str
    n_folds: int
    folds: list[FoldResult] = field(default_factory=list)

    @property
    def mean_auc(self) -> float:
        return np.mean([f.auc for f in self.folds])

    @property
    def std_auc(self) -> float:
        return np.std([f.auc for f in self.folds])

    @property
    def mean_win_rate(self) -> float:
        return np.mean([f.win_rate for f in self.folds])

    @property
    def mean_edge_win_rate(self) -> float:
        rates = [f.edge_win_rate for f in self.folds if f.edge_bets > 0]
        return np.mean(rates) if rates else 0.0

    @property
    def total_roi(self) -> float:
        """Calculate overall ROI across all folds."""
        total_bets = sum(f.test_samples for f in self.folds)
        total_profit = sum(f.roi * f.test_samples for f in self.folds)
        return total_profit / total_bets if total_bets > 0 else 0.0

    def summary(self) -> str:
        """Generate summary report."""
        lines = [
            "=" * 70,
            f"WALK-FORWARD VALIDATION RESULTS: {self.market}",
            "=" * 70,
            f"Folds: {self.n_folds}",
            f"Mean AUC: {self.mean_auc:.3f} (+/- {self.std_auc:.3f})",
            f"Mean Win Rate: {self.mean_win_rate:.1%}",
            f"Mean Edge Win Rate: {self.mean_edge_win_rate:.1%}",
            f"Total ROI: {self.total_roi:+.2%}",
            "",
            "Per-Fold Results:",
            "-" * 70,
            f"{'Fold':<5} {'Train Period':<23} {'Test Period':<23} {'AUC':<7} {'WR':<7} {'Edge WR':<8}",
            "-" * 70,
        ]

        for f in self.folds:
            train_period = f"{f.train_start[:10]} to {f.train_end[:10]}"
            test_period = f"{f.test_start[:10]} to {f.test_end[:10]}"
            edge_wr = f"{f.edge_win_rate:.1%}" if f.edge_bets > 0 else "N/A"
            lines.append(
                f"{f.fold:<5} {train_period:<23} {test_period:<23} "
                f"{f.auc:.3f}   {f.win_rate:.1%}   {edge_wr:<8}"
            )

        lines.append("-" * 70)

        # Stability assessment
        auc_stable = self.std_auc < 0.05
        wr_consistent = all(f.win_rate > 0.5 for f in self.folds)

        lines.append("")
        lines.append("Stability Assessment:")
        lines.append(
            f"  AUC Stability: {'✅ STABLE' if auc_stable else '⚠️  UNSTABLE'} "
            f"(std={self.std_auc:.3f})"
        )
        lines.append(
            f"  Win Rate Consistency: {'✅ ALL FOLDS >50%' if wr_consistent else '⚠️  SOME FOLDS <50%'}"
        )

        return "\n".join(lines)


class WalkForwardValidator:
    """
    Walk-forward validation for time series betting models.

    Uses expanding window: train on all data up to T, test on T to T+window.
    """

    def __init__(
        self,
        n_folds: int = 6,
        test_months: int = 3,
        min_train_months: int = 9,
        edge_threshold: float = 0.03,
    ):
        """
        Initialize validator.

        Training data: 2023-03-01 to 2026-01-05 (~34 months)

        Default configuration:
        - 6 folds with 3-month test windows
        - 9 months minimum training (ensures enough history)
        - Covers: train 9mo→test 3mo, train 12mo→test 3mo, etc.

        Args:
            n_folds: Number of validation folds
            test_months: Months in each test window
            min_train_months: Minimum months required for training
            edge_threshold: Minimum edge for "edge bets" calculation
        """
        self.n_folds = n_folds
        self.test_months = test_months
        self.min_train_months = min_train_months
        self.edge_threshold = edge_threshold

    def create_folds(
        self, df: pd.DataFrame, date_col: str = "game_date"
    ) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Create walk-forward folds with expanding training window.

        Args:
            df: DataFrame with game_date column
            date_col: Name of date column

        Returns:
            List of (train_df, test_df) tuples
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)

        min_date = df[date_col].min()
        max_date = df[date_col].max()

        total_months = (max_date.year - min_date.year) * 12 + (max_date.month - min_date.month)

        # Calculate fold boundaries
        test_total_months = self.n_folds * self.test_months
        available_for_test = total_months - self.min_train_months

        if available_for_test < test_total_months:
            logger.warning(
                f"Not enough data for {self.n_folds} folds. "
                f"Reducing to {available_for_test // self.test_months} folds."
            )
            self.n_folds = max(1, available_for_test // self.test_months)

        folds = []

        for i in range(self.n_folds):
            # Test window starts after min_train + i*test_months
            test_start_offset = self.min_train_months + i * self.test_months
            test_end_offset = test_start_offset + self.test_months

            test_start = min_date + pd.DateOffset(months=test_start_offset)
            test_end = min_date + pd.DateOffset(months=test_end_offset)

            # Train on everything before test_start
            train_df = df[df[date_col] < test_start]
            test_df = df[(df[date_col] >= test_start) & (df[date_col] < test_end)]

            if len(train_df) > 100 and len(test_df) > 20:
                folds.append((train_df, test_df))
                logger.info(
                    f"Fold {i+1}: Train {len(train_df)} samples "
                    f"({train_df[date_col].min().date()} to {train_df[date_col].max().date()}), "
                    f"Test {len(test_df)} samples "
                    f"({test_df[date_col].min().date()} to {test_df[date_col].max().date()})"
                )

        return folds

    def train_and_evaluate_fold(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        market: str,
        fold_num: int,
    ) -> FoldResult:
        """
        Train model on fold and evaluate.

        Args:
            train_df: Training data
            test_df: Test data
            market: Market name (POINTS, REBOUNDS, etc.)
            fold_num: Fold number for logging

        Returns:
            FoldResult with metrics
        """
        from nba.models.train_market import StackedMarketModel

        # Initialize model
        model = StackedMarketModel(market=market)

        # Prepare features for train and test
        X_train, y_value_train, y_binary_train, y_residual_train, meta_train = (
            model.prepare_features(train_df)
        )
        X_test, y_value_test, y_binary_test, y_residual_test, meta_test = model.prepare_features(
            test_df
        )

        # Train the full stacked model
        metrics = model.train(
            X_train,
            y_value_train,
            y_binary_train,
            y_residual_train,
            X_test,
            y_value_test,
            y_binary_test,
            y_residual_test,
        )

        # Get predictions on test set using trained model
        # Impute and scale test data
        X_test_imputed = model.imputer.transform(X_test)
        X_test_scaled = model.scaler.transform(X_test_imputed)

        # Get regressor predictions
        test_value_preds = model.regressor.predict(X_test_scaled)

        # Calculate expected_diff and augment features for classifier
        expected_diff = test_value_preds - meta_test["line"].values
        X_test_aug = np.column_stack([X_test_scaled, expected_diff])

        # Get classifier predictions
        test_probs_raw = model.classifier.predict_proba(X_test_aug)[:, 1]

        # Calibrate
        test_probs = model.calibrator.transform(test_probs_raw)

        # Blend
        test_residuals = y_value_test.values - test_value_preds
        blend_weight = 0.6
        residual_contrib = np.clip(test_residuals / 5.0, -0.3, 0.3)
        test_probs_blended = np.clip(
            blend_weight * test_probs + (1 - blend_weight) * (0.5 + residual_contrib), 0.05, 0.95
        )

        # Calculate metrics
        test_preds_binary = (test_probs_blended > 0.5).astype(int)
        y_binary_test_arr = y_binary_test.values

        auc = roc_auc_score(y_binary_test_arr, test_probs_blended)
        accuracy = accuracy_score(y_binary_test_arr, test_preds_binary)

        # Win rate (for OVER predictions)
        over_mask = test_preds_binary == 1
        if over_mask.sum() > 0:
            win_rate = y_binary_test_arr[over_mask].mean()
        else:
            win_rate = 0.5

        # ROI calculation (assuming -110 odds)
        wins = (test_preds_binary == y_binary_test_arr).sum()
        losses = len(y_binary_test_arr) - wins
        roi = (wins * 0.91 - losses) / len(y_binary_test_arr) if len(y_binary_test_arr) > 0 else 0

        # Edge bets (high confidence)
        edge_mask = (test_probs_blended > 0.5 + self.edge_threshold) | (
            test_probs_blended < 0.5 - self.edge_threshold
        )
        edge_bets = edge_mask.sum()
        if edge_bets > 0:
            edge_preds = (test_probs_blended[edge_mask] > 0.5).astype(int)
            edge_actuals = y_binary_test_arr[edge_mask]
            edge_win_rate = (edge_preds == edge_actuals).mean()
        else:
            edge_win_rate = 0.0

        return FoldResult(
            fold=fold_num,
            train_start=str(train_df["game_date"].min()),
            train_end=str(train_df["game_date"].max()),
            test_start=str(test_df["game_date"].min()),
            test_end=str(test_df["game_date"].max()),
            train_samples=len(train_df),
            test_samples=len(test_df),
            auc=auc,
            accuracy=accuracy,
            win_rate=win_rate,
            roi=roi,
            edge_bets=edge_bets,
            edge_win_rate=edge_win_rate,
        )

    def validate(self, df: pd.DataFrame, market: str) -> WalkForwardResults:
        """
        Run full walk-forward validation.

        Args:
            df: Full dataset with game_date column
            market: Market name

        Returns:
            WalkForwardResults with all fold metrics
        """
        logger.info(f"Starting walk-forward validation for {market}")
        logger.info(f"Folds: {self.n_folds}, Test window: {self.test_months} months")

        folds = self.create_folds(df)
        results = WalkForwardResults(market=market, n_folds=len(folds))

        for i, (train_df, test_df) in enumerate(folds):
            logger.info(f"Processing fold {i+1}/{len(folds)}...")
            try:
                fold_result = self.train_and_evaluate_fold(train_df, test_df, market, i + 1)
                results.folds.append(fold_result)
                logger.info(
                    f"Fold {i+1} complete: AUC={fold_result.auc:.3f}, "
                    f"WR={fold_result.win_rate:.1%}"
                )
            except (ValueError, KeyError, RuntimeError) as e:
                logger.error(f"Fold {i+1} failed: {e}")
                continue

        return results


def main():
    parser = argparse.ArgumentParser(description="Walk-forward validation for NBA props models")
    parser.add_argument(
        "--market",
        type=str,
        required=True,
        choices=["POINTS", "REBOUNDS", "ASSISTS", "THREES"],
        help="Market to validate",
    )
    parser.add_argument(
        "--data",
        type=str,
        help="Path to training CSV (default: auto-detect)",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of validation folds (default: 5)",
    )
    parser.add_argument(
        "--test-months",
        type=int,
        default=2,
        help="Months in each test window (default: 2)",
    )
    parser.add_argument(
        "--min-train-months",
        type=int,
        default=6,
        help="Minimum months for initial training (default: 6)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for results (default: stdout)",
    )

    args = parser.parse_args()

    # Find data file
    if args.data:
        data_path = Path(args.data)
    else:
        data_path = Path(f"nba/features/datasets/xl_training_{args.market}_2023_2025.csv")

    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        sys.exit(1)

    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} samples")

    # Run validation
    validator = WalkForwardValidator(
        n_folds=args.folds,
        test_months=args.test_months,
        min_train_months=args.min_train_months,
    )

    results = validator.validate(df, args.market)

    # Output results
    summary = results.summary()
    print(summary)

    if args.output:
        with open(args.output, "w") as f:
            f.write(summary)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
