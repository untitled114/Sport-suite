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

        # Prepare features — train first, then align test to same columns
        X_train, y_value_train, y_binary_train, y_residual_train, meta_train = (
            model.prepare_features(train_df)
        )
        X_test, y_value_test, y_binary_test, y_residual_test, meta_test = model.prepare_features(
            test_df
        )
        # Align test features to train columns (some may be missing or extra per fold)
        if isinstance(X_test, pd.DataFrame) and isinstance(X_train, pd.DataFrame):
            X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

        # Extract game dates for temporal decay
        game_dates_train = (
            pd.to_datetime(meta_train["game_date"]) if "game_date" in meta_train.columns else None
        )
        game_dates_test = (
            pd.to_datetime(meta_test["game_date"]) if "game_date" in meta_test.columns else None
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
            game_dates_train=game_dates_train,
            game_dates_test=game_dates_test,
        )

        # Get predictions on test set using trained model
        # Impute and scale test data
        X_test_arr = model.imputer.transform(X_test)
        X_test_scaled_arr = model.scaler.transform(X_test_arr)

        # Get regressor predictions
        test_value_preds = model.regressor.predict(X_test_scaled_arr)

        # Calculate expected_diff and augment features for classifier
        line_values = (
            meta_test["line"].values if hasattr(meta_test, "values") else meta_test["line"]
        )
        expected_diff = test_value_preds - line_values

        # Build classifier input — append expected_diff as last column
        X_test_aug = np.column_stack([X_test_scaled_arr, expected_diff])

        # Get classifier predictions
        test_probs_raw = model.classifier.predict_proba(X_test_aug)[:, 1]

        # Calibrate if calibrator was applied
        if model.calibrator is not None:
            test_probs = model.calibrator.transform(test_probs_raw)
        else:
            test_probs = test_probs_raw

        # Use classifier-only output (blend weights from config: 1.0/0.0)
        test_probs_blended = np.clip(test_probs, 0.05, 0.95)

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
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate walk-forward charts (saved to nba/models/model_cards/images/)",
    )

    args = parser.parse_args()

    # Find data file
    if args.data:
        data_path = Path(args.data)
    else:
        data_path = Path(f"nba/features/datasets/xl_training_{args.market}_2024_present.csv")

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

    if args.plot:
        plot_results(results)


def plot_results(results: WalkForwardResults):
    """Generate walk-forward charts and markdown for GitHub."""
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")

    market = results.market
    folds = results.folds
    n = len(folds)

    if n == 0:
        print("No folds to plot")
        return

    output_dir = Path("nba/models/model_cards/images")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Data
    fold_nums = [f.fold for f in folds]
    aucs = [f.auc for f in folds]
    win_rates = [f.win_rate * 100 for f in folds]
    edge_win_rates = [f.edge_win_rate * 100 for f in folds]
    rois = [f.roi * 100 for f in folds]
    test_labels = []
    for f in folds:
        ts = f.test_start[:10] if len(f.test_start) >= 10 else f.test_start
        te = f.test_end[:10] if len(f.test_end) >= 10 else f.test_end
        test_labels.append(f"{ts[5:]}\n{te[5:]}")

    primary = "#2563EB"
    accent = "#10B981"
    bg = "#F8FAFC"

    # === Chart 1: AUC per fold ===
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    bars = ax.bar(
        fold_nums, aucs, color=primary, alpha=0.85, width=0.6, edgecolor="white", linewidth=1.5
    )
    ax.axhline(
        y=results.mean_auc,
        color=accent,
        linestyle="--",
        linewidth=2,
        label=f"Mean: {results.mean_auc:.3f} (\u00b1{results.std_auc:.3f})",
    )
    ax.axhline(
        y=0.5, color="#94A3B8", linestyle=":", linewidth=1, alpha=0.5, label="Random (0.500)"
    )

    for bar, val in zip(bars, aucs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_xlabel("Fold (test period)", fontsize=12)
    ax.set_ylabel("AUC", fontsize=12)
    ax.set_title(f"{market} \u2014 Walk-Forward AUC by Fold", fontsize=14, fontweight="bold")
    ax.set_xticks(fold_nums)
    ax.set_xticklabels(test_labels, fontsize=9)
    ax.set_ylim(0.45, max(aucs) + 0.06)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = output_dir / f"{market}_walkforward_auc.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # === Chart 2: Win Rate + Edge Win Rate ===
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    x = np.arange(n)
    width = 0.35
    bars1 = ax.bar(
        x - width / 2,
        win_rates,
        width,
        label="Overall Win Rate",
        color=primary,
        alpha=0.85,
        edgecolor="white",
        linewidth=1.5,
    )
    bars2 = ax.bar(
        x + width / 2,
        edge_win_rates,
        width,
        label="Edge Bets Win Rate",
        color=accent,
        alpha=0.85,
        edgecolor="white",
        linewidth=1.5,
    )
    ax.axhline(
        y=52.4,
        color="#EF4444",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label="Break-even (52.4%)",
    )

    for bar, val in zip(bars1, win_rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    for bar, val in zip(bars2, edge_win_rates):
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}%",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    ax.set_xlabel("Fold (test period)", fontsize=12)
    ax.set_ylabel("Win Rate (%)", fontsize=12)
    ax.set_title(
        f"{market} \u2014 Win Rate Consistency Across Folds", fontsize=14, fontweight="bold"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(test_labels, fontsize=9)
    ax.set_ylim(40, max(max(win_rates), max(edge_win_rates)) + 8)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = output_dir / f"{market}_walkforward_winrate.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # === Chart 3: Cumulative ROI ===
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    cumulative_roi = np.cumsum(rois)
    ax.plot(
        fold_nums,
        cumulative_roi,
        color=primary,
        linewidth=2.5,
        marker="o",
        markersize=8,
        markerfacecolor="white",
        markeredgecolor=primary,
        markeredgewidth=2,
    )
    ax.fill_between(fold_nums, 0, cumulative_roi, alpha=0.15, color=primary)
    ax.axhline(y=0, color="#94A3B8", linestyle="-", linewidth=1, alpha=0.5)

    for x_val, y_val in zip(fold_nums, cumulative_roi):
        ax.annotate(
            f"{y_val:+.1f}%",
            (x_val, y_val),
            textcoords="offset points",
            xytext=(0, 12),
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xlabel("Fold (test period)", fontsize=12)
    ax.set_ylabel("Cumulative ROI (%)", fontsize=12)
    ax.set_title(f"{market} \u2014 Cumulative ROI (Walk-Forward)", fontsize=14, fontweight="bold")
    ax.set_xticks(fold_nums)
    ax.set_xticklabels(test_labels, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = output_dir / f"{market}_walkforward_roi.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # === Markdown ===
    md_path = Path("nba/models/model_cards") / f"WALKFORWARD_{market}.md"
    lines = [
        f"# {market} Walk-Forward Validation",
        "",
        f"**{results.n_folds} folds** | "
        f"Mean AUC: **{results.mean_auc:.3f}** (std {results.std_auc:.3f}) | "
        f"Win Rate: **{results.mean_win_rate:.1%}** | "
        f"Edge WR: **{results.mean_edge_win_rate:.1%}** | "
        f"ROI: **{results.total_roi:+.1%}**",
        "",
        "| Fold | Train Period | Test Period | AUC | Win Rate | Edge WR | ROI |",
        "|------|-------------|-------------|-----|----------|---------|-----|",
    ]
    for f in folds:
        lines.append(
            f"| {f.fold} | {f.train_start[:10]} to {f.train_end[:10]} | "
            f"{f.test_start[:10]} to {f.test_end[:10]} | "
            f"{f.auc:.3f} | {f.win_rate:.1%} | "
            f"{f.edge_win_rate:.1%} | {f.roi:+.1%} |"
        )
    lines.extend(
        [
            "",
            f"![AUC](images/{market}_walkforward_auc.png)",
            f"![Win Rate](images/{market}_walkforward_winrate.png)",
            f"![ROI](images/{market}_walkforward_roi.png)",
        ]
    )
    md_path.write_text("\n".join(lines))
    print(f"  Saved: {md_path}")


if __name__ == "__main__":
    main()
