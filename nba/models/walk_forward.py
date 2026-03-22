"""
Walk-Forward Validation
========================
Time-series cross-validation for comparing model approaches.

Expanding window: train on months 1-N, test on month N+1.
Tracks AUC, accuracy, calibration (Brier score), and simulated ROI
per fold, then produces a summary comparison table.

Usage:
    from nba.models.walk_forward import WalkForwardValidator

    validator = WalkForwardValidator(n_folds=6)
    results = validator.validate(
        df=training_data,
        market="POINTS",
        model_type="projection",  # or "lgbm"
    )
    validator.print_summary(results)

    # Compare two model types
    comparison = validator.compare(
        df=training_data,
        market="POINTS",
    )
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from nba.core.probability_engine import ProbabilityEngine

logger = logging.getLogger(__name__)


@dataclass
class FoldResult:
    """Results from a single walk-forward fold."""

    fold_num: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_size: int
    test_size: int
    # Regression metrics
    mae: float = 0.0
    rmse: float = 0.0
    r2: float = 0.0
    # Classification metrics
    auc: float = 0.0
    accuracy: float = 0.0
    brier_score: float = 0.0
    # Betting simulation
    win_rate: float = 0.0
    roi: float = 0.0
    total_bets: int = 0
    wins: int = 0


@dataclass
class ValidationSummary:
    """Aggregated results across all folds."""

    market: str
    model_type: str
    n_folds: int
    folds: List[FoldResult] = field(default_factory=list)

    @property
    def mean_mae(self) -> float:
        return np.mean([f.mae for f in self.folds]) if self.folds else 0.0

    @property
    def mean_rmse(self) -> float:
        return np.mean([f.rmse for f in self.folds]) if self.folds else 0.0

    @property
    def mean_r2(self) -> float:
        return np.mean([f.r2 for f in self.folds]) if self.folds else 0.0

    @property
    def mean_auc(self) -> float:
        return np.mean([f.auc for f in self.folds]) if self.folds else 0.0

    @property
    def mean_accuracy(self) -> float:
        return np.mean([f.accuracy for f in self.folds]) if self.folds else 0.0

    @property
    def mean_brier(self) -> float:
        return np.mean([f.brier_score for f in self.folds]) if self.folds else 0.0

    @property
    def mean_win_rate(self) -> float:
        return np.mean([f.win_rate for f in self.folds]) if self.folds else 0.0

    @property
    def mean_roi(self) -> float:
        return np.mean([f.roi for f in self.folds]) if self.folds else 0.0

    @property
    def total_bets(self) -> int:
        return sum(f.total_bets for f in self.folds)

    @property
    def total_wins(self) -> int:
        return sum(f.wins for f in self.folds)


class WalkForwardValidator:
    """
    Walk-forward cross-validation with ROI simulation.

    Supports both:
    - "projection": Projection model + distribution-based probabilities
    - "lgbm": Existing two-head LightGBM model
    """

    def __init__(
        self,
        n_folds: int = 6,
        test_months: int = 2,
        min_train_size: int = 5000,
        min_edge: float = 0.05,
    ):
        """
        Args:
            n_folds: Number of walk-forward folds
            test_months: Months per test period
            min_train_size: Minimum training samples per fold
            min_edge: Minimum edge for simulated betting
        """
        self.n_folds = n_folds
        self.test_months = test_months
        self.min_train_size = min_train_size
        self.min_edge = min_edge
        self._prob_engine = ProbabilityEngine()

    def generate_splits(self, df: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Generate time-series walk-forward splits.

        Expanding training window, fixed test window:
        Fold 1: Train [start, t1), Test [t1, t2)
        Fold 2: Train [start, t2), Test [t2, t3)
        ...
        """
        if "game_date" not in df.columns:
            raise ValueError("DataFrame must have 'game_date' column")

        df = df.copy()
        df["game_date"] = pd.to_datetime(df["game_date"])
        df = df.sort_values("game_date").reset_index(drop=True)

        min_date = df["game_date"].min()
        max_date = df["game_date"].max()
        total_days = (max_date - min_date).days
        test_days = self.test_months * 30

        splits = []
        for i in range(self.n_folds):
            train_end_offset = total_days - (self.n_folds - i) * test_days
            train_end = min_date + pd.Timedelta(days=train_end_offset)
            test_end = train_end + pd.Timedelta(days=test_days)

            train_df = df[df["game_date"] < train_end].copy()
            test_df = df[(df["game_date"] >= train_end) & (df["game_date"] < test_end)].copy()

            if len(train_df) >= self.min_train_size and len(test_df) > 50:
                splits.append((train_df, test_df))
                logger.info(
                    f"Fold {len(splits)}: train={len(train_df)} "
                    f"({train_df['game_date'].min().date()} to {train_df['game_date'].max().date()}), "
                    f"test={len(test_df)} "
                    f"({test_df['game_date'].min().date()} to {test_df['game_date'].max().date()})"
                )

        logger.info(f"Generated {len(splits)} valid folds from {self.n_folds} requested")
        return splits

    def validate_projection(
        self,
        df: pd.DataFrame,
        market: str,
        stat_col: str = None,
        line_col: str = "line",
        actual_col: str = None,
    ) -> ValidationSummary:
        """
        Walk-forward validation using projection + distribution approach.

        For each fold:
        1. Compute rolling stats from training data
        2. Project values for test data using weighted rolling averages
        3. Convert to probabilities using ProbabilityEngine
        4. Evaluate and simulate betting

        Args:
            df: Full dataset with game_date, line, actual_result, rolling stats
            market: Market type (POINTS, REBOUNDS, etc.)
            stat_col: Column with rolling average (auto-detected if None)
            line_col: Column with sportsbook line
            actual_col: Column with actual result (auto-detected if None)

        Returns:
            ValidationSummary with per-fold and aggregate metrics
        """
        market = market.upper()
        if actual_col is None:
            actual_col = f"actual_{market.lower()}"
            if actual_col not in df.columns:
                actual_col = "actual_result"

        if actual_col not in df.columns:
            raise ValueError(f"Cannot find actual result column. Tried: {actual_col}")

        # Detect rolling stat columns
        stat_key = market.lower()
        ema_cols = {
            "L5": f"ema_{stat_key}_L5",
            "L10": f"ema_{stat_key}_L10",
            "L20": f"ema_{stat_key}_L20",
        }

        # Fall back to non-EMA columns
        for window, col in ema_cols.items():
            if col not in df.columns:
                alt = f"rolling_{stat_key}_{window}"
                if alt in df.columns:
                    ema_cols[window] = alt

        splits = self.generate_splits(df)
        summary = ValidationSummary(market=market, model_type="projection", n_folds=len(splits))

        for fold_idx, (train_df, test_df) in enumerate(splits):
            fold_result = self._evaluate_projection_fold(
                fold_num=fold_idx + 1,
                train_df=train_df,
                test_df=test_df,
                market=market,
                ema_cols=ema_cols,
                line_col=line_col,
                actual_col=actual_col,
            )
            summary.folds.append(fold_result)

        return summary

    def _evaluate_projection_fold(
        self,
        fold_num: int,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        market: str,
        ema_cols: Dict[str, str],
        line_col: str,
        actual_col: str,
    ) -> FoldResult:
        """Evaluate a single fold for the projection model."""
        result = FoldResult(
            fold_num=fold_num,
            train_start=str(train_df["game_date"].min().date()),
            train_end=str(train_df["game_date"].max().date()),
            test_start=str(test_df["game_date"].min().date()),
            test_end=str(test_df["game_date"].max().date()),
            train_size=len(train_df),
            test_size=len(test_df),
        )

        # Build projections from rolling averages in test data
        projections = []
        actuals = []
        lines = []
        p_overs = []

        weights = {"L5": 0.50, "L10": 0.30, "L20": 0.20}

        for _, row in test_df.iterrows():
            actual = row[actual_col]
            line = row[line_col]

            if pd.isna(actual) or pd.isna(line):
                continue

            # Weighted projection from rolling averages
            proj = 0.0
            total_weight = 0.0
            for window, col_name in ema_cols.items():
                if col_name in test_df.columns and not pd.isna(row.get(col_name)):
                    w = weights.get(window, 0.2)
                    proj += float(row[col_name]) * w
                    total_weight += w

            if total_weight == 0:
                continue

            proj = proj / total_weight * total_weight  # Already weighted
            if proj <= 0:
                continue

            # Estimate std from rolling window spread
            vals = []
            for col_name in ema_cols.values():
                if col_name in test_df.columns and not pd.isna(row.get(col_name)):
                    vals.append(float(row[col_name]))
            if len(vals) >= 2:
                spread = max(vals) - min(vals)
                avg = np.mean(vals)
                std = max(avg * 0.15, spread + avg * 0.10) if avg > 0 else 5.0
            else:
                std = proj * 0.25  # Default 25% CV

            # Calculate P(OVER) from distribution
            p_over = self._prob_engine.calculate_probability(
                projected_value=proj,
                std_dev=std,
                line=float(line),
                stat_type=market,
            )

            projections.append(proj)
            actuals.append(float(actual))
            lines.append(float(line))
            p_overs.append(p_over)

        if len(projections) < 10:
            logger.warning(f"Fold {fold_num}: only {len(projections)} valid samples")
            return result

        projections = np.array(projections)
        actuals = np.array(actuals)
        lines = np.array(lines)
        p_overs = np.array(p_overs)

        # Binary labels: did actual beat the line?
        y_true = (actuals > lines).astype(int)

        # Regression metrics
        result.mae = float(mean_absolute_error(actuals, projections))
        result.rmse = float(np.sqrt(mean_squared_error(actuals, projections)))
        result.r2 = float(r2_score(actuals, projections))

        # Classification metrics
        if len(np.unique(y_true)) == 2:
            result.auc = float(roc_auc_score(y_true, p_overs))
            result.brier_score = float(brier_score_loss(y_true, p_overs))
        y_pred = (p_overs >= 0.5).astype(int)
        result.accuracy = float(accuracy_score(y_true, y_pred))

        # ROI simulation: bet when P(OVER) implies edge > min_edge
        # Assume -110 odds (implied prob = 0.524)
        implied = 0.524
        edge_mask = p_overs - implied >= self.min_edge
        if edge_mask.sum() > 0:
            bet_outcomes = y_true[edge_mask]
            result.total_bets = int(edge_mask.sum())
            result.wins = int(bet_outcomes.sum())
            result.win_rate = float(result.wins / result.total_bets)
            # ROI at -110: win pays +0.909, lose pays -1.0
            profit = result.wins * 0.909 - (result.total_bets - result.wins) * 1.0
            result.roi = float(profit / result.total_bets)

        logger.info(
            f"Fold {fold_num}: AUC={result.auc:.3f}, R²={result.r2:.3f}, "
            f"WR={result.win_rate:.1%}, ROI={result.roi:.1%} "
            f"({result.total_bets} bets)"
        )

        return result

    def compare(
        self,
        df: pd.DataFrame,
        market: str,
        lgbm_predict_fn: Optional[Callable] = None,
    ) -> Dict[str, ValidationSummary]:
        """
        Compare projection model vs LightGBM on the same folds.

        Args:
            df: Full dataset
            market: Market type
            lgbm_predict_fn: Optional function(train_df, test_df) -> (predictions, p_overs)
                             If None, only runs projection validation.

        Returns:
            Dict with "projection" and optionally "lgbm" ValidationSummary objects
        """
        results = {}

        # Always run projection
        proj_summary = self.validate_projection(df, market)
        results["projection"] = proj_summary

        if lgbm_predict_fn is not None:
            lgbm_summary = self._validate_lgbm(df, market, lgbm_predict_fn)
            results["lgbm"] = lgbm_summary

        return results

    def _validate_lgbm(
        self,
        df: pd.DataFrame,
        market: str,
        predict_fn: Callable,
    ) -> ValidationSummary:
        """Validate LightGBM model using the same walk-forward folds."""
        actual_col = f"actual_{market.lower()}"
        if actual_col not in df.columns:
            actual_col = "actual_result"

        splits = self.generate_splits(df)
        summary = ValidationSummary(market=market, model_type="lgbm", n_folds=len(splits))

        for fold_idx, (train_df, test_df) in enumerate(splits):
            fold_result = FoldResult(
                fold_num=fold_idx + 1,
                train_start=str(train_df["game_date"].min().date()),
                train_end=str(train_df["game_date"].max().date()),
                test_start=str(test_df["game_date"].min().date()),
                test_end=str(test_df["game_date"].max().date()),
                train_size=len(train_df),
                test_size=len(test_df),
            )

            try:
                predictions, p_overs = predict_fn(train_df, test_df)
                actuals = test_df[actual_col].values
                lines = test_df["line"].values

                valid = ~(np.isnan(actuals) | np.isnan(predictions) | np.isnan(lines))
                actuals = actuals[valid]
                predictions = predictions[valid]
                p_overs = p_overs[valid]
                lines = lines[valid]

                if len(actuals) < 10:
                    continue

                y_true = (actuals > lines).astype(int)

                fold_result.mae = float(mean_absolute_error(actuals, predictions))
                fold_result.rmse = float(np.sqrt(mean_squared_error(actuals, predictions)))
                fold_result.r2 = float(r2_score(actuals, predictions))

                if len(np.unique(y_true)) == 2:
                    fold_result.auc = float(roc_auc_score(y_true, p_overs))
                    fold_result.brier_score = float(brier_score_loss(y_true, p_overs))
                fold_result.accuracy = float(accuracy_score(y_true, (p_overs >= 0.5).astype(int)))

                implied = 0.524
                edge_mask = p_overs - implied >= self.min_edge
                if edge_mask.sum() > 0:
                    bet_outcomes = y_true[edge_mask]
                    fold_result.total_bets = int(edge_mask.sum())
                    fold_result.wins = int(bet_outcomes.sum())
                    fold_result.win_rate = float(fold_result.wins / fold_result.total_bets)
                    profit = (
                        fold_result.wins * 0.909 - (fold_result.total_bets - fold_result.wins) * 1.0
                    )
                    fold_result.roi = float(profit / fold_result.total_bets)

            except Exception as e:
                logger.error(f"LightGBM fold {fold_idx + 1} failed: {e}")

            summary.folds.append(fold_result)

        return summary

    @staticmethod
    def print_summary(summary: ValidationSummary):
        """Print a formatted summary table."""
        print(f"\n{'='*70}")
        print(f"Walk-Forward Validation: {summary.market} ({summary.model_type})")
        print(f"{'='*70}")
        print(
            f"{'Fold':<6} {'Train':<15} {'Test':<15} "
            f"{'AUC':<7} {'R²':<7} {'Brier':<7} {'WR':<7} {'ROI':<8} {'Bets':<5}"
        )
        print("-" * 70)

        for f in summary.folds:
            print(
                f"{f.fold_num:<6} "
                f"{f.train_start[:10]:<15} "
                f"{f.test_start[:10]:<15} "
                f"{f.auc:<7.3f} "
                f"{f.r2:<7.3f} "
                f"{f.brier_score:<7.3f} "
                f"{f.win_rate:<7.1%} "
                f"{f.roi:<8.1%} "
                f"{f.total_bets:<5}"
            )

        print("-" * 70)
        print(
            f"{'MEAN':<6} "
            f"{'':<15} "
            f"{'':<15} "
            f"{summary.mean_auc:<7.3f} "
            f"{summary.mean_r2:<7.3f} "
            f"{summary.mean_brier:<7.3f} "
            f"{summary.mean_win_rate:<7.1%} "
            f"{summary.mean_roi:<8.1%} "
            f"{summary.total_bets:<5}"
        )
        print(f"{'='*70}\n")

    @staticmethod
    def print_comparison(results: Dict[str, ValidationSummary]):
        """Print side-by-side comparison of model types."""
        print(f"\n{'='*50}")
        print("Model Comparison Summary")
        print(f"{'='*50}")
        print(f"{'Metric':<20} ", end="")
        for model_type in results:
            print(f"{model_type:<15} ", end="")
        print()
        print("-" * 50)

        metrics = [
            ("Mean AUC", "mean_auc"),
            ("Mean R²", "mean_r2"),
            ("Mean Brier", "mean_brier"),
            ("Mean Win Rate", "mean_win_rate"),
            ("Mean ROI", "mean_roi"),
            ("Total Bets", "total_bets"),
            ("Total Wins", "total_wins"),
        ]

        for label, attr in metrics:
            print(f"{label:<20} ", end="")
            for summary in results.values():
                val = getattr(summary, attr)
                if isinstance(val, float):
                    if "rate" in attr.lower() or "roi" in attr.lower():
                        print(f"{val:<15.1%} ", end="")
                    else:
                        print(f"{val:<15.3f} ", end="")
                else:
                    print(f"{val:<15} ", end="")
            print()

        print(f"{'='*50}\n")
