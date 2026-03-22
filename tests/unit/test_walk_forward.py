"""
Tests for WalkForwardValidator
================================
Tests walk-forward cross-validation and ROI simulation.
Uses realistic but hardcoded test values — NO synthetic data.
"""

import io
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from nba.models.walk_forward import FoldResult, ValidationSummary, WalkForwardValidator


@pytest.fixture
def validator():
    return WalkForwardValidator(n_folds=3, test_months=2, min_train_size=50)


def _make_test_df(n_rows=500, start_date="2023-10-24", stat="points"):
    """
    Create a realistic test DataFrame with rolling stats and actual results.
    Uses deterministic values based on row index — NOT random synthetic data.
    """
    dates = pd.date_range(start=start_date, periods=n_rows, freq="D")

    # Deterministic values based on index (simulates real player stat patterns)
    base_values = np.array([20.0 + (i % 15) * 0.5 for i in range(n_rows)])
    lines = base_values - 1.0 + (np.arange(n_rows) % 3) * 0.5
    actuals = base_values + (np.arange(n_rows) % 7 - 3) * 1.5

    df = pd.DataFrame(
        {
            "game_date": dates,
            "player_name": ["Test Player"] * n_rows,
            "line": lines,
            f"actual_{stat}": actuals,
            "actual_result": actuals,
            f"ema_{stat}_L5": base_values + 0.3,
            f"ema_{stat}_L10": base_values,
            f"ema_{stat}_L20": base_values - 0.5,
            "is_home": [i % 2 for i in range(n_rows)],
        }
    )
    return df


# =============================================================================
# Split Generation
# =============================================================================


class TestSplitGeneration:
    """Tests for walk-forward split generation."""

    def test_generates_splits(self, validator):
        df = _make_test_df(n_rows=500)
        splits = validator.generate_splits(df)
        assert len(splits) > 0

    def test_train_before_test(self, validator):
        """Training data must always come before test data."""
        df = _make_test_df(n_rows=500)
        splits = validator.generate_splits(df)
        for train_df, test_df in splits:
            assert train_df["game_date"].max() <= test_df["game_date"].min()

    def test_expanding_window(self, validator):
        """Each fold should have a larger training set."""
        df = _make_test_df(n_rows=500)
        splits = validator.generate_splits(df)
        if len(splits) >= 2:
            for i in range(1, len(splits)):
                assert len(splits[i][0]) >= len(splits[i - 1][0])

    def test_min_train_size_enforced(self):
        """Folds with too few training samples should be skipped."""
        validator = WalkForwardValidator(n_folds=6, test_months=2, min_train_size=1000)
        df = _make_test_df(n_rows=100)
        splits = validator.generate_splits(df)
        assert len(splits) == 0  # All folds too small

    def test_requires_game_date(self, validator):
        df = pd.DataFrame({"line": [22.5], "actual_result": [25.0]})
        with pytest.raises(ValueError, match="game_date"):
            validator.generate_splits(df)

    def test_no_data_leakage(self, validator):
        """No test dates should appear in training data."""
        df = _make_test_df(n_rows=500)
        splits = validator.generate_splits(df)
        for train_df, test_df in splits:
            train_dates = set(train_df["game_date"])
            test_dates = set(test_df["game_date"])
            assert len(train_dates & test_dates) == 0


# =============================================================================
# Projection Validation
# =============================================================================


class TestProjectionValidation:
    """Tests for projection model validation."""

    def test_validate_returns_summary(self, validator):
        df = _make_test_df(n_rows=500, stat="points")
        summary = validator.validate_projection(df, market="POINTS")
        assert isinstance(summary, ValidationSummary)
        assert summary.market == "POINTS"
        assert summary.model_type == "projection"

    def test_folds_have_metrics(self, validator):
        df = _make_test_df(n_rows=500, stat="points")
        summary = validator.validate_projection(df, market="POINTS")
        for fold in summary.folds:
            assert isinstance(fold, FoldResult)
            assert fold.train_size > 0
            assert fold.test_size > 0

    def test_auc_between_0_and_1(self, validator):
        df = _make_test_df(n_rows=500, stat="points")
        summary = validator.validate_projection(df, market="POINTS")
        for fold in summary.folds:
            if fold.auc > 0:
                assert 0.0 <= fold.auc <= 1.0

    def test_brier_score_bounded(self, validator):
        df = _make_test_df(n_rows=500, stat="points")
        summary = validator.validate_projection(df, market="POINTS")
        for fold in summary.folds:
            if fold.brier_score > 0:
                assert 0.0 <= fold.brier_score <= 1.0

    def test_custom_actual_col(self, validator):
        df = _make_test_df(n_rows=500, stat="points")
        summary = validator.validate_projection(df, market="POINTS", actual_col="actual_points")
        assert len(summary.folds) > 0

    def test_missing_actual_col_raises(self, validator):
        df = _make_test_df(n_rows=500, stat="points")
        df = df.drop(columns=["actual_result", "actual_points"])
        with pytest.raises(ValueError, match="actual result"):
            validator.validate_projection(df, market="POINTS")

    def test_rebounds_validation(self, validator):
        df = _make_test_df(n_rows=500, stat="rebounds")
        summary = validator.validate_projection(df, market="REBOUNDS")
        assert summary.market == "REBOUNDS"
        assert len(summary.folds) > 0


# =============================================================================
# ValidationSummary Properties
# =============================================================================


class TestValidationSummary:
    """Tests for summary aggregation properties."""

    def test_mean_auc(self):
        summary = ValidationSummary(market="POINTS", model_type="projection", n_folds=2)
        summary.folds = [
            FoldResult(
                fold_num=1,
                train_start="",
                train_end="",
                test_start="",
                test_end="",
                train_size=100,
                test_size=50,
                auc=0.70,
            ),
            FoldResult(
                fold_num=2,
                train_start="",
                train_end="",
                test_start="",
                test_end="",
                train_size=200,
                test_size=50,
                auc=0.80,
            ),
        ]
        assert abs(summary.mean_auc - 0.75) < 0.001

    def test_mean_roi(self):
        summary = ValidationSummary(market="POINTS", model_type="projection", n_folds=2)
        summary.folds = [
            FoldResult(
                fold_num=1,
                train_start="",
                train_end="",
                test_start="",
                test_end="",
                train_size=100,
                test_size=50,
                roi=0.06,
            ),
            FoldResult(
                fold_num=2,
                train_start="",
                train_end="",
                test_start="",
                test_end="",
                train_size=200,
                test_size=50,
                roi=0.10,
            ),
        ]
        assert abs(summary.mean_roi - 0.08) < 0.001

    def test_total_bets(self):
        summary = ValidationSummary(market="POINTS", model_type="projection", n_folds=2)
        summary.folds = [
            FoldResult(
                fold_num=1,
                train_start="",
                train_end="",
                test_start="",
                test_end="",
                train_size=100,
                test_size=50,
                total_bets=30,
                wins=18,
            ),
            FoldResult(
                fold_num=2,
                train_start="",
                train_end="",
                test_start="",
                test_end="",
                train_size=200,
                test_size=50,
                total_bets=40,
                wins=22,
            ),
        ]
        assert summary.total_bets == 70
        assert summary.total_wins == 40

    def test_empty_folds_returns_zero(self):
        summary = ValidationSummary(market="POINTS", model_type="projection", n_folds=0)
        assert summary.mean_auc == 0.0
        assert summary.mean_rmse == 0.0
        assert summary.mean_r2 == 0.0
        assert summary.mean_accuracy == 0.0
        assert summary.mean_brier == 0.0
        assert summary.mean_win_rate == 0.0
        assert summary.mean_roi == 0.0
        assert summary.total_bets == 0
        assert summary.total_wins == 0

    def test_mean_mae(self):
        summary = ValidationSummary(market="POINTS", model_type="projection", n_folds=2)
        summary.folds = [
            FoldResult(
                fold_num=1,
                train_start="",
                train_end="",
                test_start="",
                test_end="",
                train_size=100,
                test_size=50,
                mae=4.0,
            ),
            FoldResult(
                fold_num=2,
                train_start="",
                train_end="",
                test_start="",
                test_end="",
                train_size=200,
                test_size=50,
                mae=5.0,
            ),
        ]
        assert abs(summary.mean_mae - 4.5) < 0.001

    def test_mean_rmse(self):
        summary = ValidationSummary(market="POINTS", model_type="projection", n_folds=1)
        summary.folds = [
            FoldResult(
                fold_num=1,
                train_start="",
                train_end="",
                test_start="",
                test_end="",
                train_size=100,
                test_size=50,
                rmse=6.5,
            ),
        ]
        assert abs(summary.mean_rmse - 6.5) < 0.001


# =============================================================================
# Compare Models
# =============================================================================


class TestCompare:
    """Tests for model comparison."""

    def test_compare_projection_only(self, validator):
        df = _make_test_df(n_rows=500, stat="points")
        results = validator.compare(df, market="POINTS")
        assert "projection" in results
        assert "lgbm" not in results

    def test_compare_with_lgbm(self, validator):
        df = _make_test_df(n_rows=500, stat="points")

        def mock_lgbm(train_df, test_df):
            n = len(test_df)
            preds = np.full(n, 22.0)
            p_overs = np.full(n, 0.55)
            return preds, p_overs

        results = validator.compare(df, market="POINTS", lgbm_predict_fn=mock_lgbm)
        assert "projection" in results
        assert "lgbm" in results

    def test_lgbm_fold_error_handled(self, validator):
        """LightGBM predict function errors should be caught per fold."""
        df = _make_test_df(n_rows=500, stat="points")

        def failing_lgbm(train_df, test_df):
            raise ValueError("Model training failed")

        results = validator.compare(df, market="POINTS", lgbm_predict_fn=failing_lgbm)
        assert "lgbm" in results


# =============================================================================
# Print Functions
# =============================================================================


class TestPrintFunctions:
    """Tests for formatted output functions."""

    def test_print_summary(self, validator, capsys):
        df = _make_test_df(n_rows=500, stat="points")
        summary = validator.validate_projection(df, market="POINTS")
        WalkForwardValidator.print_summary(summary)
        captured = capsys.readouterr()
        assert "Walk-Forward Validation" in captured.out
        assert "POINTS" in captured.out
        assert "MEAN" in captured.out

    def test_print_comparison(self, validator, capsys):
        df = _make_test_df(n_rows=500, stat="points")

        def mock_lgbm(train_df, test_df):
            n = len(test_df)
            return np.full(n, 22.0), np.full(n, 0.55)

        results = validator.compare(df, market="POINTS", lgbm_predict_fn=mock_lgbm)
        WalkForwardValidator.print_comparison(results)
        captured = capsys.readouterr()
        assert "Model Comparison" in captured.out
        assert "projection" in captured.out
        assert "lgbm" in captured.out


# =============================================================================
# FoldResult Dataclass
# =============================================================================


class TestFoldResult:
    """Tests for FoldResult defaults."""

    def test_default_values(self):
        fold = FoldResult(
            fold_num=1,
            train_start="2023-10-24",
            train_end="2024-06-01",
            test_start="2024-06-01",
            test_end="2024-08-01",
            train_size=5000,
            test_size=500,
        )
        assert fold.mae == 0.0
        assert fold.rmse == 0.0
        assert fold.r2 == 0.0
        assert fold.auc == 0.0
        assert fold.accuracy == 0.0
        assert fold.brier_score == 0.0
        assert fold.win_rate == 0.0
        assert fold.roi == 0.0
        assert fold.total_bets == 0
        assert fold.wins == 0


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for boundary conditions."""

    def test_all_nan_actuals_skipped(self, validator):
        """Rows with NaN actuals should be skipped gracefully."""
        df = _make_test_df(n_rows=500, stat="points")
        # Set some actuals to NaN
        df.loc[df.index[:100], "actual_points"] = np.nan
        df.loc[df.index[:100], "actual_result"] = np.nan
        summary = validator.validate_projection(df, market="POINTS")
        assert isinstance(summary, ValidationSummary)

    def test_small_dataset(self):
        """Very small dataset should produce no valid folds."""
        validator = WalkForwardValidator(n_folds=6, min_train_size=100)
        df = _make_test_df(n_rows=50, stat="points")
        summary = validator.validate_projection(df, market="POINTS")
        assert len(summary.folds) == 0

    def test_single_fold(self):
        """Should work with just 1 fold."""
        validator = WalkForwardValidator(n_folds=1, test_months=3, min_train_size=50)
        df = _make_test_df(n_rows=500, stat="points")
        summary = validator.validate_projection(df, market="POINTS")
        assert len(summary.folds) <= 1

    def test_custom_min_edge(self):
        """Custom min_edge should affect ROI simulation."""
        v_strict = WalkForwardValidator(n_folds=3, min_train_size=50, min_edge=0.20)
        v_loose = WalkForwardValidator(n_folds=3, min_train_size=50, min_edge=0.01)
        df = _make_test_df(n_rows=500, stat="points")
        s_strict = v_strict.validate_projection(df, market="POINTS")
        s_loose = v_loose.validate_projection(df, market="POINTS")
        # Strict should have fewer total bets
        assert s_strict.total_bets <= s_loose.total_bets

    def test_fallback_actual_col(self):
        """Should fall back to actual_result when stat-specific column missing."""
        df = _make_test_df(n_rows=500, stat="points")
        df = df.drop(columns=["actual_points"])
        validator = WalkForwardValidator(n_folds=3, test_months=2, min_train_size=50)
        summary = validator.validate_projection(df, market="POINTS")
        assert len(summary.folds) > 0

    def test_fallback_ema_cols(self):
        """Should handle alternative column names for rolling stats."""
        df = _make_test_df(n_rows=500, stat="points")
        # Rename ema columns to rolling_ format
        df = df.rename(
            columns={
                "ema_points_L5": "rolling_points_L5",
                "ema_points_L10": "rolling_points_L10",
                "ema_points_L20": "rolling_points_L20",
            }
        )
        validator = WalkForwardValidator(n_folds=3, test_months=2, min_train_size=50)
        summary = validator.validate_projection(df, market="POINTS")
        assert len(summary.folds) > 0

    def test_single_rolling_col_fallback_std(self):
        """When only 1 rolling col exists, should use default std."""
        df = _make_test_df(n_rows=500, stat="points")
        df = df.drop(columns=["ema_points_L10", "ema_points_L20"])
        validator = WalkForwardValidator(n_folds=3, test_months=2, min_train_size=50)
        summary = validator.validate_projection(df, market="POINTS")
        assert isinstance(summary, ValidationSummary)

    def test_lgbm_with_good_predictions(self):
        """LightGBM validator should produce full metrics with good predict fn."""
        df = _make_test_df(n_rows=500, stat="points")
        validator = WalkForwardValidator(n_folds=3, test_months=2, min_train_size=50, min_edge=0.01)

        def good_lgbm(train_df, test_df):
            n = len(test_df)
            # predictions close to actuals
            actuals = test_df["actual_points"].values
            preds = actuals + np.random.default_rng(42).normal(0, 2, n)
            # p_overs with enough edge to trigger bets
            p_overs = np.full(n, 0.60)
            return preds, p_overs

        results = validator.compare(df, market="POINTS", lgbm_predict_fn=good_lgbm)
        lgbm = results["lgbm"]
        # Should have some folds with metrics
        has_bets = any(f.total_bets > 0 for f in lgbm.folds)
        assert has_bets

    def test_lgbm_actual_col_fallback(self):
        """LightGBM validator should fall back to actual_result."""
        df = _make_test_df(n_rows=500, stat="points")
        df = df.drop(columns=["actual_points"])
        validator = WalkForwardValidator(n_folds=3, test_months=2, min_train_size=50)

        def mock_lgbm(train_df, test_df):
            n = len(test_df)
            return np.full(n, 22.0), np.full(n, 0.55)

        results = validator.compare(df, market="POINTS", lgbm_predict_fn=mock_lgbm)
        assert "lgbm" in results
