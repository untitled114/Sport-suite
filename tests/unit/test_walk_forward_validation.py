"""
Tests for Walk-Forward Validation module.

Covers all classes, methods, and branches including the training/evaluation
pipeline and CLI entry point.
"""

from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pandas as pd
import pytest

from nba.models.walk_forward_validation import FoldResult, WalkForwardResults, WalkForwardValidator

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_fold_result(fold=1, auc=0.72, win_rate=0.56, roi=0.05, edge_bets=10, edge_win_rate=0.60):
    return FoldResult(
        fold=fold,
        train_start="2023-10-24",
        train_end="2024-06-30",
        test_start="2024-07-01",
        test_end="2024-09-30",
        train_samples=5000,
        test_samples=1000,
        auc=auc,
        accuracy=0.65,
        win_rate=win_rate,
        roi=roi,
        edge_bets=edge_bets,
        edge_win_rate=edge_win_rate,
    )


def _make_dataset(n=500, months=24):
    """Create a mock training dataset spanning `months` months."""
    dates = pd.date_range("2023-03-01", periods=n, freq="D")[:n]
    rng = np.random.RandomState(42)
    return pd.DataFrame(
        {
            "game_date": dates,
            "line": rng.uniform(10, 35, n),
            "is_home": rng.randint(0, 2, n),
            "ema_points_L5": rng.uniform(15, 30, n),
            "ema_points_L10": rng.uniform(15, 30, n),
            "actual_result": rng.uniform(5, 45, n),
            "hit_over": rng.randint(0, 2, n),
            "label": rng.randint(0, 2, n),
        }
    )


# ---------------------------------------------------------------------------
# FoldResult
# ---------------------------------------------------------------------------


class TestFoldResult:
    def test_creation(self):
        fr = _make_fold_result()
        assert fr.fold == 1
        assert fr.auc == 0.72
        assert fr.win_rate == 0.56
        assert fr.edge_bets == 10

    def test_all_fields(self):
        fr = _make_fold_result(fold=3, auc=0.80, roi=0.10)
        assert fr.train_start == "2023-10-24"
        assert fr.train_end == "2024-06-30"
        assert fr.test_start == "2024-07-01"
        assert fr.test_end == "2024-09-30"
        assert fr.train_samples == 5000
        assert fr.test_samples == 1000
        assert fr.accuracy == 0.65


# ---------------------------------------------------------------------------
# WalkForwardResults
# ---------------------------------------------------------------------------


class TestWalkForwardResults:
    def test_mean_auc(self):
        results = WalkForwardResults(market="POINTS", n_folds=3)
        results.folds = [
            _make_fold_result(auc=0.70),
            _make_fold_result(auc=0.74),
            _make_fold_result(auc=0.72),
        ]
        assert abs(results.mean_auc - 0.72) < 0.001

    def test_std_auc(self):
        results = WalkForwardResults(market="POINTS", n_folds=2)
        results.folds = [
            _make_fold_result(auc=0.70),
            _make_fold_result(auc=0.70),
        ]
        assert results.std_auc == 0.0

    def test_std_auc_nonzero(self):
        results = WalkForwardResults(market="POINTS", n_folds=2)
        results.folds = [
            _make_fold_result(auc=0.60),
            _make_fold_result(auc=0.80),
        ]
        assert results.std_auc > 0

    def test_mean_win_rate(self):
        results = WalkForwardResults(market="REBOUNDS", n_folds=2)
        results.folds = [
            _make_fold_result(win_rate=0.55),
            _make_fold_result(win_rate=0.65),
        ]
        assert abs(results.mean_win_rate - 0.60) < 0.001

    def test_mean_edge_win_rate(self):
        results = WalkForwardResults(market="POINTS", n_folds=2)
        results.folds = [
            _make_fold_result(edge_bets=10, edge_win_rate=0.70),
            _make_fold_result(edge_bets=5, edge_win_rate=0.60),
        ]
        assert abs(results.mean_edge_win_rate - 0.65) < 0.001

    def test_mean_edge_win_rate_no_edge_bets(self):
        results = WalkForwardResults(market="POINTS", n_folds=2)
        results.folds = [
            _make_fold_result(edge_bets=0, edge_win_rate=0.0),
            _make_fold_result(edge_bets=0, edge_win_rate=0.0),
        ]
        assert results.mean_edge_win_rate == 0.0

    def test_mean_edge_win_rate_partial(self):
        results = WalkForwardResults(market="POINTS", n_folds=2)
        results.folds = [
            _make_fold_result(edge_bets=10, edge_win_rate=0.70),
            _make_fold_result(edge_bets=0, edge_win_rate=0.0),
        ]
        assert abs(results.mean_edge_win_rate - 0.70) < 0.001

    def test_total_roi(self):
        results = WalkForwardResults(market="POINTS", n_folds=2)
        results.folds = [
            _make_fold_result(roi=0.10),  # test_samples=1000
            _make_fold_result(roi=0.05),  # test_samples=1000
        ]
        # (0.10*1000 + 0.05*1000) / 2000 = 0.075
        assert abs(results.total_roi - 0.075) < 0.001

    def test_total_roi_empty(self):
        results = WalkForwardResults(market="POINTS", n_folds=0)
        assert results.total_roi == 0.0

    def test_summary_output(self):
        results = WalkForwardResults(market="POINTS", n_folds=2)
        results.folds = [
            _make_fold_result(fold=1, auc=0.72, win_rate=0.56),
            _make_fold_result(fold=2, auc=0.74, win_rate=0.58),
        ]
        summary = results.summary()
        assert "POINTS" in summary
        assert "Folds: 2" in summary
        assert "Mean AUC" in summary
        assert "STABLE" in summary
        assert "ALL FOLDS >50%" in summary

    def test_summary_unstable_auc(self):
        results = WalkForwardResults(market="POINTS", n_folds=2)
        results.folds = [
            _make_fold_result(fold=1, auc=0.55, win_rate=0.55),
            _make_fold_result(fold=2, auc=0.85, win_rate=0.55),
        ]
        summary = results.summary()
        assert "UNSTABLE" in summary

    def test_summary_inconsistent_wr(self):
        results = WalkForwardResults(market="POINTS", n_folds=2)
        results.folds = [
            _make_fold_result(fold=1, auc=0.72, win_rate=0.45),
            _make_fold_result(fold=2, auc=0.74, win_rate=0.55),
        ]
        summary = results.summary()
        assert "SOME FOLDS <50%" in summary

    def test_summary_no_edge_bets_in_fold(self):
        results = WalkForwardResults(market="REBOUNDS", n_folds=1)
        results.folds = [
            _make_fold_result(fold=1, edge_bets=0, edge_win_rate=0.0),
        ]
        summary = results.summary()
        assert "N/A" in summary


# ---------------------------------------------------------------------------
# WalkForwardValidator.__init__ and create_folds
# ---------------------------------------------------------------------------


class TestWalkForwardValidatorInit:
    def test_defaults(self):
        v = WalkForwardValidator()
        assert v.n_folds == 6
        assert v.test_months == 3
        assert v.min_train_months == 9
        assert v.edge_threshold == 0.03

    def test_custom(self):
        v = WalkForwardValidator(n_folds=4, test_months=2, min_train_months=6, edge_threshold=0.05)
        assert v.n_folds == 4
        assert v.test_months == 2
        assert v.min_train_months == 6
        assert v.edge_threshold == 0.05


class TestCreateFolds:
    def test_basic_folding(self):
        v = WalkForwardValidator(n_folds=2, test_months=3, min_train_months=6)
        df = _make_dataset(n=500, months=18)
        folds = v.create_folds(df)
        assert len(folds) >= 1
        for train_df, test_df in folds:
            assert len(train_df) > 0
            assert len(test_df) > 0
            # Train dates should be before test dates
            assert train_df["game_date"].max() <= test_df["game_date"].min()

    def test_reduces_folds_when_not_enough_data(self):
        v = WalkForwardValidator(n_folds=10, test_months=3, min_train_months=9)
        # Only 12 months of data → can't do 10 folds of 3 months each
        df = _make_dataset(n=365, months=12)
        folds = v.create_folds(df)
        assert v.n_folds < 10  # Should have been reduced
        assert len(folds) <= v.n_folds

    def test_expanding_window(self):
        v = WalkForwardValidator(n_folds=3, test_months=2, min_train_months=6)
        df = _make_dataset(n=500, months=18)
        folds = v.create_folds(df)
        if len(folds) >= 2:
            # Each fold should have a larger training set than the previous
            assert len(folds[1][0]) >= len(folds[0][0])

    def test_empty_folds_filtered(self):
        v = WalkForwardValidator(n_folds=2, test_months=1, min_train_months=1)
        # Very small dataset — some folds may not meet the 100/20 threshold
        df = _make_dataset(n=50, months=6)
        folds = v.create_folds(df)
        # Should not crash; may return fewer folds
        assert isinstance(folds, list)


# ---------------------------------------------------------------------------
# train_and_evaluate_fold (mocked training)
# ---------------------------------------------------------------------------


class TestTrainAndEvaluateFold:
    def test_full_fold_evaluation(self):
        v = WalkForwardValidator(edge_threshold=0.03)
        n_train, n_test = 200, 50

        rng = np.random.RandomState(42)
        train_df = pd.DataFrame(
            {
                "game_date": pd.date_range("2023-10-01", periods=n_train),
                "line": rng.uniform(15, 30, n_train),
            }
        )
        test_df = pd.DataFrame(
            {
                "game_date": pd.date_range("2024-06-01", periods=n_test),
                "line": rng.uniform(15, 30, n_test),
            }
        )

        # Mock the entire StackedMarketModel
        mock_model = MagicMock()
        mock_model.prepare_features.side_effect = [
            # Train return
            (
                rng.randn(n_train, 10),  # X_train
                pd.Series(rng.uniform(15, 30, n_train)),  # y_value_train
                pd.Series(rng.randint(0, 2, n_train)),  # y_binary_train
                pd.Series(rng.randn(n_train)),  # y_residual_train
                pd.DataFrame({"line": rng.uniform(15, 30, n_train)}),  # meta_train
            ),
            # Test return
            (
                rng.randn(n_test, 10),  # X_test
                pd.Series(rng.uniform(15, 30, n_test)),  # y_value_test
                pd.Series(rng.randint(0, 2, n_test)),  # y_binary_test
                pd.Series(rng.randn(n_test)),  # y_residual_test
                pd.DataFrame({"line": rng.uniform(15, 30, n_test)}),  # meta_test
            ),
        ]
        mock_model.train.return_value = {"auc": 0.72}
        mock_model.imputer.transform.return_value = rng.randn(n_test, 10)
        mock_model.scaler.transform.return_value = rng.randn(n_test, 10)
        mock_model.regressor.predict.return_value = rng.uniform(15, 30, n_test)
        mock_model.classifier.predict_proba.return_value = np.column_stack(
            [
                rng.uniform(0.2, 0.5, n_test),
                rng.uniform(0.5, 0.8, n_test),
            ]
        )
        mock_model.calibrator.transform.return_value = rng.uniform(0.4, 0.9, n_test)

        with patch("nba.models.train_market.StackedMarketModel", return_value=mock_model):
            result = v.train_and_evaluate_fold(train_df, test_df, "POINTS", 1)

        assert isinstance(result, FoldResult)
        assert result.fold == 1
        assert 0 <= result.auc <= 1
        assert 0 <= result.accuracy <= 1
        assert result.train_samples == n_train
        assert result.test_samples == n_test
        assert result.train_start == str(train_df["game_date"].min())
        assert result.edge_bets >= 0

    def test_no_over_predictions(self):
        """When classifier predicts all UNDER, win_rate defaults to 0.5."""
        v = WalkForwardValidator(edge_threshold=0.03)
        n_train, n_test = 100, 30

        rng = np.random.RandomState(99)
        train_df = pd.DataFrame(
            {
                "game_date": pd.date_range("2023-10-01", periods=n_train),
                "line": rng.uniform(15, 30, n_train),
            }
        )
        test_df = pd.DataFrame(
            {
                "game_date": pd.date_range("2024-06-01", periods=n_test),
                "line": rng.uniform(15, 30, n_test),
            }
        )

        mock_model = MagicMock()
        mock_model.prepare_features.side_effect = [
            (
                rng.randn(n_train, 5),
                pd.Series(rng.uniform(15, 30, n_train)),
                pd.Series(rng.randint(0, 2, n_train)),
                pd.Series(rng.randn(n_train)),
                pd.DataFrame({"line": rng.uniform(15, 30, n_train)}),
            ),
            (
                rng.randn(n_test, 5),
                pd.Series(rng.uniform(15, 30, n_test)),
                pd.Series(rng.randint(0, 2, n_test)),
                pd.Series(rng.randn(n_test)),
                pd.DataFrame({"line": rng.uniform(15, 30, n_test)}),
            ),
        ]
        mock_model.train.return_value = {}
        mock_model.imputer.transform.return_value = rng.randn(n_test, 5)
        mock_model.scaler.transform.return_value = rng.randn(n_test, 5)
        mock_model.regressor.predict.return_value = rng.uniform(15, 30, n_test)
        mock_model.classifier.predict_proba.return_value = np.column_stack(
            [
                np.ones(n_test) * 0.8,  # High P(UNDER)
                np.ones(n_test) * 0.2,  # Low P(OVER)
            ]
        )
        # After calibration, all below 0.5 → all UNDER predictions
        mock_model.calibrator.transform.return_value = np.ones(n_test) * 0.3

        with patch("nba.models.train_market.StackedMarketModel", return_value=mock_model):
            result = v.train_and_evaluate_fold(train_df, test_df, "POINTS", 1)

        # All predictions are UNDER, so over_mask.sum() == 0 → win_rate == 0.5
        assert result.win_rate == 0.5

    def test_no_edge_bets(self):
        """When all probabilities are near 0.5, no edge bets."""
        v = WalkForwardValidator(edge_threshold=0.20)  # Very high threshold
        n_train, n_test = 100, 30

        rng = np.random.RandomState(7)
        train_df = pd.DataFrame(
            {
                "game_date": pd.date_range("2023-10-01", periods=n_train),
                "line": rng.uniform(15, 30, n_train),
            }
        )
        test_df = pd.DataFrame(
            {
                "game_date": pd.date_range("2024-06-01", periods=n_test),
                "line": rng.uniform(15, 30, n_test),
            }
        )

        mock_model = MagicMock()
        mock_model.prepare_features.side_effect = [
            (
                rng.randn(n_train, 5),
                pd.Series(rng.uniform(15, 30, n_train)),
                pd.Series(rng.randint(0, 2, n_train)),
                pd.Series(rng.randn(n_train)),
                pd.DataFrame({"line": rng.uniform(15, 30, n_train)}),
            ),
            (
                rng.randn(n_test, 5),
                pd.Series(rng.uniform(15, 30, n_test)),
                pd.Series(rng.randint(0, 2, n_test)),
                pd.Series(rng.randn(n_test)),
                pd.DataFrame({"line": rng.uniform(15, 30, n_test)}),
            ),
        ]
        mock_model.train.return_value = {}
        mock_model.imputer.transform.return_value = rng.randn(n_test, 5)
        mock_model.scaler.transform.return_value = rng.randn(n_test, 5)
        mock_model.regressor.predict.return_value = rng.uniform(15, 30, n_test)
        mock_model.classifier.predict_proba.return_value = np.column_stack(
            [
                np.ones(n_test) * 0.48,
                np.ones(n_test) * 0.52,
            ]
        )
        # After calibration + blending, all near 0.5 → no edge bets at threshold 0.20
        mock_model.calibrator.transform.return_value = np.ones(n_test) * 0.51

        with patch("nba.models.train_market.StackedMarketModel", return_value=mock_model):
            result = v.train_and_evaluate_fold(train_df, test_df, "POINTS", 1)

        assert result.edge_bets == 0
        assert result.edge_win_rate == 0.0


# ---------------------------------------------------------------------------
# validate (full pipeline)
# ---------------------------------------------------------------------------


class TestValidate:
    def test_validate_runs_all_folds(self):
        v = WalkForwardValidator(n_folds=2, test_months=3, min_train_months=6)

        mock_fold_result = _make_fold_result()

        with (
            patch.object(v, "create_folds") as mock_create,
            patch.object(v, "train_and_evaluate_fold", return_value=mock_fold_result) as mock_eval,
        ):
            df1, df2 = pd.DataFrame({"game_date": ["2023-10-01"]}), pd.DataFrame(
                {"game_date": ["2024-01-01"]}
            )
            mock_create.return_value = [(df1, df2), (df1, df2)]

            results = v.validate(pd.DataFrame(), "POINTS")

        assert isinstance(results, WalkForwardResults)
        assert results.market == "POINTS"
        assert len(results.folds) == 2
        assert mock_eval.call_count == 2

    def test_validate_handles_fold_failure(self):
        v = WalkForwardValidator(n_folds=3, test_months=2, min_train_months=6)

        mock_fold_result = _make_fold_result()

        def side_effect(train, test, market, fold_num):
            if fold_num == 2:
                raise ValueError("Fold 2 failed")
            return mock_fold_result

        with (
            patch.object(v, "create_folds") as mock_create,
            patch.object(v, "train_and_evaluate_fold", side_effect=side_effect),
        ):
            df1 = pd.DataFrame({"game_date": ["2023-10-01"]})
            df2 = pd.DataFrame({"game_date": ["2024-01-01"]})
            mock_create.return_value = [(df1, df2), (df1, df2), (df1, df2)]

            results = v.validate(pd.DataFrame(), "REBOUNDS")

        # Fold 2 failed, so only 2 results
        assert len(results.folds) == 2

    def test_validate_empty_folds(self):
        v = WalkForwardValidator()
        with patch.object(v, "create_folds", return_value=[]):
            results = v.validate(pd.DataFrame(), "POINTS")
        assert len(results.folds) == 0
        assert results.n_folds == 0


# ---------------------------------------------------------------------------
# main() CLI
# ---------------------------------------------------------------------------


class TestMain:
    @patch("nba.models.walk_forward_validation.WalkForwardValidator")
    @patch("nba.models.walk_forward_validation.pd.read_csv")
    @patch("nba.models.walk_forward_validation.Path.exists", return_value=True)
    @patch(
        "sys.argv",
        [
            "prog",
            "--market",
            "POINTS",
            "--folds",
            "3",
            "--test-months",
            "2",
            "--min-train-months",
            "6",
        ],
    )
    def test_main_runs(self, mock_exists, mock_read_csv, MockValidator):
        from nba.models.walk_forward_validation import main

        mock_df = pd.DataFrame({"game_date": ["2023-10-01"]})
        mock_read_csv.return_value = mock_df

        mock_results = MagicMock()
        mock_results.summary.return_value = "Test summary"
        mock_validator_instance = MagicMock()
        mock_validator_instance.validate.return_value = mock_results
        MockValidator.return_value = mock_validator_instance

        main()

        MockValidator.assert_called_once_with(n_folds=3, test_months=2, min_train_months=6)
        mock_validator_instance.validate.assert_called_once()

    @patch("nba.models.walk_forward_validation.Path.exists", return_value=False)
    @patch("sys.argv", ["prog", "--market", "POINTS"])
    def test_main_missing_data_file(self, mock_exists):
        from nba.models.walk_forward_validation import main

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

    @patch("nba.models.walk_forward_validation.WalkForwardValidator")
    @patch("nba.models.walk_forward_validation.pd.read_csv")
    @patch("nba.models.walk_forward_validation.Path.exists", return_value=True)
    @patch(
        "sys.argv",
        ["prog", "--market", "REBOUNDS", "--data", "/tmp/test.csv", "--output", "/tmp/out.txt"],
    )
    def test_main_with_custom_data_and_output(self, mock_exists, mock_read_csv, MockValidator):
        from nba.models.walk_forward_validation import main

        mock_df = pd.DataFrame({"game_date": ["2023-10-01"]})
        mock_read_csv.return_value = mock_df

        mock_results = MagicMock()
        mock_results.summary.return_value = "Custom summary"
        mock_validator_instance = MagicMock()
        mock_validator_instance.validate.return_value = mock_results
        MockValidator.return_value = mock_validator_instance

        m = mock_open()
        with patch("builtins.open", m):
            main()

        m.assert_called_once_with("/tmp/out.txt", "w")
        m().write.assert_called_once_with("Custom summary")
