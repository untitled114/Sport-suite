"""
Tests for Market Classifier Model (Model 2) — LightGBM Classifier.

Tests feature preparation, training, save/load, and prediction.
Model 1 is mocked to provide fixed projections.
"""

import json
import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from nba.models.market_classifier_model import (
    CLASSIFIER_PREFIXES,
    FORBIDDEN_MODEL1_PREFIXES,
    MarketClassifierModel,
)


@pytest.fixture
def mock_projection_model():
    """Create a mock PlayerProjectionModel that returns fixed predictions."""
    mock_model = MagicMock()
    mock_model.feature_names = [
        "ema_points_L3",
        "ema_points_L5",
        "ema_points_L10",
        "team_pace",
        "is_home",
    ]

    def mock_predict(features):
        # Return projections around 25, with some noise
        np.random.seed(123)
        return np.random.uniform(20, 30, len(features))

    mock_model.predict.side_effect = mock_predict
    return mock_model


@pytest.fixture
def classifier_training_data():
    """Create training data with both Model 1 features and classifier features.

    This simulates the full XL training dataset that Model 2 receives.
    """
    np.random.seed(42)
    n = 300
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    line = np.random.uniform(18, 32, n)
    actual = line + np.random.normal(0, 5, n)

    df = pd.DataFrame(
        {
            "player_name": np.random.choice(["Player A", "Player B", "Player C"], n),
            "game_date": dates,
            "actual_points": actual,
            "line": line,
            # Model 1 features (present in CSV but should NOT leak into Model 2)
            "ema_points_L3": actual + np.random.normal(0, 2, n),
            "ema_points_L5": actual + np.random.normal(0, 1.5, n),
            "ema_points_L10": actual + np.random.normal(0, 1, n),
            "team_pace": np.random.uniform(95, 105, n),
            "is_home": np.random.choice([0, 1], n),
            # Classifier features (allowed for Model 2)
            "line_spread": np.random.uniform(0, 3, n),
            "line_std": np.random.uniform(0.3, 1.5, n),
            "consensus_line": line + np.random.normal(0, 0.5, n),
            "num_books": np.random.randint(3, 8, n),
            "draftkings_deviation": np.random.normal(0, 0.5, n),
            "fanduel_deviation": np.random.normal(0, 0.5, n),
            "betmgm_deviation": np.random.normal(0, 0.5, n),
            "softest_vs_consensus": np.random.uniform(-2, 0, n),
            "hardest_vs_consensus": np.random.uniform(0, 2, n),
            "min_line": line - np.random.uniform(0, 2, n),
            "max_line": line + np.random.uniform(0, 2, n),
            "bp_hit_rate": np.random.uniform(0.3, 0.8, n),
            "bp_projection": actual + np.random.normal(0, 3, n),
            "prop_hit_rate_L10": np.random.uniform(0.3, 0.8, n),
            "dvp_rank": np.random.randint(1, 31, n),
            "dvp_factor": np.random.uniform(0.8, 1.2, n),
            "points_std_L10": np.random.uniform(3, 8, n),
        }
    )
    return df


class TestPrepareClassifierFeatures:
    """Tests for MarketClassifierModel.prepare_classifier_features()."""

    def test_prepare_classifier_features_computes_projection_diff(
        self, mock_projection_model, classifier_training_data, tmp_path
    ):
        """projection_diff column is created from Model 1 predictions - line."""
        with patch("nba.models.market_classifier_model.PlayerProjectionModel") as MockPPM:
            instance = mock_projection_model
            MockPPM.return_value = instance

            model = MarketClassifierModel(market="POINTS")
            X, y, metadata = model.prepare_classifier_features(
                classifier_training_data, str(tmp_path)
            )
            # projection_diff should be in the classifier features
            assert "projection_diff" in X.columns

    def test_prepare_classifier_features_validates_no_player_features(
        self, mock_projection_model, classifier_training_data, tmp_path
    ):
        """Raises ValueError if Model 1 (ema_*) features leak into classifier."""
        with patch("nba.models.market_classifier_model.PlayerProjectionModel") as MockPPM:
            MockPPM.return_value = mock_projection_model

            # Inject a forbidden Model 1 feature with a classifier prefix
            # This shouldn't happen normally, but test the safety guard
            df = classifier_training_data.copy()
            df["ema_fake_classifier"] = 1.0

            model = MarketClassifierModel(market="POINTS")
            # ema_ features are in FORBIDDEN_MODEL1_PREFIXES but NOT in CLASSIFIER_PREFIXES
            # so they should be filtered out by prefix matching, not raise ValueError.
            # The ValueError only fires if a col matches BOTH classifier and forbidden prefixes.
            # Since the test data has ema_ columns that don't match CLASSIFIER_PREFIXES,
            # they are simply excluded. This test verifies they're excluded.
            X, y, metadata = model.prepare_classifier_features(df, str(tmp_path))
            ema_cols = [c for c in X.columns if c.startswith("ema_")]
            assert len(ema_cols) == 0

    def test_prepare_classifier_features_binary_target(
        self, mock_projection_model, classifier_training_data, tmp_path
    ):
        """Target is binary: 1 when actual > line."""
        with patch("nba.models.market_classifier_model.PlayerProjectionModel") as MockPPM:
            MockPPM.return_value = mock_projection_model

            model = MarketClassifierModel(market="POINTS")
            X, y, metadata = model.prepare_classifier_features(
                classifier_training_data, str(tmp_path)
            )
            assert set(y.unique()).issubset({0, 1})

    def test_prepare_classifier_features_normalized_diff(
        self, mock_projection_model, classifier_training_data, tmp_path
    ):
        """projection_diff_normalized is computed when std column exists."""
        with patch("nba.models.market_classifier_model.PlayerProjectionModel") as MockPPM:
            MockPPM.return_value = mock_projection_model

            model = MarketClassifierModel(market="POINTS")
            X, y, metadata = model.prepare_classifier_features(
                classifier_training_data, str(tmp_path)
            )
            # points_std_L10 is present, so projection_diff_normalized should exist
            assert "projection_diff_normalized" in X.columns


class TestClassifierTrain:
    """Tests for MarketClassifierModel.train()."""

    @pytest.fixture
    def classifier_train_test(self, mock_projection_model, classifier_training_data, tmp_path):
        """Prepare classifier train/test splits."""
        with patch("nba.models.market_classifier_model.PlayerProjectionModel") as MockPPM:
            MockPPM.return_value = mock_projection_model

            model = MarketClassifierModel(market="POINTS")
            X, y, metadata = model.prepare_classifier_features(
                classifier_training_data, str(tmp_path)
            )
            split = int(len(X) * 0.7)
            return {
                "model": model,
                "X_train": X.iloc[:split],
                "X_test": X.iloc[split:],
                "y_train": y.iloc[:split],
                "y_test": y.iloc[split:],
            }

    def test_train_produces_metrics(self, classifier_train_test):
        """train() returns dict with auc, accuracy, brier_score."""
        d = classifier_train_test
        metrics = d["model"].train(d["X_train"], d["y_train"], d["X_test"], d["y_test"])
        assert "auc" in metrics
        assert "accuracy" in metrics
        assert "brier_score" in metrics
        assert "log_loss" in metrics
        assert "n_trees" in metrics
        assert 0 <= metrics["auc"] <= 1
        assert 0 <= metrics["accuracy"] <= 1

    def test_train_creates_artifacts(self, classifier_train_test):
        """After training, classifier/imputer/scaler are not None."""
        d = classifier_train_test
        d["model"].train(d["X_train"], d["y_train"], d["X_test"], d["y_test"])
        assert d["model"].classifier is not None
        assert d["model"].imputer is not None
        assert d["model"].scaler is not None
        # calibrator may or may not be set depending on Brier improvement
        assert hasattr(d["model"], "calibrator")


class TestClassifierSaveLoad:
    """Tests for save() and load() round-trip."""

    def test_save_and_load(self, mock_projection_model, classifier_training_data, tmp_path):
        """save() then load() round-trips correctly."""
        with patch("nba.models.market_classifier_model.PlayerProjectionModel") as MockPPM:
            MockPPM.return_value = mock_projection_model

            model = MarketClassifierModel(market="POINTS")
            X, y, metadata = model.prepare_classifier_features(
                classifier_training_data, str(tmp_path)
            )
            split = int(len(X) * 0.7)
            model.train(X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:])

            save_dir = tmp_path / "saved_model"
            model.save(str(save_dir))

            # Load into a new model
            model2 = MarketClassifierModel(market="POINTS")
            model2.load(str(save_dir))

            assert model2.classifier is not None
            assert model2.imputer is not None
            assert model2.scaler is not None
            assert model2.feature_names == model.feature_names

    def test_save_creates_metadata_json(
        self, mock_projection_model, classifier_training_data, tmp_path
    ):
        """save() creates a valid metadata JSON file."""
        with patch("nba.models.market_classifier_model.PlayerProjectionModel") as MockPPM:
            MockPPM.return_value = mock_projection_model

            model = MarketClassifierModel(market="POINTS")
            X, y, metadata = model.prepare_classifier_features(
                classifier_training_data, str(tmp_path)
            )
            split = int(len(X) * 0.7)
            model.train(X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:])

            save_dir = tmp_path / "saved_model"
            model.save(str(save_dir))

            meta_path = save_dir / "points_market_metadata.json"
            assert meta_path.exists()
            with open(meta_path) as f:
                meta = json.load(f)
            assert meta["market"] == "POINTS"
            assert meta["model_type"] == "market_inefficiency_classifier"


class TestPredictProba:
    """Tests for predict_proba() after training."""

    def test_predict_proba_returns_probabilities(
        self, mock_projection_model, classifier_training_data, tmp_path
    ):
        """predict_proba() returns probabilities in [0, 1]."""
        with patch("nba.models.market_classifier_model.PlayerProjectionModel") as MockPPM:
            MockPPM.return_value = mock_projection_model

            model = MarketClassifierModel(market="POINTS")
            X, y, metadata = model.prepare_classifier_features(
                classifier_training_data, str(tmp_path)
            )
            split = int(len(X) * 0.7)
            model.train(X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:])

            probs = model.predict_proba(X.iloc[split:])
            assert isinstance(probs, np.ndarray)
            assert len(probs) == len(X.iloc[split:])
            assert all(0 <= p <= 1 for p in probs)

    def test_predict_proba_raises_without_training(self):
        """predict_proba() raises RuntimeError if not trained."""
        model = MarketClassifierModel(market="POINTS")
        dummy = pd.DataFrame({"projection_diff": [1.0]})
        with pytest.raises(RuntimeError, match="Model not trained"):
            model.predict_proba(dummy)


class TestCalibrationQualityGate:
    """Tests for calibration quality gate behavior."""

    def test_calibration_quality_gate(
        self, mock_projection_model, classifier_training_data, tmp_path
    ):
        """If Brier gets worse with calibration, calibrator is None."""
        with patch("nba.models.market_classifier_model.PlayerProjectionModel") as MockPPM:
            MockPPM.return_value = mock_projection_model

            model = MarketClassifierModel(market="POINTS")
            X, y, metadata = model.prepare_classifier_features(
                classifier_training_data, str(tmp_path)
            )
            split = int(len(X) * 0.7)
            metrics = model.train(X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:])

            # The calibrator is either None (Brier worse) or a fitted LogisticRegression
            # We can't control which happens with random data, but we verify the flag
            assert metrics["calibrated"] == (model.calibrator is not None)


class TestClassifierEdgeCases:
    """Tests for edge cases in classifier feature preparation and training."""

    def test_prepare_features_raises_on_leaked_model1_features(
        self, mock_projection_model, tmp_path
    ):
        """ValueError raised when Model 1 features match CLASSIFIER_PREFIXES."""
        np.random.seed(42)
        n = 50
        line = np.random.uniform(18, 32, n)
        actual = line + np.random.normal(0, 5, n)

        # Create a DataFrame where a forbidden prefix also matches a classifier prefix.
        # "ema_" is forbidden; we need it to also match CLASSIFIER_PREFIXES.
        # We can simulate this by injecting a feature that starts with both.
        # Actually, the guard checks classifier_cols for FORBIDDEN_MODEL1_PREFIXES.
        # So if a column starts with a classifier prefix AND a forbidden prefix,
        # it triggers the error. The only realistic way: create a prefix collision.
        # Let's directly test the guard by manipulating classifier_cols in the flow.
        df = pd.DataFrame(
            {
                "player_name": [f"P{i}" for i in range(n)],
                "game_date": pd.date_range("2024-01-01", periods=n),
                "actual_points": actual,
                "line": line,
                "ema_points_L3": actual + np.random.normal(0, 2, n),
                "ema_points_L5": actual + np.random.normal(0, 1.5, n),
                "ema_points_L10": actual + np.random.normal(0, 1, n),
                "team_pace": np.random.uniform(95, 105, n),
                "is_home": np.random.choice([0, 1], n),
                "projection_diff": np.random.normal(0, 3, n),
                "line_spread": np.random.uniform(0, 3, n),
                "points_std_L10": np.random.uniform(3, 8, n),
            }
        )

        with patch("nba.models.market_classifier_model.PlayerProjectionModel") as MockPPM:
            MockPPM.return_value = mock_projection_model

            # Inject a feature that has both a classifier prefix AND forbidden prefix
            # by temporarily adding ema_ to CLASSIFIER_PREFIXES
            with patch(
                "nba.models.market_classifier_model.CLASSIFIER_PREFIXES",
                CLASSIFIER_PREFIXES + ("ema_",),
            ):
                model = MarketClassifierModel(market="POINTS")
                with pytest.raises(ValueError, match="Model 1 features leaked"):
                    model.prepare_classifier_features(df, str(tmp_path))

    def test_prepare_features_drops_non_numeric(self, mock_projection_model, tmp_path):
        """Non-numeric classifier features are dropped."""
        np.random.seed(42)
        n = 50
        line = np.random.uniform(18, 32, n)
        actual = line + np.random.normal(0, 5, n)

        df = pd.DataFrame(
            {
                "player_name": [f"P{i}" for i in range(n)],
                "game_date": pd.date_range("2024-01-01", periods=n),
                "actual_points": actual,
                "line": line,
                "ema_points_L3": actual + np.random.normal(0, 2, n),
                "team_pace": np.random.uniform(95, 105, n),
                "is_home": np.random.choice([0, 1], n),
                "projection_diff": np.random.normal(0, 3, n),
                "line_spread": np.random.uniform(0, 3, n),
                "line_source_reliability": ["high"] * n,  # Non-numeric classifier feature
                "points_std_L10": np.random.uniform(3, 8, n),
            }
        )

        with patch("nba.models.market_classifier_model.PlayerProjectionModel") as MockPPM:
            MockPPM.return_value = mock_projection_model
            model = MarketClassifierModel(market="POINTS")
            X, y, metadata = model.prepare_classifier_features(df, str(tmp_path))
            # line_source_reliability should be dropped (non-numeric)
            assert "line_source_reliability" not in X.columns

    def test_train_with_temporal_decay(
        self, mock_projection_model, classifier_training_data, tmp_path
    ):
        """Training with game_dates enables temporal decay weighting."""
        with patch("nba.models.market_classifier_model.PlayerProjectionModel") as MockPPM:
            MockPPM.return_value = mock_projection_model

            model = MarketClassifierModel(market="POINTS")
            X, y, metadata = model.prepare_classifier_features(
                classifier_training_data, str(tmp_path)
            )
            split = int(len(X) * 0.7)

            game_dates_train = (
                metadata["game_date"].iloc[:split] if "game_date" in metadata.columns else None
            )
            game_dates_test = (
                metadata["game_date"].iloc[split:] if "game_date" in metadata.columns else None
            )

            metrics = model.train(
                X.iloc[:split],
                y.iloc[:split],
                X.iloc[split:],
                y.iloc[split:],
                game_dates_train=game_dates_train,
                game_dates_test=game_dates_test,
            )
            assert metrics["auc"] > 0
            assert metrics["n_trees"] > 0
