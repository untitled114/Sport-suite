"""
Tests for Player Projection Model (Model 1) — LightGBM Regressor.

Tests data loading, feature preparation, training, save/load, and prediction.
Uses small DataFrames with numpy random for feature matrices.
"""

import json
import pickle
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from nba.models.player_projection_model import (
    EXCLUDE_COLS,
    FORBIDDEN_FEATURE_PREFIXES,
    PlayerProjectionModel,
)


@pytest.fixture
def projection_training_data():
    """Create a small realistic training DataFrame (~200 rows).

    Contains player/team features that Model 1 expects.
    No line features, no book features, no BP features.
    """
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    players = np.random.choice(["Player A", "Player B", "Player C", "Player D"], n)
    actual_pts = np.random.uniform(10, 35, n)

    df = pd.DataFrame(
        {
            "player_name": players,
            "game_date": dates,
            "actual_points": actual_pts,
            # Player EMA features
            "ema_points_L3": actual_pts + np.random.normal(0, 2, n),
            "ema_points_L5": actual_pts + np.random.normal(0, 1.5, n),
            "ema_points_L10": actual_pts + np.random.normal(0, 1, n),
            "ema_rebounds_L3": np.random.uniform(3, 10, n),
            "ema_rebounds_L5": np.random.uniform(3, 10, n),
            "ema_rebounds_L10": np.random.uniform(3, 10, n),
            "ema_minutes_L5": np.random.uniform(25, 38, n),
            # Variance features
            "minutes_std_L5": np.random.uniform(1, 5, n),
            "points_std_L10": np.random.uniform(3, 8, n),
            # Team/game context features
            "is_home": np.random.choice([0, 1], n),
            "team_pace": np.random.uniform(95, 105, n),
            "team_off_rating": np.random.uniform(108, 120, n),
            "team_def_rating": np.random.uniform(108, 118, n),
            "opp_pace": np.random.uniform(95, 105, n),
            "opp_def_rating": np.random.uniform(108, 118, n),
            "rest_days": np.random.choice([1, 2, 3], n),
            "is_back_to_back": np.random.choice([0, 1], n, p=[0.8, 0.2]),
            "games_played_season": np.random.randint(10, 70, n),
            "season_phase": np.random.uniform(0.1, 0.9, n),
        }
    )
    return df


@pytest.fixture
def projection_csv(tmp_path, projection_training_data):
    """Write projection training data to a CSV file."""
    csv_path = tmp_path / "projection_training_POINTS.csv"
    projection_training_data.to_csv(csv_path, index=False)
    return csv_path


class TestLoadData:
    """Tests for PlayerProjectionModel.load_data()."""

    def test_load_data_reads_csv(self, projection_csv):
        """load_data reads CSV and returns DataFrame."""
        model = PlayerProjectionModel(market="POINTS")
        df = model.load_data(str(projection_csv))
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_load_data_deduplicates(self, tmp_path):
        """load_data deduplicates by player_name + game_date."""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "player_name": ["Player A", "Player A", "Player B"],
                "game_date": ["2024-01-01", "2024-01-01", "2024-01-01"],
                "actual_points": [20, 22, 18],
                "ema_points_L3": [19, 21, 17],
            }
        )
        csv_path = tmp_path / "dup_test.csv"
        df.to_csv(csv_path, index=False)

        model = PlayerProjectionModel(market="POINTS")
        result = model.load_data(str(csv_path))
        # Player A duplicated on same date -> should dedup to 2 rows
        assert len(result) == 2

    def test_load_data_validates_target(self, tmp_path):
        """load_data raises ValueError when target column is missing."""
        df = pd.DataFrame(
            {
                "player_name": ["A"],
                "game_date": ["2024-01-01"],
                "ema_points_L3": [20.0],
            }
        )
        csv_path = tmp_path / "no_target.csv"
        df.to_csv(csv_path, index=False)

        model = PlayerProjectionModel(market="POINTS")
        with pytest.raises(ValueError, match="Target column.*not found"):
            model.load_data(str(csv_path))

    def test_load_data_filters_by_stat_type(self, tmp_path):
        """load_data filters by stat_type when multiple are present."""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "player_name": ["A", "B", "C", "D"],
                "game_date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
                "stat_type": ["points", "rebounds", "points", "rebounds"],
                "actual_points": [25.0, 8.0, 22.0, 6.0],
                "ema_points_L3": [24.0, 7.0, 21.0, 5.0],
            }
        )
        csv_path = tmp_path / "multi_stat.csv"
        df.to_csv(csv_path, index=False)

        model = PlayerProjectionModel(market="POINTS")
        result = model.load_data(str(csv_path))
        # Should only have the 2 "points" rows
        assert len(result) == 2

    def test_load_data_uses_actual_result_fallback(self, tmp_path):
        """load_data uses actual_result as fallback for target column."""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "player_name": ["A", "B"],
                "game_date": ["2024-01-01", "2024-01-02"],
                "actual_result": [25.0, 30.0],
                "ema_points_L3": [24.0, 29.0],
            }
        )
        csv_path = tmp_path / "fallback.csv"
        df.to_csv(csv_path, index=False)

        model = PlayerProjectionModel(market="POINTS")
        result = model.load_data(str(csv_path))
        assert "actual_points" in result.columns


class TestPrepareFeatures:
    """Tests for PlayerProjectionModel.prepare_features()."""

    def test_prepare_features_excludes_metadata(self, projection_training_data):
        """Metadata columns are excluded from feature matrix."""
        model = PlayerProjectionModel(market="POINTS")
        X, y, metadata = model.prepare_features(projection_training_data)
        for col in EXCLUDE_COLS:
            assert col not in X.columns

    def test_prepare_features_validates_no_line_features(self, projection_training_data):
        """Raises ValueError if forbidden line/book/BP features are present."""
        df = projection_training_data.copy()
        df["line_spread"] = np.random.uniform(0, 3, len(df))
        df["bp_hit_rate"] = np.random.uniform(0, 1, len(df))

        model = PlayerProjectionModel(market="POINTS")
        with pytest.raises(ValueError, match="Forbidden line/book/BP features"):
            model.prepare_features(df)

    def test_prepare_features_returns_correct_shapes(self, projection_training_data):
        """X, y, metadata have matching row counts."""
        model = PlayerProjectionModel(market="POINTS")
        X, y, metadata = model.prepare_features(projection_training_data)
        assert len(X) == len(y) == len(metadata)
        assert len(X.columns) > 0

    def test_prepare_features_drops_zero_variance(self, projection_training_data):
        """Zero-variance features are dropped."""
        df = projection_training_data.copy()
        df["constant_feature"] = 5.0  # Zero variance

        model = PlayerProjectionModel(market="POINTS")
        X, y, metadata = model.prepare_features(df)
        assert "constant_feature" not in X.columns

    def test_prepare_features_sets_feature_names(self, projection_training_data):
        """feature_names attribute is set after prepare_features."""
        model = PlayerProjectionModel(market="POINTS")
        X, y, metadata = model.prepare_features(projection_training_data)
        assert model.feature_names is not None
        assert len(model.feature_names) == len(X.columns)

    def test_prepare_features_drops_non_numeric(self, projection_training_data):
        """Non-numeric columns are dropped from features."""
        df = projection_training_data.copy()
        df["team_name"] = "Lakers"  # Non-numeric feature

        model = PlayerProjectionModel(market="POINTS")
        X, y, metadata = model.prepare_features(df)
        assert "team_name" not in X.columns


class TestTrain:
    """Tests for PlayerProjectionModel.train()."""

    @pytest.fixture
    def train_test_split(self, projection_training_data):
        """Split projection data into train/test sets."""
        model = PlayerProjectionModel(market="POINTS")
        df = projection_training_data
        X, y, metadata = model.prepare_features(df)
        split = int(len(X) * 0.7)
        return {
            "model": model,
            "X_train": X.iloc[:split],
            "X_test": X.iloc[split:],
            "y_train": y.iloc[:split],
            "y_test": y.iloc[split:],
            "game_dates_train": (
                metadata["game_date"].iloc[:split] if "game_date" in metadata.columns else None
            ),
            "game_dates_test": (
                metadata["game_date"].iloc[split:] if "game_date" in metadata.columns else None
            ),
        }

    def test_train_produces_metrics(self, train_test_split):
        """train() returns dict with rmse_train, rmse_test, etc."""
        d = train_test_split
        metrics = d["model"].train(
            d["X_train"],
            d["y_train"],
            d["X_test"],
            d["y_test"],
            game_dates_train=d["game_dates_train"],
            game_dates_test=d["game_dates_test"],
        )
        assert "rmse_train" in metrics
        assert "rmse_test" in metrics
        assert "mae_test" in metrics
        assert "r2_test" in metrics
        assert "n_trees" in metrics
        assert metrics["rmse_train"] > 0
        assert metrics["rmse_test"] > 0

    def test_train_creates_artifacts(self, train_test_split):
        """After training, regressor/imputer/scaler are not None."""
        d = train_test_split
        d["model"].train(
            d["X_train"],
            d["y_train"],
            d["X_test"],
            d["y_test"],
        )
        assert d["model"].regressor is not None
        assert d["model"].imputer is not None
        assert d["model"].scaler is not None

    def test_train_without_temporal_decay(self, train_test_split):
        """Training works without game dates (no temporal decay)."""
        d = train_test_split
        metrics = d["model"].train(
            d["X_train"],
            d["y_train"],
            d["X_test"],
            d["y_test"],
            game_dates_train=None,
            game_dates_test=None,
        )
        assert metrics["rmse_test"] > 0


class TestSaveAndLoad:
    """Tests for save() and load() round-trip."""

    def test_save_and_load(self, projection_training_data, tmp_path):
        """save() then load() round-trips correctly."""
        model = PlayerProjectionModel(market="POINTS")
        df = projection_training_data
        X, y, metadata = model.prepare_features(df)
        split = int(len(X) * 0.7)

        model.train(X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:])

        # Save
        model.save(str(tmp_path), model_version="test_v1")

        # Load into a new model
        model2 = PlayerProjectionModel(market="POINTS")
        model2.load(str(tmp_path), model_version="test_v1")

        assert model2.regressor is not None
        assert model2.imputer is not None
        assert model2.scaler is not None
        assert model2.feature_names == model.feature_names

    def test_save_creates_metadata_json(self, projection_training_data, tmp_path):
        """save() creates a valid metadata JSON file."""
        model = PlayerProjectionModel(market="POINTS")
        df = projection_training_data
        X, y, metadata = model.prepare_features(df)
        split = int(len(X) * 0.7)
        model.train(X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:])
        model.save(str(tmp_path), model_version="test_v1")

        meta_path = tmp_path / "points_test_v1_metadata.json"
        assert meta_path.exists()
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["market"] == "POINTS"
        assert meta["model_type"] == "player_projection_regressor"
        assert "features" in meta
        assert meta["features"]["count"] == len(model.feature_names)


class TestPredict:
    """Tests for predict() after training."""

    def test_predict_returns_correct_shape(self, projection_training_data):
        """After training, predict() returns array of correct shape."""
        model = PlayerProjectionModel(market="POINTS")
        df = projection_training_data
        X, y, metadata = model.prepare_features(df)
        split = int(len(X) * 0.7)
        model.train(X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:])

        preds = model.predict(X.iloc[split:])
        assert isinstance(preds, np.ndarray)
        assert len(preds) == len(X.iloc[split:])

    def test_predict_raises_without_training(self):
        """predict() raises RuntimeError if model not trained."""
        model = PlayerProjectionModel(market="POINTS")
        dummy = pd.DataFrame({"feature_1": [1.0]})
        with pytest.raises(RuntimeError, match="Model not trained"):
            model.predict(dummy)

    def test_predict_values_reasonable(self, projection_training_data):
        """Predictions should be in a reasonable range for NBA stat values."""
        model = PlayerProjectionModel(market="POINTS")
        df = projection_training_data
        X, y, metadata = model.prepare_features(df)
        split = int(len(X) * 0.7)
        model.train(X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:])

        preds = model.predict(X.iloc[split:])
        # NBA points typically range 0-60
        assert all(p > -10 for p in preds)
        assert all(p < 80 for p in preds)
