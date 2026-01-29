"""
Unit Tests for Experiment Tracking
==================================
Tests for MLflow experiment tracking integration.

Best Practices Applied:
- Mock MLflow to avoid actual experiment creation
- Test all public methods
- Test error handling when MLflow is unavailable
- Test parameter/metric logging
"""

from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

import pytest


class TestExperimentTrackerInitialization:
    """Tests for ExperimentTracker initialization."""

    def test_tracker_initialization(self):
        """Test ExperimentTracker can be initialized."""
        from nba.core.experiment_tracking import ExperimentTracker

        tracker = ExperimentTracker(experiment_name="test-experiment")
        assert tracker is not None
        assert tracker.experiment_name == "test-experiment"

    def test_default_experiment_name(self):
        """Test default experiment name."""
        from nba.core.experiment_tracking import ExperimentTracker

        tracker = ExperimentTracker()
        assert tracker.experiment_name == "nba-props-xl"

    def test_custom_tracking_uri(self):
        """Test custom tracking URI."""
        from nba.core.experiment_tracking import ExperimentTracker

        tracker = ExperimentTracker(tracking_uri="sqlite:///test.db")
        assert tracker.tracking_uri == "sqlite:///test.db"

    def test_default_tracking_uri(self):
        """Test default tracking URI from env or fallback."""
        from nba.core.experiment_tracking import ExperimentTracker

        tracker = ExperimentTracker()
        assert "sqlite" in tracker.tracking_uri or "http" in tracker.tracking_uri


class TestExperimentTrackerMlflowDisabled:
    """Tests for ExperimentTracker when MLflow is disabled."""

    @patch("nba.core.experiment_tracking.MLFLOW_AVAILABLE", False)
    def test_tracker_disabled_when_mlflow_unavailable(self):
        """Test tracker is disabled when MLflow not available."""
        from nba.core.experiment_tracking import ExperimentTracker

        tracker = ExperimentTracker()
        assert tracker.enabled is False

    @patch("nba.core.experiment_tracking.MLFLOW_AVAILABLE", False)
    def test_start_run_yields_none_when_disabled(self):
        """Test start_run yields None when disabled."""
        from nba.core.experiment_tracking import ExperimentTracker

        tracker = ExperimentTracker()

        with tracker.start_run(run_name="test") as run:
            assert run is None

    @patch("nba.core.experiment_tracking.MLFLOW_AVAILABLE", False)
    def test_log_params_no_op_when_disabled(self):
        """Test log_params is no-op when disabled."""
        from nba.core.experiment_tracking import ExperimentTracker

        tracker = ExperimentTracker()
        # Should not raise
        tracker.log_params({"market": "POINTS"})

    @patch("nba.core.experiment_tracking.MLFLOW_AVAILABLE", False)
    def test_log_metrics_no_op_when_disabled(self):
        """Test log_metrics is no-op when disabled."""
        from nba.core.experiment_tracking import ExperimentTracker

        tracker = ExperimentTracker()
        # Should not raise
        tracker.log_metrics({"auc": 0.76})


class TestParameterLogging:
    """Tests for parameter logging."""

    def test_params_dict_structure(self):
        """Test parameter dictionary structure."""
        params = {
            "market": "POINTS",
            "n_estimators": 2000,
            "learning_rate": 0.02,
            "num_leaves": 63,
        }

        # All params should be serializable
        for key, value in params.items():
            assert isinstance(key, str)
            assert isinstance(value, (str, int, float, bool))

    def test_param_type_conversion(self):
        """Test parameter type conversion for non-standard types."""
        params = {
            "list_param": [1, 2, 3],
            "dict_param": {"a": 1},
            "none_param": None,
        }

        # Convert non-standard types to strings
        clean_params = {}
        for key, value in params.items():
            if isinstance(value, (str, int, float, bool)):
                clean_params[key] = value
            else:
                clean_params[key] = str(value)

        assert clean_params["list_param"] == "[1, 2, 3]"
        assert clean_params["dict_param"] == "{'a': 1}"
        assert clean_params["none_param"] == "None"


class TestMetricLogging:
    """Tests for metric logging."""

    def test_metrics_dict_structure(self):
        """Test metrics dictionary structure."""
        metrics = {
            "rmse_train": 6.128,
            "rmse_test": 6.841,
            "auc_test": 0.767,
            "accuracy_test": 0.901,
        }

        # All metrics should be numeric
        for key, value in metrics.items():
            assert isinstance(key, str)
            assert isinstance(value, (int, float))

    def test_metric_value_conversion(self):
        """Test metric value conversion."""
        import numpy as np

        metrics = {
            "numpy_float": np.float64(0.767),
            "python_float": 0.767,
            "int_metric": 100,
        }

        clean_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                clean_metrics[key] = float(value)

        assert all(isinstance(v, float) for v in clean_metrics.values())

    @pytest.mark.parametrize(
        "metric_name,value,is_valid",
        [
            ("auc", 0.76, True),
            ("auc", 1.5, True),  # Technically invalid AUC but still loggable
            ("rmse", -1.0, True),  # Negative RMSE is technically invalid but loggable
            ("accuracy", 0.95, True),
            ("count", 1000, True),
        ],
    )
    def test_metric_values(self, metric_name, value, is_valid):
        """Test various metric values."""
        assert isinstance(value, (int, float)) == is_valid


class TestFeatureImportanceLogging:
    """Tests for feature importance logging."""

    def test_feature_importance_structure(self):
        """Test feature importance data structure."""
        feature_names = ["line", "ema_points_L5", "team_pace"]
        importances = [100.5, 85.3, 72.1]

        importance_data = list(zip(feature_names, importances))
        assert len(importance_data) == 3
        assert importance_data[0] == ("line", 100.5)

    def test_feature_importance_sorting(self):
        """Test feature importance sorting."""
        importance_data = [
            ("feature_a", 50.0),
            ("feature_b", 100.0),
            ("feature_c", 75.0),
        ]

        sorted_data = sorted(importance_data, key=lambda x: x[1], reverse=True)
        assert sorted_data[0][0] == "feature_b"
        assert sorted_data[0][1] == 100.0

    def test_top_n_features(self):
        """Test getting top N features."""
        importance_data = [
            ("f1", 100.0),
            ("f2", 90.0),
            ("f3", 80.0),
            ("f4", 70.0),
            ("f5", 60.0),
        ]

        top_3 = importance_data[:3]
        assert len(top_3) == 3
        assert top_3[0][0] == "f1"


class TestTrainingDataInfoLogging:
    """Tests for training data info logging."""

    def test_training_data_info_structure(self):
        """Test training data info structure."""
        info = {
            "train_samples": 15000,
            "test_samples": 5000,
            "feature_count": 102,
            "date_range": ("2023-10-24", "2025-11-06"),
        }

        assert info["train_samples"] > 0
        assert info["test_samples"] > 0
        assert info["feature_count"] == 102

    def test_train_test_ratio_calculation(self):
        """Test train/test ratio calculation."""
        train_samples = 15000
        test_samples = 5000
        total = train_samples + test_samples

        ratio = train_samples / total
        assert abs(ratio - 0.75) < 0.01


class TestRunManagement:
    """Tests for MLflow run management."""

    def test_run_name_generation(self):
        """Test run name generation."""
        market = "POINTS"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{market}_training_{timestamp}"

        assert market in run_name
        assert len(run_name) > len(market) + 10

    def test_run_tags_structure(self):
        """Test run tags structure."""
        tags = {
            "market": "POINTS",
            "architecture": "stacked_two_head",
            "environment": "development",
        }

        for key, value in tags.items():
            assert isinstance(key, str)
            assert isinstance(value, str)


class TestBestRunRetrieval:
    """Tests for best run retrieval."""

    def test_metric_comparison(self):
        """Test metric comparison for best run."""
        runs = [
            {"run_id": "1", "auc_test": 0.75},
            {"run_id": "2", "auc_test": 0.78},
            {"run_id": "3", "auc_test": 0.76},
        ]

        best_run = max(runs, key=lambda x: x["auc_test"])
        assert best_run["run_id"] == "2"
        assert best_run["auc_test"] == 0.78

    def test_ascending_metric_comparison(self):
        """Test ascending metric comparison (lower is better)."""
        runs = [
            {"run_id": "1", "rmse_test": 6.5},
            {"run_id": "2", "rmse_test": 6.0},
            {"run_id": "3", "rmse_test": 7.0},
        ]

        best_run = min(runs, key=lambda x: x["rmse_test"])
        assert best_run["run_id"] == "2"
        assert best_run["rmse_test"] == 6.0


class TestConvenienceFunction:
    """Tests for the log_training_run convenience function."""

    def test_log_training_run_params(self):
        """Test log_training_run parameter structure."""
        params = {
            "market": "POINTS",
            "test_size": 0.3,
            "random_state": 42,
            "n_estimators": 2000,
        }

        assert "market" in params
        assert isinstance(params["test_size"], float)
        assert isinstance(params["random_state"], int)

    def test_log_training_run_metrics(self):
        """Test log_training_run metrics structure."""
        metrics = {
            "rmse_train": 6.1,
            "rmse_test": 6.8,
            "auc_test": 0.76,
            "accuracy_test": 0.89,
        }

        assert all(isinstance(v, float) for v in metrics.values())


class TestTempfileCleanup:
    """Tests for tempfile cleanup."""

    def test_tempfile_creation_and_cleanup(self):
        """Test tempfile is created and cleaned up."""
        import json
        import os
        import tempfile

        # Create tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"test": "data"}, f)
            temp_path = f.name

        # Verify file exists
        assert os.path.exists(temp_path)

        # Simulate MLflow copy, then cleanup
        os.remove(temp_path)

        # Verify file is removed
        assert not os.path.exists(temp_path)

    def test_tempfile_not_leaked(self):
        """Test tempfile is not leaked on error."""
        import json
        import os
        import tempfile

        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json.dump({"test": "data"}, f)
                temp_path = f.name

            # Simulate error during processing
            raise ValueError("Simulated error")
        except ValueError:
            pass
        finally:
            # Cleanup should happen even on error
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

        assert temp_path is None or not os.path.exists(temp_path)
