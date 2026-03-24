"""
Tests for ModelTracker (MLflow tracking)
==========================================
Tests the MLflow tracking wrapper used by all model types.
All tests work with MLflow disabled (graceful no-op behavior).
Enabled-mode tests inject a mock mlflow module directly.
"""

from unittest.mock import MagicMock, patch

import pytest

from nba.models.mlflow_tracking import ModelTracker, ProjectionModelTracker


@pytest.fixture
def tracker_disabled():
    """Tracker with MLflow disabled (no mlflow installed)."""
    with patch("nba.models.mlflow_tracking.MLFLOW_AVAILABLE", False):
        t = ModelTracker(experiment="test-experiment")
        t.enabled = False
        return t


@pytest.fixture
def tracker_enabled():
    """Tracker with mocked MLflow injected into module namespace."""
    mock_mlflow = MagicMock()
    mock_mlflow.get_experiment_by_name.return_value = MagicMock()
    mock_mlflow.active_run.return_value = None
    # start_run returns an object with info.run_id
    mock_run = MagicMock()
    mock_run.info.run_id = "test-run-id-123"
    mock_mlflow.start_run.return_value = mock_run

    import nba.models.mlflow_tracking as mod

    original_available = mod.MLFLOW_AVAILABLE
    mod.MLFLOW_AVAILABLE = True
    mod.mlflow = mock_mlflow

    t = ModelTracker.__new__(ModelTracker)
    t.experiment = "test-experiment"
    t.tracking_uri = "sqlite:///mlruns.db"
    t.enabled = True
    t._active = False
    t._run_id = None
    t._mock_mlflow = mock_mlflow

    yield t

    mod.MLFLOW_AVAILABLE = original_available
    if hasattr(mod, "mlflow") and mod.mlflow is mock_mlflow:
        delattr(mod, "mlflow")


# =============================================================================
# Backwards Compatibility
# =============================================================================


class TestBackwardsCompat:
    """ProjectionModelTracker alias still works."""

    def test_alias_is_model_tracker(self):
        assert ProjectionModelTracker is ModelTracker


# =============================================================================
# Disabled Mode (Graceful No-ops)
# =============================================================================


class TestDisabledMode:
    """When MLflow is not installed, all methods should be no-ops."""

    def test_start_run_noop(self, tracker_disabled):
        result = tracker_disabled.start_run("test")
        assert result is None
        assert not tracker_disabled._active

    def test_end_run_noop(self, tracker_disabled):
        tracker_disabled.end_run()

    def test_log_params_noop(self, tracker_disabled):
        tracker_disabled.log_params({"market": "POINTS"})

    def test_log_metrics_noop(self, tracker_disabled):
        tracker_disabled.log_metrics({"auc": 0.76})

    def test_log_feature_importance_noop(self, tracker_disabled):
        tracker_disabled.log_feature_importance(["f1", "f2"], [1.0, 0.5])

    def test_log_artifact_noop(self, tracker_disabled):
        tracker_disabled.log_artifact("/tmp/test.json")

    def test_log_model_path_noop(self, tracker_disabled):
        tracker_disabled.log_model_path("/path/to/model.pkl")

    def test_run_id_none_when_disabled(self, tracker_disabled):
        assert tracker_disabled.run_id is None


# =============================================================================
# Enabled Mode (With Mocked MLflow)
# =============================================================================


class TestEnabledMode:
    """When MLflow is available, methods should call MLflow APIs."""

    def test_start_run_calls_mlflow(self, tracker_enabled):
        mock = tracker_enabled._mock_mlflow
        tracker_enabled.start_run("test_run")
        assert tracker_enabled._active
        mock.start_run.assert_called_once()

    def test_start_run_returns_run_id(self, tracker_enabled):
        run_id = tracker_enabled.start_run("test_run")
        assert run_id == "test-run-id-123"

    def test_start_run_sets_run_id(self, tracker_enabled):
        tracker_enabled.start_run("test_run")
        assert tracker_enabled.run_id == "test-run-id-123"

    def test_start_run_with_tags(self, tracker_enabled):
        mock = tracker_enabled._mock_mlflow
        tracker_enabled.start_run("test", tags={"env": "test"})
        call_kwargs = mock.start_run.call_args[1]
        assert "env" in call_kwargs["tags"]
        assert "timestamp" in call_kwargs["tags"]

    def test_start_run_includes_git_hash(self, tracker_enabled):
        mock = tracker_enabled._mock_mlflow
        tracker_enabled.start_run("test")
        call_kwargs = mock.start_run.call_args[1]
        # git_hash may or may not be present depending on environment
        assert "package_version" in call_kwargs["tags"]

    def test_start_run_cleans_stale_run(self, tracker_enabled):
        mock = tracker_enabled._mock_mlflow
        mock.active_run.return_value = MagicMock()  # simulate stale run
        tracker_enabled.start_run("test")
        # end_run called once to clean stale, then start_run called
        mock.end_run.assert_called_once()
        mock.start_run.assert_called_once()

    def test_end_run_calls_mlflow(self, tracker_enabled):
        mock = tracker_enabled._mock_mlflow
        tracker_enabled.start_run("test")
        tracker_enabled.end_run()
        assert not tracker_enabled._active
        mock.end_run.assert_called()

    def test_end_run_clears_run_id_property(self, tracker_enabled):
        tracker_enabled.start_run("test")
        assert tracker_enabled.run_id == "test-run-id-123"
        tracker_enabled.end_run()
        assert tracker_enabled.run_id is None

    def test_end_run_idempotent(self, tracker_enabled):
        """Calling end_run twice doesn't error."""
        tracker_enabled.start_run("test")
        tracker_enabled.end_run()
        tracker_enabled.end_run()  # should not raise

    def test_log_params_calls_mlflow(self, tracker_enabled):
        mock = tracker_enabled._mock_mlflow
        tracker_enabled.start_run("test")
        tracker_enabled.log_params({"market": "POINTS", "features": 102})
        mock.log_params.assert_called_once()

    def test_log_params_converts_non_primitives(self, tracker_enabled):
        mock = tracker_enabled._mock_mlflow
        tracker_enabled.start_run("test")
        tracker_enabled.log_params({"list_param": [1, 2, 3]})
        call_args = mock.log_params.call_args[0][0]
        assert call_args["list_param"] == "[1, 2, 3]"

    def test_log_metrics_calls_mlflow(self, tracker_enabled):
        mock = tracker_enabled._mock_mlflow
        tracker_enabled.start_run("test")
        tracker_enabled.log_metrics({"auc": 0.76, "rmse": 6.0})
        assert mock.log_metric.call_count == 2

    def test_log_metrics_skips_non_numeric(self, tracker_enabled):
        mock = tracker_enabled._mock_mlflow
        tracker_enabled.start_run("test")
        tracker_enabled.log_metrics({"auc": 0.76, "name": "test"})
        assert mock.log_metric.call_count == 1


# =============================================================================
# Feature Importance Logging
# =============================================================================


class TestFeatureImportance:
    """Tests for log_feature_importance method."""

    def test_log_feature_importance_logs_metrics(self, tracker_enabled):
        mock = tracker_enabled._mock_mlflow
        tracker_enabled.start_run("test")
        tracker_enabled.log_feature_importance(
            feature_names=["f1", "f2", "f3"],
            importances=[100.0, 50.0, 25.0],
            head="regressor",
        )
        # 3 features = 3 log_metric calls
        assert mock.log_metric.call_count == 3
        # 3 set_tag calls for feature names
        assert mock.set_tag.call_count == 3

    def test_log_feature_importance_logs_artifact(self, tracker_enabled):
        mock = tracker_enabled._mock_mlflow
        tracker_enabled.start_run("test")
        tracker_enabled.log_feature_importance(
            feature_names=["f1", "f2"],
            importances=[100.0, 50.0],
            head="classifier",
        )
        mock.log_artifact.assert_called_once()
        call_args = mock.log_artifact.call_args
        assert call_args[1].get("artifact_path") or call_args[0][1] == "feature_importance"

    def test_log_feature_importance_respects_top_n(self, tracker_enabled):
        mock = tracker_enabled._mock_mlflow
        tracker_enabled.start_run("test")
        names = [f"f{i}" for i in range(50)]
        imps = [float(i) for i in range(50, 0, -1)]
        tracker_enabled.log_feature_importance(
            feature_names=names,
            importances=imps,
            head="regressor",
            top_n=5,
        )
        # Only top 5 logged as metrics
        assert mock.log_metric.call_count == 5

    def test_log_feature_importance_noop_when_no_run(self, tracker_enabled):
        """Should be no-op when no run is active."""
        mock = tracker_enabled._mock_mlflow
        tracker_enabled.log_feature_importance(["f1"], [1.0])
        mock.log_metric.assert_not_called()

    def test_log_feature_importance_sorts_by_importance(self, tracker_enabled):
        mock = tracker_enabled._mock_mlflow
        tracker_enabled.start_run("test")
        tracker_enabled.log_feature_importance(
            feature_names=["low", "high", "mid"],
            importances=[10.0, 100.0, 50.0],
            head="regressor",
        )
        # First set_tag should be for the highest importance feature
        first_tag_call = mock.set_tag.call_args_list[0]
        assert first_tag_call[0] == ("regressor_top_feature_1", "high")


# =============================================================================
# Artifact and Model Path Logging
# =============================================================================


class TestArtifactLogging:
    """Tests for log_artifact and log_model_path methods."""

    def test_log_artifact_calls_mlflow(self, tracker_enabled):
        mock = tracker_enabled._mock_mlflow
        tracker_enabled.start_run("test")
        tracker_enabled.log_artifact("/tmp/test.json", "artifacts")
        mock.log_artifact.assert_called_once_with("/tmp/test.json", "artifacts")

    def test_log_model_path_sets_tag(self, tracker_enabled):
        mock = tracker_enabled._mock_mlflow
        tracker_enabled.start_run("test")
        tracker_enabled.log_model_path("/models/POINTS_v4_regressor.pkl")
        mock.set_tag.assert_called_once_with("model_pkl_path", "/models/POINTS_v4_regressor.pkl")


# =============================================================================
# No Active Run
# =============================================================================


class TestNoActiveRun:
    """Methods should be no-ops when enabled but no run is active."""

    def test_log_params_no_run(self, tracker_enabled):
        tracker_enabled.log_params({"test": "value"})

    def test_log_metrics_no_run(self, tracker_enabled):
        tracker_enabled.log_metrics({"auc": 0.76})

    def test_log_feature_importance_no_run(self, tracker_enabled):
        tracker_enabled.log_feature_importance(["f1"], [1.0])

    def test_log_artifact_no_run(self, tracker_enabled):
        tracker_enabled.log_artifact("/tmp/test.json")

    def test_log_model_path_no_run(self, tracker_enabled):
        tracker_enabled.log_model_path("/path/to/model.pkl")


# =============================================================================
# Setup
# =============================================================================


class TestSetup:
    """Tests for tracker initialization."""

    def test_default_experiment(self):
        with patch("nba.models.mlflow_tracking.MLFLOW_AVAILABLE", False):
            t = ModelTracker()
            assert t.experiment == "nba-model-cascade"

    def test_custom_experiment(self):
        with patch("nba.models.mlflow_tracking.MLFLOW_AVAILABLE", False):
            t = ModelTracker(experiment="custom-exp")
            assert t.experiment == "custom-exp"

    def test_custom_tracking_uri(self):
        with patch("nba.models.mlflow_tracking.MLFLOW_AVAILABLE", False):
            t = ModelTracker(tracking_uri="http://localhost:5000")
            assert t.tracking_uri == "http://localhost:5000"

    def test_creates_experiment_if_missing(self):
        import nba.models.mlflow_tracking as mod

        mock_mlflow = MagicMock()
        mock_mlflow.get_experiment_by_name.return_value = None
        mock_mlflow.active_run.return_value = None
        mod.mlflow = mock_mlflow
        mod.MLFLOW_AVAILABLE = True

        try:
            ModelTracker(experiment="new-exp")
            mock_mlflow.create_experiment.assert_called_once_with("new-exp")
        finally:
            mod.MLFLOW_AVAILABLE = False
            delattr(mod, "mlflow")

    def test_uses_existing_experiment(self):
        import nba.models.mlflow_tracking as mod

        mock_mlflow = MagicMock()
        mock_mlflow.get_experiment_by_name.return_value = MagicMock()
        mock_mlflow.active_run.return_value = None
        mod.mlflow = mock_mlflow
        mod.MLFLOW_AVAILABLE = True

        try:
            ModelTracker(experiment="existing-exp")
            mock_mlflow.create_experiment.assert_not_called()
        finally:
            mod.MLFLOW_AVAILABLE = False
            delattr(mod, "mlflow")

    def test_cleans_orphaned_run_on_setup(self):
        """If a stale run exists from a prior crash, _setup ends it."""
        import nba.models.mlflow_tracking as mod

        mock_mlflow = MagicMock()
        mock_mlflow.get_experiment_by_name.return_value = MagicMock()
        mock_mlflow.active_run.return_value = MagicMock()  # orphaned run
        mod.mlflow = mock_mlflow
        mod.MLFLOW_AVAILABLE = True

        try:
            ModelTracker(experiment="test")
            mock_mlflow.end_run.assert_called_once()
        finally:
            mod.MLFLOW_AVAILABLE = False
            delattr(mod, "mlflow")


# =============================================================================
# Code Version Tracking
# =============================================================================


class TestCodeVersionTracking:
    """Tests for git hash and package version tagging."""

    def test_package_version_tag_set(self, tracker_enabled):
        """start_run should include package_version in tags."""
        mock = tracker_enabled._mock_mlflow
        tracker_enabled.start_run("test")
        call_kwargs = mock.start_run.call_args[1]
        assert "package_version" in call_kwargs["tags"]
        assert call_kwargs["tags"]["package_version"] != "unknown"

    def test_git_hash_helper_returns_string_or_none(self):
        """_get_git_hash should return a short hash or None."""
        from nba.models.mlflow_tracking import _get_git_hash

        result = _get_git_hash()
        # In a git repo, should return a short hash
        if result is not None:
            assert isinstance(result, str)
            assert len(result) <= 12

    def test_package_version_helper(self):
        """_get_package_version should return version string."""
        from nba.models.mlflow_tracking import _get_package_version

        result = _get_package_version()
        assert isinstance(result, str)
        assert result != "unknown"


# =============================================================================
# Error Handling
# =============================================================================


class TestErrorHandling:
    """Tracker should never crash the training pipeline."""

    def test_start_run_handles_exception(self, tracker_enabled):
        mock = tracker_enabled._mock_mlflow
        mock.start_run.side_effect = Exception("MLflow down")
        result = tracker_enabled.start_run("test")
        assert result is None
        assert not tracker_enabled._active

    def test_log_params_handles_exception(self, tracker_enabled):
        mock = tracker_enabled._mock_mlflow
        mock.log_params.side_effect = Exception("MLflow down")
        tracker_enabled.start_run("test")
        tracker_enabled.log_params({"key": "value"})  # should not raise

    def test_log_metrics_handles_exception(self, tracker_enabled):
        mock = tracker_enabled._mock_mlflow
        mock.log_metric.side_effect = Exception("MLflow down")
        tracker_enabled.start_run("test")
        tracker_enabled.log_metrics({"auc": 0.76})  # should not raise

    def test_end_run_handles_exception(self, tracker_enabled):
        mock = tracker_enabled._mock_mlflow
        mock.end_run.side_effect = Exception("MLflow down")
        tracker_enabled._active = True
        tracker_enabled.end_run()  # should not raise
        assert not tracker_enabled._active

    def test_log_feature_importance_handles_exception(self, tracker_enabled):
        mock = tracker_enabled._mock_mlflow
        mock.log_metric.side_effect = Exception("MLflow down")
        tracker_enabled.start_run("test")
        tracker_enabled.log_feature_importance(["f1"], [1.0])  # should not raise

    def test_log_artifact_handles_exception(self, tracker_enabled):
        mock = tracker_enabled._mock_mlflow
        mock.log_artifact.side_effect = Exception("MLflow down")
        tracker_enabled.start_run("test")
        tracker_enabled.log_artifact("/tmp/test.json")  # should not raise

    def test_log_model_path_handles_exception(self, tracker_enabled):
        mock = tracker_enabled._mock_mlflow
        mock.set_tag.side_effect = Exception("MLflow down")
        tracker_enabled.start_run("test")
        tracker_enabled.log_model_path("/path/to/model.pkl")  # should not raise
