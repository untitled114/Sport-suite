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

    import nba.models.mlflow_tracking as mod

    original_available = mod.MLFLOW_AVAILABLE
    mod.MLFLOW_AVAILABLE = True
    mod.mlflow = mock_mlflow

    t = ModelTracker.__new__(ModelTracker)
    t.experiment = "test-experiment"
    t.tracking_uri = "sqlite:///mlruns.db"
    t.enabled = True
    t._active = False
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
        tracker_disabled.start_run("test")
        assert not tracker_disabled._active

    def test_end_run_noop(self, tracker_disabled):
        tracker_disabled.end_run()

    def test_log_params_noop(self, tracker_disabled):
        tracker_disabled.log_params({"market": "POINTS"})

    def test_log_metrics_noop(self, tracker_disabled):
        tracker_disabled.log_metrics({"auc": 0.76})


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

    def test_start_run_with_tags(self, tracker_enabled):
        mock = tracker_enabled._mock_mlflow
        tracker_enabled.start_run("test", tags={"env": "test"})
        call_kwargs = mock.start_run.call_args[1]
        assert "env" in call_kwargs["tags"]
        assert "timestamp" in call_kwargs["tags"]

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
# No Active Run
# =============================================================================


class TestNoActiveRun:
    """Methods should be no-ops when enabled but no run is active."""

    def test_log_params_no_run(self, tracker_enabled):
        tracker_enabled.log_params({"test": "value"})

    def test_log_metrics_no_run(self, tracker_enabled):
        tracker_enabled.log_metrics({"auc": 0.76})


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
# Error Handling
# =============================================================================


class TestErrorHandling:
    """Tracker should never crash the training pipeline."""

    def test_start_run_handles_exception(self, tracker_enabled):
        mock = tracker_enabled._mock_mlflow
        mock.start_run.side_effect = Exception("MLflow down")
        tracker_enabled.start_run("test")
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
