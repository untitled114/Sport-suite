"""
Tests for ProjectionModelTracker (MLflow tracking)
====================================================
Tests the MLflow tracking wrapper for the projection model.
All tests work with MLflow disabled (graceful no-op behavior).
Since mlflow is not installed in test env, enabled-mode tests
inject a mock mlflow module directly.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

from nba.models.mlflow_tracking import ProjectionModelTracker


@pytest.fixture
def tracker_disabled():
    """Tracker with MLflow disabled (no mlflow installed)."""
    with patch("nba.models.mlflow_tracking.MLFLOW_AVAILABLE", False):
        t = ProjectionModelTracker(experiment_name="test-experiment")
        t.enabled = False
        return t


@pytest.fixture
def tracker_enabled():
    """Tracker with mocked MLflow injected into module namespace."""
    mock_mlflow = MagicMock()
    mock_mlflow.get_experiment_by_name.return_value = MagicMock()

    import nba.models.mlflow_tracking as mod

    # Inject mock mlflow into module
    original_available = mod.MLFLOW_AVAILABLE
    mod.MLFLOW_AVAILABLE = True
    mod.mlflow = mock_mlflow

    t = ProjectionModelTracker.__new__(ProjectionModelTracker)
    t.experiment_name = "test-experiment"
    t.tracking_uri = "sqlite:///mlruns.db"
    t.enabled = True
    t._run = None
    t._mock_mlflow = mock_mlflow

    yield t

    # Cleanup
    mod.MLFLOW_AVAILABLE = original_available
    if hasattr(mod, "mlflow") and mod.mlflow is mock_mlflow:
        delattr(mod, "mlflow")


# =============================================================================
# Disabled Mode (Graceful No-ops)
# =============================================================================


class TestDisabledMode:
    """When MLflow is not installed, all methods should be no-ops."""

    def test_start_run_yields_none(self, tracker_disabled):
        with tracker_disabled.start_run(run_name="test") as run:
            assert run is None

    def test_log_params_noop(self, tracker_disabled):
        tracker_disabled.log_params({"market": "POINTS"})

    def test_log_metrics_noop(self, tracker_disabled):
        tracker_disabled.log_metrics({"auc": 0.76})

    def test_log_projection_config_noop(self, tracker_disabled):
        tracker_disabled.log_projection_config(
            market="POINTS",
            rolling_weights={"L5": 0.5, "L10": 0.3, "L20": 0.2},
            home_advantage=0.015,
            league_avg_pace=100.0,
        )

    def test_log_projection_metrics_noop(self, tracker_disabled):
        tracker_disabled.log_projection_metrics(mae=4.5, rmse=6.0, r2=0.45)

    def test_log_walk_forward_fold_noop(self, tracker_disabled):
        tracker_disabled.log_walk_forward_fold(
            fold_num=1,
            train_start="2023-10-24",
            train_end="2024-06-01",
            test_start="2024-06-01",
            test_end="2024-08-01",
            metrics={"auc": 0.72},
        )

    def test_log_comparison_noop(self, tracker_disabled):
        tracker_disabled.log_comparison(
            lgbm_metrics={"auc": 0.76},
            projection_metrics={"auc": 0.72},
        )

    def test_log_comparison_artifact_noop(self, tracker_disabled):
        tracker_disabled.log_comparison_artifact({"model": "test"})


# =============================================================================
# Enabled Mode (With Mocked MLflow)
# =============================================================================


class TestEnabledMode:
    """When MLflow is available, methods should call MLflow APIs."""

    def test_start_run_calls_mlflow(self, tracker_enabled):
        mock = tracker_enabled._mock_mlflow
        mock.start_run.return_value = MagicMock()

        with tracker_enabled.start_run(run_name="test_run") as run:
            assert run is not None

        mock.start_run.assert_called_once()
        mock.end_run.assert_called_once()

    def test_start_run_with_tags(self, tracker_enabled):
        mock = tracker_enabled._mock_mlflow
        mock.start_run.return_value = MagicMock()

        with tracker_enabled.start_run(run_name="test", tags={"env": "test"}):
            pass

        call_kwargs = mock.start_run.call_args
        tags = call_kwargs[1]["tags"] if "tags" in call_kwargs[1] else call_kwargs[0][1]
        assert "env" in tags

    def test_log_params_calls_mlflow(self, tracker_enabled):
        mock = tracker_enabled._mock_mlflow
        mock.start_run.return_value = MagicMock()

        with tracker_enabled.start_run(run_name="test"):
            tracker_enabled.log_params({"market": "POINTS", "features": 102})

        mock.log_params.assert_called_once()

    def test_log_params_converts_non_primitives(self, tracker_enabled):
        mock = tracker_enabled._mock_mlflow
        mock.start_run.return_value = MagicMock()

        with tracker_enabled.start_run(run_name="test"):
            tracker_enabled.log_params({"list_param": [1, 2, 3]})

        call_args = mock.log_params.call_args[0][0]
        assert call_args["list_param"] == "[1, 2, 3]"

    def test_log_metrics_calls_mlflow(self, tracker_enabled):
        mock = tracker_enabled._mock_mlflow
        mock.start_run.return_value = MagicMock()

        with tracker_enabled.start_run(run_name="test"):
            tracker_enabled.log_metrics({"auc": 0.76, "rmse": 6.0})

        assert mock.log_metric.call_count == 2

    def test_log_metrics_skips_non_numeric(self, tracker_enabled):
        mock = tracker_enabled._mock_mlflow
        mock.start_run.return_value = MagicMock()

        with tracker_enabled.start_run(run_name="test"):
            tracker_enabled.log_metrics({"auc": 0.76, "name": "test"})

        assert mock.log_metric.call_count == 1

    def test_log_metrics_with_step(self, tracker_enabled):
        mock = tracker_enabled._mock_mlflow
        mock.start_run.return_value = MagicMock()

        with tracker_enabled.start_run(run_name="test"):
            tracker_enabled.log_metrics({"auc": 0.76}, step=1)

        mock.log_metric.assert_called_with("auc", 0.76, step=1)

    def test_log_projection_config(self, tracker_enabled):
        mock = tracker_enabled._mock_mlflow
        mock.start_run.return_value = MagicMock()

        with tracker_enabled.start_run(run_name="test"):
            tracker_enabled.log_projection_config(
                market="POINTS",
                rolling_weights={"L5": 0.5, "L10": 0.3, "L20": 0.2},
                home_advantage=0.015,
                league_avg_pace=100.0,
            )

        mock.log_params.assert_called_once()
        params = mock.log_params.call_args[0][0]
        assert params["market"] == "POINTS"
        assert params["model_type"] == "pace_adjusted_projection"
        assert params["weight_L5"] == 0.5

    def test_log_projection_metrics(self, tracker_enabled):
        mock = tracker_enabled._mock_mlflow
        mock.start_run.return_value = MagicMock()

        with tracker_enabled.start_run(run_name="test"):
            tracker_enabled.log_projection_metrics(
                mae=4.5,
                rmse=6.0,
                r2=0.45,
                auc=0.72,
                brier_score=0.23,
                win_rate=0.58,
                roi=0.06,
            )

        assert mock.log_metric.call_count == 7

    def test_log_projection_metrics_optional_fields(self, tracker_enabled):
        mock = tracker_enabled._mock_mlflow
        mock.start_run.return_value = MagicMock()

        with tracker_enabled.start_run(run_name="test"):
            tracker_enabled.log_projection_metrics(mae=4.5, rmse=6.0, r2=0.45)

        assert mock.log_metric.call_count == 3

    def test_log_walk_forward_fold(self, tracker_enabled):
        mock = tracker_enabled._mock_mlflow
        mock.start_run.return_value = MagicMock()

        with tracker_enabled.start_run(run_name="test"):
            tracker_enabled.log_walk_forward_fold(
                fold_num=1,
                train_start="2023-10-24",
                train_end="2024-06-01",
                test_start="2024-06-01",
                test_end="2024-08-01",
                metrics={"auc": 0.72, "roi": 0.05},
            )

        assert mock.set_tag.call_count == 2
        assert mock.log_metric.call_count == 2

    def test_log_comparison(self, tracker_enabled):
        mock = tracker_enabled._mock_mlflow
        mock.start_run.return_value = MagicMock()

        with tracker_enabled.start_run(run_name="test"):
            tracker_enabled.log_comparison(
                lgbm_metrics={"auc": 0.76, "rmse": 6.0},
                projection_metrics={"auc": 0.72, "rmse": 5.5},
            )

        # 2 lgbm + 2 proj + 2 deltas = 6
        assert mock.log_metric.call_count == 6

    def test_log_comparison_artifact(self, tracker_enabled):
        mock = tracker_enabled._mock_mlflow
        mock.start_run.return_value = MagicMock()

        with tracker_enabled.start_run(run_name="test"):
            tracker_enabled.log_comparison_artifact(
                {"model_a": "lgbm", "model_b": "projection", "winner": "lgbm"}
            )

        mock.log_artifact.assert_called_once()


# =============================================================================
# No Active Run
# =============================================================================


class TestNoActiveRun:
    """Methods should be no-ops when enabled but no run is active."""

    def test_log_params_no_run(self, tracker_enabled):
        # _run is None, so all methods should silently return
        tracker_enabled.log_params({"test": "value"})

    def test_log_metrics_no_run(self, tracker_enabled):
        tracker_enabled.log_metrics({"auc": 0.76})

    def test_log_projection_config_no_run(self, tracker_enabled):
        tracker_enabled.log_projection_config(
            market="POINTS",
            rolling_weights={"L5": 0.5},
            home_advantage=0.015,
            league_avg_pace=100.0,
        )

    def test_log_projection_metrics_no_run(self, tracker_enabled):
        tracker_enabled.log_projection_metrics(mae=4.5, rmse=6.0, r2=0.45)

    def test_log_walk_forward_fold_no_run(self, tracker_enabled):
        tracker_enabled.log_walk_forward_fold(
            fold_num=1,
            train_start="2023-10-24",
            train_end="2024-06-01",
            test_start="2024-06-01",
            test_end="2024-08-01",
            metrics={"auc": 0.72},
        )

    def test_log_comparison_no_run(self, tracker_enabled):
        tracker_enabled.log_comparison(
            lgbm_metrics={"auc": 0.76},
            projection_metrics={"auc": 0.72},
        )

    def test_log_comparison_artifact_no_run(self, tracker_enabled):
        tracker_enabled.log_comparison_artifact({"data": "test"})


# =============================================================================
# Setup
# =============================================================================


class TestSetup:
    """Tests for tracker initialization."""

    def test_default_experiment_name(self):
        with patch("nba.models.mlflow_tracking.MLFLOW_AVAILABLE", False):
            t = ProjectionModelTracker()
            assert t.experiment_name == "nba-projection-model"

    def test_custom_experiment_name(self):
        with patch("nba.models.mlflow_tracking.MLFLOW_AVAILABLE", False):
            t = ProjectionModelTracker(experiment_name="custom-exp")
            assert t.experiment_name == "custom-exp"

    def test_custom_tracking_uri(self):
        with patch("nba.models.mlflow_tracking.MLFLOW_AVAILABLE", False):
            t = ProjectionModelTracker(tracking_uri="http://localhost:5000")
            assert t.tracking_uri == "http://localhost:5000"

    def test_creates_experiment_if_missing(self):
        import nba.models.mlflow_tracking as mod

        mock_mlflow = MagicMock()
        mock_mlflow.get_experiment_by_name.return_value = None
        mod.mlflow = mock_mlflow
        mod.MLFLOW_AVAILABLE = True

        try:
            ProjectionModelTracker(experiment_name="new-exp")
            mock_mlflow.create_experiment.assert_called_once()
        finally:
            mod.MLFLOW_AVAILABLE = False
            delattr(mod, "mlflow")

    def test_uses_existing_experiment(self):
        import nba.models.mlflow_tracking as mod

        mock_mlflow = MagicMock()
        mock_mlflow.get_experiment_by_name.return_value = MagicMock()
        mod.mlflow = mock_mlflow
        mod.MLFLOW_AVAILABLE = True

        try:
            ProjectionModelTracker(experiment_name="existing-exp")
            mock_mlflow.create_experiment.assert_not_called()
        finally:
            mod.MLFLOW_AVAILABLE = False
            delattr(mod, "mlflow")
