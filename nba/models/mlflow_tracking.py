"""
MLflow Tracking for Sport-Suite Models
=======================================
Single tracker used by all model types (stacked two-head, projection, market classifier).

Handles:
- Experiment creation and selection
- Run lifecycle (start/end with proper cleanup)
- Parameter, metric, and artifact logging
- Stale run cleanup

Usage:
    from nba.models.mlflow_tracking import ModelTracker

    tracker = ModelTracker(experiment="nba-model-cascade")
    tracker.start_run("POINTS_projection_20260323")
    tracker.log_params({"market": "POINTS", "features": 30})
    tracker.log_metrics({"rmse_test": 6.53, "r2_test": 0.39})
    tracker.end_run()

Environment:
    pip install mlflow
    Tracking URI defaults to sqlite:///mlruns.db
"""

import logging
import os
from datetime import datetime
from typing import Any, Optional
from zoneinfo import ZoneInfo

EST = ZoneInfo("America/New_York")
log = logging.getLogger("nba.mlflow_tracking")

try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class ModelTracker:
    """MLflow tracker for all Sport-Suite models.

    Not a context manager — uses explicit start_run/end_run to avoid the
    generator-based __enter__/__exit__ bug that caused orphaned runs.
    """

    def __init__(
        self,
        experiment: str = "nba-model-cascade",
        tracking_uri: Optional[str] = None,
    ):
        self.experiment = experiment
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db")
        self.enabled = MLFLOW_AVAILABLE
        self._active = False

        if self.enabled:
            self._setup()

    def _setup(self):
        """Configure MLflow, cleaning up any stale runs."""
        mlflow.set_tracking_uri(self.tracking_uri)

        # End any orphaned active run from a prior crash
        if mlflow.active_run():
            log.warning("Ending orphaned MLflow run: %s", mlflow.active_run().info.run_id)
            mlflow.end_run()

        if not mlflow.get_experiment_by_name(self.experiment):
            mlflow.create_experiment(self.experiment)
        mlflow.set_experiment(self.experiment)

    def start_run(self, run_name: str, tags: Optional[dict[str, str]] = None):
        """Start a new MLflow run. Safe to call even if a prior run leaked."""
        if not self.enabled:
            return

        # Safety: end any lingering run
        if mlflow.active_run():
            mlflow.end_run()

        all_tags = {"timestamp": datetime.now(EST).isoformat()}
        if tags:
            all_tags.update(tags)

        try:
            mlflow.start_run(run_name=run_name, tags=all_tags)
            self._active = True
        except Exception as e:
            log.warning("MLflow start_run failed: %s", e)
            self._active = False

    def end_run(self):
        """End the active MLflow run."""
        if not self.enabled or not self._active:
            return
        try:
            mlflow.end_run()
        except Exception as e:
            log.warning("MLflow end_run failed: %s", e)
        finally:
            self._active = False

    def log_params(self, params: dict[str, Any]):
        """Log parameters. Converts non-primitive values to strings."""
        if not self.enabled or not self._active:
            return
        clean = {}
        for k, v in params.items():
            if isinstance(v, (str, int, float, bool)):
                clean[k] = v
            else:
                clean[k] = str(v)
        try:
            mlflow.log_params(clean)
        except Exception as e:
            log.debug("MLflow log_params failed: %s", e)

    def log_metrics(self, metrics: dict[str, float]):
        """Log metrics. Skips non-numeric values."""
        if not self.enabled or not self._active:
            return
        try:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, float(v))
        except Exception as e:
            log.debug("MLflow log_metrics failed: %s", e)


# Backwards compatibility — old code imports ProjectionModelTracker
ProjectionModelTracker = ModelTracker
