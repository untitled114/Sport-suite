"""
MLflow Tracking for Sport-Suite Models
=======================================
Single tracker used by all model types (stacked two-head, projection, market classifier).

Handles:
- Experiment creation and selection
- Run lifecycle (start/end with proper cleanup)
- Parameter, metric, and artifact logging
- Feature importance logging
- Code version tracking (git hash + package version)
- Stale run cleanup

Usage:
    from nba.models.mlflow_tracking import ModelTracker

    tracker = ModelTracker()
    run_id = tracker.start_run("POINTS_projection_20260323_143022")
    tracker.log_params({"market": "POINTS", "features": 30})
    tracker.log_metrics({"rmse_test": 6.53, "r2_test": 0.39})
    tracker.log_feature_importance(feature_names, importances, head="regressor")
    tracker.end_run()

Environment:
    pip install mlflow
    export MLFLOW_TRACKING_URI=sqlite:///path/to/mlruns.db
"""

import json
import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional
from zoneinfo import ZoneInfo

EST = ZoneInfo("America/New_York")
log = logging.getLogger("nba.mlflow_tracking")

try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


def _get_git_hash() -> Optional[str]:
    """Get current git commit hash, or None if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).parent.parent.parent,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _get_package_version() -> str:
    """Get nba package version."""
    try:
        from nba import __version__

        return __version__
    except Exception:
        return "unknown"


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
        self._run_id: Optional[str] = None

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

    def start_run(self, run_name: str, tags: Optional[dict[str, str]] = None) -> Optional[str]:
        """Start a new MLflow run. Returns run_id or None.

        Safe to call even if a prior run leaked.
        """
        if not self.enabled:
            return None

        # Safety: end any lingering run
        if mlflow.active_run():
            mlflow.end_run()

        all_tags = {"timestamp": datetime.now(EST).isoformat()}

        # Code version tracking
        git_hash = _get_git_hash()
        if git_hash:
            all_tags["git_hash"] = git_hash
        all_tags["package_version"] = _get_package_version()

        if tags:
            all_tags.update(tags)

        try:
            run = mlflow.start_run(run_name=run_name, tags=all_tags)
            self._active = True
            self._run_id = run.info.run_id
            return self._run_id
        except Exception as e:
            log.warning("MLflow start_run failed: %s", e)
            self._active = False
            self._run_id = None
            return None

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

    @property
    def run_id(self) -> Optional[str]:
        """Current run ID, or None if no run is active."""
        return self._run_id if self._active else None

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
            log.warning("MLflow log_params failed: %s", e)

    def log_metrics(self, metrics: dict[str, float]):
        """Log metrics. Skips non-numeric values."""
        if not self.enabled or not self._active:
            return
        try:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, float(v))
        except Exception as e:
            log.warning("MLflow log_metrics failed: %s", e)

    def log_feature_importance(
        self,
        feature_names: List[str],
        importances: List[float],
        head: str = "regressor",
        top_n: int = 20,
    ):
        """Log feature importance as metrics + JSON artifact.

        Args:
            feature_names: Feature names from the model.
            importances: Corresponding importance values.
            head: Which model head ("regressor" or "classifier").
            top_n: Number of top features to log individually.
        """
        if not self.enabled or not self._active:
            return

        importance_data = sorted(
            zip(feature_names, importances),
            key=lambda x: x[1],
            reverse=True,
        )[:top_n]

        try:
            for i, (name, imp) in enumerate(importance_data):
                mlflow.log_metric(f"{head}_feature_importance_{i + 1}", float(imp))
                mlflow.set_tag(f"{head}_top_feature_{i + 1}", name)

            # Log full importance as named JSON artifact
            importance_dict = {name: float(imp) for name, imp in importance_data}
            artifact_path = Path(f"{head}_feature_importance.json")
            artifact_path.write_text(json.dumps(importance_dict, indent=2))
            try:
                mlflow.log_artifact(str(artifact_path), "feature_importance")
            finally:
                artifact_path.unlink(missing_ok=True)
        except Exception as e:
            log.warning("MLflow log_feature_importance failed: %s", e)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log a file as an artifact.

        Args:
            local_path: Path to the local file.
            artifact_path: Optional subdirectory in the artifact store.
        """
        if not self.enabled or not self._active:
            return
        try:
            mlflow.log_artifact(local_path, artifact_path)
        except Exception as e:
            log.warning("MLflow log_artifact failed: %s", e)

    def log_model_path(self, pkl_path: str):
        """Log the path to the saved model pkl for lineage tracking.

        Args:
            pkl_path: Path to the saved .pkl file.
        """
        if not self.enabled or not self._active:
            return
        try:
            mlflow.set_tag("model_pkl_path", pkl_path)
        except Exception as e:
            log.warning("MLflow log_model_path failed: %s", e)


# Backwards compatibility — old code imports ProjectionModelTracker
ProjectionModelTracker = ModelTracker
