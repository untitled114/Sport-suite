"""
MLflow Tracking for Projection Model
======================================
Extends the existing ExperimentTracker with projection-model-specific
tracking capabilities.

Wraps training runs for both the existing two-head LightGBM model and
the new projection + distribution model, enabling side-by-side comparison
in the MLflow UI.

Usage:
    from nba.models.mlflow_tracking import ProjectionModelTracker

    tracker = ProjectionModelTracker()

    with tracker.start_run(run_name="projection_POINTS_v1"):
        tracker.log_projection_config(...)
        tracker.log_projection_metrics(...)
        tracker.log_comparison(lgbm_metrics, projection_metrics)

Environment:
    pip install mlflow  # or: pip install -e ".[mlops]"
    export MLFLOW_TRACKING_URI=sqlite:///mlruns.db
"""

import json
import logging
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

EST = ZoneInfo("America/New_York")

logger = logging.getLogger(__name__)

try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class ProjectionModelTracker:
    """
    MLflow tracker specialized for projection model experiments.

    Builds on top of ExperimentTracker but adds:
    - Projection-specific parameter logging
    - Distribution fit quality metrics
    - Side-by-side comparison with LightGBM models
    - Walk-forward fold tracking
    """

    def __init__(
        self,
        experiment_name: str = "nba-projection-model",
        tracking_uri: Optional[str] = None,
    ):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db")
        self.enabled = MLFLOW_AVAILABLE
        self._run = None

        if self.enabled:
            self._setup()

    def _setup(self):
        """Configure MLflow."""
        mlflow.set_tracking_uri(self.tracking_uri)
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            mlflow.create_experiment(
                self.experiment_name,
                tags={"project": "nba-props-ml", "model_type": "projection"},
            )
        mlflow.set_experiment(self.experiment_name)

    @contextmanager
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        nested: bool = False,
    ):
        """Start a new MLflow run."""
        if not self.enabled:
            yield None
            return

        default_tags = {
            "timestamp": datetime.now(EST).isoformat(),
            "model_type": "projection",
        }
        if tags:
            default_tags.update(tags)

        try:
            self._run = mlflow.start_run(run_name=run_name, tags=default_tags, nested=nested)
            yield self._run
        finally:
            if self._run:
                mlflow.end_run()
                self._run = None

    def log_projection_config(
        self,
        market: str,
        rolling_weights: Dict[str, float],
        home_advantage: float,
        league_avg_pace: float,
    ):
        """Log projection model configuration."""
        if not self.enabled or not self._run:
            return

        params = {
            "market": market,
            "model_type": "pace_adjusted_projection",
            "league_avg_pace": league_avg_pace,
            "home_advantage": home_advantage,
        }
        for window, weight in rolling_weights.items():
            params[f"weight_{window}"] = weight

        mlflow.log_params(params)

    def log_projection_metrics(
        self,
        mae: float,
        rmse: float,
        r2: float,
        auc: Optional[float] = None,
        brier_score: Optional[float] = None,
        win_rate: Optional[float] = None,
        roi: Optional[float] = None,
        step: Optional[int] = None,
    ):
        """Log projection model performance metrics."""
        if not self.enabled or not self._run:
            return

        metrics = {"mae": mae, "rmse": rmse, "r2": r2}
        if auc is not None:
            metrics["auc"] = auc
        if brier_score is not None:
            metrics["brier_score"] = brier_score
        if win_rate is not None:
            metrics["win_rate"] = win_rate
        if roi is not None:
            metrics["roi"] = roi

        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)

    def log_walk_forward_fold(
        self,
        fold_num: int,
        train_start: str,
        train_end: str,
        test_start: str,
        test_end: str,
        metrics: Dict[str, float],
    ):
        """Log metrics for a single walk-forward fold."""
        if not self.enabled or not self._run:
            return

        # Log fold boundaries as params
        mlflow.set_tag(f"fold_{fold_num}_train", f"{train_start} to {train_end}")
        mlflow.set_tag(f"fold_{fold_num}_test", f"{test_start} to {test_end}")

        # Log fold metrics
        for key, value in metrics.items():
            mlflow.log_metric(f"fold_{fold_num}_{key}", value)

    def log_comparison(
        self,
        lgbm_metrics: Dict[str, float],
        projection_metrics: Dict[str, float],
    ):
        """
        Log side-by-side comparison between LightGBM and projection model.

        Args:
            lgbm_metrics: Metrics from the existing two-head model
            projection_metrics: Metrics from the new projection model
        """
        if not self.enabled or not self._run:
            return

        for key, value in lgbm_metrics.items():
            mlflow.log_metric(f"lgbm_{key}", value)
        for key, value in projection_metrics.items():
            mlflow.log_metric(f"proj_{key}", value)

        # Log deltas
        common_keys = set(lgbm_metrics.keys()) & set(projection_metrics.keys())
        for key in common_keys:
            delta = projection_metrics[key] - lgbm_metrics[key]
            mlflow.log_metric(f"delta_{key}", delta)

    def log_comparison_artifact(
        self,
        comparison_data: Dict[str, Any],
        filename: str = "model_comparison.json",
    ):
        """Save comparison data as a JSON artifact."""
        if not self.enabled or not self._run:
            return

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(comparison_data, f, indent=2, default=str)
            temp_path = f.name

        try:
            mlflow.log_artifact(temp_path, "comparison")
        finally:
            os.remove(temp_path)

    def log_params(self, params: Dict[str, Any]):
        """Log parameters."""
        if not self.enabled or not self._run:
            return

        clean = {}
        for k, v in params.items():
            clean[k] = v if isinstance(v, (str, int, float, bool)) else str(v)
        mlflow.log_params(clean)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics."""
        if not self.enabled or not self._run:
            return

        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, float(v), step=step)
