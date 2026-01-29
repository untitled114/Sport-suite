"""
MLflow Experiment Tracking Integration
======================================
Wrapper for MLflow to track model training experiments.

Usage:
    from nba.core.experiment_tracking import ExperimentTracker

    tracker = ExperimentTracker(experiment_name="nba-props-xl")

    with tracker.start_run(run_name="POINTS_training_v2"):
        # Log parameters
        tracker.log_params({
            "market": "POINTS",
            "n_estimators": 2000,
            "learning_rate": 0.02,
        })

        # Train model...

        # Log metrics
        tracker.log_metrics({
            "auc_test": 0.767,
            "rmse_test": 6.84,
        })

        # Log model artifact
        tracker.log_model(model, "points_classifier")

Environment Setup:
    export MLFLOW_TRACKING_URI=sqlite:///mlruns.db  # Local SQLite
    # OR
    export MLFLOW_TRACKING_URI=http://localhost:5000  # MLflow server
"""

import json
import logging
import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Check if MLflow is available
try:
    import mlflow
    from mlflow.tracking import MlflowClient

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not installed. Install with: pip install mlflow")


class ExperimentTracker:
    """
    MLflow experiment tracker for NBA props models.

    Provides a simplified interface for:
    - Creating/managing experiments
    - Logging parameters, metrics, and artifacts
    - Model versioning and registration
    """

    def __init__(
        self,
        experiment_name: str = "nba-props-xl",
        tracking_uri: Optional[str] = None,
    ):
        """
        Initialize experiment tracker.

        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking server URI (defaults to env var or local)
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db")
        self.enabled = MLFLOW_AVAILABLE
        self._run = None
        self._client = None

        if self.enabled:
            self._setup_mlflow()

    def _setup_mlflow(self):
        """Configure MLflow tracking."""
        mlflow.set_tracking_uri(self.tracking_uri)

        # Create or get experiment
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            mlflow.create_experiment(
                self.experiment_name,
                tags={"project": "nba-props-ml", "team": "sports-analytics"},
            )

        mlflow.set_experiment(self.experiment_name)
        self._client = MlflowClient()
        logger.info(f"MLflow tracking URI: {self.tracking_uri}")

    @contextmanager
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        nested: bool = False,
    ):
        """
        Start a new MLflow run.

        Args:
            run_name: Name for the run (e.g., "POINTS_training_v2")
            tags: Additional tags for the run
            nested: Whether this is a nested run

        Yields:
            The active MLflow run

        Example:
            with tracker.start_run(run_name="POINTS_model"):
                tracker.log_params({"market": "POINTS"})
                tracker.log_metrics({"auc": 0.76})
        """
        if not self.enabled:
            yield None
            return

        default_tags = {
            "timestamp": datetime.now().isoformat(),
            "environment": os.getenv("ENV", "development"),
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

    def log_params(self, params: Dict[str, Any]):
        """
        Log parameters for the current run.

        Args:
            params: Dictionary of parameter names and values
        """
        if not self.enabled or not self._run:
            return

        # MLflow only accepts strings, numbers, booleans
        clean_params = {}
        for key, value in params.items():
            if isinstance(value, (str, int, float, bool)):
                clean_params[key] = value
            else:
                clean_params[key] = str(value)

        mlflow.log_params(clean_params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics for the current run.

        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number for tracking over time
        """
        if not self.enabled or not self._run:
            return

        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, float(value), step=step)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """
        Log a file or directory as an artifact.

        Args:
            local_path: Path to the local file/directory
            artifact_path: Optional subdirectory in the artifact store
        """
        if not self.enabled or not self._run:
            return

        mlflow.log_artifact(local_path, artifact_path)

    def log_model(
        self,
        model: Any,
        artifact_path: str,
        registered_model_name: Optional[str] = None,
    ):
        """
        Log a trained model.

        Args:
            model: The trained model object
            artifact_path: Path within artifacts to store the model
            registered_model_name: Optional name to register in Model Registry
        """
        if not self.enabled or not self._run:
            return

        # Determine model flavor
        model_type = type(model).__name__

        if "LGBMClassifier" in model_type or "LGBMRegressor" in model_type:
            mlflow.lightgbm.log_model(
                model,
                artifact_path,
                registered_model_name=registered_model_name,
            )
        else:
            # Fall back to sklearn
            mlflow.sklearn.log_model(
                model,
                artifact_path,
                registered_model_name=registered_model_name,
            )

    def log_feature_importance(
        self,
        feature_names: List[str],
        importances: List[float],
        top_n: int = 20,
    ):
        """
        Log feature importance as a table artifact.

        Args:
            feature_names: List of feature names
            importances: List of importance values
            top_n: Number of top features to log
        """
        if not self.enabled or not self._run:
            return

        # Create importance dict
        importance_data = sorted(
            zip(feature_names, importances),
            key=lambda x: x[1],
            reverse=True,
        )[:top_n]

        # Log as params
        for i, (name, importance) in enumerate(importance_data):
            mlflow.log_metric(f"feature_importance_{i+1}", importance)
            mlflow.set_tag(f"top_feature_{i+1}", name)

        # Also log as JSON artifact
        importance_dict = {name: float(imp) for name, imp in importance_data}
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(importance_dict, f, indent=2)
            temp_path = f.name

        try:
            mlflow.log_artifact(temp_path, "feature_importance")
        finally:
            # Clean up tempfile after MLflow copies it
            os.remove(temp_path)

    def log_training_data_info(
        self,
        train_samples: int,
        test_samples: int,
        feature_count: int,
        date_range: Optional[tuple[str, str]] = None,
    ):
        """
        Log information about the training data.

        Args:
            train_samples: Number of training samples
            test_samples: Number of test samples
            feature_count: Number of features
            date_range: Optional (start_date, end_date) tuple
        """
        if not self.enabled or not self._run:
            return

        mlflow.log_params(
            {
                "train_samples": train_samples,
                "test_samples": test_samples,
                "feature_count": feature_count,
                "train_test_ratio": f"{train_samples / (train_samples + test_samples):.2%}",
            }
        )

        if date_range:
            mlflow.log_params(
                {
                    "data_start_date": date_range[0],
                    "data_end_date": date_range[1],
                }
            )

    def get_best_run(
        self,
        metric: str = "auc_test",
        ascending: bool = False,
    ) -> Optional[dict]:
        """
        Get the best run based on a metric.

        Args:
            metric: Metric name to optimize
            ascending: If True, lower is better

        Returns:
            Dictionary with run info or None
        """
        if not self.enabled:
            return None

        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if not experiment:
            return None

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"],
            max_results=1,
        )

        if runs.empty:
            return None

        best_run = runs.iloc[0]
        return {
            "run_id": best_run.run_id,
            "run_name": best_run.get("tags.mlflow.runName", "unnamed"),
            metric: best_run.get(f"metrics.{metric}"),
            "params": {
                k.replace("params.", ""): v for k, v in best_run.items() if k.startswith("params.")
            },
        }

    def compare_runs(self, metric: str = "auc_test", top_n: int = 5) -> List[Dict]:
        """
        Compare top runs by a metric.

        Args:
            metric: Metric to compare
            top_n: Number of runs to return

        Returns:
            List of run dictionaries
        """
        if not self.enabled:
            return []

        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if not experiment:
            return []

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} DESC"],
            max_results=top_n,
        )

        results = []
        for _, run in runs.iterrows():
            results.append(
                {
                    "run_id": run.run_id,
                    "run_name": run.get("tags.mlflow.runName", "unnamed"),
                    metric: run.get(f"metrics.{metric}"),
                    "start_time": run.start_time,
                }
            )

        return results


# Convenience function for quick logging without full setup
def log_training_run(
    market: str,
    metrics: Dict[str, float],
    params: Dict[str, Any],
    model_path: Optional[str] = None,
    experiment_name: str = "nba-props-xl",
):
    """
    Quick function to log a complete training run.

    Args:
        market: Market type (POINTS, REBOUNDS, etc.)
        metrics: Dictionary of training metrics
        params: Dictionary of training parameters
        model_path: Optional path to saved model files
        experiment_name: MLflow experiment name
    """
    tracker = ExperimentTracker(experiment_name=experiment_name)

    run_name = f"{market}_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with tracker.start_run(run_name=run_name, tags={"market": market}):
        tracker.log_params({"market": market, **params})
        tracker.log_metrics(metrics)

        if model_path and Path(model_path).exists():
            tracker.log_artifact(model_path, "model_artifacts")

    logger.info(f"Logged training run: {run_name}")
