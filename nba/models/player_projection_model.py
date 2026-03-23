#!/usr/bin/env python3
"""
Player Projection Model (Model 1) — LightGBM Regressor
========================================================
Predicts raw stat values (points, rebounds) using ONLY player/team features.
No line features, no book features, no BP features.

This is the first stage of a 3-model cascade that will replace the current
two-head stacked model. It produces a clean projection that feeds into
Model 2 (line-aware classifier) and Model 3 (market intelligence).

Architecture:
    Single LightGBM Regressor → predicted stat value (e.g., 25.3 points)

Usage:
    python nba/models/player_projection_model.py --market POINTS \\
        --data nba/features/datasets/projection_training_POINTS.csv

    python nba/models/player_projection_model.py --market REBOUNDS \\
        --data nba/features/datasets/projection_training_REBOUNDS.csv
"""

import argparse
import json
import logging
import pickle
import time as _time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple
from zoneinfo import ZoneInfo

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from nba.config.thresholds import TEMPORAL_DECAY_CONFIG, TRAINING_HYPERPARAMETERS
from nba.core.logging_config import get_logger, setup_logging
from nba.models.mlflow_tracking import ModelTracker

EST = ZoneInfo("America/New_York")

logger = get_logger(__name__)

# Prefixes that indicate line/book/BP features — MUST NOT be present
FORBIDDEN_FEATURE_PREFIXES = (
    "line",
    "book_",
    "bp_",
    "consensus_",
    "prop_hit_rate",
    "softest_",
    "hardest_",
    "draftkings_",
    "fanduel_",
    "betmgm_",
    "caesars_",
    "betrivers_",
)

# Metadata/target columns excluded from features
EXCLUDE_COLS = {
    "player_name",
    "player_id",
    "game_date",
    "date",
    "team_abbrev",
    "opponent_abbrev",
    "season",
    "actual_points",
    "actual_rebounds",
    "actual_assists",
    "actual_threes",
    "actual_result",
    "stat_type",
    "source",
    "prop_id",
    "opponent_team",
    "opponent",
    "player",
    "split",
    "label",
    "hit_over",
    "residual",
    "is_over",
}


class PlayerProjectionModel:
    """
    Single-head LightGBM regressor for player stat projection.

    Predicts raw stat values using only player/team features.
    No line, book, or betting features allowed.
    """

    def __init__(self, market: str):
        """
        Args:
            market: 'POINTS' or 'REBOUNDS'
        """
        self.market = market.upper()
        self.regressor = None
        self.imputer = None
        self.scaler = None
        self.feature_names = None

        self.target_map = {
            "POINTS": "actual_points",
            "REBOUNDS": "actual_rebounds",
            "ASSISTS": "actual_assists",
            "THREES": "actual_threes",
        }

    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load projection training dataset from CSV.

        Args:
            data_path: Path to the projection dataset CSV.

        Returns:
            Loaded and filtered DataFrame.
        """
        logger.info("Loading projection data", extra={"path": data_path})

        df = pd.read_csv(data_path)
        logger.info(
            "Loaded projection data",
            extra={"samples": len(df), "columns": len(df.columns)},
        )

        if "game_date" in df.columns:
            df["game_date"] = pd.to_datetime(df["game_date"])

        # Filter by stat_type if multiple are present
        if "stat_type" in df.columns and df["stat_type"].nunique() > 1:
            stat_type_val = self.market.lower()
            df = df[df["stat_type"] == stat_type_val].copy()
            logger.info(
                "Filtered by stat_type",
                extra={"market": self.market, "count": len(df)},
            )

        # Dedup per player/game
        before = len(df)
        if "player_name" in df.columns and "game_date" in df.columns:
            df = df.drop_duplicates(subset=["player_name", "game_date"], keep="first")
        after = len(df)
        if before != after:
            logger.info(
                "Dropped duplicates",
                extra={"dropped": before - after, "remaining": after},
            )

        # Verify target column exists
        target_col = self.target_map[self.market]
        if target_col not in df.columns and "actual_result" in df.columns:
            df[target_col] = df["actual_result"]

        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset")

        # Log summary
        if "game_date" in df.columns:
            logger.info(
                "Data date range",
                extra={
                    "min_date": str(df["game_date"].min()),
                    "max_date": str(df["game_date"].max()),
                },
            )
        if "season" in df.columns:
            logger.info(
                "Data seasons",
                extra={"seasons": sorted(df["season"].unique().tolist())},
            )

        logger.info(
            "Projection dataset ready",
            extra={"samples": len(df), "market": self.market},
        )

        return df

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """Prepare feature matrix, target, and metadata.

        Validates that NO line/book/BP features are present, selects numeric
        columns, and drops zero-variance features.

        Args:
            df: Raw DataFrame from load_data.

        Returns:
            Tuple of (X features, y target, metadata DataFrame).

        Raises:
            ValueError: If forbidden line/book/BP features are detected.
        """
        logger.info("Preparing projection features", extra={"market": self.market})

        # Validate: NO line/book/BP features allowed
        forbidden_found = [
            col
            for col in df.columns
            if any(col.startswith(prefix) for prefix in FORBIDDEN_FEATURE_PREFIXES)
        ]
        if forbidden_found:
            raise ValueError(
                f"Forbidden line/book/BP features detected in projection dataset: "
                f"{forbidden_found[:10]}. Model 1 must use ONLY player/team features."
            )

        # Select feature columns (exclude metadata + targets)
        feature_cols = [col for col in df.columns if col not in EXCLUDE_COLS]

        X = df[feature_cols].copy()

        # Convert booleans to int
        for col in X.columns:
            if X[col].dtype == "bool":
                X[col] = X[col].astype(int)

        # Drop non-numeric columns
        non_numeric = X.select_dtypes(exclude="number").columns.tolist()
        if non_numeric:
            logger.info(f"Dropping {len(non_numeric)} non-numeric columns: {non_numeric}")
            X = X.drop(columns=non_numeric)

        # Drop zero-variance features
        zero_var = [col for col in X.columns if X[col].std() == 0]
        if zero_var:
            logger.info(f"Dropping {len(zero_var)} zero-variance features: {zero_var[:5]}...")
            X = X.drop(columns=zero_var)

        self.feature_names = X.columns.tolist()

        # Target
        target_col = self.target_map[self.market]
        y = df[target_col].copy()

        # Metadata (for temporal weighting and evaluation)
        meta_cols = [c for c in ["player_name", "game_date"] if c in df.columns]
        metadata = df[meta_cols].copy() if meta_cols else pd.DataFrame(index=df.index)

        logger.info(
            "Projection feature summary",
            extra={
                "feature_count": len(self.feature_names),
                "target_mean": round(float(y.mean()), 2),
                "target_std": round(float(y.std()), 2),
                "samples": len(X),
            },
        )

        return X, y, metadata

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        game_dates_train: Optional[pd.Series] = None,
        game_dates_test: Optional[pd.Series] = None,
    ) -> Dict:
        """Train the LightGBM regressor with temporal decay weighting.

        Args:
            X_train: Training features.
            y_train: Training targets.
            X_test: Test features.
            y_test: Test targets.
            game_dates_train: Optional game dates for temporal decay (train).
            game_dates_test: Optional game dates for temporal decay (test).

        Returns:
            Dictionary of training metrics.
        """
        logger.info(
            "Training projection regressor",
            extra={
                "market": self.market,
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "features": len(X_train.columns),
            },
        )
        train_start = _time.monotonic()

        # MLflow tracking
        self._tracker = ModelTracker(experiment="nba-model-cascade")
        self._tracker.start_run(
            f"{self.market}_projection_{datetime.now(EST).strftime('%Y%m%d_%H%M')}",
            tags={"model_type": "projection", "market": self.market},
        )
        self._tracker.log_params(
            {
                "market": self.market,
                "model_type": "projection",
                "feature_count": len(X_train.columns),
                "train_samples": len(X_train),
                "test_samples": len(X_test),
            }
        )

        # Store feature names from input (in case prepare_features was skipped)
        if self.feature_names is None:
            self.feature_names = X_train.columns.tolist()

        # Impute missing values
        self.imputer = SimpleImputer(strategy="median")
        X_train_imp = pd.DataFrame(
            self.imputer.fit_transform(X_train),
            columns=self.imputer.get_feature_names_out(X_train.columns),
            index=X_train.index,
        )
        X_test_imp = pd.DataFrame(
            self.imputer.transform(X_test),
            columns=self.imputer.get_feature_names_out(X_test.columns),
            index=X_test.index,
        )

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train_imp),
            columns=X_train_imp.columns,
            index=X_train_imp.index,
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test_imp),
            columns=X_test_imp.columns,
            index=X_test_imp.index,
        )

        # Temporal decay weighting
        temporal_weights = None
        if TEMPORAL_DECAY_CONFIG.enabled and game_dates_train is not None:
            tau = TEMPORAL_DECAY_CONFIG.get_tau(self.market)
            max_date = game_dates_train.max()
            days_ago = (max_date - game_dates_train).dt.days.values
            temporal_weights = np.maximum(np.exp(-days_ago / tau), TEMPORAL_DECAY_CONFIG.min_weight)
            logger.info(
                "Temporal decay applied",
                extra={
                    "tau": tau,
                    "mean_weight": round(float(temporal_weights.mean()), 3),
                    "min_weight": round(float(temporal_weights.min()), 3),
                    "oldest_days": int(days_ago.max()),
                },
            )

        # Train LightGBM regressor
        hp = TRAINING_HYPERPARAMETERS
        self.regressor = LGBMRegressor(
            objective="regression",
            boosting_type="gbdt",
            num_leaves=hp.num_leaves,
            learning_rate=hp.learning_rate,
            n_estimators=hp.n_estimators,
            feature_fraction=hp.feature_fraction,
            bagging_fraction=hp.bagging_fraction,
            bagging_freq=hp.bagging_freq,
            reg_alpha=hp.lambda_l1,
            reg_lambda=hp.lambda_l2,
            min_child_samples=hp.min_child_samples,
            verbose=-1,
            random_state=hp.random_state,
        )

        self.regressor.fit(
            X_train_scaled,
            y_train,
            sample_weight=temporal_weights,
            eval_set=[(X_test_scaled, y_test)],
            eval_metric="rmse",
            callbacks=[
                lgb.early_stopping(stopping_rounds=hp.early_stopping_rounds),
                lgb.log_evaluation(period=100),
            ],
        )

        # Evaluate
        y_pred_train = self.regressor.predict(X_train_scaled)
        y_pred_test = self.regressor.predict(X_test_scaled)

        rmse_train = float(np.sqrt(mean_squared_error(y_train, y_pred_train)))
        rmse_test = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
        mae_test = float(mean_absolute_error(y_test, y_pred_test))
        r2_test = float(r2_score(y_test, y_pred_test))

        train_duration = round(_time.monotonic() - train_start, 1)

        logger.info(
            "Projection regressor metrics",
            extra={
                "rmse_train": round(rmse_train, 3),
                "rmse_test": round(rmse_test, 3),
                "mae_test": round(mae_test, 3),
                "r2_test": round(r2_test, 4),
                "n_trees": self.regressor.n_estimators_,
                "duration_s": train_duration,
            },
        )

        # Feature importance (top 20)
        importance_df = (
            pd.DataFrame(
                {
                    "feature": X_train_scaled.columns,
                    "importance": self.regressor.feature_importances_,
                }
            )
            .sort_values("importance", ascending=False)
            .head(20)
        )
        top_features = importance_df.to_dict("records")
        logger.info("Top 20 projection features", extra={"features": top_features})

        metrics = {
            "rmse_train": rmse_train,
            "rmse_test": rmse_test,
            "mae_test": mae_test,
            "r2_test": r2_test,
            "n_trees": self.regressor.n_estimators_,
            "training_duration_s": train_duration,
            "top_features": top_features,
        }

        self._tracker.log_metrics(
            {
                "rmse_train": rmse_train,
                "rmse_test": rmse_test,
                "mae_test": mae_test,
                "r2_test": r2_test,
                "n_trees": float(self.regressor.n_estimators_),
                "training_duration_s": train_duration,
            }
        )
        self._tracker.end_run()

        return metrics

    def save(self, output_dir: str, model_version: str = "projection") -> None:
        """Save model artifacts to disk.

        Args:
            output_dir: Directory to save artifacts.
            model_version: Version tag (default: 'projection').
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        prefix = f"{self.market.lower()}_{model_version}"

        with open(output_path / f"{prefix}_regressor.pkl", "wb") as f:
            pickle.dump(self.regressor, f)

        with open(output_path / f"{prefix}_imputer.pkl", "wb") as f:
            pickle.dump(self.imputer, f)

        with open(output_path / f"{prefix}_scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)

        with open(output_path / f"{prefix}_features.pkl", "wb") as f:
            pickle.dump(self.feature_names, f)

        # Metadata JSON
        metadata = {
            "market": self.market,
            "model_type": "player_projection_regressor",
            "model_version": model_version,
            "trained_date": datetime.now(EST).strftime("%Y-%m-%d %H:%M:%S"),
            "features": {
                "count": len(self.feature_names),
                "names": self.feature_names,
            },
        }

        # Attach latest metrics if available
        if hasattr(self, "_last_metrics"):
            metadata["metrics"] = self._last_metrics

        with open(output_path / f"{prefix}_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(
            "Projection model saved",
            extra={
                "output_dir": str(output_path),
                "prefix": prefix,
                "feature_count": len(self.feature_names),
            },
        )

    def load(self, output_dir: str, model_version: str = "projection") -> None:
        """Load model artifacts from disk.

        Args:
            output_dir: Directory containing saved artifacts.
            model_version: Version tag (default: 'projection').
        """
        output_path = Path(output_dir)
        prefix = f"{self.market.lower()}_{model_version}"

        with open(output_path / f"{prefix}_regressor.pkl", "rb") as f:
            self.regressor = pickle.load(f)

        with open(output_path / f"{prefix}_imputer.pkl", "rb") as f:
            self.imputer = pickle.load(f)

        with open(output_path / f"{prefix}_scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)

        with open(output_path / f"{prefix}_features.pkl", "rb") as f:
            self.feature_names = pickle.load(f)

        logger.info(
            "Projection model loaded",
            extra={
                "market": self.market,
                "feature_count": len(self.feature_names),
                "output_dir": str(output_path),
            },
        )

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Generate predictions for new data.

        Args:
            features: DataFrame with feature columns matching training features.

        Returns:
            Array of predicted stat values.
        """
        if self.regressor is None:
            raise RuntimeError("Model not trained or loaded. Call train() or load() first.")

        # Align columns to training feature order, filling missing with NaN
        X = features.reindex(columns=self.feature_names).copy()

        X_imp = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imp)

        return self.regressor.predict(X_scaled)


def main():
    parser = argparse.ArgumentParser(
        description="Train Player Projection Model (Model 1 — LightGBM Regressor)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--market",
        required=True,
        choices=["POINTS", "REBOUNDS"],
        help="Market to train",
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to projection training dataset CSV",
    )
    parser.add_argument(
        "--output",
        default="nba/models/saved_xl",
        help="Output directory for saved models (default: nba/models/saved_xl)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.3,
        help="Test set fraction (default: 0.3)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / args.data
    output_dir = project_root / args.output

    setup_logging(
        "player_projection_model",
        level=logging.DEBUG if args.debug else logging.INFO,
    )

    if not data_path.exists():
        logger.error("Dataset not found", extra={"path": str(data_path)})
        print(f"ERROR: Dataset not found: {data_path}")
        return 1

    try:
        model = PlayerProjectionModel(market=args.market)

        # Load data
        df = model.load_data(str(data_path))

        # Prepare features
        X, y, metadata = model.prepare_features(df)

        # Temporal split (no shuffle — chronological)
        X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
            X, y, metadata, test_size=args.test_size, shuffle=False
        )

        # Extract game dates for temporal weighting
        game_dates_train = meta_train["game_date"] if "game_date" in meta_train.columns else None
        game_dates_test = meta_test["game_date"] if "game_date" in meta_test.columns else None

        logger.info(
            "Train/Test split",
            extra={"train_samples": len(X_train), "test_samples": len(X_test)},
        )

        # Train
        metrics = model.train(
            X_train,
            y_train,
            X_test,
            y_test,
            game_dates_train=game_dates_train,
            game_dates_test=game_dates_test,
        )

        # Store metrics for save
        model._last_metrics = metrics

        # Print results
        print(f"\n{'=' * 60}")
        print(f"PLAYER PROJECTION MODEL — {args.market}")
        print(f"{'=' * 60}")
        print(f"  Train samples: {len(X_train):,}")
        print(f"  Test samples:  {len(X_test):,}")
        print(f"  Features:      {len(model.feature_names)}")
        print(f"  Trees:         {metrics['n_trees']}")
        print(f"  Duration:      {metrics['training_duration_s']}s")
        print(f"\n  RMSE (train):  {metrics['rmse_train']:.3f}")
        print(f"  RMSE (test):   {metrics['rmse_test']:.3f}")
        print(f"  MAE  (test):   {metrics['mae_test']:.3f}")
        print(f"  R2   (test):   {metrics['r2_test']:.4f}")
        print("\n  Top 5 features:")
        for feat in metrics["top_features"][:5]:
            print(f"    {feat['feature']:<40s} {feat['importance']}")
        print(f"{'=' * 60}\n")

        # Save
        model.save(str(output_dir))

        logger.info(
            "Training completed successfully",
            extra={"market": args.market, "output": str(output_dir)},
        )

        return 0

    except (OSError, KeyError, ValueError, TypeError) as e:
        logger.error("Training failed", extra={"error": str(e)}, exc_info=True)
        print(f"ERROR: {e}")
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
