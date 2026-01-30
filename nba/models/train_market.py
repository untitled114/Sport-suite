#!/usr/bin/env python3
"""
NBA Market-Aware Model Trainer - Stacked Two-Head Architecture
===============================================================
Trains stacked models that predict BOTH absolute value AND beat-the-line probability

ARCHITECTURE:
1. HEAD 1 (Regressor): Predict actual stat value
   - Input: 73 features (68 base + line + 4 book encoding)
   - Output: Predicted value (e.g., 25.3 points)
   - Model: LightGBM Regressor

2. HEAD 2 (Classifier): Predict P(actual > line)
   - Input: 74 features (73 from Head 1 + expected_diff from regressor)
   - Output: Probability of OVER winning (0.0 to 1.0)
   - Model: LightGBM Classifier
   - Training: Binary label (1 if actual > line, 0 otherwise)

CRITICAL FEATURES:
- Sportsbook line is a FEATURE (markets are efficient, line is informative)
- Expected diff = regressor_prediction - line (key signal for classifier)
- Book encoding (some books have better lines than others)

Usage:
    python nba/models/train_market.py --market POINTS
    python nba/models/train_market.py --market REBOUNDS --data custom_dataset.csv
    python nba/models/train_market.py --market THREES --test-size 0.3
"""

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy.special import expit
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from nba.core.logging_config import get_logger, setup_logging

# MLflow integration
try:
    from nba.core.experiment_tracking import ExperimentTracker

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Logger will be configured in main()
logger = get_logger(__name__)


# Null context manager for when MLflow is disabled
from contextlib import contextmanager


@contextmanager
def _null_context():
    """Null context manager that does nothing."""
    yield None


class StackedMarketModel:
    """
    Stacked two-head model:
    - Head 1: Regressor (predict absolute value)
    - Head 2: Classifier (predict P(actual > line) using regressor output)
    """

    def __init__(self, market: str):
        """
        Args:
            market: 'POINTS', 'REBOUNDS', 'ASSISTS', or 'THREES'
        """
        self.market = market.upper()
        self.regressor = None
        self.classifier = None
        self.imputer = None
        self.scaler = None
        self.feature_names = None
        self.calibrator = None
        self.blend_config = {}

        # Target column mapping
        self.target_map = {
            "POINTS": "actual_points",
            "REBOUNDS": "actual_rebounds",
            "ASSISTS": "actual_assists",
            "THREES": "actual_threes",
        }

        # Stat type filtering
        self.stat_type_map = {
            "POINTS": "points",
            "REBOUNDS": "rebounds",
            "ASSISTS": "assists",
            "THREES": ["threes", "threePointersMade"],
        }

    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load market-aware training dataset"""
        logger.info("Loading training data", extra={"path": data_path})

        df = pd.read_csv(data_path)
        logger.info("Loaded training data", extra={"samples": len(df), "columns": len(df.columns)})

        # Convert game_date
        if "game_date" in df.columns:
            df["game_date"] = pd.to_datetime(df["game_date"])

        # Filter by stat_type (skip if data already filtered - single stat_type present)
        if "stat_type" in df.columns and df["stat_type"].nunique() > 1:
            stat_types = self.stat_type_map[self.market]
            if isinstance(stat_types, list):
                df = df[df["stat_type"].isin(stat_types)].copy()
            else:
                df = df[df["stat_type"] == stat_types].copy()
            logger.info("Filtered by stat_type", extra={"market": self.market, "count": len(df)})
        else:
            logger.info("Data already filtered", extra={"market": self.market, "count": len(df)})

        # Drop duplicates (keep first occurrence per player/game)
        before_dedup = len(df)
        df = df.drop_duplicates(subset=["player_name", "game_date"], keep="first")
        after_dedup = len(df)
        if before_dedup != after_dedup:
            logger.info(
                "Dropped duplicate props",
                extra={"dropped": before_dedup - after_dedup, "remaining": after_dedup},
            )

        # Handle column compatibility (support both old and new extractor formats)
        target_col = self.target_map[self.market]  # e.g., 'actual_assists'

        # If stat-specific column missing, create from actual_result
        if target_col not in df.columns and "actual_result" in df.columns:
            logger.debug(
                "Creating target column from actual_result", extra={"target_col": target_col}
            )
            df[target_col] = df["actual_result"]

        # If source missing, add default
        if "source" not in df.columns:
            logger.debug("Adding default source", extra={"source": "bettingpros"})
            df["source"] = "bettingpros"

        # Verify required columns
        required = ["line", "source", target_col]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Log date range if available (optional for historical datasets)
        if "game_date" in df.columns:
            logger.info(
                "Data date range",
                extra={
                    "min_date": str(df["game_date"].min()),
                    "max_date": str(df["game_date"].max()),
                },
            )
        elif "season" in df.columns:
            logger.info("Data seasons", extra={"seasons": sorted(df["season"].unique())})

        logger.info("Data sources", extra={"sources": df["source"].value_counts().to_dict()})

        return df

    def prepare_features(self, df: pd.DataFrame):
        """
        Prepare features for training

        Returns:
            X: Feature matrix (68 base + line + 4 book encoding = 73 features)
            y_value: Actual stat value (for regressor)
            y_binary: Binary label (1 if actual > line, 0 otherwise, for classifier)
            metadata: player_name, game_date, line, actual
        """
        logger.info("Preparing features", extra={"market": self.market})

        # Exclude metadata and targets
        exclude_cols = [
            "player_id",
            "player_name",
            "game_date",
            "date",
            "player",
            "team_abbrev",
            "opponent_abbrev",
            "actual_points",
            "actual_rebounds",
            "actual_assists",
            "actual_threes",
            "actual_result",
            "season",
            "opponent",
            "stat_type",
            "source",
            "prop_id",
            "opponent_team",  # Metadata (categorical, not useful as feature)
            # NOTE: 'is_home' is NOW INCLUDED as a feature (provides home/away performance signal)
            "split",
            "label",  # Train/val split indicator and binary label (targets, must exclude!)
            "hit_over",
            "residual",  # CRITICAL: Binary target + residual - exclude to prevent data leakage
        ]

        # Select features (MUST include 'line' and book encoding)
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        # Verify critical features
        assert "line" in feature_cols, "FATAL: 'line' feature missing"

        # Book encoding optional (production data may not have it)
        book_features = [col for col in feature_cols if col.startswith("book_")]
        if len(book_features) == 0:
            logger.warning(
                "No book encoding features found - training without market-aware features"
            )

        logger.info(
            "Feature summary",
            extra={"total_features": len(feature_cols), "book_features": book_features},
        )

        X = df[feature_cols].copy()
        self.feature_names = feature_cols

        # Target value (for regressor)
        target_col = self.target_map[self.market]
        y_value = df[target_col].copy()

        # Calculate residual (attempt1.md Step 1)
        df["residual"] = df[target_col] - df["line"]
        y_residual = df["residual"].copy()

        # Binary label (for classifier) - RESIDUAL-BASED
        y_binary = (df["residual"] > 0).astype(int)

        logger.info(
            "Feature and target stats",
            extra={
                "feature_shape": list(X.shape),
                "target_value_mean": round(y_value.mean(), 2),
                "target_value_std": round(y_value.std(), 2),
                "target_residual_mean": round(y_residual.mean(), 2),
                "target_residual_std": round(y_residual.std(), 2),
                "over_rate": round(y_binary.mean(), 4),
            },
        )

        # Metadata for evaluation (only include columns that exist)
        metadata_cols = []
        metadata_col_names = []

        if "player_name" in df.columns:
            metadata_cols.append("player_name")
            metadata_col_names.append("player_name")
        if "game_date" in df.columns:
            metadata_cols.append("game_date")
            metadata_col_names.append("game_date")

        # Always include line and actual (required)
        metadata_cols.extend(["line", target_col])
        metadata_col_names.extend(["line", "actual"])

        metadata = df[metadata_cols].copy()
        metadata.columns = metadata_col_names

        return X, y_value, y_binary, y_residual, metadata

    def train(
        self,
        X_train,
        y_value_train,
        y_binary_train,
        y_residual_train,
        X_test,
        y_value_test,
        y_binary_test,
        y_residual_test,
    ):
        """
        Train stacked two-head model

        Step 1: Train regressor on X_train → predict absolute value
        Step 2: Calculate expected_diff = prediction - line
        Step 3: Augment X_train with expected_diff
        Step 4: Train classifier on augmented features → predict P(actual > line)
        """
        logger.info("Training stacked two-head model", extra={"market": self.market})

        # ==========================
        # STEP 1: Train Regressor
        # ==========================
        logger.info("HEAD 1: Training Regressor (absolute value prediction)")

        # Impute missing values
        self.imputer = SimpleImputer(strategy="median")
        X_train_imputed_array = self.imputer.fit_transform(X_train)
        X_test_imputed_array = self.imputer.transform(X_test)

        # Get feature names after imputation (some columns may have been dropped)
        imputed_feature_names = self.imputer.get_feature_names_out(X_train.columns)

        X_train_imputed = pd.DataFrame(
            X_train_imputed_array, columns=imputed_feature_names, index=X_train.index
        )
        X_test_imputed = pd.DataFrame(
            X_test_imputed_array, columns=imputed_feature_names, index=X_test.index
        )

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train_imputed),
            columns=X_train_imputed.columns,
            index=X_train_imputed.index,
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test_imputed),
            columns=X_test_imputed.columns,
            index=X_test_imputed.index,
        )

        # Temporal drift modeling (attempt1.md Advanced #4) - emphasize recent games
        # NOTE: Requires 'game_date' in features (currently excluded). Commented for reference.
        # if 'game_date' in X_train.columns:
        #     max_date = X_train['game_date'].max()
        #     days_ago_train = (max_date - X_train['game_date']).dt.days
        #     days_ago_test = (max_date - X_test['game_date']).dt.days
        #
        #     # Exponential decay: 60-day half-life
        #     sample_weight_train = np.exp(-days_ago_train / 60.0)
        #     sample_weight_test = np.exp(-days_ago_test / 60.0)
        #
        #     logger.info(f"   Temporal weighting: mean_weight={sample_weight_train.mean():.3f}")
        #
        #     # Pass to LightGBM
        #     lgb_train = lgb.Dataset(X_train_scaled, y_value_train, weight=sample_weight_train)
        #     lgb_test = lgb.Dataset(X_test_scaled, y_value_test, reference=lgb_train, weight=sample_weight_test)
        # else:
        #     sample_weight_train = None
        #     sample_weight_test = None

        # Train regressor using sklearn API
        self.regressor = LGBMRegressor(
            objective="regression",
            boosting_type="gbdt",
            num_leaves=63,
            learning_rate=0.02,
            n_estimators=2000,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            verbose=-1,
            random_state=42,
        )

        self.regressor.fit(
            X_train_scaled,
            y_value_train,
            eval_set=[(X_test_scaled, y_value_test)],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=100)],
        )

        # Evaluate regressor
        y_pred_train = self.regressor.predict(X_train_scaled)
        y_pred_test = self.regressor.predict(X_test_scaled)

        rmse_train = np.sqrt(mean_squared_error(y_value_train, y_pred_train))
        rmse_test = np.sqrt(mean_squared_error(y_value_test, y_pred_test))
        mae_test = mean_absolute_error(y_value_test, y_pred_test)
        r2_test = r2_score(y_value_test, y_pred_test)

        logger.info(
            "Regressor metrics",
            extra={
                "rmse_train": round(rmse_train, 3),
                "rmse_test": round(rmse_test, 3),
                "mae_test": round(mae_test, 3),
                "r2_test": round(r2_test, 3),
            },
        )

        # ==========================
        # STEP 2: Augment with Expected Diff
        # ==========================
        logger.info("Augmenting features with expected_diff")

        # Calculate expected_diff = prediction - line
        line_train = X_train["line"].values
        line_test = X_test["line"].values

        expected_diff_train = y_pred_train - line_train
        expected_diff_test = y_pred_test - line_test

        logger.debug(
            "Expected diff stats",
            extra={
                "train_mean": round(expected_diff_train.mean(), 2),
                "train_std": round(expected_diff_train.std(), 2),
                "test_mean": round(expected_diff_test.mean(), 2),
                "test_std": round(expected_diff_test.std(), 2),
            },
        )

        # Augment features
        X_train_augmented = X_train_scaled.copy()
        X_train_augmented["expected_diff"] = expected_diff_train

        X_test_augmented = X_test_scaled.copy()
        X_test_augmented["expected_diff"] = expected_diff_test

        # ==========================
        # Enhanced Class Balancing (attempt1.md Step 2)
        # ==========================
        class_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_binary_train)
        weight_dict = {0: class_weights[0], 1: class_weights[1]}

        # Create sample weights for LightGBM
        sample_weights = np.array([weight_dict[int(y)] for y in y_binary_train])

        logger.info(
            "Class balancing",
            extra={
                "under_weight": round(weight_dict[0], 3),
                "over_weight": round(weight_dict[1], 3),
                "under_count": int(sum(y_binary_train == 0)),
                "over_count": int(sum(y_binary_train == 1)),
            },
        )

        # ==========================
        # STEP 3: Train Classifier
        # ==========================
        logger.info("HEAD 2: Training Classifier (P(actual > line) prediction)")

        # Train classifier using sklearn API
        self.classifier = LGBMClassifier(
            objective="binary",
            boosting_type="gbdt",
            num_leaves=63,
            learning_rate=0.02,
            n_estimators=2000,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            verbose=-1,
            random_state=42,
        )

        self.classifier.fit(
            X_train_augmented,
            y_binary_train,
            sample_weight=sample_weights,
            eval_set=[(X_test_augmented, y_binary_test)],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=100)],
        )

        # Evaluate classifier
        y_prob_train = self.classifier.predict_proba(X_train_augmented)[:, 1]
        y_prob_test = self.classifier.predict_proba(X_test_augmented)[:, 1]

        y_pred_binary_train = (y_prob_train > 0.5).astype(int)
        y_pred_binary_test = (y_prob_test > 0.5).astype(int)

        acc_train = accuracy_score(y_binary_train, y_pred_binary_train)
        acc_test = accuracy_score(y_binary_test, y_pred_binary_test)
        auc_test = roc_auc_score(y_binary_test, y_prob_test)
        logloss_test = log_loss(y_binary_test, y_prob_test)

        logger.info(
            "Classifier metrics",
            extra={
                "accuracy_train": round(acc_train, 3),
                "accuracy_test": round(acc_test, 3),
                "auc_test": round(auc_test, 3),
                "logloss_test": round(logloss_test, 3),
            },
        )

        # Log classification report as debug
        report = classification_report(
            y_binary_test, y_pred_binary_test, target_names=["UNDER", "OVER"], output_dict=True
        )
        logger.debug("Classification report", extra={"report": report})

        # ==========================
        # STEP 4: Isotonic Calibration (attempt1.md Step 5)
        # FIX: Train calibrator on held-out portion of TRAINING data, not test data
        # This prevents data leakage - calibrator never sees test data during fitting
        # ==========================
        logger.info("Calibrating classifier probabilities")

        # Split training predictions into calibration train/val (80/20)
        # Using shuffled split since we already have temporal split for train/test
        from sklearn.model_selection import train_test_split

        cal_train_idx, cal_val_idx = train_test_split(
            np.arange(len(y_prob_train)), test_size=0.2, random_state=42, stratify=y_binary_train
        )

        y_prob_cal_train = y_prob_train[cal_train_idx]
        y_prob_cal_val = y_prob_train[cal_val_idx]
        y_binary_cal_train = (
            y_binary_train.iloc[cal_train_idx].values
            if hasattr(y_binary_train, "iloc")
            else y_binary_train[cal_train_idx]
        )
        y_binary_cal_val = (
            y_binary_train.iloc[cal_val_idx].values
            if hasattr(y_binary_train, "iloc")
            else y_binary_train[cal_val_idx]
        )

        # Fit calibrator on the calibration validation set (not test data!)
        self.calibrator = IsotonicRegression(out_of_bounds="clip")
        self.calibrator.fit(y_prob_cal_val, y_binary_cal_val)

        logger.info(
            "Calibrator fitted (no test data leakage)",
            extra={"calibration_samples": len(y_prob_cal_val)},
        )

        # Generate calibrated probabilities
        y_prob_test_cal = self.calibrator.transform(y_prob_test)
        y_prob_train_cal = self.calibrator.transform(y_prob_train)

        # Evaluate calibration
        brier_before = brier_score_loss(y_binary_test, y_prob_test)
        brier_after = brier_score_loss(y_binary_test, y_prob_test_cal)
        auc_cal = roc_auc_score(y_binary_test, y_prob_test_cal)

        logger.info(
            "Calibration results",
            extra={
                "brier_before": round(brier_before, 4),
                "brier_after": round(brier_after, 4),
                "calibration_improvement": round(brier_before - brier_after, 4),
                "auc_calibrated": round(auc_cal, 4),
            },
        )

        # ==========================
        # STEP 5: Ensemble Blending (attempt1.md Step 6)
        # ==========================
        logger.info("Blending regressor and classifier predictions")

        # Convert regressor residuals to probabilities using sigmoid
        residual_train = y_pred_train - line_train
        residual_test = y_pred_test - line_test

        # Scale factor tuned for typical prop magnitude
        scale_factor = 5.0 if self.market in ["POINTS", "ASSISTS"] else 2.0

        prob_from_residual_train = expit(residual_train / scale_factor)
        prob_from_residual_test = expit(residual_test / scale_factor)

        # Blend: 60% classifier, 40% residual-based probability
        blend_weight_cls = 0.6
        blend_weight_res = 0.4

        y_prob_blend_test = (
            blend_weight_cls * y_prob_test_cal + blend_weight_res * prob_from_residual_test
        )
        y_prob_blend_train = (
            blend_weight_cls * y_prob_train_cal + blend_weight_res * prob_from_residual_train
        )

        # Evaluate blended performance
        auc_blend = roc_auc_score(y_binary_test, y_prob_blend_test)
        brier_blend = brier_score_loss(y_binary_test, y_prob_blend_test)

        logger.info(
            "Blending results",
            extra={
                "auc_classifier_only": round(auc_cal, 4),
                "auc_blended": round(auc_blend, 4),
                "blend_improvement": round(auc_blend - auc_cal, 4),
                "brier_blended": round(brier_blend, 4),
            },
        )

        # Store blend config
        self.blend_config = {
            "classifier_weight": blend_weight_cls,
            "residual_weight": blend_weight_res,
            "residual_scale_factor": scale_factor,
        }

        # ==========================
        # Feature Importance
        # ==========================
        feature_importance_reg = (
            pd.DataFrame(
                {
                    "feature": X_train_scaled.columns,
                    "importance": self.regressor.feature_importances_,
                }
            )
            .sort_values("importance", ascending=False)
            .head(20)
        )
        top_features_reg = feature_importance_reg.to_dict("records")
        logger.info("Top 20 features (Regressor)", extra={"features": top_features_reg})

        feature_importance_cls = (
            pd.DataFrame(
                {
                    "feature": X_train_augmented.columns,
                    "importance": self.classifier.feature_importances_,
                }
            )
            .sort_values("importance", ascending=False)
            .head(20)
        )
        top_features_cls = feature_importance_cls.to_dict("records")
        logger.info("Top 20 features (Classifier)", extra={"features": top_features_cls})

        return {
            "regressor": {
                "rmse_train": rmse_train,
                "rmse_test": rmse_test,
                "mae_test": mae_test,
                "r2_test": r2_test,
            },
            "classifier": {
                "acc_train": acc_train,
                "acc_test": acc_test,
                "auc_test": auc_test,
                "auc_calibrated": auc_cal,
                "auc_blended": auc_blend,
                "logloss_test": logloss_test,
                "brier_before": brier_before,
                "brier_after": brier_after,
                "brier_blended": brier_blend,
            },
        }

    def save(self, output_dir: str, metrics: dict):
        """
        Save both models and metadata

        Files saved:
        - {market}_regressor_model.pkl
        - {market}_classifier_model.pkl
        - {market}_market_imputer.pkl
        - {market}_market_scaler.pkl
        - {market}_market_features.pkl
        - {market}_market_metadata.json
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        prefix = f"{self.market.lower()}_market"

        # Save models
        with open(output_path / f"{prefix}_regressor.pkl", "wb") as f:
            pickle.dump(self.regressor, f)

        with open(output_path / f"{prefix}_classifier.pkl", "wb") as f:
            pickle.dump(self.classifier, f)

        # Save preprocessors
        with open(output_path / f"{prefix}_imputer.pkl", "wb") as f:
            pickle.dump(self.imputer, f)

        with open(output_path / f"{prefix}_scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)

        # Save calibrator (attempt1.md Step 5)
        with open(output_path / f"{prefix}_calibrator.pkl", "wb") as f:
            pickle.dump(self.calibrator, f)

        # Save feature names
        with open(output_path / f"{prefix}_features.pkl", "wb") as f:
            pickle.dump(self.feature_names, f)

        # Save metadata
        metadata = {
            "market": self.market,
            "trained_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "architecture": "stacked_two_head_calibrated_blended",
            "features": {"count": len(self.feature_names), "names": self.feature_names},
            "blend_config": self.blend_config,
            "metrics": metrics,
        }

        with open(output_path / f"{prefix}_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(
            "Models saved successfully",
            extra={
                "output_dir": str(output_path),
                "files": [
                    f"{prefix}_regressor.pkl",
                    f"{prefix}_classifier.pkl",
                    f"{prefix}_imputer.pkl",
                    f"{prefix}_scaler.pkl",
                    f"{prefix}_calibrator.pkl",
                    f"{prefix}_features.pkl",
                    f"{prefix}_metadata.json",
                ],
            },
        )


def main():
    parser = argparse.ArgumentParser(
        description="Train stacked two-head NBA props model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--market",
        required=True,
        choices=["POINTS", "REBOUNDS", "ASSISTS", "THREES"],
        help="Market to train",
    )

    parser.add_argument(
        "--data",
        default="nba/features/datasets/nba_training_MARKET_AWARE.csv",
        help="Path to training dataset",
    )

    parser.add_argument(
        "--output", default="nba/models/saved_xl", help="Output directory for saved models"
    )

    parser.add_argument("--test-size", type=float, default=0.3, help="Test set size (default: 0.3)")

    parser.add_argument(
        "--random-state", type=int, default=42, help="Random state for reproducibility"
    )

    parser.add_argument(
        "--track",
        action="store_true",
        help="Enable MLflow experiment tracking",
    )

    parser.add_argument(
        "--experiment-name",
        default="nba-props-xl",
        help="MLflow experiment name",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # Resolve paths
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / args.data
    output_dir = project_root / args.output

    # Setup logging
    import logging

    setup_logging(
        "train_market",
        level=logging.DEBUG if getattr(args, "debug", False) else logging.INFO,
    )

    if not data_path.exists():
        logger.error("Dataset not found", extra={"path": str(data_path)})
        return 1

    # Initialize MLflow tracker if enabled
    tracker: Optional[ExperimentTracker] = None
    if args.track and MLFLOW_AVAILABLE:
        tracker = ExperimentTracker(experiment_name=args.experiment_name)
        logger.info("MLflow tracking enabled", extra={"experiment": args.experiment_name})
    elif args.track and not MLFLOW_AVAILABLE:
        logger.warning(
            "MLflow tracking requested but not available. Install with: pip install mlflow"
        )

    try:
        # Initialize model
        model = StackedMarketModel(market=args.market)

        # Load data
        df = model.load_data(str(data_path))

        # Prepare features
        X, y_value, y_binary, y_residual, metadata = model.prepare_features(df)

        # Split (temporal or random - temporal recommended)
        (
            X_train,
            X_test,
            y_value_train,
            y_value_test,
            y_binary_train,
            y_binary_test,
            y_residual_train,
            y_residual_test,
        ) = train_test_split(
            X,
            y_value,
            y_binary,
            y_residual,
            test_size=args.test_size,
            random_state=args.random_state,
            shuffle=False,  # Temporal split (no shuffle)
        )

        logger.info(
            "Train/Test split",
            extra={"train_samples": len(X_train), "test_samples": len(X_test)},
        )

        # Start MLflow run if tracking enabled
        mlflow_context = (
            tracker.start_run(
                run_name=f"{args.market}_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                tags={"market": args.market, "architecture": "stacked_two_head"},
            )
            if tracker
            else None
        )

        with mlflow_context if mlflow_context else _null_context():
            # Log parameters to MLflow
            if tracker:
                tracker.log_params(
                    {
                        "market": args.market,
                        "test_size": args.test_size,
                        "random_state": args.random_state,
                        "regressor_n_estimators": 2000,
                        "regressor_learning_rate": 0.02,
                        "regressor_num_leaves": 63,
                        "classifier_n_estimators": 2000,
                        "classifier_learning_rate": 0.02,
                        "classifier_num_leaves": 63,
                    }
                )
                tracker.log_training_data_info(
                    train_samples=len(X_train),
                    test_samples=len(X_test),
                    feature_count=len(X.columns),
                )

            # Train
            metrics = model.train(
                X_train,
                y_value_train,
                y_binary_train,
                y_residual_train,
                X_test,
                y_value_test,
                y_binary_test,
                y_residual_test,
            )

            # Log metrics to MLflow
            if tracker:
                tracker.log_metrics(
                    {
                        "rmse_train": metrics["regressor"]["rmse_train"],
                        "rmse_test": metrics["regressor"]["rmse_test"],
                        "mae_test": metrics["regressor"]["mae_test"],
                        "r2_test": metrics["regressor"]["r2_test"],
                        "accuracy_train": metrics["classifier"]["acc_train"],
                        "accuracy_test": metrics["classifier"]["acc_test"],
                        "auc_test": metrics["classifier"]["auc_test"],
                        "auc_calibrated": metrics["classifier"]["auc_calibrated"],
                        "auc_blended": metrics["classifier"]["auc_blended"],
                        "brier_before": metrics["classifier"]["brier_before"],
                        "brier_after": metrics["classifier"]["brier_after"],
                        "brier_blended": metrics["classifier"]["brier_blended"],
                    }
                )

                # Log feature importance
                if model.regressor is not None:
                    tracker.log_feature_importance(
                        feature_names=list(model.feature_names),
                        importances=list(model.regressor.feature_importances_),
                        top_n=20,
                    )

        # Save
        model.save(str(output_dir), metrics)

        logger.info(
            "Training completed successfully",
            extra={
                "market": args.market,
                "mlflow_experiment": args.experiment_name if tracker else None,
            },
        )

        return 0

    except (OSError, KeyError, ValueError, TypeError) as e:
        logger.error("Training failed", extra={"error": str(e)}, exc_info=True)
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
