#!/usr/bin/env python3
"""
Market Classifier Model (Model 2) — LightGBM Classifier
=========================================================
Predicts P(OVER) using Model 1's projection output + market/book features.

This is the second stage of the model cascade:
    Model 1 (PlayerProjectionModel) → raw stat prediction (e.g., 25.3 points)
    Model 2 (MarketClassifierModel) → P(OVER) from projection_diff + book features
    Model 2's P(OVER) replaces the old 60/40 classifier+residual blend.

Key design:
    - ONLY sees projection_diff, book features, BP features, prop features, DVP features
    - NEVER sees player EMA stats, team ratings, or any Model 1 feature
    - Platt scaling calibration (LogisticRegression, NOT isotonic)

Usage:
    python nba/models/market_classifier_model.py --market POINTS \
        --data nba/features/datasets/xl_training_POINTS_2024_present.csv \
        --projection-model-dir nba/models/saved_xl

    python nba/models/market_classifier_model.py --market REBOUNDS \
        --data nba/features/datasets/xl_training_REBOUNDS_2024_present.csv \
        --projection-model-dir nba/models/saved_xl
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
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from nba.config.thresholds import TEMPORAL_DECAY_CONFIG, TRAINING_HYPERPARAMETERS
from nba.core.logging_config import get_logger, setup_logging
from nba.models.mlflow_tracking import ModelTracker
from nba.models.player_projection_model import PlayerProjectionModel

EST = ZoneInfo("America/New_York")

logger = get_logger(__name__)

# Features Model 2 is allowed to see (prefix matching)
CLASSIFIER_PREFIXES = (
    "projection_diff",
    "line",
    "consensus_line",
    "line_spread",
    "line_std",
    "num_books",
    "line_coef",
    "books_agree",
    "books_disagree",
    "draftkings_deviation",
    "fanduel_deviation",
    "betmgm_deviation",
    "caesars_deviation",
    "betrivers_deviation",
    "softest_",
    "hardest_",
    "min_line",
    "max_line",
    "line_spread_percentile",
    "bp_",
    "prop_hit_rate",
    "prop_line_vs",
    "prop_line_percentile",
    "prop_bayesian",
    "prop_consecutive",
    "prop_days_since",
    "prop_sample",
    "dvp_",
    "snapshot_count",
    "line_delta",
    "line_movement_std",
    "consensus_strength",
    "volume_proxy",
    "line_source_reliability",
    "hours_tracked",
)

# Model 1 features that MUST NOT leak into Model 2
FORBIDDEN_MODEL1_PREFIXES = (
    "ema_",
    "fg_pct_",
    "true_shooting",
    "plus_minus_L",
    "team_pace",
    "team_off_",
    "team_def_",
    "opp_pace",
    "opp_off_",
    "opp_def_",
)

# Target column mapping
TARGET_MAP = {
    "POINTS": "actual_points",
    "REBOUNDS": "actual_rebounds",
}

# Rolling std column for normalization
STD_COL_MAP = {
    "POINTS": "points_std_L10",
    "REBOUNDS": "rebounds_std_L10",
}


class MarketClassifierModel:
    """
    LightGBM classifier for market inefficiency detection.

    Takes Model 1's projection_diff + book/line/BP features to predict P(OVER).
    """

    def __init__(self, market: str):
        self.market = market.upper()
        self.classifier = None
        self.imputer = None
        self.scaler = None
        self.calibrator = None
        self.feature_names = None

    def prepare_classifier_features(
        self,
        df: pd.DataFrame,
        projection_model_dir: str,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """Prepare classifier features using Model 1 projections.

        Loads Model 1, runs predictions, computes projection_diff,
        selects only classifier-allowed features.

        Args:
            df: Full training DataFrame (includes all features).
            projection_model_dir: Directory containing saved Model 1 artifacts.

        Returns:
            Tuple of (X features, y_binary target, metadata DataFrame).

        Raises:
            ValueError: If Model 1 features leak into classifier feature set.
        """
        logger.info(
            "Preparing classifier features",
            extra={"market": self.market, "samples": len(df)},
        )

        # Load Model 1
        model1 = PlayerProjectionModel(market=self.market)
        model1.load(projection_model_dir)
        logger.info(
            "Model 1 loaded",
            extra={"features": len(model1.feature_names)},
        )

        # Get Model 1 predictions — fill missing columns with NaN (imputer handles them)
        missing = set(model1.feature_names) - set(df.columns)
        if missing:
            logger.warning(
                "Model 1 features missing from CSV, filling NaN",
                extra={"missing_count": len(missing), "missing": sorted(missing)},
            )

        model1_input = df.reindex(columns=model1.feature_names)
        projections = model1.predict(model1_input)

        # Compute projection_diff
        df = df.copy()
        df["projection_diff"] = projections - df["line"]

        # Compute projection_diff_normalized
        std_col = STD_COL_MAP[self.market]
        if std_col in df.columns:
            std_vals = df[std_col].clip(lower=0.5)
            df["projection_diff_normalized"] = df["projection_diff"] / std_vals
        else:
            logger.warning(f"Std column {std_col} not found, skipping normalization")

        # Select classifier features by prefix matching
        classifier_cols = [
            col
            for col in df.columns
            if any(col.startswith(prefix) for prefix in CLASSIFIER_PREFIXES)
        ]

        # Validate: NO Model 1 features leaked through
        leaked = [
            col
            for col in classifier_cols
            if any(col.startswith(prefix) for prefix in FORBIDDEN_MODEL1_PREFIXES)
        ]
        if leaked:
            raise ValueError(
                f"Model 1 features leaked into classifier: {leaked}. "
                "These belong to Model 1, not Model 2."
            )

        X = df[classifier_cols].copy()

        # Drop non-numeric columns
        non_numeric = X.select_dtypes(exclude="number").columns.tolist()
        if non_numeric:
            logger.info(f"Dropping {len(non_numeric)} non-numeric columns: {non_numeric}")
            X = X.drop(columns=non_numeric)

        # Drop zero-variance features
        zero_var = [col for col in X.columns if X[col].std() == 0]
        if zero_var:
            logger.info(f"Dropping {len(zero_var)} zero-variance features")
            X = X.drop(columns=zero_var)

        self.feature_names = X.columns.tolist()

        # Binary target: actual > line
        target_col = TARGET_MAP[self.market]
        y_binary = (df[target_col] > df["line"]).astype(int)

        # Metadata for temporal weighting
        meta_cols = [c for c in ["player_name", "game_date"] if c in df.columns]
        metadata = df[meta_cols].copy() if meta_cols else pd.DataFrame(index=df.index)

        logger.info(
            "Classifier features ready",
            extra={
                "feature_count": len(self.feature_names),
                "positive_rate": round(float(y_binary.mean()), 3),
                "samples": len(X),
            },
        )

        return X, y_binary, metadata

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        game_dates_train: Optional[pd.Series] = None,
        game_dates_test: Optional[pd.Series] = None,
    ) -> Dict:
        """Train LightGBM classifier with temporal decay and Platt calibration.

        Args:
            X_train: Training features.
            y_train: Training binary targets.
            X_test: Test features.
            y_test: Test binary targets.
            game_dates_train: Optional game dates for temporal decay (train).
            game_dates_test: Optional game dates for temporal decay (test).

        Returns:
            Dictionary of training metrics.
        """
        logger.info(
            "Training market classifier",
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
            f"{self.market}_classifier_{datetime.now(EST).strftime('%Y%m%d_%H%M')}",
            tags={"model_type": "market_classifier", "market": self.market},
        )
        self._tracker.log_params(
            {
                "market": self.market,
                "model_type": "market_classifier",
                "feature_count": len(X_train.columns),
                "train_samples": len(X_train),
                "test_samples": len(X_test),
            }
        )

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
        temporal_weights = np.ones(len(X_train))
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
                },
            )

        # Class weights (balanced)
        classes = np.array([0, 1])
        class_weights = compute_class_weight("balanced", classes=classes, y=y_train)
        class_weight_map = dict(zip(classes, class_weights))
        sample_class_weights = np.array([class_weight_map[y] for y in y_train])

        # Combined weights = class_weight * temporal_weight
        combined_weights = sample_class_weights * temporal_weights

        # Train LightGBM classifier
        hp = TRAINING_HYPERPARAMETERS
        self.classifier = LGBMClassifier(
            objective="binary",
            boosting_type="gbdt",
            num_leaves=hp.classifier_num_leaves,
            learning_rate=hp.classifier_learning_rate,
            n_estimators=hp.classifier_n_estimators,
            feature_fraction=hp.feature_fraction,
            bagging_fraction=hp.bagging_fraction,
            bagging_freq=hp.bagging_freq,
            reg_alpha=hp.lambda_l1,
            reg_lambda=hp.lambda_l2,
            min_child_samples=hp.min_child_samples,
            verbose=-1,
            random_state=hp.random_state,
        )

        self.classifier.fit(
            X_train_scaled,
            y_train,
            sample_weight=combined_weights,
            eval_set=[(X_test_scaled, y_test)],
            eval_metric="auc",
            callbacks=[
                lgb.early_stopping(stopping_rounds=hp.classifier_early_stopping_rounds),
                lgb.log_evaluation(period=100),
            ],
        )

        # Raw probabilities
        probs_train = self.classifier.predict_proba(X_train_scaled)[:, 1]
        probs_test = self.classifier.predict_proba(X_test_scaled)[:, 1]

        # Platt scaling calibration
        self.calibrator = None
        brier_before = brier_score_loss(y_test, probs_test)

        # Use 80/20 split of training data for calibration holdout
        cal_split_idx = int(len(probs_train) * 0.8)
        cal_probs = probs_train[cal_split_idx:]
        cal_labels = y_train.iloc[cal_split_idx:]

        if len(cal_probs) >= 100:
            platt = LogisticRegression(max_iter=1000, random_state=42)
            platt.fit(cal_probs.reshape(-1, 1), cal_labels)
            calibrated_test = platt.predict_proba(probs_test.reshape(-1, 1))[:, 1]
            brier_after = brier_score_loss(y_test, calibrated_test)

            if brier_after < brier_before:
                self.calibrator = platt
                probs_test = calibrated_test
                logger.info(
                    "Platt calibration applied",
                    extra={
                        "brier_before": round(brier_before, 5),
                        "brier_after": round(brier_after, 5),
                    },
                )
            else:
                logger.info(
                    "Platt calibration skipped (no improvement)",
                    extra={
                        "brier_before": round(brier_before, 5),
                        "brier_after": round(brier_after, 5),
                    },
                )

        # Compute metrics
        auc = roc_auc_score(y_test, probs_test)
        accuracy = accuracy_score(y_test, (probs_test >= 0.5).astype(int))
        brier = brier_score_loss(y_test, probs_test)
        logloss = log_loss(y_test, probs_test)

        train_duration = round(_time.monotonic() - train_start, 1)

        # Feature importance (top 20)
        importance_df = (
            pd.DataFrame(
                {
                    "feature": X_train_scaled.columns,
                    "importance": self.classifier.feature_importances_,
                }
            )
            .sort_values("importance", ascending=False)
            .head(20)
        )
        top_features = importance_df.to_dict("records")

        logger.info(
            "Market classifier metrics",
            extra={
                "auc": round(auc, 4),
                "accuracy": round(accuracy, 4),
                "brier": round(brier, 5),
                "logloss": round(logloss, 5),
                "n_trees": self.classifier.n_estimators_,
                "calibrated": self.calibrator is not None,
                "duration_s": train_duration,
            },
        )

        metrics = {
            "auc": auc,
            "accuracy": accuracy,
            "brier_score": brier,
            "log_loss": logloss,
            "n_trees": self.classifier.n_estimators_,
            "calibrated": self.calibrator is not None,
            "training_duration_s": train_duration,
            "top_features": top_features,
        }

        self._tracker.log_metrics(
            {
                "auc_test": auc,
                "accuracy_test": accuracy,
                "brier_score": brier,
                "log_loss": logloss,
                "n_trees": float(self.classifier.n_estimators_),
                "training_duration_s": train_duration,
            }
        )

        return metrics

    def save(self, output_dir: str) -> None:
        """Save model artifacts to disk.

        Args:
            output_dir: Directory to save artifacts.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        prefix = self.market.lower()

        with open(output_path / f"{prefix}_market_classifier.pkl", "wb") as f:
            pickle.dump(self.classifier, f)

        with open(output_path / f"{prefix}_market_imputer.pkl", "wb") as f:
            pickle.dump(self.imputer, f)

        with open(output_path / f"{prefix}_market_scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)

        with open(output_path / f"{prefix}_market_calibrator.pkl", "wb") as f:
            pickle.dump(self.calibrator, f)

        with open(output_path / f"{prefix}_market_features.pkl", "wb") as f:
            pickle.dump(self.feature_names, f)

        metadata = {
            "market": self.market,
            "model_type": "market_inefficiency_classifier",
            "trained_date": datetime.now(EST).strftime("%Y-%m-%d %H:%M:%S"),
            "calibrated": self.calibrator is not None,
            "features": {
                "count": len(self.feature_names),
                "names": self.feature_names,
            },
        }

        if hasattr(self, "_last_metrics"):
            metadata["metrics"] = {
                k: v for k, v in self._last_metrics.items() if k != "top_features"
            }

        with open(output_path / f"{prefix}_market_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # End MLflow run
        if hasattr(self, "_tracker"):
            self._tracker.end_run()

        logger.info(
            "Market classifier saved",
            extra={
                "output_dir": str(output_path),
                "prefix": prefix,
                "feature_count": len(self.feature_names),
            },
        )

    def load(self, output_dir: str) -> None:
        """Load model artifacts from disk.

        Args:
            output_dir: Directory containing saved artifacts.
        """
        output_path = Path(output_dir)
        prefix = self.market.lower()

        with open(output_path / f"{prefix}_market_classifier.pkl", "rb") as f:
            self.classifier = pickle.load(f)

        with open(output_path / f"{prefix}_market_imputer.pkl", "rb") as f:
            self.imputer = pickle.load(f)

        with open(output_path / f"{prefix}_market_scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)

        with open(output_path / f"{prefix}_market_calibrator.pkl", "rb") as f:
            self.calibrator = pickle.load(f)

        with open(output_path / f"{prefix}_market_features.pkl", "rb") as f:
            self.feature_names = pickle.load(f)

        logger.info(
            "Market classifier loaded",
            extra={
                "market": self.market,
                "feature_count": len(self.feature_names),
                "calibrated": self.calibrator is not None,
            },
        )

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Generate P(OVER) predictions.

        Args:
            features: DataFrame with classifier feature columns.

        Returns:
            Array of calibrated P(OVER) probabilities.
        """
        if self.classifier is None:
            raise RuntimeError("Model not trained or loaded. Call train() or load() first.")

        X = features[self.feature_names].copy()
        X_imp = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imp)
        probs = self.classifier.predict_proba(X_scaled)[:, 1]

        if self.calibrator is not None:
            probs = self.calibrator.predict_proba(probs.reshape(-1, 1))[:, 1]

        return probs


def main():
    parser = argparse.ArgumentParser(
        description="Train Market Classifier Model (Model 2 — P(OVER) from projection_diff)",
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
        help="Path to XL training dataset CSV (with all features)",
    )
    parser.add_argument(
        "--projection-model-dir",
        default="nba/models/saved_xl",
        help="Directory containing saved Model 1 artifacts",
    )
    parser.add_argument(
        "--output",
        default="nba/models/saved_xl",
        help="Output directory for saved models",
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

    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / args.data
    output_dir = project_root / args.output
    projection_dir = project_root / args.projection_model_dir

    setup_logging(
        "market_classifier_model",
        level=logging.DEBUG if args.debug else logging.INFO,
    )

    if not data_path.exists():
        print(f"ERROR: Dataset not found: {data_path}")
        return 1

    try:
        model = MarketClassifierModel(market=args.market)

        # Load data
        logger.info("Loading data", extra={"path": str(data_path)})
        df = pd.read_csv(str(data_path))
        if "game_date" in df.columns:
            df["game_date"] = pd.to_datetime(df["game_date"])

        # Dedup per player/game
        if "player_name" in df.columns and "game_date" in df.columns:
            before = len(df)
            df = df.drop_duplicates(subset=["player_name", "game_date"], keep="first")
            if len(df) < before:
                logger.info(f"Deduped: {before} -> {len(df)}")

        # Drop rows missing target or line
        target_col = TARGET_MAP[args.market]
        df = df.dropna(subset=[target_col, "line"])

        # Prepare classifier features (runs Model 1 internally)
        X, y, metadata = model.prepare_classifier_features(df, str(projection_dir))

        # Temporal split (no shuffle — chronological)
        X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
            X, y, metadata, test_size=args.test_size, shuffle=False
        )

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

        model._last_metrics = metrics

        # Print results
        print(f"\n{'=' * 60}")
        print(f"MARKET CLASSIFIER (Model 2) — {args.market}")
        print(f"{'=' * 60}")
        print(f"  Train samples: {len(X_train):,}")
        print(f"  Test samples:  {len(X_test):,}")
        print(f"  Features:      {len(model.feature_names)}")
        print(f"  Trees:         {metrics['n_trees']}")
        print(f"  Calibrated:    {metrics['calibrated']}")
        print(f"  Duration:      {metrics['training_duration_s']}s")
        print(f"\n  AUC:           {metrics['auc']:.4f}")
        print(f"  Accuracy:      {metrics['accuracy']:.4f}")
        print(f"  Brier Score:   {metrics['brier_score']:.5f}")
        print(f"  Log Loss:      {metrics['log_loss']:.5f}")
        print("\n  Top 5 features:")
        for feat in metrics["top_features"][:5]:
            print(f"    {feat['feature']:<40s} {feat['importance']}")
        print(f"{'=' * 60}\n")

        # Save
        model.save(str(output_dir))

        return 0

    except (OSError, KeyError, ValueError, TypeError) as e:
        logger.error("Training failed", extra={"error": str(e)}, exc_info=True)
        print(f"ERROR: {e}")
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
