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
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy.special import expit
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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

from nba.config.thresholds import (
    BLEND_WEIGHTS,
    FEATURE_PREPROCESSING,
    TEMPORAL_DECAY_CONFIG,
    TRAINING_HYPERPARAMETERS,
)
from nba.core.logging_config import get_logger, setup_logging
from nba.models.mlflow_tracking import MLFLOW_AVAILABLE, ModelTracker

# Logger will be configured in main()
logger = get_logger(__name__)


class PlattCalibrator:
    """Platt scaling wrapper with .transform() for API compatibility.

    The prediction pipeline (xl_predictor.py, walk_forward_validation.py) calls
    calibrator.transform([prob])[0]. IsotonicRegression has .transform() natively.
    LogisticRegression uses .predict_proba(). This wrapper bridges the two APIs
    so the calibrator pickle is a drop-in replacement.
    """

    def __init__(self, lr: LogisticRegression):
        self._lr = lr

    def transform(self, X):
        """Transform raw probabilities to calibrated probabilities.

        Args:
            X: array-like of raw probabilities (1D).

        Returns:
            array of calibrated probabilities.
        """
        X_arr = np.asarray(X).reshape(-1, 1)
        return self._lr.predict_proba(X_arr)[:, 1]

    def predict_proba(self, X):
        """Direct access to predict_proba for code that uses it."""
        return self._lr.predict_proba(X)


def walk_forward_cv(
    df: pd.DataFrame,
    n_splits: int = 5,
    min_train_size: int = 10000,
    test_size_months: int = 2,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Generate walk-forward cross-validation splits.

    Each fold expands the training window and tests on the next period:
    Fold 1: Train Oct 2023 - Mar 2024, Test Apr-May 2024
    Fold 2: Train Oct 2023 - May 2024, Test Jun-Jul 2024
    ...

    This approach is better for time-series data because it:
    - Tests on multiple future periods (not just one)
    - Detects concept drift over time
    - Provides more reliable performance estimates

    Args:
        df: DataFrame with 'game_date' column, will be sorted by date
        n_splits: Number of CV folds (default: 5)
        min_train_size: Minimum training samples required per fold (default: 10000)
        test_size_months: Months in each test period (default: 2)

    Returns:
        List of (train_df, test_df) tuples for each valid fold
    """
    # Ensure game_date is datetime
    if "game_date" not in df.columns:
        raise ValueError("DataFrame must have 'game_date' column for walk-forward CV")

    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values("game_date").reset_index(drop=True)

    min_date = df["game_date"].min()
    max_date = df["game_date"].max()

    # Calculate split points
    total_days = (max_date - min_date).days
    test_days = test_size_months * 30

    logger.info(
        "Walk-forward CV setup",
        extra={
            "min_date": str(min_date.date()),
            "max_date": str(max_date.date()),
            "total_days": total_days,
            "test_days_per_fold": test_days,
            "requested_splits": n_splits,
        },
    )

    splits = []
    for i in range(n_splits):
        # Expanding training window
        # Each fold adds one test period worth of data to training
        train_end_offset = total_days - (n_splits - i) * test_days
        train_end = min_date + pd.Timedelta(days=train_end_offset)
        test_end = train_end + pd.Timedelta(days=test_days)

        train_df = df[df["game_date"] < train_end].copy()
        test_df = df[(df["game_date"] >= train_end) & (df["game_date"] < test_end)].copy()

        # Validate split has enough data
        if len(train_df) >= min_train_size and len(test_df) > 100:
            splits.append((train_df, test_df))
            logger.debug(
                f"Fold {len(splits)} created",
                extra={
                    "train_start": str(train_df["game_date"].min().date()),
                    "train_end": str(train_df["game_date"].max().date()),
                    "test_start": str(test_df["game_date"].min().date()),
                    "test_end": str(test_df["game_date"].max().date()),
                    "train_size": len(train_df),
                    "test_size": len(test_df),
                },
            )
        else:
            logger.debug(
                f"Skipping fold {i + 1}",
                extra={
                    "train_size": len(train_df),
                    "test_size": len(test_df),
                    "min_train_required": min_train_size,
                },
            )

    logger.info(
        "Walk-forward CV splits generated",
        extra={"valid_splits": len(splits), "requested_splits": n_splits},
    )

    return splits


EST = ZoneInfo("America/New_York")


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

        # Drop constant columns (market-aware: keeps H2H for target market)
        cols_to_drop = [
            col for col in FEATURE_PREPROCESSING.get_cols_to_drop(self.market) if col in df.columns
        ]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            logger.info(
                "Dropped constant columns",
                extra={
                    "market": self.market,
                    "dropped": len(cols_to_drop),
                    "columns": cols_to_drop[:5],
                },
            )

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

        # Select features (MUST include 'line' and book deviation features)
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        # Verify critical features
        assert "line" in feature_cols, "FATAL: 'line' feature missing"

        # Book features: per-book deviations + line shopping columns
        _book_names = {
            "line_spread",
            "consensus_line",
            "num_books_offering",
            "draftkings_deviation",
            "fanduel_deviation",
            "betmgm_deviation",
            "caesars_deviation",
            "betrivers_deviation",
        }
        book_features = [col for col in feature_cols if col in _book_names]
        if len(book_features) == 0:
            logger.warning("No book features found - training without line shopping features")

        logger.info(
            "Feature summary",
            extra={"total_features": len(feature_cols), "book_features": len(book_features)},
        )

        X = df[feature_cols].copy()

        # Convert boolean columns to int (is_home stored as True/False)
        for col in X.columns:
            if X[col].dtype == "bool":
                X[col] = X[col].astype(int)

        # Drop zero-variance features (constants add noise, hurt classifier)
        zero_var = [col for col in X.columns if X[col].std() == 0]
        if zero_var:
            logger.info(f"Dropping {len(zero_var)} zero-variance features: {zero_var[:5]}...")
            X = X.drop(columns=zero_var)

        # Drop non-numeric columns that slipped through
        non_numeric = X.select_dtypes(exclude="number").columns.tolist()
        if non_numeric:
            logger.info(f"Dropping {len(non_numeric)} non-numeric columns: {non_numeric}")
            X = X.drop(columns=non_numeric)

        feature_cols = X.columns.tolist()
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
        game_dates_train=None,
        game_dates_test=None,
    ):
        """
        Train stacked two-head model

        Step 1: Train regressor on X_train → predict absolute value
        Step 2: Calculate expected_diff = prediction - line
        Step 3: Augment X_train with expected_diff
        Step 4: Train classifier on augmented features → predict P(actual > line)

        Args:
            game_dates_train: Optional Series of game dates for temporal weighting
            game_dates_test: Optional Series of test game dates
        """
        logger.info("Training stacked two-head model", extra={"market": self.market})
        import hashlib
        import time as _time

        _train_start = _time.monotonic()

        # Dataset fingerprint for reproducibility (Pillar 4: Security/Integrity)
        _dataset_hash = hashlib.md5(
            pd.util.hash_pandas_object(X_train).values.tobytes(),
            usedforsecurity=False,
        ).hexdigest()

        # Feature default rate — % of NaN/zero per feature (Pillar 5: Cost Optimization)
        _null_rates = X_train.isnull().mean()
        _zero_rates = (X_train == 0).mean()
        _default_rate = (_null_rates + _zero_rates) / 2
        _high_default_features = _default_rate[_default_rate > 0.5].index.tolist()

        # MLflow tracking (non-blocking — no-ops if mlflow not installed)
        self._tracker = ModelTracker(experiment="nba-model-cascade")
        self._tracker.start_run(
            f"{self.market}_stacked_{datetime.now(EST).strftime('%Y%m%d_%H%M%S')}",
            tags={"model_type": "stacked_two_head"},
        )
        if self._tracker.enabled:
            hp = TRAINING_HYPERPARAMETERS
            self._tracker.log_params(
                {
                    "market": self.market,
                    "feature_count": X_train.shape[1],
                    "train_samples": X_train.shape[0],
                    "test_samples": X_test.shape[0],
                    "num_leaves": hp.num_leaves,
                    "learning_rate": hp.learning_rate,
                    "n_estimators": hp.n_estimators,
                    "feature_fraction": hp.feature_fraction,
                    "temporal_decay_enabled": TEMPORAL_DECAY_CONFIG.enabled,
                    "dataset_hash": _dataset_hash,
                    "high_default_features": str(_high_default_features[:10]),
                    "high_default_count": len(_high_default_features),
                }
            )

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

        # ==========================
        # TEMPORAL DECAY WEIGHTING
        # ==========================
        # Recent games weighted more heavily (older data can poison the signal)
        # POINTS: tau=30 days (usage changes fast)
        # REBOUNDS: tau=45 days (more stable)
        temporal_weights_train = None
        temporal_weights_test = None

        if TEMPORAL_DECAY_CONFIG.enabled and game_dates_train is not None:
            tau = TEMPORAL_DECAY_CONFIG.get_tau(self.market)
            max_date = game_dates_train.max()

            # Calculate days ago
            days_ago_train = (max_date - game_dates_train).dt.days.values
            days_ago_test = (
                (max_date - game_dates_test).dt.days.values if game_dates_test is not None else None
            )

            # Exponential decay with floor
            temporal_weights_train = np.maximum(
                np.exp(-days_ago_train / tau), TEMPORAL_DECAY_CONFIG.min_weight
            )

            if days_ago_test is not None:
                temporal_weights_test = np.maximum(
                    np.exp(-days_ago_test / tau), TEMPORAL_DECAY_CONFIG.min_weight
                )

            logger.info(
                "Temporal decay weighting enabled",
                extra={
                    "market": self.market,
                    "tau": tau,
                    "mean_weight": round(temporal_weights_train.mean(), 3),
                    "min_weight": round(temporal_weights_train.min(), 3),
                    "max_weight": round(temporal_weights_train.max(), 3),
                    "oldest_days": int(days_ago_train.max()),
                    "newest_days": int(days_ago_train.min()),
                },
            )
        else:
            if not TEMPORAL_DECAY_CONFIG.enabled:
                logger.info("Temporal decay disabled in config")
            elif game_dates_train is None:
                logger.warning("Temporal decay enabled but game_dates not provided - skipping")

        # Train regressor using sklearn API (hyperparameters from centralized config)
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
            y_value_train,
            sample_weight=temporal_weights_train,
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
        # STEP 2: Augment with Expected Diff (out-of-fold for training)
        # ==========================
        # CRITICAL: Using in-sample regressor predictions for expected_diff_train
        # causes classifier early-stopping at ~5 trees. The regressor's training
        # predictions are overfit (RMSE 5.4 train vs 6.2 test), making
        # expected_diff_train ≈ actual - line. The classifier learns
        # "expected_diff > 0 → OVER" immediately and finds nothing more to improve.
        #
        # Fix: compute expected_diff_train via 5-fold CV so each sample's prediction
        # comes from a regressor that never saw it. This gives expected_diff_train
        # honest noise levels matching test, letting the classifier build 300+ trees.
        logger.info("Computing out-of-fold expected_diff for training data")

        from sklearn.model_selection import KFold

        line_train = X_train["line"].values
        line_test = X_test["line"].values

        # Out-of-fold predictions for training set
        oof_predictions = np.zeros(len(X_train_scaled))
        kf = KFold(n_splits=5, shuffle=False)  # No shuffle: preserve temporal order

        for _fold_idx, (tr_idx, val_idx) in enumerate(kf.split(X_train_scaled)):
            fold_reg = LGBMRegressor(
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
            fold_weights = (
                temporal_weights_train[tr_idx] if temporal_weights_train is not None else None
            )
            fold_reg.fit(
                X_train_scaled.iloc[tr_idx],
                (
                    y_value_train.iloc[tr_idx]
                    if hasattr(y_value_train, "iloc")
                    else y_value_train[tr_idx]
                ),
                sample_weight=fold_weights,
                eval_set=[
                    (
                        X_train_scaled.iloc[val_idx],
                        (
                            y_value_train.iloc[val_idx]
                            if hasattr(y_value_train, "iloc")
                            else y_value_train[val_idx]
                        ),
                    )
                ],
                eval_metric="rmse",
                callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=0)],
            )
            oof_predictions[val_idx] = fold_reg.predict(X_train_scaled.iloc[val_idx])

        expected_diff_train = oof_predictions - line_train
        expected_diff_test = y_pred_test - line_test

        # Noise injection: match train variance to test variance.
        # Even with OOF, expected_diff_train is slightly cleaner than test
        # (each OOF fold's regressor still overfits its 80% slice). If the
        # classifier sees cleaner expected_diff during training, it over-relies
        # on it and early-stops when the noisier test expected_diff offers no
        # further gradient. Injecting noise forces the classifier to learn
        # from other features too, building more trees with distributed signal.
        train_std = expected_diff_train.std()
        test_std = expected_diff_test.std()
        if train_std < test_std:
            noise_std = np.sqrt(test_std**2 - train_std**2)
            rng = np.random.RandomState(hp.random_state)
            expected_diff_train = expected_diff_train + rng.normal(
                0, noise_std, len(expected_diff_train)
            )
            logger.info(
                "Noise injected into expected_diff_train to match test variance",
                extra={
                    "original_std": round(train_std, 3),
                    "target_std": round(test_std, 3),
                    "noise_std": round(noise_std, 3),
                    "final_std": round(expected_diff_train.std(), 3),
                },
            )

        logger.info(
            "Out-of-fold expected_diff stats",
            extra={
                "train_mean": round(expected_diff_train.mean(), 2),
                "train_std": round(expected_diff_train.std(), 2),
                "test_mean": round(expected_diff_test.mean(), 2),
                "test_std": round(expected_diff_test.std(), 2),
                "train_test_std_ratio": round(
                    expected_diff_train.std() / max(expected_diff_test.std(), 0.01), 2
                ),
            },
        )

        # Classifier sees ALL features + expected_diff (matching XL architecture).
        # The XL production model (AUC 0.765) gave the classifier full visibility
        # into player rolling stats, team context, and book features. Restricting
        # to line/book/BP only dropped AUC to 0.557-0.680 because the classifier
        # lost cross-feature interactions (e.g. high usage + weak defense + low line).
        X_train_augmented = X_train_scaled.copy()
        X_train_augmented["expected_diff"] = expected_diff_train

        X_test_augmented = X_test_scaled.copy()
        X_test_augmented["expected_diff"] = expected_diff_test

        self.classifier_feature_names = list(X_train_augmented.columns)
        logger.info(
            "Classifier features",
            extra={
                "classifier_features": len(self.classifier_feature_names),
            },
        )

        # ==========================
        # Enhanced Class Balancing + Temporal Decay
        # ==========================
        class_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_binary_train)
        weight_dict = {0: class_weights[0], 1: class_weights[1]}

        # Create sample weights: class balance * temporal decay
        class_sample_weights = np.array([weight_dict[int(y)] for y in y_binary_train])

        if temporal_weights_train is not None:
            # Combine class weights with temporal weights (multiplicative)
            sample_weights = class_sample_weights * temporal_weights_train
            logger.info(
                "Combined class + temporal weighting",
                extra={
                    "under_weight": round(weight_dict[0], 3),
                    "over_weight": round(weight_dict[1], 3),
                    "mean_combined_weight": round(sample_weights.mean(), 3),
                },
            )
        else:
            sample_weights = class_sample_weights
            logger.info(
                "Class balancing only (no temporal decay)",
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

        # Train classifier using SEPARATE hyperparams (simpler trees, faster learning)
        # The classifier needs fewer leaves + higher LR to avoid early-stopping at 13 trees
        # when feature count is high. XL (0.767 AUC) used num_leaves=31, lr=0.05 → 369 trees.
        cls_leaves = getattr(hp, "classifier_num_leaves", hp.num_leaves)
        cls_lr = getattr(hp, "classifier_learning_rate", hp.learning_rate)
        cls_n = getattr(hp, "classifier_n_estimators", hp.n_estimators)
        # ==========================
        # STEP 3b: Holdout split for calibration
        # ==========================
        # Carve 15% from training data BEFORE classifier training. The main
        # classifier never sees these samples, so its predictions on them are
        # truly out-of-sample — matching the test distribution. Platt scaling
        # on in-sample predictions failed because the probability distributions
        # differ (65% train acc vs 62% test acc → different confidence levels).
        cal_size = int(len(X_train_augmented) * 0.15)
        # Take from the end (most recent) since data is temporal-ordered
        X_cls_train = X_train_augmented.iloc[:-cal_size]
        X_cal_holdout = X_train_augmented.iloc[-cal_size:]
        y_cls_train = (
            y_binary_train.iloc[:-cal_size]
            if hasattr(y_binary_train, "iloc")
            else y_binary_train[:-cal_size]
        )
        y_cal_holdout = (
            y_binary_train.iloc[-cal_size:]
            if hasattr(y_binary_train, "iloc")
            else y_binary_train[-cal_size:]
        )
        cls_weights = sample_weights[:-cal_size] if sample_weights is not None else None

        logger.info(
            "Classifier train/cal split",
            extra={"cls_train": len(X_cls_train), "cal_holdout": len(X_cal_holdout)},
        )

        # ==========================
        # STEP 3c: Train Classifier
        # ==========================
        logger.info("HEAD 2: Training Classifier (P(actual > line) prediction)")

        logger.info(
            "Classifier hyperparams",
            extra={"num_leaves": cls_leaves, "learning_rate": cls_lr, "n_estimators": cls_n},
        )
        self.classifier = LGBMClassifier(
            objective="binary",
            boosting_type="gbdt",
            num_leaves=cls_leaves,
            learning_rate=cls_lr,
            n_estimators=cls_n,
            feature_fraction=hp.feature_fraction,
            bagging_fraction=hp.bagging_fraction,
            bagging_freq=hp.bagging_freq,
            reg_alpha=hp.lambda_l1,
            reg_lambda=hp.lambda_l2,
            min_child_samples=hp.min_child_samples,
            verbose=-1,
            random_state=hp.random_state,
        )

        cls_es = getattr(hp, "classifier_early_stopping_rounds", hp.early_stopping_rounds)
        self.classifier.fit(
            X_cls_train,
            y_cls_train,
            sample_weight=cls_weights,
            eval_set=[(X_test_augmented, y_binary_test)],
            eval_metric="binary_logloss",
            callbacks=[
                lgb.early_stopping(stopping_rounds=cls_es),
                lgb.log_evaluation(period=100),
            ],
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
        # STEP 4: Platt Scaling Calibration (holdout)
        # ==========================
        # Fit Platt on the main classifier's predictions on the calibration
        # holdout — data the classifier never trained on. These out-of-sample
        # predictions have the same noise profile as test predictions.
        logger.info("Calibrating classifier probabilities (Platt scaling on holdout)")

        y_prob_cal = self.classifier.predict_proba(X_cal_holdout)[:, 1]
        y_binary_cal = y_cal_holdout.values if hasattr(y_cal_holdout, "values") else y_cal_holdout

        platt_lr = LogisticRegression(max_iter=1000, random_state=42)
        platt_lr.fit(y_prob_cal.reshape(-1, 1), y_binary_cal)
        calibrator = PlattCalibrator(platt_lr)

        logger.info(
            "Platt calibrator fitted on holdout predictions",
            extra={"calibration_samples": len(y_prob_cal)},
        )

        # Generate calibrated probabilities
        y_prob_test_cal = calibrator.transform(y_prob_test)
        y_prob_train_cal = calibrator.transform(y_prob_train)

        # Evaluate calibration
        brier_before = brier_score_loss(y_binary_test, y_prob_test)
        brier_after = brier_score_loss(y_binary_test, y_prob_test_cal)
        auc_cal = roc_auc_score(y_binary_test, y_prob_test_cal)

        # Quality gate — skipping calibration is expected when the classifier
        # is well-calibrated (train/test accuracy gap < 5%). LightGBM with
        # binary_logloss already optimizes calibration directly.
        if brier_after >= brier_before:
            logger.info(
                "Raw probabilities already well-calibrated — Platt scaling skipped",
                extra={
                    "brier_before": round(brier_before, 5),
                    "brier_after": round(brier_after, 5),
                },
            )
            y_prob_test_cal = y_prob_test
            y_prob_train_cal = y_prob_train
            brier_after = brier_before
            auc_cal = roc_auc_score(y_binary_test, y_prob_test)
            self.calibrator = None
        else:
            self.calibrator = calibrator

        logger.info(
            "Calibration results",
            extra={
                "brier_before": round(brier_before, 5),
                "brier_after": round(brier_after, 5),
                "calibration_improvement": round(brier_before - brier_after, 5),
                "auc_calibrated": round(auc_cal, 4),
                "calibration_applied": self.calibrator is not None,
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
        scale_factor = (
            BLEND_WEIGHTS.residual_scale_factor if self.market in ["POINTS", "ASSISTS"] else 2.0
        )

        prob_from_residual_train = expit(residual_train / scale_factor)
        prob_from_residual_test = expit(residual_test / scale_factor)

        # Blend weights from centralized config
        blend_weight_cls = BLEND_WEIGHTS.classifier_weight
        blend_weight_res = BLEND_WEIGHTS.residual_weight

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

        metrics = {
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

        # Training duration (Pillar 5: Cost Optimization)
        _train_duration_s = round(_time.monotonic() - _train_start, 1)

        # Log all metrics to MLflow
        if self._tracker.enabled:
            self._tracker.log_metrics(
                {
                    "regressor_rmse_train": rmse_train,
                    "regressor_rmse_test": rmse_test,
                    "regressor_mae_test": mae_test,
                    "regressor_r2_test": r2_test,
                    "classifier_acc_train": acc_train,
                    "classifier_acc_test": acc_test,
                    "classifier_auc_test": auc_test,
                    "classifier_auc_calibrated": auc_cal,
                    "classifier_auc_blended": auc_blend,
                    "classifier_logloss_test": logloss_test,
                    "calibration_brier_before": brier_before,
                    "calibration_brier_after": brier_after,
                    "blend_brier": brier_blend,
                    "blend_classifier_weight": blend_weight_cls,
                    "blend_residual_weight": blend_weight_res,
                    "regressor_n_trees": self.regressor.n_estimators_,
                    "classifier_n_trees": self.classifier.n_estimators_,
                    "training_duration_seconds": _train_duration_s,
                    "feature_default_rate_mean": round(float(_default_rate.mean()), 4),
                    "high_default_feature_count": len(_high_default_features),
                }
            )

            # Feature importance for both heads
            self._tracker.log_feature_importance(
                feature_names=list(X_train_scaled.columns),
                importances=list(self.regressor.feature_importances_),
                head="regressor",
            )
            self._tracker.log_feature_importance(
                feature_names=list(X_train_augmented.columns),
                importances=list(self.classifier.feature_importances_),
                head="classifier",
            )

        # Store integrity data for reproducibility
        metrics["integrity"] = {
            "dataset_hash": _dataset_hash,
            "training_duration_seconds": _train_duration_s,
            "high_default_features": _high_default_features,
            "feature_default_rate_mean": round(float(_default_rate.mean()), 4),
        }

        return metrics

    def save(self, output_dir: str, metrics: dict, model_version: str = "v3"):
        """
        Save both models and metadata

        Files saved:
        - {market}_{version}_regressor.pkl
        - {market}_{version}_classifier.pkl
        - {market}_{version}_calibrator.pkl
        - {market}_{version}_imputer.pkl
        - {market}_{version}_scaler.pkl
        - {market}_{version}_features.pkl
        - {market}_{version}_metadata.json
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        prefix = f"{self.market.lower()}_{model_version}"

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

        # Save classifier feature names (reduced set)
        cls_feat_names = getattr(self, "classifier_feature_names", self.feature_names)
        with open(output_path / f"{prefix}_classifier_features.pkl", "wb") as f:
            pickle.dump(cls_feat_names, f)

        # Save metadata
        metadata = {
            "market": self.market,
            "trained_date": datetime.now(EST).strftime("%Y-%m-%d %H:%M:%S"),
            "architecture": "stacked_two_head_calibrated_blended",
            "features": {"count": len(self.feature_names), "names": self.feature_names},
            "classifier_features": {
                "count": len(cls_feat_names),
                "names": cls_feat_names,
            },
            "calibration_applied": self.calibrator is not None,
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

        # Log model path for lineage and end the MLflow run
        if hasattr(self, "_tracker"):
            self._tracker.log_params(
                {
                    "model_version": model_version,
                    "output_dir": str(output_path),
                }
            )
            self._tracker.log_model_path(str(output_path / f"{prefix}_regressor.pkl"))
            self._tracker.end_run()


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
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Use walk-forward cross-validation instead of single split",
    )

    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of CV folds for walk-forward validation (default: 5)",
    )

    parser.add_argument(
        "--cv-test-months",
        type=int,
        default=2,
        help="Months in each test period for walk-forward CV (default: 2)",
    )

    parser.add_argument(
        "--cv-min-train",
        type=int,
        default=10000,
        help="Minimum training samples per fold for walk-forward CV (default: 10000)",
    )

    parser.add_argument(
        "--model-version",
        default="v3",
        help="Model version prefix for saved files (default: v3). Use v4 for new features.",
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

    if args.track and not MLFLOW_AVAILABLE:
        logger.warning(
            "MLflow tracking requested but not available. Install with: pip install mlflow"
        )

    try:
        # Initialize model
        model = StackedMarketModel(market=args.market)

        # Load data
        df = model.load_data(str(data_path))

        # ========================================
        # WALK-FORWARD CROSS-VALIDATION MODE
        # ========================================
        if args.walk_forward:
            print(f"\n{'=' * 60}")
            print("WALK-FORWARD CROSS-VALIDATION")
            print(f"{'=' * 60}")

            # Verify game_date column exists
            if "game_date" not in df.columns:
                logger.error("Walk-forward CV requires 'game_date' column in dataset")
                return 1

            # Generate walk-forward splits
            splits = walk_forward_cv(
                df,
                n_splits=args.cv_folds,
                min_train_size=args.cv_min_train,
                test_size_months=args.cv_test_months,
            )

            if len(splits) == 0:
                logger.error(
                    "No valid walk-forward splits generated. "
                    "Try reducing --cv-min-train or --cv-folds"
                )
                return 1

            cv_metrics: List[Dict] = []

            for fold_idx, (train_df, test_df) in enumerate(splits):
                print(f"\n--- Fold {fold_idx + 1}/{len(splits)} ---")
                print(
                    f"Train: {train_df['game_date'].min().date()} to "
                    f"{train_df['game_date'].max().date()} ({len(train_df):,} rows)"
                )
                print(
                    f"Test:  {test_df['game_date'].min().date()} to "
                    f"{test_df['game_date'].max().date()} ({len(test_df):,} rows)"
                )

                # Prepare features for this fold
                X_train, y_value_train, y_binary_train, y_residual_train, metadata_train = (
                    model.prepare_features(train_df)
                )
                X_test, y_value_test, y_binary_test, y_residual_test, metadata_test = (
                    model.prepare_features(test_df)
                )

                # Extract game_dates for temporal weighting
                game_dates_train = (
                    metadata_train["game_date"] if "game_date" in metadata_train.columns else None
                )
                game_dates_test = (
                    metadata_test["game_date"] if "game_date" in metadata_test.columns else None
                )

                # Train model on this fold
                fold_metrics = model.train(
                    X_train,
                    y_value_train,
                    y_binary_train,
                    y_residual_train,
                    X_test,
                    y_value_test,
                    y_binary_test,
                    y_residual_test,
                    game_dates_train=game_dates_train,
                    game_dates_test=game_dates_test,
                )

                # Collect metrics for this fold
                cv_metrics.append(
                    {
                        "fold": fold_idx + 1,
                        "train_start": str(train_df["game_date"].min().date()),
                        "train_end": str(train_df["game_date"].max().date()),
                        "test_start": str(test_df["game_date"].min().date()),
                        "test_end": str(test_df["game_date"].max().date()),
                        "train_size": len(train_df),
                        "test_size": len(test_df),
                        "rmse_test": fold_metrics["regressor"]["rmse_test"],
                        "mae_test": fold_metrics["regressor"]["mae_test"],
                        "r2_test": fold_metrics["regressor"]["r2_test"],
                        "auc_test": fold_metrics["classifier"]["auc_test"],
                        "auc_calibrated": fold_metrics["classifier"]["auc_calibrated"],
                        "auc_blended": fold_metrics["classifier"]["auc_blended"],
                        "accuracy_test": fold_metrics["classifier"]["acc_test"],
                        "brier_blended": fold_metrics["classifier"]["brier_blended"],
                    }
                )

                print(f"  AUC (blended): {fold_metrics['classifier']['auc_blended']:.4f}")
                print(f"  Accuracy:      {fold_metrics['classifier']['acc_test']:.4f}")
                print(f"  RMSE:          {fold_metrics['regressor']['rmse_test']:.3f}")

            # ========================================
            # WALK-FORWARD CV SUMMARY
            # ========================================
            print(f"\n{'=' * 60}")
            print("WALK-FORWARD CV SUMMARY")
            print(f"{'=' * 60}")

            # Calculate aggregate statistics
            aucs = [m["auc_blended"] for m in cv_metrics]
            accs = [m["accuracy_test"] for m in cv_metrics]
            rmses = [m["rmse_test"] for m in cv_metrics]
            r2s = [m["r2_test"] for m in cv_metrics]
            briers = [m["brier_blended"] for m in cv_metrics]

            print(f"\nMetric               Mean      Std       Min       Max")
            print(f"-" * 60)
            print(
                f"AUC (blended)       {np.mean(aucs):.4f}    {np.std(aucs):.4f}    "
                f"{np.min(aucs):.4f}    {np.max(aucs):.4f}"
            )
            print(
                f"Accuracy            {np.mean(accs):.4f}    {np.std(accs):.4f}    "
                f"{np.min(accs):.4f}    {np.max(accs):.4f}"
            )
            print(
                f"RMSE                {np.mean(rmses):.3f}     {np.std(rmses):.3f}     "
                f"{np.min(rmses):.3f}     {np.max(rmses):.3f}"
            )
            print(
                f"R2                  {np.mean(r2s):.4f}    {np.std(r2s):.4f}    "
                f"{np.min(r2s):.4f}    {np.max(r2s):.4f}"
            )
            print(
                f"Brier Score         {np.mean(briers):.4f}    {np.std(briers):.4f}    "
                f"{np.min(briers):.4f}    {np.max(briers):.4f}"
            )

            # Check for concept drift (significant variance across folds)
            auc_cv = np.std(aucs) / np.mean(aucs) if np.mean(aucs) > 0 else 0
            print(f"\nCoefficient of Variation (AUC): {auc_cv:.4f}")
            if auc_cv > 0.10:
                print("  WARNING: High variance across folds - possible concept drift detected")
            elif auc_cv > 0.05:
                print("  NOTICE: Moderate variance across folds - monitor for drift")
            else:
                print("  OK: Low variance - model appears stable over time")

            # Trend analysis (is model getting worse over time?)
            if len(aucs) >= 3:
                first_half = np.mean(aucs[: len(aucs) // 2])
                second_half = np.mean(aucs[len(aucs) // 2 :])
                trend_diff = second_half - first_half
                print(f"\nTemporal Trend (AUC):")
                print(f"  First half avg:  {first_half:.4f}")
                print(f"  Second half avg: {second_half:.4f}")
                print(f"  Difference:      {trend_diff:+.4f}")
                if trend_diff < -0.02:
                    print("  WARNING: Performance degrading over time - consider retraining")
                elif trend_diff > 0.02:
                    print("  OK: Performance improving over time")
                else:
                    print("  OK: Performance stable over time")

            # Per-fold details table
            print(f"\n{'=' * 60}")
            print("PER-FOLD DETAILS")
            print(f"{'=' * 60}")
            print(f"{'Fold':<6}{'Train Period':<25}{'Test Period':<25}{'AUC':<8}{'Acc':<8}")
            print("-" * 72)
            for m in cv_metrics:
                train_period = f"{m['train_start']} to {m['train_end']}"
                test_period = f"{m['test_start']} to {m['test_end']}"
                print(
                    f"{m['fold']:<6}{train_period:<25}{test_period:<25}"
                    f"{m['auc_blended']:<8.4f}{m['accuracy_test']:<8.4f}"
                )

            # Log walk-forward CV summary to MLflow
            if MLFLOW_AVAILABLE:
                cv_tracker = ModelTracker(experiment="nba-model-cascade")
                cv_tracker.start_run(
                    f"{args.market}_walkforward_cv_{datetime.now(EST).strftime('%Y%m%d_%H%M%S')}",
                    tags={
                        "market": args.market,
                        "model_type": "stacked_two_head",
                        "validation": "walk_forward_cv",
                    },
                )
                cv_tracker.log_params(
                    {
                        "market": args.market,
                        "cv_folds": args.cv_folds,
                        "cv_test_months": args.cv_test_months,
                        "cv_min_train": args.cv_min_train,
                        "actual_folds": len(splits),
                    }
                )
                cv_tracker.log_metrics(
                    {
                        "cv_auc_mean": np.mean(aucs),
                        "cv_auc_std": np.std(aucs),
                        "cv_accuracy_mean": np.mean(accs),
                        "cv_accuracy_std": np.std(accs),
                        "cv_rmse_mean": np.mean(rmses),
                        "cv_rmse_std": np.std(rmses),
                        "cv_r2_mean": np.mean(r2s),
                        "cv_brier_mean": np.mean(briers),
                        "cv_auc_cv": auc_cv,
                    }
                )
                cv_tracker.end_run()

            logger.info(
                "Walk-forward CV completed",
                extra={
                    "market": args.market,
                    "folds": len(splits),
                    "mean_auc": round(np.mean(aucs), 4),
                    "std_auc": round(np.std(aucs), 4),
                },
            )

            # NOTE: Walk-forward CV does NOT save models (it's for validation only)
            # To train a production model, run without --walk-forward flag
            print(f"\n{'=' * 60}")
            print("NOTE: Walk-forward CV is for validation only.")
            print("No models were saved. To train a production model:")
            print(f"  python train_market.py --market {args.market} --data {args.data}")
            print(f"{'=' * 60}\n")

            return 0

        # ========================================
        # STANDARD SINGLE-SPLIT TRAINING MODE
        # ========================================
        # Prepare features
        X, y_value, y_binary, y_residual, metadata = model.prepare_features(df)

        # Split (temporal or random - temporal recommended)
        # Also split metadata to preserve game_dates for temporal weighting
        (
            X_train,
            X_test,
            y_value_train,
            y_value_test,
            y_binary_train,
            y_binary_test,
            y_residual_train,
            y_residual_test,
            metadata_train,
            metadata_test,
        ) = train_test_split(
            X,
            y_value,
            y_binary,
            y_residual,
            metadata,
            test_size=args.test_size,
            random_state=args.random_state,
            shuffle=False,  # Temporal split (no shuffle)
        )

        # Extract game_dates for temporal weighting
        game_dates_train = (
            metadata_train["game_date"] if "game_date" in metadata_train.columns else None
        )
        game_dates_test = (
            metadata_test["game_date"] if "game_date" in metadata_test.columns else None
        )

        logger.info(
            "Train/Test split",
            extra={"train_samples": len(X_train), "test_samples": len(X_test)},
        )

        # MLflow tracking is handled inside train() via ModelTracker.
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
            game_dates_train=game_dates_train,
            game_dates_test=game_dates_test,
        )

        # Save
        model.save(str(output_dir), metrics, model_version=args.model_version)

        logger.info(
            "Training completed successfully",
            extra={"market": args.market},
        )

        return 0

    except (OSError, KeyError, ValueError, TypeError) as e:
        logger.error("Training failed", extra={"error": str(e)}, exc_info=True)
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
