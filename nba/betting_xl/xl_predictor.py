#!/usr/bin/env python3
"""
NBA XL Predictor Class
======================
Loads XL models and generates predictions for player props.

Returns: {player, market, prediction, p_over, edge, confidence}

Part of Phase 5: XL Betting Pipeline
"""

import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Any, List, Union
from datetime import datetime
import logging
import warnings
import sys

# Suppress sklearn feature name warnings
warnings.filterwarnings('ignore', message='.*feature names.*')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent / 'models' / 'saved_xl'
BOOK_INTELLIGENCE_DIR = Path(__file__).parent.parent / 'models' / 'saved_book_intelligence'

# Import calibrators - prefer JSONCalibrator (uses real predictions from JSON files)
sys.path.insert(0, str(Path(__file__).parent.parent / 'models'))
try:
    from json_calibrator import JSONCalibrator
    DYNAMIC_CALIBRATION_AVAILABLE = True
    CALIBRATOR_TYPE = 'json'
    logger.info("JSONCalibrator loaded - using REAL predictions from JSON files")
except ImportError:
    try:
        from dynamic_calibrator import DynamicCalibrator as JSONCalibrator
        DYNAMIC_CALIBRATION_AVAILABLE = True
        CALIBRATOR_TYPE = 'dynamic'
        logger.warning("JSONCalibrator not available, falling back to DynamicCalibrator")
    except ImportError:
        DYNAMIC_CALIBRATION_AVAILABLE = False
        CALIBRATOR_TYPE = None
        logger.warning("No calibrator available - predictions will not be dynamically adjusted")


class BookIntelligencePredictor:
    """
    Book Intelligence predictor (third head).
    Loads book intelligence model and generates predictions.
    """

    market: str
    market_lower: str
    classifier: Optional[Any]
    calibrator: Optional[Any]
    metadata: Optional[Dict[str, Any]]

    def __init__(self, market: str) -> None:
        """
        Initialize book intelligence predictor for a specific market.

        Args:
            market: 'POINTS' or 'REBOUNDS'
        """
        self.market = market.upper()
        self.market_lower = market.lower()

        # Model components
        self.classifier = None
        self.calibrator = None
        self.metadata = None

        # Load model
        self.load_model()

    def load_model(self):
        """Load book intelligence model components"""
        try:
            model_prefix = BOOK_INTELLIGENCE_DIR / f"{self.market_lower}_book_intelligence"

            with open(f"{model_prefix}_classifier.pkl", 'rb') as f:
                self.classifier = pickle.load(f)
            with open(f"{model_prefix}_calibrator.pkl", 'rb') as f:
                self.calibrator = pickle.load(f)

            # Load metadata (optional)
            try:
                import json
                with open(f"{model_prefix}_metadata.json", 'r') as f:
                    self.metadata = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                self.metadata = None

            logger.info(f"[OK] {self.market}: Loaded book intelligence model")

        except FileNotFoundError:
            logger.warning(f"[WARN]  {self.market}: Book intelligence model not found - skipping")
            raise
        except Exception as e:
            logger.error(f"[ERROR] {self.market}: Failed to load book intelligence model: {e}")
            raise

    def predict(self, book_features: Dict) -> Optional[float]:
        """
        Generate book intelligence prediction.

        Args:
            book_features: Dict with 4 book intelligence features:
                - softest_book_historical_accuracy
                - line_spread
                - books_in_agreement
                - softest_vs_consensus

        Returns:
            Calibrated P(softest line OVER will hit), or None if features missing
        """
        try:
            # Expected features
            expected_features = [
                'softest_book_historical_accuracy',
                'line_spread',
                'books_in_agreement',
                'softest_vs_consensus'
            ]

            # Check if all features present
            if not all(feat in book_features for feat in expected_features):
                logger.debug(f"Book intelligence features missing: {set(expected_features) - set(book_features.keys())}")
                return None

            # Build feature vector
            feature_vector = [book_features[feat] for feat in expected_features]
            X = pd.DataFrame([feature_vector], columns=expected_features)

            # Predict
            prob_over_raw = self.classifier.predict_proba(X)[0, 1]

            # Calibrate
            prob_over_cal = self.calibrator.transform([prob_over_raw])[0]

            return float(np.clip(prob_over_cal, 0.01, 0.99))

        except Exception as e:
            logger.error(f"Book intelligence prediction error: {e}")
            return None


class XLPredictor:
    """
    XL Predictor for NBA player props.
    Supports both 2-head (legacy) and 3-head (matchup) architectures.
    Includes optional dynamic calibration based on recent performance.
    """

    # Type annotations for instance attributes
    market: str
    market_lower: str
    use_3head: bool
    model_version: str
    enable_book_intelligence: bool
    enable_dynamic_calibration: bool
    as_of_date: Optional[datetime]
    backtest_mode: bool
    predictions_dir: Optional[str]

    # Model components
    regressor: Optional[Any]
    classifier: Optional[Any]
    calibrator: Optional[Any]
    imputer: Optional[Any]
    scaler: Optional[Any]
    features: Optional[List[str]]
    metadata: Optional[Dict[str, Any]]

    def __init__(
        self,
        market: str,
        use_3head: bool = False,
        enable_book_intelligence: bool = False,
        enable_dynamic_calibration: bool = True,
        dynamic_lookback_days: int = 21,
        as_of_date: Optional[datetime] = None,
        backtest_mode: bool = False,
        predictions_dir: Optional[str] = None,
        model_version: str = 'xl'
    ) -> None:
        """
        Initialize XL predictor for a specific market.

        Args:
            market: 'POINTS', 'REBOUNDS', 'ASSISTS', or 'THREES'
            use_3head: If True, load 3-head architecture (base + matchup + classifier)
                      If False, load 2-head architecture (base + classifier) - legacy
            enable_book_intelligence: Whether to load book intelligence model (separate from 3-head)
            enable_dynamic_calibration: Whether to apply dynamic calibration adjustments
            dynamic_lookback_days: Days to look back for dynamic calibration (7, 14, or 30)
            as_of_date: Historical date for backtesting (default: None = production mode)
            backtest_mode: If True, relaxes freshness checks (but still uses JSON calibration)
            predictions_dir: Directory to read predictions from for calibration (default: standard location)
            model_version: 'xl' for legacy models (*_xl_*.pkl), 'v3' for new models (*_market_*.pkl)
        """
        # Backtest support
        self.as_of_date = as_of_date
        self.backtest_mode = backtest_mode
        self.predictions_dir = predictions_dir
        self.market = market.upper()
        self.market_lower = market.lower()
        self.use_3head = use_3head
        self.model_version = model_version  # 'xl' for legacy, 'v3' for new models
        self.enable_book_intelligence = enable_book_intelligence
        self.enable_dynamic_calibration = enable_dynamic_calibration and DYNAMIC_CALIBRATION_AVAILABLE

        # Model components (2-head legacy)
        self.regressor = None
        self.classifier = None
        self.calibrator = None
        self.imputer = None
        self.scaler = None
        self.features = None
        self.metadata = None

        # 3-head matchup architecture components
        self.matchup_head = None
        self.matchup_imputer = None
        self.matchup_scaler = None
        self.matchup_features = None
        self.enhanced_classifier = None
        self.enhanced_calibrator = None
        self.enhanced_imputer = None
        self.enhanced_scaler = None
        self.enhanced_features = None
        self.blend_config = None

        # Book intelligence (separate optional head)
        self.book_intelligence_predictor = None

        # Dynamic calibration (now uses JSONCalibrator with real predictions)
        # IMPORTANT: Always use JSON files for calibration (never db_only which uses fake p_over)
        # In backtest mode, pass predictions_dir to read from backtest output directory
        self.dynamic_calibrator = None
        if self.enable_dynamic_calibration:
            try:
                self.dynamic_calibrator = JSONCalibrator(
                    market=self.market,
                    lookback_days=dynamic_lookback_days,
                    predictions_dir=self.predictions_dir,  # Custom dir for backtest, None for production
                    as_of_date=self.as_of_date,            # Temporal boundary for lookback
                    db_only=False,                         # NEVER use db_only (fake p_over)
                    model_version=self.model_version       # Calibrate per model (xl vs v3)
                )
                # Pre-fetch metrics to warm up the cache
                self.dynamic_calibrator.get_recent_performance()
                cal_type = CALIBRATOR_TYPE or 'unknown'
                dir_str = self.predictions_dir or "default"
                logger.info(f"   {self.market}: Calibration enabled ({cal_type}, lookback={dynamic_lookback_days} days, dir={dir_str})")
            except Exception as e:
                logger.warning(f"   {self.market}: Dynamic calibration failed to initialize: {e}")
                self.dynamic_calibrator = None
                self.enable_dynamic_calibration = False

        # Load models
        if self.use_3head:
            self.load_3head_models()
        else:
            self.load_model()  # Legacy 2-head

        if self.enable_book_intelligence:
            self.load_book_intelligence()

    def load_model(self) -> None:
        """Load all 6 XL model components."""
        try:
            # Determine model prefix based on version
            # 'xl' = legacy (*_xl_*.pkl), 'v3' = new (*_market_*.pkl)
            if self.model_version == 'v3':
                model_prefix = MODELS_DIR / f"{self.market_lower}_market"
                version_label = "V3"
            else:
                model_prefix = MODELS_DIR / f"{self.market_lower}_xl"
                version_label = "XL"

            with open(f"{model_prefix}_regressor.pkl", 'rb') as f:
                self.regressor = pickle.load(f)
            with open(f"{model_prefix}_classifier.pkl", 'rb') as f:
                self.classifier = pickle.load(f)
            with open(f"{model_prefix}_calibrator.pkl", 'rb') as f:
                self.calibrator = pickle.load(f)
            with open(f"{model_prefix}_imputer.pkl", 'rb') as f:
                self.imputer = pickle.load(f)
            with open(f"{model_prefix}_scaler.pkl", 'rb') as f:
                self.scaler = pickle.load(f)
            with open(f"{model_prefix}_features.pkl", 'rb') as f:
                self.features = pickle.load(f)

            # Load metadata (optional)
            try:
                import json
                with open(f"{model_prefix}_metadata.json", 'r') as f:
                    self.metadata = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                self.metadata = None

            logger.info(f"[OK] {self.market}: Loaded {version_label} model ({len(self.features)} features)")

        except Exception as e:
            logger.error(f"[ERROR] {self.market}: Failed to load model: {e}")
            raise

    def load_3head_models(self):
        """Load 3-head matchup architecture: base regressor + matchup head + enhanced classifier"""
        try:
            import json
            model_prefix = MODELS_DIR / f"{self.market_lower}"

            # HEAD 1: Base Regressor (uses 102 base features)
            with open(f"{model_prefix}_xl_regressor.pkl", 'rb') as f:
                self.regressor = pickle.load(f)
            with open(f"{model_prefix}_xl_imputer.pkl", 'rb') as f:
                self.imputer = pickle.load(f)
            with open(f"{model_prefix}_xl_scaler.pkl", 'rb') as f:
                self.scaler = pickle.load(f)
            with open(f"{model_prefix}_xl_features.pkl", 'rb') as f:
                self.features = pickle.load(f)

            # HEAD 2: Matchup Head (uses 32 H2H features)
            with open(f"{model_prefix}_matchup_head.pkl", 'rb') as f:
                self.matchup_head = pickle.load(f)
            with open(f"{model_prefix}_matchup_imputer.pkl", 'rb') as f:
                self.matchup_imputer = pickle.load(f)
            with open(f"{model_prefix}_matchup_scaler.pkl", 'rb') as f:
                self.matchup_scaler = pickle.load(f)
            with open(f"{model_prefix}_matchup_features.pkl", 'rb') as f:
                self.matchup_features = pickle.load(f)

            # HEAD 3: Enhanced Classifier (uses all 142 features + expected_diff)
            with open(f"{model_prefix}_matchup_classifier.pkl", 'rb') as f:
                self.enhanced_classifier = pickle.load(f)
            with open(f"{model_prefix}_matchup_calibrator.pkl", 'rb') as f:
                self.enhanced_calibrator = pickle.load(f)
            with open(f"{model_prefix}_matchup_classifier_imputer.pkl", 'rb') as f:
                self.enhanced_imputer = pickle.load(f)
            with open(f"{model_prefix}_matchup_classifier_scaler.pkl", 'rb') as f:
                self.enhanced_scaler = pickle.load(f)
            with open(f"{model_prefix}_matchup_classifier_features.pkl", 'rb') as f:
                self.enhanced_features = pickle.load(f)

            # Load blend config from metadata
            try:
                with open(f"{model_prefix}_matchup_classifier_metadata.json", 'r') as f:
                    self.metadata = json.load(f)
                    self.blend_config = self.metadata.get('blend_config', None)
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                self.metadata = None
                self.blend_config = None

            logger.info(f"[OK] {self.market}: Loaded 3-head matchup model")
            logger.info(f"   Base features: {len(self.features)}")
            logger.info(f"   Matchup features: {len(self.matchup_features)}")
            logger.info(f"   Classifier features: {len(self.enhanced_features)}")

        except FileNotFoundError as e:
            logger.error(f"[ERROR] {self.market}: 3-head model files not found: {e}")
            logger.info(f"   Falling back to 2-head model...")
            self.use_3head = False
            self.load_model()
        except Exception as e:
            logger.error(f"[ERROR] {self.market}: Failed to load 3-head model: {e}")
            raise

    def load_book_intelligence(self):
        """Load book intelligence model (third head) if available"""
        try:
            # Only POINTS and REBOUNDS have book intelligence models
            if self.market not in ['POINTS', 'REBOUNDS']:
                logger.info(f"   {self.market}: Book intelligence not available for this market")
                self.enable_book_intelligence = False
                return

            self.book_intelligence_predictor = BookIntelligencePredictor(self.market)
            logger.info(f"   {self.market}: Book intelligence enabled (ensemble mode)")

        except FileNotFoundError:
            logger.warning(f"   {self.market}: Book intelligence model not found - using base model only")
            self.enable_book_intelligence = False
        except Exception as e:
            logger.error(f"   {self.market}: Failed to load book intelligence: {e}")
            self.enable_book_intelligence = False

    def predict(
        self,
        features_dict: Dict[str, Any],
        line: float,
        book_features: Optional[Dict[str, float]] = None,
        player_name: Optional[str] = None,
        game_date: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Generate prediction using loaded model architecture (2-head or 3-head).

        Args:
            features_dict: Dict of features from LiveFeatureExtractorXL
                          - 102 features for 2-head model
                          - 142 features for 3-head model
            line: Sportsbook line value
            book_features: Optional dict with 4 book intelligence features (for ensemble)
            player_name: Optional player name (for dynamic calibration logging)
            game_date: Optional game date (for dynamic calibration logging)

        Returns:
            Dict with:
            - predicted_value: Model's predicted stat value
            - p_over: Final probability of OVER
            - p_over_base: Base model probability (without book intelligence)
            - p_over_book: Book intelligence probability (if available)
            - edge: predicted_value - line
            - confidence: 'HIGH', 'MEDIUM', or 'LOW'
            - dynamic_adjustment: Dict with adjustment details (if enabled)
        """
        try:
            # Route to appropriate prediction pipeline
            if self.use_3head:
                return self._predict_3head(features_dict, line, book_features, player_name, game_date)
            else:
                return self._predict_2head(features_dict, line, book_features, player_name, game_date)
        except Exception as e:
            logger.error(f"Prediction error for {self.market}: {e}")
            return None

    def _predict_2head(self, features_dict: Dict, line: float, book_features: Optional[Dict] = None,
                        player_name: str = None, game_date: str = None) -> Dict:
        """Legacy 2-head prediction pipeline (regressor + classifier)"""
        try:
            # Ensure line is float (PostgreSQL NUMERIC returns Decimal)
            line = float(line)

            # Build feature vector in model's expected order
            # CRITICAL: For regressor, we need to handle expected_diff specially
            # since it can only be computed AFTER the regressor runs
            feature_vector = []
            for feat in self.features:
                if feat == 'expected_diff':
                    # Use 0 as placeholder for first pass (will be corrected for classifier)
                    feature_vector.append(0.0)
                else:
                    feature_vector.append(features_dict.get(feat, np.nan))

            X = pd.DataFrame([feature_vector], columns=self.features)

            # Preprocess
            X_imputed = self.imputer.transform(X)
            X_scaled = self.scaler.transform(X_imputed)

            # Stage 1: Regressor predicts stat value
            # Note: Regressor gets expected_diff=0, but if properly trained it shouldn't depend heavily on it
            predicted_value = self.regressor.predict(X_scaled)[0]

            # Stage 2: Classifier predicts P(actual > line)
            # CRITICAL: Now compute and inject the REAL expected_diff
            expected_diff = predicted_value - line
            X_cls_df = pd.DataFrame(X_scaled, columns=self.features)
            X_cls_df['expected_diff'] = expected_diff
            X_cls = X_cls_df.values

            prob_over_raw = self.classifier.predict_proba(X_cls)[0, 1]

            # Stage 3: Calibrate probability (Isotonic Regression)
            prob_over = self.calibrator.transform([prob_over_raw])[0]

            # Blend (60% classifier, 40% residual signal)
            residual_signal = 1 / (1 + np.exp(-expected_diff / 5.0))
            prob_over_blended = 0.6 * prob_over + 0.4 * residual_signal
            prob_over_base = np.clip(prob_over_blended, 0.01, 0.99)

            # Stage 4: Dynamic calibration adjustment (if enabled)
            dynamic_adjustment = None
            if self.enable_dynamic_calibration and self.dynamic_calibrator:
                try:
                    adjustment_result = self.dynamic_calibrator.apply_adjustment(
                        raw_prob=prob_over_base,
                        player_name=player_name,
                        game_date=game_date,
                        line=line
                    )
                    prob_over_base = adjustment_result['adjusted_prob']
                    dynamic_adjustment = {
                        'adjustment': adjustment_result['adjustment_applied'],
                        'reason': adjustment_result['reason'],
                        'was_adjusted': adjustment_result['was_adjusted']
                    }
                except Exception as e:
                    logger.debug(f"Dynamic calibration failed: {e}")

            # Book intelligence ensemble (third head)
            prob_over_book = None
            if self.enable_book_intelligence and self.book_intelligence_predictor:
                try:
                    # Extract book intelligence features from features_dict if not provided separately
                    if book_features is None:
                        book_features = {
                            'softest_book_historical_accuracy': features_dict.get('softest_book_historical_accuracy'),
                            'line_spread': features_dict.get('line_spread'),
                            'books_in_agreement': features_dict.get('books_in_agreement'),
                            'softest_vs_consensus': features_dict.get('softest_vs_consensus')
                        }

                    # Only predict if all features are present
                    if all(v is not None for v in book_features.values()):
                        prob_over_book = self.book_intelligence_predictor.predict(book_features)
                    else:
                        logger.debug(f"Book intelligence features incomplete: {book_features}")
                except Exception as e:
                    logger.debug(f"Book intelligence prediction failed: {e}")
                    prob_over_book = None

            # Ensemble blend (70% base, 30% book intelligence)
            if prob_over_book is not None:
                prob_over_final = 0.70 * prob_over_base + 0.30 * prob_over_book
                prob_over_final = np.clip(prob_over_final, 0.01, 0.99)
            else:
                prob_over_final = prob_over_base

            # Calculate edge
            edge = predicted_value - line

            # Determine confidence level
            if abs(edge) >= 5.0:
                confidence = 'HIGH'
            elif abs(edge) >= 3.0:
                confidence = 'MEDIUM'
            else:
                confidence = 'LOW'

            result = {
                'predicted_value': float(predicted_value),
                'p_over': float(prob_over_final),
                'p_over_base': float(prob_over_base),
                'edge': float(edge),
                'confidence': confidence
            }

            # Add book intelligence prob if available
            if prob_over_book is not None:
                result['p_over_book'] = float(prob_over_book)

            # Add dynamic calibration info if available
            if dynamic_adjustment is not None:
                result['dynamic_adjustment'] = dynamic_adjustment

            return result

        except Exception as e:
            logger.error(f"2-head prediction error for {self.market}: {e}")
            return None

    def _predict_3head(self, features_dict: Dict, line: float, book_features: Optional[Dict] = None,
                       player_name: str = None, game_date: str = None) -> Dict:
        """3-head matchup prediction pipeline (base + matchup + enhanced classifier)"""
        try:
            # Ensure line is float (PostgreSQL NUMERIC returns Decimal)
            line = float(line)

            # STAGE 1: Base regressor (uses 102 base features)
            base_vector = [features_dict.get(feat, np.nan) for feat in self.features]
            X_base = pd.DataFrame([base_vector], columns=self.features)
            X_base_imputed = self.imputer.transform(X_base)
            X_base_scaled = self.scaler.transform(X_base_imputed)
            base_prediction = self.regressor.predict(X_base_scaled)[0]

            # STAGE 2: Matchup head (uses 32 H2H features)
            matchup_vector = [features_dict.get(feat, np.nan) for feat in self.matchup_features]
            X_matchup = pd.DataFrame([matchup_vector], columns=self.matchup_features)
            X_matchup_imputed = self.matchup_imputer.transform(X_matchup)
            X_matchup_scaled = self.matchup_scaler.transform(X_matchup_imputed)
            matchup_adjustment = self.matchup_head.predict(X_matchup_scaled)[0]

            # Combine predictions
            predicted_value = base_prediction + matchup_adjustment
            expected_diff = predicted_value - line

            # STAGE 3: Enhanced classifier (uses all 142 features + expected_diff)
            # Add expected_diff to features
            features_with_diff = features_dict.copy()
            features_with_diff['expected_diff'] = expected_diff

            classifier_vector = [features_with_diff.get(feat, np.nan) for feat in self.enhanced_features]
            X_classifier = pd.DataFrame([classifier_vector], columns=self.enhanced_features)
            X_classifier_imputed = self.enhanced_imputer.transform(X_classifier)
            X_classifier_scaled = self.enhanced_scaler.transform(X_classifier_imputed)

            # Predict probability
            prob_over_raw = self.enhanced_classifier.predict_proba(X_classifier_scaled)[0, 1]
            prob_over_calibrated = self.enhanced_calibrator.transform([prob_over_raw])[0]

            # Optional blending (if configured in metadata)
            if self.blend_config:
                residual_scale = self.blend_config.get('residual_scale_factor', 5.0)
                classifier_weight = self.blend_config.get('classifier_weight', 0.6)
                residual_weight = self.blend_config.get('residual_weight', 0.4)

                # Sigmoid of expected_diff
                residual_prob = 1 / (1 + np.exp(-expected_diff / residual_scale))

                # Weighted blend
                prob_over_base = classifier_weight * prob_over_calibrated + residual_weight * residual_prob
                prob_over_base = np.clip(prob_over_base, 0.01, 0.99)
            else:
                prob_over_base = np.clip(prob_over_calibrated, 0.01, 0.99)

            # Stage 4: Dynamic calibration adjustment (if enabled)
            dynamic_adjustment = None
            if self.enable_dynamic_calibration and self.dynamic_calibrator:
                try:
                    adjustment_result = self.dynamic_calibrator.apply_adjustment(
                        raw_prob=prob_over_base,
                        player_name=player_name,
                        game_date=game_date,
                        line=line
                    )
                    prob_over_base = adjustment_result['adjusted_prob']
                    dynamic_adjustment = {
                        'adjustment': adjustment_result['adjustment_applied'],
                        'reason': adjustment_result['reason'],
                        'was_adjusted': adjustment_result['was_adjusted']
                    }
                except Exception as e:
                    logger.debug(f"Dynamic calibration failed: {e}")

            # Book intelligence ensemble (optional separate head)
            prob_over_book = None
            if self.enable_book_intelligence and self.book_intelligence_predictor:
                try:
                    if book_features is None:
                        book_features = {
                            'softest_book_historical_accuracy': features_dict.get('softest_book_historical_accuracy'),
                            'line_spread': features_dict.get('line_spread'),
                            'books_in_agreement': features_dict.get('books_in_agreement'),
                            'softest_vs_consensus': features_dict.get('softest_vs_consensus')
                        }

                    if all(v is not None for v in book_features.values()):
                        prob_over_book = self.book_intelligence_predictor.predict(book_features)
                except Exception as e:
                    logger.debug(f"Book intelligence prediction failed: {e}")
                    prob_over_book = None

            # Final ensemble blend
            if prob_over_book is not None:
                prob_over_final = 0.70 * prob_over_base + 0.30 * prob_over_book
                prob_over_final = np.clip(prob_over_final, 0.01, 0.99)
            else:
                prob_over_final = prob_over_base

            # Calculate edge
            edge = predicted_value - line

            # Determine confidence level
            if abs(edge) >= 5.0:
                confidence = 'HIGH'
            elif abs(edge) >= 3.0:
                confidence = 'MEDIUM'
            else:
                confidence = 'LOW'

            result = {
                'predicted_value': float(predicted_value),
                'base_prediction': float(base_prediction),
                'matchup_adjustment': float(matchup_adjustment),
                'p_over': float(prob_over_final),
                'p_over_base': float(prob_over_base),
                'edge': float(edge),
                'confidence': confidence
            }

            # Add book intelligence prob if available
            if prob_over_book is not None:
                result['p_over_book'] = float(prob_over_book)

            # Add dynamic calibration info if available
            if dynamic_adjustment is not None:
                result['dynamic_adjustment'] = dynamic_adjustment

            return result

        except Exception as e:
            logger.error(f"3-head prediction error for {self.market}: {e}")
            return None

    def get_dynamic_calibration_status(self) -> Optional[Dict]:
        """
        Get current dynamic calibration status and metrics.

        Returns:
            Dict with calibration metrics, or None if not enabled
        """
        if not self.enable_dynamic_calibration or not self.dynamic_calibrator:
            return None

        return self.dynamic_calibrator.get_recent_performance()

    def get_adjustment_summary(self) -> Optional[Dict]:
        """
        Get summary of adjustments applied today.

        Returns:
            Dict with adjustment statistics, or None if not enabled
        """
        if not self.enable_dynamic_calibration or not self.dynamic_calibrator:
            return None

        return self.dynamic_calibrator.get_adjustment_summary()


if __name__ == '__main__':
    # Test XL predictor
    print("Testing XL Predictor...")

    # Test loading all markets
    for market in ['POINTS', 'REBOUNDS', 'ASSISTS', 'THREES']:
        try:
            predictor = XLPredictor(market)
            print(f"  {market}: {len(predictor.features)} features loaded (base model)")
        except Exception as e:
            print(f"  {market}: {e}")

    # Test loading with book intelligence
    print("\nTesting XL Predictor with Book Intelligence...")
    for market in ['POINTS', 'REBOUNDS']:
        try:
            predictor = XLPredictor(market, enable_book_intelligence=True)
            if predictor.book_intelligence_predictor:
                print(f"  {market}: Book intelligence enabled (ensemble mode)")
            else:
                print(f"  {market}: Book intelligence not available (base mode)")
        except Exception as e:
            print(f"  {market}: {e}")

    # Test dynamic calibration
    print("\nTesting Dynamic Calibration...")
    for market in ['POINTS', 'REBOUNDS', 'ASSISTS', 'THREES']:
        try:
            predictor = XLPredictor(market, enable_dynamic_calibration=True)
            if predictor.dynamic_calibrator:
                status = predictor.get_dynamic_calibration_status()
                if status and status.get('status') == 'ok':
                    bias = status.get('bias', 0) * 100
                    win_rate = status.get('win_rate', 0) * 100
                    print(f"  {market}: Dynamic calibration enabled "
                          f"(WR: {win_rate:.1f}%, Bias: {bias:+.1f}%)")
                else:
                    print(f"  {market}: Dynamic calibration enabled (insufficient data)")
            else:
                print(f"  {market}: Dynamic calibration not available")
        except Exception as e:
            print(f"  {market}: {e}")
