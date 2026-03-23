"""
API Dependencies
================
Dependency injection for FastAPI endpoints.

Provides:
- Singleton model manager for lazy loading
- Feature extractor injection
- Database connection management
"""

import logging
import os
from functools import lru_cache
from typing import Dict, Optional

import psycopg2

logger = logging.getLogger(__name__)


# ==============================================================================
# Model Manager - Singleton Pattern with Lazy Loading
# ==============================================================================


class ModelManager:
    """
    Singleton manager for XL prediction models.

    Implements lazy loading to avoid startup delay - models are loaded
    on first prediction request, not at API startup.

    Attributes:
        predictors: Dict of market -> XLPredictor instances
        feature_extractor: Shared LiveFeatureExtractorXL instance
        enabled_markets: List of markets enabled for predictions
    """

    _instance: Optional["ModelManager"] = None
    _initialized: bool = False

    def __new__(cls) -> "ModelManager":
        """Ensure only one instance exists (singleton pattern)."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the model manager (only runs once due to singleton)."""
        if ModelManager._initialized:
            return

        self.predictors: Dict = {}
        self.feature_extractor = None
        # Only POINTS and REBOUNDS are enabled (ASSISTS/THREES disabled per CLAUDE.md)
        self.enabled_markets = ["POINTS", "REBOUNDS"]
        self._models_loaded = False

        ModelManager._initialized = True
        logger.info("ModelManager initialized (models will load on first request)")

    def load_models(self) -> None:
        """
        Load all XL models for enabled markets.

        Called lazily on first prediction request.
        """
        if self._models_loaded:
            return

        logger.info("Loading XL models...")

        try:
            from nba.betting_xl.xl_predictor import XLPredictor

            for market in self.enabled_markets:
                try:
                    # Load without dynamic calibration for API (simpler, faster)
                    predictor = XLPredictor(
                        market=market,
                        use_3head=False,  # Use 2-head for stability
                        enable_book_intelligence=False,
                        enable_dynamic_calibration=False,
                        model_version="xl",
                    )
                    self.predictors[market] = predictor
                    feature_count = len(predictor.features) if predictor.features else 0
                    logger.info(f"  [OK] {market}: Loaded XL model ({feature_count} features)")
                except (psycopg2.Error, KeyError, TypeError, ValueError) as e:
                    logger.error(f"  [ERROR] {market}: Failed to load model: {e}")
                    self.predictors[market] = None

            self._models_loaded = True
            loaded_count = sum(1 for p in self.predictors.values() if p is not None)
            logger.info(f"Models loaded: {loaded_count}/{len(self.enabled_markets)}")

        except ImportError as e:
            logger.error(f"Failed to import XLPredictor: {e}")
            raise RuntimeError("XLPredictor not available") from e

    def load_feature_extractor(self) -> None:
        """
        Load the feature extractor.

        Called lazily on first prediction request.
        """
        if self.feature_extractor is not None:
            return

        logger.info("Loading feature extractor...")

        try:
            from nba.features.extract_live_features_xl import LiveFeatureExtractorXL

            self.feature_extractor = LiveFeatureExtractorXL()
            logger.info("  [OK] Feature extractor loaded")

        except (psycopg2.Error, KeyError, TypeError, ValueError) as e:
            logger.error(f"  [ERROR] Failed to load feature extractor: {e}")
            raise RuntimeError(f"Feature extractor not available: {e}") from e

    def ensure_loaded(self) -> None:
        """Ensure models and feature extractor are loaded."""
        if not self._models_loaded:
            self.load_models()
        if self.feature_extractor is None:
            self.load_feature_extractor()

    def get_predictor(self, market: str):
        """
        Get predictor for a specific market.

        Args:
            market: Market type ('POINTS', 'REBOUNDS', etc.)

        Returns:
            XLPredictor instance or None if not loaded

        Raises:
            ValueError: If market is not enabled
        """
        market = market.upper()
        if market not in self.enabled_markets:
            raise ValueError(
                f"Market '{market}' not enabled. " f"Enabled markets: {self.enabled_markets}"
            )

        self.ensure_loaded()
        return self.predictors.get(market)

    def get_feature_extractor(self):
        """
        Get the feature extractor instance.

        Returns:
            LiveFeatureExtractorXL instance
        """
        self.ensure_loaded()
        return self.feature_extractor

    def is_market_enabled(self, market: str) -> bool:
        """Check if a market is enabled for predictions."""
        return market.upper() in self.enabled_markets

    def get_model_status(self) -> Dict:
        """
        Get status of all models.

        Returns:
            Dict with model status information
        """
        status = {
            "loaded": self._models_loaded,
            "enabled_markets": self.enabled_markets,
            "models": {},
        }

        for market in self.enabled_markets:
            predictor = self.predictors.get(market)
            if predictor is not None:
                status["models"][market] = {
                    "loaded": True,
                    "feature_count": len(predictor.features) if predictor.features else 0,
                    "model_version": getattr(predictor, "model_version", "xl"),
                }
            else:
                status["models"][market] = {
                    "loaded": False,
                    "feature_count": None,
                    "model_version": None,
                }

        return status


# ==============================================================================
# Dependency Injection Functions
# ==============================================================================


@lru_cache()
def get_model_manager() -> ModelManager:
    """
    Get the singleton ModelManager instance.

    Uses lru_cache for efficient repeated access.

    Returns:
        ModelManager instance
    """
    return ModelManager()


def get_predictor(market: str):
    """
    Dependency for getting a predictor for a specific market.

    Usage:
        @app.post("/predict")
        def predict(predictor = Depends(lambda: get_predictor("POINTS"))):
            ...

    Args:
        market: Market type

    Returns:
        XLPredictor instance
    """
    manager = get_model_manager()
    return manager.get_predictor(market)


def get_feature_extractor():
    """
    Dependency for getting the feature extractor.

    Usage:
        @app.post("/predict")
        def predict(extractor = Depends(get_feature_extractor)):
            ...

    Returns:
        LiveFeatureExtractorXL instance
    """
    manager = get_model_manager()
    return manager.get_feature_extractor()


# ==============================================================================
# Database Connection Helpers
# ==============================================================================


class DatabaseConfig:
    """Database configuration — consolidated DB on port 5500, schema-based isolation."""

    # Schemas to health-check in the consolidated sportsuite database
    SCHEMAS = ("players", "games", "teams", "intelligence", "axiom", "features")

    @classmethod
    def get_all_configs(cls) -> Dict:
        """Get connection configs for all schemas in the consolidated DB."""
        from nba.config.database import get_schema_config

        return {schema: get_schema_config(schema) for schema in cls.SCHEMAS}


def check_database_connection(config: Dict) -> Dict:
    """
    Check if a database connection is available.

    Args:
        config: Database configuration dict (from get_schema_config)

    Returns:
        Dict with connection status
    """
    schema = "unknown"
    options = config.get("options", "")
    if "search_path=" in options:
        schema = options.split("search_path=")[1].split(",")[0]

    result = {
        "name": schema,
        "host": config["host"],
        "port": config["port"],
        "connected": False,
        "error": None,
    }

    try:
        conn = psycopg2.connect(**config)
        conn.close()
        result["connected"] = True
    except (psycopg2.Error, KeyError, TypeError, ValueError) as e:
        result["error"] = str(e)

    return result


def check_all_databases() -> Dict:
    """
    Check connectivity to all schemas in the consolidated database.

    Returns:
        Dict with overall status and individual schema statuses
    """
    configs = DatabaseConfig.get_all_configs()
    results = []
    connected_count = 0

    for _name, config in configs.items():
        status = check_database_connection(config)
        results.append(status)
        if status["connected"]:
            connected_count += 1

    return {
        "status": "healthy" if connected_count == len(configs) else "degraded",
        "databases": results,
        "total_connected": connected_count,
    }
