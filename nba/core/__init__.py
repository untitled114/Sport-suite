"""
NBA Props ML Core Module
========================
Core utilities for ML operations including:
- Data validation (Pydantic schemas)
- Experiment tracking (MLflow integration)
- Feature monitoring (drift detection)
- Custom exceptions
- Structured logging configuration
"""

from nba.core.exceptions import (
    APIConnectionError,
    APIError,
    APIRateLimitError,
    ConfigurationError,
    DatabaseConnectionError,
    DatabaseError,
    DataNotFoundError,
    DataValidationError,
    FeatureDriftError,
    FeatureError,
    FeatureExtractionError,
    InvalidConfigError,
    InvalidPredictionError,
    InvalidPropLineError,
    MissingConfigError,
    MissingFeatureError,
    ModelError,
    ModelLoadError,
    ModelNotFoundError,
    ModelPredictionError,
    NBAPropsError,
)
from nba.core.logging_config import add_logging_args, get_logger, setup_logging
from nba.core.schemas import (
    FeatureVector,
    MarketType,
    Prediction,
    PredictionBatch,
    PropLine,
    ValidationResult,
)

__all__ = [
    # Logging
    "setup_logging",
    "get_logger",
    "add_logging_args",
    # Schemas
    "PropLine",
    "Prediction",
    "PredictionBatch",
    "FeatureVector",
    "MarketType",
    "ValidationResult",
    # Base Exceptions
    "NBAPropsError",
    # Model Exceptions
    "ModelError",
    "ModelNotFoundError",
    "ModelLoadError",
    "ModelPredictionError",
    # Feature Exceptions
    "FeatureError",
    "FeatureExtractionError",
    "MissingFeatureError",
    "FeatureDriftError",
    # Data Validation Exceptions
    "DataValidationError",
    "InvalidPropLineError",
    "InvalidPredictionError",
    # Database Exceptions
    "DatabaseError",
    "DatabaseConnectionError",
    "DataNotFoundError",
    # API Exceptions
    "APIError",
    "APIConnectionError",
    "APIRateLimitError",
    # Configuration Exceptions
    "ConfigurationError",
    "MissingConfigError",
    "InvalidConfigError",
]
