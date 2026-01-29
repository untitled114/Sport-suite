"""
Custom Exceptions for NBA Props ML System
=========================================
Centralized exception definitions for better error handling and debugging.

Usage:
    from nba.core.exceptions import (
        ModelNotFoundError,
        FeatureExtractionError,
        DataValidationError,
    )

    # Raise custom exception
    if not model_path.exists():
        raise ModelNotFoundError(f"Model not found: {model_path}")
"""


class NBAPropsError(Exception):
    """Base exception for all NBA Props ML errors."""

    pass


# =============================================================================
# Model Errors
# =============================================================================


class ModelError(NBAPropsError):
    """Base exception for model-related errors."""

    pass


class ModelNotFoundError(ModelError):
    """Raised when a required model file is not found."""

    def __init__(self, model_path: str, market: str = None):
        self.model_path = model_path
        self.market = market
        message = f"Model not found: {model_path}"
        if market:
            message = f"{market} model not found: {model_path}"
        super().__init__(message)


class ModelLoadError(ModelError):
    """Raised when a model fails to load (corrupted, version mismatch, etc.)."""

    def __init__(self, model_path: str, reason: str = None):
        self.model_path = model_path
        self.reason = reason
        message = f"Failed to load model: {model_path}"
        if reason:
            message = f"{message} - {reason}"
        super().__init__(message)


class ModelPredictionError(ModelError):
    """Raised when model prediction fails."""

    def __init__(self, message: str, player_name: str = None, stat_type: str = None):
        self.player_name = player_name
        self.stat_type = stat_type
        if player_name and stat_type:
            message = f"{message} (player={player_name}, stat={stat_type})"
        super().__init__(message)


# =============================================================================
# Feature Extraction Errors
# =============================================================================


class FeatureError(NBAPropsError):
    """Base exception for feature-related errors."""

    pass


class FeatureExtractionError(FeatureError):
    """Raised when feature extraction fails."""

    def __init__(self, message: str, player_name: str = None, feature_name: str = None):
        self.player_name = player_name
        self.feature_name = feature_name
        if player_name:
            message = f"{message} (player={player_name})"
        if feature_name:
            message = f"{message} (feature={feature_name})"
        super().__init__(message)


class MissingFeatureError(FeatureError):
    """Raised when a required feature is missing or NaN."""

    def __init__(self, feature_name: str, player_name: str = None):
        self.feature_name = feature_name
        self.player_name = player_name
        message = f"Missing required feature: {feature_name}"
        if player_name:
            message = f"{message} (player={player_name})"
        super().__init__(message)


class FeatureDriftError(FeatureError):
    """Raised when significant feature drift is detected."""

    def __init__(self, drifted_features: list, drift_percentage: float):
        self.drifted_features = drifted_features
        self.drift_percentage = drift_percentage
        message = (
            f"Feature drift detected: {drift_percentage:.1f}% of features drifted "
            f"({len(drifted_features)} features)"
        )
        super().__init__(message)


# =============================================================================
# Data Validation Errors
# =============================================================================


class DataValidationError(NBAPropsError):
    """Base exception for data validation errors."""

    pass


class InvalidPropLineError(DataValidationError):
    """Raised when a prop line fails validation."""

    def __init__(self, message: str, player_name: str = None, line: float = None):
        self.player_name = player_name
        self.line = line
        if player_name and line:
            message = f"{message} (player={player_name}, line={line})"
        super().__init__(message)


class InvalidPredictionError(DataValidationError):
    """Raised when a prediction fails validation."""

    def __init__(self, message: str, field: str = None):
        self.field = field
        if field:
            message = f"{message} (field={field})"
        super().__init__(message)


# =============================================================================
# Database Errors
# =============================================================================


class DatabaseError(NBAPropsError):
    """Base exception for database-related errors."""

    pass


class DatabaseConnectionError(DatabaseError):
    """Raised when database connection fails."""

    def __init__(self, database: str, host: str = None, port: int = None):
        self.database = database
        self.host = host
        self.port = port
        message = f"Failed to connect to database: {database}"
        if host and port:
            message = f"{message} ({host}:{port})"
        super().__init__(message)


class DatabaseQueryError(DatabaseError):
    """Raised when a database query fails."""

    def __init__(self, message: str, query: str = None):
        self.query = query
        if query:
            # Truncate long queries
            truncated = query[:100] + "..." if len(query) > 100 else query
            message = f"{message} (query: {truncated})"
        super().__init__(message)


class DataNotFoundError(DatabaseError):
    """Raised when expected data is not found in database."""

    def __init__(self, entity: str, identifier: str = None):
        self.entity = entity
        self.identifier = identifier
        message = f"{entity} not found"
        if identifier:
            message = f"{message}: {identifier}"
        super().__init__(message)


# =============================================================================
# API Errors
# =============================================================================


class APIError(NBAPropsError):
    """Base exception for API-related errors."""

    pass


class APIConnectionError(APIError):
    """Raised when API connection fails."""

    def __init__(self, api_name: str, url: str = None, status_code: int = None):
        self.api_name = api_name
        self.url = url
        self.status_code = status_code
        message = f"Failed to connect to {api_name} API"
        if status_code:
            message = f"{message} (status={status_code})"
        super().__init__(message)


class APIRateLimitError(APIError):
    """Raised when API rate limit is exceeded."""

    def __init__(self, api_name: str, retry_after: int = None):
        self.api_name = api_name
        self.retry_after = retry_after
        message = f"{api_name} API rate limit exceeded"
        if retry_after:
            message = f"{message} (retry after {retry_after}s)"
        super().__init__(message)


class APIResponseError(APIError):
    """Raised when API response is invalid or unexpected."""

    def __init__(self, api_name: str, message: str, response_data: dict = None):
        self.api_name = api_name
        self.response_data = response_data
        full_message = f"{api_name} API error: {message}"
        super().__init__(full_message)


# =============================================================================
# Configuration Errors
# =============================================================================


class ConfigurationError(NBAPropsError):
    """Base exception for configuration errors."""

    pass


class MissingConfigError(ConfigurationError):
    """Raised when required configuration is missing."""

    def __init__(self, config_key: str, source: str = None):
        self.config_key = config_key
        self.source = source
        message = f"Missing required configuration: {config_key}"
        if source:
            message = f"{message} (expected in {source})"
        super().__init__(message)


class InvalidConfigError(ConfigurationError):
    """Raised when configuration value is invalid."""

    def __init__(self, config_key: str, value: str, reason: str = None):
        self.config_key = config_key
        self.value = value
        self.reason = reason
        message = f"Invalid configuration: {config_key}={value}"
        if reason:
            message = f"{message} ({reason})"
        super().__init__(message)
