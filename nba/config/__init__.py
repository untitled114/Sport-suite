# NBA Configuration Module
import os

# =============================================================================
# Environment Detection
# =============================================================================
ENVIRONMENT = os.getenv("ENVIRONMENT", "development").lower()
IS_PRODUCTION = ENVIRONMENT == "production"
IS_DEVELOPMENT = ENVIRONMENT == "development"


def get_environment() -> str:
    """Get current environment name."""
    return ENVIRONMENT


def is_production() -> bool:
    """Check if running in production."""
    return IS_PRODUCTION


def is_development() -> bool:
    """Check if running in development."""
    return IS_DEVELOPMENT


# Database configuration
# Shared constants
from .constants import (
    ACTIVE_MARKETS,
    ALL_BOOKS,
    BETTINGPROS_MARKET_IDS,
    BOOK_MARKET_EXCLUSIONS,
    COMBO_STAT_MAP,
    DISABLED_MARKETS,
    EXTENDED_STAT_TYPES,
    PRIORITY_BOOKS,
    STAT_COLUMN_MAP,
    TEAM_ABBREV_MAP,
    VALID_STAT_TYPES,
)
from .database import (
    DB_DEFAULT_PASSWORD,
    DB_DEFAULT_USER,
    get_db_config,
    get_games_db_config,
    get_intelligence_db_config,
    get_mongo_config,
    get_players_db_config,
    get_team_db_config,
)

# Threshold configurations (centralized magic numbers)
from .thresholds import (  # Core dataclasses; Singleton instances; Market configs; Star player configs; Book configs; Helper functions
    ASSISTS_CONFIG,
    BLEND_WEIGHTS,
    EDGE_THRESHOLDS,
    POINTS_CONFIG,
    PROBABILITY_THRESHOLDS,
    REBOUNDS_CONFIG,
    RELIABLE_BOOKS,
    SPREAD_THRESHOLDS,
    STAR_POINTS_CONFIG,
    STAR_REBOUNDS_CONFIG,
    THREES_CONFIG,
    TRAINING_HYPERPARAMETERS,
    TRAP_BOOKS,
    BlendWeights,
    EdgeThresholds,
    LineConstraints,
    MarketConfig,
    ProbabilityThresholds,
    SpreadThresholds,
    StarPlayerConfig,
    TierConfig,
    TrainingHyperparameters,
    TrapBookConfig,
    get_market_config,
    get_star_config,
    get_tier_config,
    is_reliable_book,
    is_trap_book,
)

__all__ = [
    # Environment
    "ENVIRONMENT",
    "IS_PRODUCTION",
    "IS_DEVELOPMENT",
    "get_environment",
    "is_production",
    "is_development",
    # Constants
    "ACTIVE_MARKETS",
    "ALL_BOOKS",
    "VALID_STAT_TYPES",
    # Database
    "get_db_config",
    "get_players_db_config",
    "get_games_db_config",
    "get_team_db_config",
    "get_intelligence_db_config",
    # Thresholds
    "get_market_config",
    "get_tier_config",
    "is_trap_book",
]
