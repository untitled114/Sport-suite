"""
Centralized Threshold Configuration for NBA Props ML System
============================================================
Frozen dataclasses containing all magic numbers and thresholds.

This module eliminates scattered magic numbers across the codebase by
centralizing all configuration values in type-safe, immutable dataclasses.

Usage:
    from nba.config.thresholds import (
        PROBABILITY_THRESHOLDS,
        EDGE_THRESHOLDS,
        BLEND_WEIGHTS,
        get_market_config,
    )

    # Access thresholds
    min_p = PROBABILITY_THRESHOLDS.min_p_over  # 0.55

    # Get market-specific config
    points_config = get_market_config("POINTS")
    print(points_config.min_line)  # 12.0

Example:
    >>> from nba.config.thresholds import POINTS_CONFIG
    >>> POINTS_CONFIG.min_edge_points
    3.0
    >>> POINTS_CONFIG.enabled
    True
"""

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Optional, Set

__all__ = [
    # Core threshold dataclasses
    "ProbabilityThresholds",
    "EdgeThresholds",
    "SpreadThresholds",
    "LineConstraints",
    "BlendWeights",
    "TierConfig",
    "MarketConfig",
    "TrainingHyperparameters",
    # Singleton instances
    "PROBABILITY_THRESHOLDS",
    "EDGE_THRESHOLDS",
    "SPREAD_THRESHOLDS",
    "BLEND_WEIGHTS",
    "TRAINING_HYPERPARAMETERS",
    # Market configs
    "POINTS_CONFIG",
    "REBOUNDS_CONFIG",
    "ASSISTS_CONFIG",
    "THREES_CONFIG",
    # Helper functions
    "get_market_config",
    "get_tier_config",
]


# =============================================================================
# CORE THRESHOLD DATACLASSES
# =============================================================================


@dataclass(frozen=True)
class ProbabilityThresholds:
    """Probability thresholds for model predictions.

    Attributes:
        min_p_over: Minimum P(OVER) to consider for betting
        max_p_over: Maximum P(OVER) (avoid overconfident predictions)
        high_confidence_threshold: P(OVER) indicating high confidence
        clip_lower: Lower bound for probability clipping
        clip_upper: Upper bound for probability clipping
        confidence_high: P(OVER) threshold for HIGH confidence label
        confidence_medium: P(OVER) threshold for MEDIUM confidence label
    """

    min_p_over: float = 0.55
    max_p_over: float = 0.95
    high_confidence_threshold: float = 0.65
    clip_lower: float = 0.01
    clip_upper: float = 0.99
    confidence_high: float = 0.70
    confidence_medium: float = 0.60


@dataclass(frozen=True)
class EdgeThresholds:
    """Edge thresholds for line shopping and filtering.

    Attributes:
        min_edge_points: Minimum edge in points to consider bet
        max_edge_points: Maximum edge (avoid outliers/data errors)
        min_edge_low_variance: Edge threshold for low-variance situations
        high_edge_threshold: Edge value indicating strong opportunity
        medium_edge_threshold: Edge value indicating moderate opportunity
        edge_confidence_high: Edge threshold for HIGH confidence
        edge_confidence_medium: Edge threshold for MEDIUM confidence
    """

    min_edge_points: float = 2.5
    max_edge_points: float = 5.0
    min_edge_low_variance: float = 3.0
    high_edge_threshold: float = 5.0
    medium_edge_threshold: float = 3.0
    edge_confidence_high: float = 5.0
    edge_confidence_medium: float = 3.0


@dataclass(frozen=True)
class SpreadThresholds:
    """Line spread thresholds for book disagreement analysis.

    Attributes:
        min_spread: Minimum line spread for tier filtering
        high_spread_goldmine: Spread threshold for high-value opportunities
        low_spread_warning: Spread below which to issue warning
        min_books_for_confidence: Minimum number of books for reliable spread
    """

    min_spread: float = 2.0
    high_spread_goldmine: float = 2.5
    low_spread_warning: float = 0.5
    min_books_for_confidence: int = 3


@dataclass(frozen=True)
class LineConstraints:
    """Line value constraints per market.

    Attributes:
        min_line: Minimum line value to consider
        max_line: Maximum line value to consider
    """

    min_line: float = 0.0
    max_line: float = 999.0


@dataclass(frozen=True)
class BlendWeights:
    """Weights for blending regressor and classifier predictions.

    Attributes:
        classifier_weight: Weight for classifier probability
        residual_weight: Weight for residual-based probability
        residual_scale_factor: Scale factor for residual sigmoid
        book_intelligence_weight: Weight for book intelligence head
        base_model_weight: Weight for base model in ensemble
    """

    classifier_weight: float = 0.6
    residual_weight: float = 0.4
    residual_scale_factor: float = 5.0
    book_intelligence_weight: float = 0.30
    base_model_weight: float = 0.70


@dataclass(frozen=True)
class TierConfig:
    """Configuration for a specific tier filter.

    Attributes:
        name: Tier identifier (e.g., "X", "V3", "META")
        direction: "OVER" or "UNDER"
        min_p_over: Minimum P(OVER) for this tier
        max_p_over: Maximum P(OVER) for this tier
        min_spread: Minimum line spread
        min_edge_points: Minimum edge in points
        min_edge_pct: Minimum edge percentage (for META tier)
        min_line: Minimum line value
        max_line: Maximum line value
        require_positive_edge: Whether positive edge is required
        require_both: If True, use AND logic (spread AND edge required)
        expected_wr: Expected win rate for this tier
        model_version: Model version this tier applies to
    """

    name: str
    direction: str = "OVER"
    min_p_over: float = 0.55
    max_p_over: float = 1.0
    min_spread: float = 0.0
    min_edge_points: float = 0.0
    min_edge_pct: float = 0.0
    min_line: float = 0.0
    max_line: float = 999.0
    require_positive_edge: bool = False
    require_both: bool = False
    expected_wr: float = 0.60
    model_version: str = "xl"


@dataclass(frozen=True)
class MarketConfig:
    """Complete configuration for a betting market.

    Attributes:
        market: Market name (POINTS, REBOUNDS, etc.)
        enabled: Whether this market is active for predictions
        min_probability: Global minimum probability threshold
        min_line: Minimum line value for this market
        max_line: Maximum line value for this market
        min_edge_points: Minimum edge in points
        max_edge_points: Maximum edge in points
        min_spread: Minimum line spread
        max_edge_low_variance: Edge cap for low-variance situations
        high_confidence_p_over: P(OVER) threshold for high confidence
        tiers: Tuple of TierConfig for this market
        avoid_books: Books to avoid when softest
        blacklisted_books: Books to skip entirely
    """

    market: str
    enabled: bool = True
    min_probability: float = 0.55
    min_line: float = 0.0
    max_line: float = 999.0
    min_edge_points: float = 2.5
    max_edge_points: float = 5.0
    min_spread: float = 2.0
    max_edge_low_variance: float = 3.0
    high_confidence_p_over: float = 0.65
    tiers: tuple = field(default_factory=tuple)
    avoid_books: FrozenSet[str] = field(default_factory=frozenset)
    blacklisted_books: FrozenSet[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class TrainingHyperparameters:
    """Hyperparameters for model training.

    Attributes:
        n_estimators: Number of boosting iterations
        learning_rate: Learning rate for gradient boosting
        num_leaves: Maximum number of leaves per tree
        feature_fraction: Fraction of features to use per tree
        bagging_fraction: Fraction of samples to use per tree
        bagging_freq: Frequency of bagging
        early_stopping_rounds: Rounds for early stopping
        random_state: Random seed for reproducibility
        test_size: Fraction of data for testing
    """

    n_estimators: int = 2000
    learning_rate: float = 0.02
    num_leaves: int = 63
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    bagging_freq: int = 5
    early_stopping_rounds: int = 50
    random_state: int = 42
    test_size: float = 0.3


# =============================================================================
# SINGLETON INSTANCES (GLOBAL DEFAULTS)
# =============================================================================

PROBABILITY_THRESHOLDS = ProbabilityThresholds()

EDGE_THRESHOLDS = EdgeThresholds()

SPREAD_THRESHOLDS = SpreadThresholds()

BLEND_WEIGHTS = BlendWeights()

TRAINING_HYPERPARAMETERS = TrainingHyperparameters()


# =============================================================================
# TIER CONFIGURATIONS
# =============================================================================

# POINTS market tiers
POINTS_TIER_X = TierConfig(
    name="X",
    direction="OVER",
    min_p_over=0.70,
    min_spread=0.0,
    min_edge_points=3.0,
    require_positive_edge=False,
    require_both=False,
    model_version="xl",
    expected_wr=0.67,
)

POINTS_TIER_V3 = TierConfig(
    name="V3",
    direction="OVER",
    min_p_over=0.85,
    min_spread=0.0,
    min_edge_points=3.0,
    require_positive_edge=False,
    require_both=False,
    model_version="v3",
    expected_wr=0.84,
)

POINTS_TIER_JAN_CONFIDENT = TierConfig(
    name="JAN_CONFIDENT_OVER",
    direction="OVER",
    min_p_over=0.75,
    max_p_over=0.85,
    min_line=0.0,
    max_line=25.0,
    min_edge_points=4.0,
    min_spread=0.0,
    expected_wr=0.875,
)

# REBOUNDS market tiers
REBOUNDS_TIER_META = TierConfig(
    name="META",
    direction="OVER",
    min_p_over=0.55,
    min_spread=1.5,
    min_edge_pct=20.0,
    require_positive_edge=True,
    require_both=True,
    expected_wr=0.706,
)

REBOUNDS_TIER_A = TierConfig(
    name="A",
    direction="OVER",
    min_p_over=0.70,
    min_spread=2.0,
    min_edge_points=1.0,
    require_positive_edge=False,
    require_both=False,
    expected_wr=0.61,
)


# =============================================================================
# MARKET CONFIGURATIONS
# =============================================================================

POINTS_CONFIG = MarketConfig(
    market="POINTS",
    enabled=True,
    min_probability=0.65,
    min_line=12.0,
    max_line=999.0,
    min_edge_points=3.0,
    max_edge_points=5.0,
    min_spread=2.5,
    max_edge_low_variance=3.0,
    high_confidence_p_over=0.65,
    tiers=(POINTS_TIER_X, POINTS_TIER_V3, POINTS_TIER_JAN_CONFIDENT),
    avoid_books=frozenset({"draftkings", "DraftKings", "betrivers", "BetRivers"}),
    blacklisted_books=frozenset({"FanDuel", "fanduel", "BetRivers", "betrivers"}),
)

REBOUNDS_CONFIG = MarketConfig(
    market="REBOUNDS",
    enabled=True,
    min_probability=0.55,
    min_line=3.0,
    max_line=999.0,
    min_edge_points=1.0,
    max_edge_points=999.0,
    min_spread=0.5,
    max_edge_low_variance=2.0,
    high_confidence_p_over=0.60,
    tiers=(REBOUNDS_TIER_META, REBOUNDS_TIER_A),
    avoid_books=frozenset(),
    blacklisted_books=frozenset(),
)

ASSISTS_CONFIG = MarketConfig(
    market="ASSISTS",
    enabled=False,  # Disabled: 14.6% WR, -72.05% ROI
    min_probability=0.58,
    min_line=3.0,
    max_line=999.0,
    min_edge_points=1.5,
    max_edge_points=5.0,
    min_spread=2.5,
)

THREES_CONFIG = MarketConfig(
    market="THREES",
    enabled=False,  # Disabled: 46.5% WR, -11.23% ROI
    min_probability=0.58,
    min_line=1.0,
    max_line=999.0,
    min_edge_points=1.5,
    max_edge_points=5.0,
    min_spread=1.5,
)


# =============================================================================
# STAR PLAYER CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class StarPlayerConfig:
    """Configuration for star player tier filtering.

    Attributes:
        min_p_over: Minimum P(OVER) for star players
        max_p_over: Maximum P(OVER) to avoid overconfidence
        min_spread: Minimum line spread
        min_line: Minimum line value
        max_line: Maximum line value
        min_edge: Minimum edge in points
        avoid_books_softest: Books to avoid when they are softest
        model_preference: Preferred model version for star picks
    """

    min_p_over: float = 0.70
    max_p_over: float = 0.80
    min_spread: float = 2.0
    min_line: float = 0.0
    max_line: float = 29.0
    min_edge: float = 3.0
    avoid_books_softest: FrozenSet[str] = field(default_factory=frozenset)
    model_preference: str = "v3"


STAR_POINTS_CONFIG = StarPlayerConfig(
    min_p_over=0.70,
    max_p_over=0.80,
    min_spread=2.0,
    max_line=29.0,
    min_edge=3.0,
    avoid_books_softest=frozenset({"draftkings", "DraftKings", "betrivers", "BetRivers"}),
    model_preference="v3",
)

STAR_REBOUNDS_CONFIG = StarPlayerConfig(
    min_p_over=0.55,
    max_p_over=0.80,
    min_spread=0.5,
    min_line=3.0,
    max_line=8.0,
    min_edge=0.25,
    avoid_books_softest=frozenset(),
)


# =============================================================================
# BOOK CONFIGURATIONS
# =============================================================================


@dataclass(frozen=True)
class TrapBookConfig:
    """Configuration for books that show trap behavior when softest.

    Attributes:
        min_spread_required: Minimum spread to accept when this book is softest
        min_p_over_boost: Additional P(OVER) threshold boost when trap book
    """

    min_spread_required: float = 3.5
    min_p_over_boost: float = 0.05


TRAP_BOOKS: Dict[str, TrapBookConfig] = {
    "DraftKings": TrapBookConfig(min_spread_required=3.5, min_p_over_boost=0.05),
    "BetMGM": TrapBookConfig(min_spread_required=3.5, min_p_over_boost=0.05),
    "BetRivers": TrapBookConfig(min_spread_required=3.0, min_p_over_boost=0.03),
    "Caesars": TrapBookConfig(min_spread_required=3.0, min_p_over_boost=0.03),
}

RELIABLE_BOOKS: FrozenSet[str] = frozenset({"Underdog", "Underdog Fantasy", "ESPNBet"})


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

_MARKET_CONFIGS: Dict[str, MarketConfig] = {
    "POINTS": POINTS_CONFIG,
    "REBOUNDS": REBOUNDS_CONFIG,
    "ASSISTS": ASSISTS_CONFIG,
    "THREES": THREES_CONFIG,
}


def get_market_config(market: str) -> MarketConfig:
    """Get configuration for a specific market.

    Args:
        market: Market name (POINTS, REBOUNDS, ASSISTS, or THREES)

    Returns:
        MarketConfig for the specified market

    Raises:
        ValueError: If market name is not recognized
    """
    market_upper = market.upper()
    if market_upper not in _MARKET_CONFIGS:
        raise ValueError(f"Unknown market: {market}. Valid markets: {list(_MARKET_CONFIGS.keys())}")
    return _MARKET_CONFIGS[market_upper]


def get_tier_config(market: str, tier_name: str) -> Optional[TierConfig]:
    """Get a specific tier configuration for a market.

    Args:
        market: Market name (POINTS, REBOUNDS, etc.)
        tier_name: Tier identifier (X, V3, META, etc.)

    Returns:
        TierConfig if found, None otherwise
    """
    market_config = get_market_config(market)
    for tier in market_config.tiers:
        if tier.name == tier_name:
            return tier
    return None


def get_star_config(market: str) -> Optional[StarPlayerConfig]:
    """Get star player configuration for a market.

    Args:
        market: Market name (POINTS, REBOUNDS, etc.)

    Returns:
        StarPlayerConfig if available, None otherwise
    """
    configs = {
        "POINTS": STAR_POINTS_CONFIG,
        "REBOUNDS": STAR_REBOUNDS_CONFIG,
    }
    return configs.get(market.upper())


def is_trap_book(book_name: str) -> bool:
    """Check if a book is considered a trap when softest.

    Args:
        book_name: Sportsbook name

    Returns:
        True if book is in trap list
    """
    return book_name in TRAP_BOOKS


def is_reliable_book(book_name: str) -> bool:
    """Check if a book is considered reliable when softest.

    Args:
        book_name: Sportsbook name

    Returns:
        True if book is in reliable list
    """
    return book_name in RELIABLE_BOOKS
