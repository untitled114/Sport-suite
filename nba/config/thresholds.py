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
    "FeaturePreprocessing",
    "ConditionalThresholds",
    "SeasonPhaseConfig",
    # Singleton instances
    "PROBABILITY_THRESHOLDS",
    "EDGE_THRESHOLDS",
    "SPREAD_THRESHOLDS",
    "BLEND_WEIGHTS",
    "TRAINING_HYPERPARAMETERS",
    "FEATURE_PREPROCESSING",
    "CONDITIONAL_THRESHOLDS",
    "TEMPORAL_DECAY_CONFIG",
    "PHASE_DECISION_THRESHOLDS",
    "STAKE_SIZING_CONFIG",
    "TemporalDecayConfig",
    "PhaseDecisionThresholds",
    "StakeSizingConfig",
    "calculate_stake",
    # Season phase configs
    "EARLY_SEASON_CONFIG",
    "MID_SEASON_CONFIG",
    "LATE_SEASON_CONFIG",
    "PLAYOFFS_CONFIG",
    # Market configs
    "POINTS_CONFIG",
    "REBOUNDS_CONFIG",
    "ASSISTS_CONFIG",
    "THREES_CONFIG",
    # Helper functions
    "get_market_config",
    "get_tier_config",
    "get_season_phase_config",
    "check_conditional_thresholds",
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
class ConditionalThresholds:
    """Conditional thresholds based on market confidence and line movement.

    Key finding: LINE DELTA (movement direction) is the strongest signal.
    When line drops, OVER hits more often (+3-6% lift).

    Attributes:
        min_consensus_strength: Minimum consensus_strength (0-1, 1 = all 8 books)
        max_line_delta: Maximum line_delta (0 = only bet when line dropped)
        min_line_delta: Minimum line_delta (-2.5 = exclude extreme drops)
        max_line_volatility: Maximum acceptable line volatility
        min_snapshot_count: Minimum line snapshots for confidence
        apply_line_delta_in_playoffs: Whether to apply line_delta filter in playoffs
    """

    min_consensus_strength: float = 0.375  # At least 3 books (optimized from 0.5)
    max_line_delta: float = 0.0  # Only bet when line dropped or stable
    min_line_delta: float = -2.5  # Exclude extreme drops (injury news)
    max_line_volatility: float = 5.0  # Relaxed (not predictive)
    min_snapshot_count: int = 2  # Relaxed
    apply_line_delta_in_playoffs: bool = False  # Line delta filter hurts in playoffs


@dataclass(frozen=True)
class SeasonPhaseConfig:
    """Season phase-specific configuration.

    Applies different thresholds based on calendar phase to maximize ROI.

    Phase behavior from walk-forward CV:
    - Early (Oct-Nov): POINTS fragile (0.67), REBOUNDS stable (0.70)
    - Mid (Dec-Feb): POINTS improving (0.74), REBOUNDS strong (0.73)
    - Late/Playoffs (Mar+): Both peak (0.77+)
    """

    phase: str  # "early", "mid", "late", "playoffs"
    points_enabled: bool = True
    rebounds_enabled: bool = True
    min_p_over_boost: float = 0.0  # Added to base min_p_over
    min_edge_boost: float = 0.0  # Added to base min_edge
    min_spread_boost: float = 0.0  # Added to base min_spread
    max_picks_per_day: Optional[int] = None  # None = unlimited


# Season phase configurations
# Key insight: line_delta filter helps MOST in early season (+5% lift)
# Don't disable markets - let the line_delta filter do the work
EARLY_SEASON_CONFIG = SeasonPhaseConfig(
    phase="early",
    points_enabled=True,  # Keep enabled - line_delta filter helps here
    rebounds_enabled=True,
    min_p_over_boost=0.02,  # Slightly stricter
    min_edge_boost=0.5,
    min_spread_boost=0.0,
    max_picks_per_day=5,
)

MID_SEASON_CONFIG = SeasonPhaseConfig(
    phase="mid",
    points_enabled=True,
    rebounds_enabled=True,
    min_p_over_boost=0.0,
    min_edge_boost=0.0,
    min_spread_boost=0.0,
    max_picks_per_day=None,
)

LATE_SEASON_CONFIG = SeasonPhaseConfig(
    phase="late",
    points_enabled=True,
    rebounds_enabled=True,
    min_p_over_boost=0.0,
    min_edge_boost=0.0,
    min_spread_boost=0.0,
    max_picks_per_day=None,
)

# Playoffs: line_delta filter HURTS here, so it's disabled in check_conditional_thresholds
PLAYOFFS_CONFIG = SeasonPhaseConfig(
    phase="playoffs",
    points_enabled=True,
    rebounds_enabled=True,
    min_p_over_boost=-0.02,  # Slightly lower threshold (more picks)
    min_edge_boost=-0.5,
    min_spread_boost=0.0,
    max_picks_per_day=None,
)


def get_season_phase_config(game_date) -> SeasonPhaseConfig:
    """Get season phase config based on game date."""
    from datetime import date

    if isinstance(game_date, str):
        from datetime import datetime

        game_date = datetime.strptime(game_date, "%Y-%m-%d").date()

    month = game_date.month
    if month in [10, 11]:
        return EARLY_SEASON_CONFIG
    elif month in [12, 1, 2]:
        return MID_SEASON_CONFIG
    elif month == 3:
        return LATE_SEASON_CONFIG
    else:  # April onwards
        return PLAYOFFS_CONFIG


@dataclass(frozen=True)
class TemporalDecayConfig:
    """Time-decay configuration for training sample weights.

    Recent games are weighted more heavily to prevent older data from
    poisoning the signal (especially for POINTS where usage changes fast).

    weight = exp(-(current_date - game_date) / tau)

    Attributes:
        enabled: Whether to use temporal decay
        tau_points: Half-life in days for POINTS market (faster decay)
        tau_rebounds: Half-life in days for REBOUNDS market (slower decay)
        tau_default: Default half-life for other markets
        min_weight: Minimum weight floor (prevent near-zero weights)
    """

    enabled: bool = True
    tau_points: float = 30.0  # ~1 month half-life (usage changes fast)
    tau_rebounds: float = 45.0  # ~1.5 month half-life (more stable)
    tau_default: float = 40.0
    min_weight: float = 0.1  # Floor to prevent near-zero weights

    def get_tau(self, market: str) -> float:
        """Get decay rate for a specific market."""
        market_upper = market.upper()
        if market_upper == "POINTS":
            return self.tau_points
        elif market_upper == "REBOUNDS":
            return self.tau_rebounds
        return self.tau_default


@dataclass(frozen=True)
class PhaseDecisionThresholds:
    """Phase-aware decision thresholds for betting exposure control.

    Same model predictions, different decision thresholds by season phase.
    Early season: require higher confidence (less exposure)
    Late season: allow lower confidence (more exposure)

    Attributes:
        early_min_p_over: Min P(OVER) for early season (days 0-20)
        mid_min_p_over: Min P(OVER) for mid season (days 20-60)
        late_min_p_over: Min P(OVER) for late season (days 60+)
        early_days_threshold: Days into season for "early" cutoff
        mid_days_threshold: Days into season for "mid" cutoff
    """

    early_min_p_over: float = 0.62  # Stricter early
    mid_min_p_over: float = 0.58
    late_min_p_over: float = 0.55  # Looser late
    early_days_threshold: int = 20
    mid_days_threshold: int = 60

    def get_min_p_over(self, days_into_season: int) -> float:
        """Get minimum P(OVER) threshold based on season phase."""
        if days_into_season < self.early_days_threshold:
            return self.early_min_p_over
        elif days_into_season < self.mid_days_threshold:
            return self.mid_min_p_over
        return self.late_min_p_over


TEMPORAL_DECAY_CONFIG = TemporalDecayConfig()
PHASE_DECISION_THRESHOLDS = PhaseDecisionThresholds()


# =============================================================================
# VOLATILITY-AWARE STAKE SIZING (Feb 3, 2026)
# =============================================================================
# Key insight: Same expected value, different variance = different optimal stake
# Low volatility players: PRESS (higher stake, more reliable)
# High volatility players: FADE (lower stake, too much variance)
#
# Uses features already computed in training:
#   - usage_volatility_score: combined CV of stat + minutes
#   - {stat}_std_L5: recent variance
#   - {stat}_trend_ratio: recent vs long-term trend
#
# Kelly-inspired but capped for bankroll preservation
# =============================================================================


@dataclass(frozen=True)
class StakeSizingConfig:
    """Volatility-aware stake sizing configuration.

    Final stake = base_units * confidence_mult * volatility_mult * edge_mult

    Attributes:
        enabled: Whether to use volatility-aware sizing
        base_units: Default stake in units
        max_units: Maximum stake allowed
        min_units: Minimum stake allowed

        confidence_floor: p_over below this = no bet (filtered earlier)
        confidence_full: p_over at or above = full confidence multiplier
        max_confidence_mult: Maximum multiplier from confidence

        volatility_weight: How aggressively to penalize volatility (0-1)
        max_volatility_penalty: Maximum stake reduction from volatility
        volatility_threshold_low: Below this = "stable" player
        volatility_threshold_high: Above this = "volatile" player

        edge_floor: Edge below this = minimum edge multiplier
        edge_full: Edge at or above = full edge multiplier
        max_edge_mult: Maximum multiplier from edge
    """

    enabled: bool = True
    base_units: float = 1.0
    max_units: float = 3.0  # Never bet more than 3 units (bankroll protection)
    min_units: float = 0.25  # Never bet less than 0.25 units

    # Confidence scaling (p_over drives base sizing)
    confidence_floor: float = 0.55  # Minimum to even consider
    confidence_full: float = 0.80  # Full confidence at this level
    max_confidence_mult: float = 1.5  # 50% boost for very confident picks

    # Volatility penalty (this is the key innovation)
    volatility_weight: float = 0.6  # 60% of volatility score becomes penalty
    max_volatility_penalty: float = 0.5  # Max 50% reduction for volatile players
    volatility_threshold_low: float = 0.15  # Below = stable, no penalty
    volatility_threshold_high: float = 0.40  # Above = max penalty

    # Edge scaling (predicted edge drives sizing)
    edge_floor: float = 1.0  # Below this = minimum multiplier
    edge_full: float = 4.0  # Full multiplier at this edge
    max_edge_mult: float = 1.25  # 25% boost for high-edge picks

    # Market-specific volatility normalization
    # POINTS has higher natural variance than REBOUNDS
    points_vol_scale: float = 1.0  # Standard scale
    rebounds_vol_scale: float = 1.2  # Rebounds std needs scaling up

    def get_vol_scale(self, market: str) -> float:
        """Get volatility scale factor for market normalization."""
        market_upper = market.upper()
        if market_upper == "POINTS":
            return self.points_vol_scale
        elif market_upper == "REBOUNDS":
            return self.rebounds_vol_scale
        return 1.0


STAKE_SIZING_CONFIG = StakeSizingConfig()


def calculate_stake(
    p_over: float,
    edge: float,
    volatility_score: float,
    market: str = "POINTS",
    config: StakeSizingConfig = None,
) -> dict:
    """Calculate volatility-adjusted stake size.

    Args:
        p_over: Model's P(OVER) probability
        edge: Predicted edge in points
        volatility_score: Player's usage_volatility_score (CV of stat + minutes)
        market: Market type for volatility normalization
        config: Optional custom config (uses STAKE_SIZING_CONFIG if None)

    Returns:
        Dict with:
        - stake_units: Recommended stake in units
        - confidence_mult: Multiplier from p_over
        - volatility_mult: Multiplier from volatility (1.0 = no penalty)
        - edge_mult: Multiplier from edge
        - volatility_label: "STABLE", "MODERATE", or "VOLATILE"
        - stake_label: "PRESS", "STANDARD", or "FADE"

    Example:
        >>> calculate_stake(p_over=0.75, edge=3.0, volatility_score=0.20)
        {'stake_units': 1.35, 'confidence_mult': 1.25, 'volatility_mult': 0.94, ...}
    """
    if config is None:
        config = STAKE_SIZING_CONFIG

    if not config.enabled:
        return {
            "stake_units": config.base_units,
            "confidence_mult": 1.0,
            "volatility_mult": 1.0,
            "edge_mult": 1.0,
            "volatility_label": "N/A",
            "stake_label": "STANDARD",
        }

    # 1. CONFIDENCE MULTIPLIER (0.5 to max_confidence_mult)
    # Linear scale from confidence_floor to confidence_full
    conf_range = config.confidence_full - config.confidence_floor
    if conf_range > 0:
        conf_pct = min(max((p_over - config.confidence_floor) / conf_range, 0.0), 1.0)
    else:
        conf_pct = 1.0 if p_over >= config.confidence_floor else 0.0
    confidence_mult = 0.5 + conf_pct * (config.max_confidence_mult - 0.5)

    # 2. VOLATILITY MULTIPLIER (1.0 - penalty)
    # Normalize volatility by market (rebounds naturally less variable)
    vol_scale = config.get_vol_scale(market)
    normalized_vol = volatility_score * vol_scale

    # Calculate penalty: linear from threshold_low to threshold_high
    vol_range = config.volatility_threshold_high - config.volatility_threshold_low
    if vol_range > 0 and normalized_vol > config.volatility_threshold_low:
        vol_pct = min((normalized_vol - config.volatility_threshold_low) / vol_range, 1.0)
    else:
        vol_pct = 0.0

    # Apply weighted penalty
    vol_penalty = vol_pct * config.volatility_weight * config.max_volatility_penalty
    volatility_mult = 1.0 - vol_penalty

    # Determine volatility label
    if normalized_vol < config.volatility_threshold_low:
        volatility_label = "STABLE"
    elif normalized_vol > config.volatility_threshold_high:
        volatility_label = "VOLATILE"
    else:
        volatility_label = "MODERATE"

    # 3. EDGE MULTIPLIER (0.75 to max_edge_mult)
    edge_range = config.edge_full - config.edge_floor
    if edge_range > 0:
        edge_pct = min(max((edge - config.edge_floor) / edge_range, 0.0), 1.0)
    else:
        edge_pct = 1.0 if edge >= config.edge_floor else 0.0
    edge_mult = 0.75 + edge_pct * (config.max_edge_mult - 0.75)

    # 4. FINAL STAKE
    stake = config.base_units * confidence_mult * volatility_mult * edge_mult
    stake = max(config.min_units, min(config.max_units, stake))

    # Determine stake action label
    if stake >= 1.5:
        stake_label = "PRESS"
    elif stake <= 0.5:
        stake_label = "FADE"
    else:
        stake_label = "STANDARD"

    return {
        "stake_units": round(stake, 2),
        "confidence_mult": round(confidence_mult, 3),
        "volatility_mult": round(volatility_mult, 3),
        "edge_mult": round(edge_mult, 3),
        "volatility_label": volatility_label,
        "stake_label": stake_label,
        "raw_volatility": round(volatility_score, 3),
        "normalized_volatility": round(normalized_vol, 3),
    }


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


@dataclass(frozen=True)
class FeaturePreprocessing:
    """Feature preprocessing configuration for training.

    Attributes:
        common_cols_to_drop: Columns always dropped (injury bugs, logic bugs)
        h2h_cols_by_stat: H2H columns grouped by stat type (drop non-matching markets)
    """

    # Always drop these (data/logic issues or redundancy)
    common_cols_to_drop: tuple = (
        # Injury features (team_abbrev not populated in player_profile)
        "injured_teammates_count",
        "teammate_absences_last_3",
        # Logic bugs (constant values)
        "games_in_L7",
        "prime_time_flag",
        # Multicollinearity - redundant with line_movement_std
        "line_volatility",  # r=0.97 with line_movement_std
        "market_stability",  # r=-0.74 with line_movement_std (inverse)
    )

    # H2H columns by stat type - drop columns for OTHER markets, keep matching market
    h2h_points_cols: tuple = (
        "h2h_avg_points",
        "h2h_std_points",
        "h2h_L3_points",
        "h2h_L5_points",
        "h2h_L10_points",
        "h2h_L20_points",
        "h2h_home_avg_points",
        "h2h_away_avg_points",
    )

    h2h_rebounds_cols: tuple = (
        "h2h_avg_rebounds",
        "h2h_std_rebounds",
        "h2h_L3_rebounds",
        "h2h_L5_rebounds",
        "h2h_L10_rebounds",
        "h2h_L20_rebounds",
        "h2h_home_avg_rebounds",
        "h2h_away_avg_rebounds",
    )

    h2h_assists_cols: tuple = (
        "h2h_avg_assists",
        "h2h_std_assists",
        "h2h_L3_assists",
        "h2h_L5_assists",
        "h2h_L10_assists",
        "h2h_L20_assists",
        "h2h_home_avg_assists",
        "h2h_away_avg_assists",
    )

    h2h_threes_cols: tuple = (
        "h2h_avg_threes",
        "h2h_std_threes",
        "h2h_L3_threes",
        "h2h_L5_threes",
        "h2h_L10_threes",
        "h2h_L20_threes",
        "h2h_home_avg_threes",
        "h2h_away_avg_threes",
    )

    def get_cols_to_drop(self, market: str) -> tuple:
        """Get columns to drop for a specific market.

        Drops H2H columns for OTHER markets, keeps H2H for the target market.
        """
        cols = list(self.common_cols_to_drop)

        market_upper = market.upper()

        # Drop H2H columns for non-matching markets
        if market_upper != "POINTS":
            cols.extend(self.h2h_points_cols)
        if market_upper != "REBOUNDS":
            cols.extend(self.h2h_rebounds_cols)
        if market_upper != "ASSISTS":
            cols.extend(self.h2h_assists_cols)
        if market_upper != "THREES":
            cols.extend(self.h2h_threes_cols)

        return tuple(cols)


# =============================================================================
# SINGLETON INSTANCES (GLOBAL DEFAULTS)
# =============================================================================

PROBABILITY_THRESHOLDS = ProbabilityThresholds()

EDGE_THRESHOLDS = EdgeThresholds()

SPREAD_THRESHOLDS = SpreadThresholds()

BLEND_WEIGHTS = BlendWeights()

TRAINING_HYPERPARAMETERS = TrainingHyperparameters()

FEATURE_PREPROCESSING = FeaturePreprocessing()

CONDITIONAL_THRESHOLDS = ConditionalThresholds()


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


def check_conditional_thresholds(
    features: dict,
    market: str,
    p_over: float,
    edge: float,
    thresholds: ConditionalThresholds = None,
) -> tuple[bool, str]:
    """Check if a pick passes conditional thresholds.

    Key optimization: LINE DELTA is the strongest signal.
    When line drops (delta < 0), OVER hits 3-6% more often.

    Args:
        features: Feature dict containing market confidence features
        market: Market name (POINTS, REBOUNDS)
        p_over: Predicted probability of over
        edge: Predicted edge in points
        thresholds: Optional custom thresholds (uses CONDITIONAL_THRESHOLDS if None)

    Returns:
        Tuple of (passes: bool, reason: str)
        - passes: True if pick passes all conditional thresholds
        - reason: Empty string if passes, otherwise explanation of failure
    """
    if thresholds is None:
        thresholds = CONDITIONAL_THRESHOLDS

    # Get season phase config
    game_date = features.get("game_date")
    is_playoffs = False
    if game_date:
        phase_config = get_season_phase_config(game_date)
        is_playoffs = phase_config.phase == "playoffs"
    else:
        phase_config = MID_SEASON_CONFIG

    # Check consensus strength (3+ books minimum)
    consensus_strength = features.get("consensus_strength", 0.5)
    if consensus_strength < thresholds.min_consensus_strength:
        return (
            False,
            f"consensus_strength {consensus_strength:.2f} < {thresholds.min_consensus_strength} (need 3+ books)",
        )

    # KEY OPTIMIZATION: Line delta filter (skip during playoffs)
    line_delta = features.get("line_delta", 0.0)

    if not is_playoffs or thresholds.apply_line_delta_in_playoffs:
        # Only bet when line dropped (or stable)
        if line_delta > thresholds.max_line_delta:
            return (
                False,
                f"line_delta {line_delta:.2f} > {thresholds.max_line_delta} (line rose - avoid)",
            )

        # Exclude extreme drops (likely injury news)
        if line_delta < thresholds.min_line_delta:
            return (
                False,
                f"line_delta {line_delta:.2f} < {thresholds.min_line_delta} (extreme drop - uncertain)",
            )

    # Check line volatility (relaxed - not very predictive)
    line_volatility = features.get("line_volatility", 0.0)
    if line_volatility > thresholds.max_line_volatility:
        return False, f"line_volatility {line_volatility:.2f} > {thresholds.max_line_volatility}"

    # Check snapshot count
    snapshot_count = features.get("snapshot_count", 1)
    if snapshot_count < thresholds.min_snapshot_count:
        return False, f"snapshot_count {snapshot_count} < {thresholds.min_snapshot_count}"

    return True, ""
