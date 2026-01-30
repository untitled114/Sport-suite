"""
Threshold Configuration Unit Tests
===================================
Tests for centralized threshold configuration dataclasses.
"""

from dataclasses import FrozenInstanceError

import pytest


class TestProbabilityThresholds:
    """Tests for ProbabilityThresholds dataclass."""

    def test_default_values(self):
        """Test default probability threshold values."""
        from nba.config.thresholds import PROBABILITY_THRESHOLDS

        assert PROBABILITY_THRESHOLDS.min_p_over == 0.55
        assert PROBABILITY_THRESHOLDS.max_p_over == 0.95
        assert PROBABILITY_THRESHOLDS.clip_lower == 0.01
        assert PROBABILITY_THRESHOLDS.clip_upper == 0.99

    def test_frozen_immutable(self):
        """Test that thresholds are immutable."""
        from nba.config.thresholds import PROBABILITY_THRESHOLDS

        with pytest.raises(FrozenInstanceError):
            PROBABILITY_THRESHOLDS.min_p_over = 0.60

    def test_custom_instance(self):
        """Test creating custom probability thresholds."""
        from nba.config.thresholds import ProbabilityThresholds

        custom = ProbabilityThresholds(
            min_p_over=0.60,
            max_p_over=0.90,
        )

        assert custom.min_p_over == 0.60
        assert custom.max_p_over == 0.90
        # Other values should have defaults
        assert custom.clip_lower == 0.01


class TestEdgeThresholds:
    """Tests for EdgeThresholds dataclass."""

    def test_default_values(self):
        """Test default edge threshold values."""
        from nba.config.thresholds import EDGE_THRESHOLDS

        assert EDGE_THRESHOLDS.min_edge_points == 2.5
        assert EDGE_THRESHOLDS.max_edge_points == 5.0
        assert EDGE_THRESHOLDS.high_edge_threshold == 5.0

    def test_frozen_immutable(self):
        """Test that edge thresholds are immutable."""
        from nba.config.thresholds import EDGE_THRESHOLDS

        with pytest.raises(FrozenInstanceError):
            EDGE_THRESHOLDS.min_edge_points = 3.0


class TestSpreadThresholds:
    """Tests for SpreadThresholds dataclass."""

    def test_default_values(self):
        """Test default spread threshold values."""
        from nba.config.thresholds import SPREAD_THRESHOLDS

        assert SPREAD_THRESHOLDS.min_spread == 2.0
        assert SPREAD_THRESHOLDS.high_spread_goldmine == 2.5
        assert SPREAD_THRESHOLDS.low_spread_warning == 0.5


class TestBlendWeights:
    """Tests for BlendWeights dataclass."""

    def test_default_values(self):
        """Test default blend weight values."""
        from nba.config.thresholds import BLEND_WEIGHTS

        assert BLEND_WEIGHTS.classifier_weight == 0.6
        assert BLEND_WEIGHTS.residual_weight == 0.4
        assert BLEND_WEIGHTS.residual_scale_factor == 5.0

    def test_weights_sum_to_one(self):
        """Test that classifier + residual weights sum to 1."""
        from nba.config.thresholds import BLEND_WEIGHTS

        total = BLEND_WEIGHTS.classifier_weight + BLEND_WEIGHTS.residual_weight
        assert abs(total - 1.0) < 0.001


class TestMarketConfigs:
    """Tests for market configuration."""

    def test_points_config(self):
        """Test POINTS market configuration."""
        from nba.config.thresholds import POINTS_CONFIG

        assert POINTS_CONFIG.market == "POINTS"
        assert POINTS_CONFIG.enabled is True
        assert POINTS_CONFIG.min_line == 12.0
        assert len(POINTS_CONFIG.tiers) >= 1

    def test_rebounds_config(self):
        """Test REBOUNDS market configuration."""
        from nba.config.thresholds import REBOUNDS_CONFIG

        assert REBOUNDS_CONFIG.market == "REBOUNDS"
        assert REBOUNDS_CONFIG.enabled is True
        assert REBOUNDS_CONFIG.min_line == 3.0

    def test_assists_disabled(self):
        """Test ASSISTS market is disabled."""
        from nba.config.thresholds import ASSISTS_CONFIG

        assert ASSISTS_CONFIG.enabled is False

    def test_threes_disabled(self):
        """Test THREES market is disabled."""
        from nba.config.thresholds import THREES_CONFIG

        assert THREES_CONFIG.enabled is False


class TestTierConfigs:
    """Tests for tier configuration."""

    def test_tier_config_structure(self):
        """Test TierConfig has required fields."""
        from nba.config.thresholds import TierConfig

        tier = TierConfig(
            name="TEST",
            direction="OVER",
            min_p_over=0.70,
            min_edge_points=2.5,
        )

        assert tier.name == "TEST"
        assert tier.direction == "OVER"
        assert tier.min_p_over == 0.70
        assert tier.min_edge_points == 2.5

    def test_points_tiers(self):
        """Test POINTS tier configurations."""
        from nba.config.thresholds import POINTS_TIER_V3, POINTS_TIER_X

        assert POINTS_TIER_X.name == "X"
        assert POINTS_TIER_X.min_p_over == 0.70

        assert POINTS_TIER_V3.name == "V3"
        assert POINTS_TIER_V3.min_p_over == 0.85

    def test_rebounds_meta_tier(self):
        """Test REBOUNDS META tier configuration."""
        from nba.config.thresholds import REBOUNDS_TIER_META

        assert REBOUNDS_TIER_META.name == "META"
        assert REBOUNDS_TIER_META.min_spread == 1.5
        assert REBOUNDS_TIER_META.min_edge_pct == 20.0
        assert REBOUNDS_TIER_META.require_both is True


class TestStarPlayerConfigs:
    """Tests for star player configuration."""

    def test_star_points_config(self):
        """Test star player POINTS configuration."""
        from nba.config.thresholds import STAR_POINTS_CONFIG

        assert STAR_POINTS_CONFIG.min_p_over == 0.70
        assert STAR_POINTS_CONFIG.max_p_over == 0.80
        assert STAR_POINTS_CONFIG.min_spread == 2.0
        assert STAR_POINTS_CONFIG.min_edge == 3.0

    def test_star_rebounds_config(self):
        """Test star player REBOUNDS configuration."""
        from nba.config.thresholds import STAR_REBOUNDS_CONFIG

        assert STAR_REBOUNDS_CONFIG.min_p_over == 0.55
        assert STAR_REBOUNDS_CONFIG.max_line == 8.0


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_market_config(self):
        """Test get_market_config function."""
        from nba.config.thresholds import POINTS_CONFIG, get_market_config

        config = get_market_config("POINTS")
        assert config == POINTS_CONFIG

        # Should work case-insensitive
        config_lower = get_market_config("points")
        assert config_lower == POINTS_CONFIG

    def test_get_market_config_invalid(self):
        """Test get_market_config with invalid market."""
        from nba.config.thresholds import get_market_config

        with pytest.raises(ValueError):
            get_market_config("INVALID")

    def test_get_tier_config(self):
        """Test get_tier_config function."""
        from nba.config.thresholds import POINTS_TIER_X, get_tier_config

        tier = get_tier_config("POINTS", "X")
        assert tier == POINTS_TIER_X

    def test_get_tier_config_not_found(self):
        """Test get_tier_config with invalid tier."""
        from nba.config.thresholds import get_tier_config

        tier = get_tier_config("POINTS", "NONEXISTENT")
        assert tier is None

    def test_get_star_config(self):
        """Test get_star_config function."""
        from nba.config.thresholds import STAR_POINTS_CONFIG, get_star_config

        config = get_star_config("POINTS")
        assert config == STAR_POINTS_CONFIG

    def test_is_trap_book(self):
        """Test is_trap_book function."""
        from nba.config.thresholds import is_trap_book

        assert is_trap_book("DraftKings") is True
        assert is_trap_book("Underdog") is False

    def test_is_reliable_book(self):
        """Test is_reliable_book function."""
        from nba.config.thresholds import is_reliable_book

        assert is_reliable_book("Underdog") is True
        assert is_reliable_book("DraftKings") is False


class TestTrainingHyperparameters:
    """Tests for training hyperparameters."""

    def test_default_hyperparameters(self):
        """Test default training hyperparameters."""
        from nba.config.thresholds import TRAINING_HYPERPARAMETERS

        assert TRAINING_HYPERPARAMETERS.n_estimators == 2000
        assert TRAINING_HYPERPARAMETERS.learning_rate == 0.02
        assert TRAINING_HYPERPARAMETERS.num_leaves == 63
        assert TRAINING_HYPERPARAMETERS.test_size == 0.3

    def test_frozen_immutable(self):
        """Test that hyperparameters are immutable."""
        from nba.config.thresholds import TRAINING_HYPERPARAMETERS

        with pytest.raises(FrozenInstanceError):
            TRAINING_HYPERPARAMETERS.n_estimators = 1000
