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


# =============================================================================
# SEASON PHASE TESTS
# =============================================================================


class TestSeasonPhaseConfig:
    """Tests for season phase configurations."""

    def test_early_season_config(self):
        """Test early season configuration."""
        from nba.config.thresholds import EARLY_SEASON_CONFIG

        assert EARLY_SEASON_CONFIG.phase == "early"
        assert EARLY_SEASON_CONFIG.points_enabled is True
        assert EARLY_SEASON_CONFIG.min_p_over_boost == 0.02
        assert EARLY_SEASON_CONFIG.max_picks_per_day == 5

    def test_mid_season_config(self):
        """Test mid season configuration."""
        from nba.config.thresholds import MID_SEASON_CONFIG

        assert MID_SEASON_CONFIG.phase == "mid"
        assert MID_SEASON_CONFIG.max_picks_per_day is None

    def test_late_season_config(self):
        """Test late season configuration."""
        from nba.config.thresholds import LATE_SEASON_CONFIG

        assert LATE_SEASON_CONFIG.phase == "late"

    def test_playoffs_config(self):
        """Test playoffs configuration."""
        from nba.config.thresholds import PLAYOFFS_CONFIG

        assert PLAYOFFS_CONFIG.phase == "playoffs"
        assert PLAYOFFS_CONFIG.min_p_over_boost == -0.02


class TestGetSeasonPhaseConfig:
    """Tests for get_season_phase_config function."""

    def test_early_season_october(self):
        """Test early season detection (October)."""
        from datetime import date

        from nba.config.thresholds import EARLY_SEASON_CONFIG, get_season_phase_config

        result = get_season_phase_config(date(2025, 10, 25))
        assert result == EARLY_SEASON_CONFIG

    def test_early_season_november(self):
        """Test early season detection (November)."""
        from datetime import date

        from nba.config.thresholds import EARLY_SEASON_CONFIG, get_season_phase_config

        result = get_season_phase_config(date(2025, 11, 15))
        assert result == EARLY_SEASON_CONFIG

    def test_mid_season_december(self):
        """Test mid season detection (December)."""
        from datetime import date

        from nba.config.thresholds import MID_SEASON_CONFIG, get_season_phase_config

        result = get_season_phase_config(date(2025, 12, 15))
        assert result == MID_SEASON_CONFIG

    def test_mid_season_january(self):
        """Test mid season detection (January)."""
        from datetime import date

        from nba.config.thresholds import MID_SEASON_CONFIG, get_season_phase_config

        result = get_season_phase_config(date(2026, 1, 15))
        assert result == MID_SEASON_CONFIG

    def test_mid_season_february(self):
        """Test mid season detection (February)."""
        from datetime import date

        from nba.config.thresholds import MID_SEASON_CONFIG, get_season_phase_config

        result = get_season_phase_config(date(2026, 2, 15))
        assert result == MID_SEASON_CONFIG

    def test_late_season_march(self):
        """Test late season detection (March)."""
        from datetime import date

        from nba.config.thresholds import LATE_SEASON_CONFIG, get_season_phase_config

        result = get_season_phase_config(date(2026, 3, 15))
        assert result == LATE_SEASON_CONFIG

    def test_playoffs_april(self):
        """Test playoffs detection (April)."""
        from datetime import date

        from nba.config.thresholds import PLAYOFFS_CONFIG, get_season_phase_config

        result = get_season_phase_config(date(2026, 4, 20))
        assert result == PLAYOFFS_CONFIG

    def test_playoffs_may(self):
        """Test playoffs detection (May)."""
        from datetime import date

        from nba.config.thresholds import PLAYOFFS_CONFIG, get_season_phase_config

        result = get_season_phase_config(date(2026, 5, 10))
        assert result == PLAYOFFS_CONFIG

    def test_string_date_input(self):
        """Test with string date input."""
        from nba.config.thresholds import EARLY_SEASON_CONFIG, get_season_phase_config

        result = get_season_phase_config("2025-10-25")
        assert result == EARLY_SEASON_CONFIG


# =============================================================================
# TEMPORAL DECAY TESTS
# =============================================================================


class TestTemporalDecayConfig:
    """Tests for TemporalDecayConfig."""

    def test_default_values(self):
        """Test default temporal decay values."""
        from nba.config.thresholds import TEMPORAL_DECAY_CONFIG

        assert TEMPORAL_DECAY_CONFIG.enabled is True
        assert TEMPORAL_DECAY_CONFIG.tau_points == 30.0
        assert TEMPORAL_DECAY_CONFIG.tau_rebounds == 45.0
        assert TEMPORAL_DECAY_CONFIG.min_weight == 0.1

    def test_get_tau_points(self):
        """Test get_tau for POINTS market."""
        from nba.config.thresholds import TEMPORAL_DECAY_CONFIG

        tau = TEMPORAL_DECAY_CONFIG.get_tau("POINTS")
        assert tau == 30.0

    def test_get_tau_rebounds(self):
        """Test get_tau for REBOUNDS market."""
        from nba.config.thresholds import TEMPORAL_DECAY_CONFIG

        tau = TEMPORAL_DECAY_CONFIG.get_tau("REBOUNDS")
        assert tau == 45.0

    def test_get_tau_default(self):
        """Test get_tau for unknown market returns default."""
        from nba.config.thresholds import TEMPORAL_DECAY_CONFIG

        tau = TEMPORAL_DECAY_CONFIG.get_tau("ASSISTS")
        assert tau == 40.0

    def test_get_tau_case_insensitive(self):
        """Test get_tau is case insensitive."""
        from nba.config.thresholds import TEMPORAL_DECAY_CONFIG

        tau_lower = TEMPORAL_DECAY_CONFIG.get_tau("points")
        tau_upper = TEMPORAL_DECAY_CONFIG.get_tau("POINTS")
        assert tau_lower == tau_upper == 30.0


# =============================================================================
# PHASE DECISION THRESHOLDS TESTS
# =============================================================================


class TestPhaseDecisionThresholds:
    """Tests for PhaseDecisionThresholds."""

    def test_default_values(self):
        """Test default phase decision values."""
        from nba.config.thresholds import PHASE_DECISION_THRESHOLDS

        assert PHASE_DECISION_THRESHOLDS.early_min_p_over == 0.62
        assert PHASE_DECISION_THRESHOLDS.mid_min_p_over == 0.58
        assert PHASE_DECISION_THRESHOLDS.late_min_p_over == 0.55

    def test_get_min_p_over_early(self):
        """Test get_min_p_over for early season."""
        from nba.config.thresholds import PHASE_DECISION_THRESHOLDS

        # Days 0-19 are early season
        p_over = PHASE_DECISION_THRESHOLDS.get_min_p_over(10)
        assert p_over == 0.62

    def test_get_min_p_over_mid(self):
        """Test get_min_p_over for mid season."""
        from nba.config.thresholds import PHASE_DECISION_THRESHOLDS

        # Days 20-59 are mid season
        p_over = PHASE_DECISION_THRESHOLDS.get_min_p_over(40)
        assert p_over == 0.58

    def test_get_min_p_over_late(self):
        """Test get_min_p_over for late season."""
        from nba.config.thresholds import PHASE_DECISION_THRESHOLDS

        # Days 60+ are late season
        p_over = PHASE_DECISION_THRESHOLDS.get_min_p_over(80)
        assert p_over == 0.55


# =============================================================================
# STAKE SIZING TESTS
# =============================================================================


class TestStakeSizingConfig:
    """Tests for StakeSizingConfig."""

    def test_default_values(self):
        """Test default stake sizing values."""
        from nba.config.thresholds import STAKE_SIZING_CONFIG

        assert STAKE_SIZING_CONFIG.enabled is True
        assert STAKE_SIZING_CONFIG.base_units == 1.0
        assert STAKE_SIZING_CONFIG.max_units == 3.0
        assert STAKE_SIZING_CONFIG.min_units == 0.25

    def test_get_vol_scale_points(self):
        """Test get_vol_scale for POINTS."""
        from nba.config.thresholds import STAKE_SIZING_CONFIG

        scale = STAKE_SIZING_CONFIG.get_vol_scale("POINTS")
        assert scale == 1.0

    def test_get_vol_scale_rebounds(self):
        """Test get_vol_scale for REBOUNDS."""
        from nba.config.thresholds import STAKE_SIZING_CONFIG

        scale = STAKE_SIZING_CONFIG.get_vol_scale("REBOUNDS")
        assert scale == 1.2

    def test_get_vol_scale_default(self):
        """Test get_vol_scale for unknown market."""
        from nba.config.thresholds import STAKE_SIZING_CONFIG

        scale = STAKE_SIZING_CONFIG.get_vol_scale("ASSISTS")
        assert scale == 1.0


class TestCalculateStake:
    """Tests for calculate_stake function."""

    def test_basic_stake_calculation(self):
        """Test basic stake calculation."""
        from nba.config.thresholds import calculate_stake

        result = calculate_stake(p_over=0.70, edge=3.0, volatility_score=0.20)

        assert "stake_units" in result
        assert "confidence_mult" in result
        assert "volatility_mult" in result
        assert "edge_mult" in result
        assert "volatility_label" in result
        assert "stake_label" in result

    def test_high_confidence_increases_stake(self):
        """Test that high confidence increases stake."""
        from nba.config.thresholds import calculate_stake

        low_conf = calculate_stake(p_over=0.56, edge=3.0, volatility_score=0.15)
        high_conf = calculate_stake(p_over=0.78, edge=3.0, volatility_score=0.15)

        assert high_conf["confidence_mult"] > low_conf["confidence_mult"]
        assert high_conf["stake_units"] >= low_conf["stake_units"]

    def test_high_volatility_decreases_stake(self):
        """Test that high volatility decreases stake."""
        from nba.config.thresholds import calculate_stake

        low_vol = calculate_stake(p_over=0.70, edge=3.0, volatility_score=0.10)
        high_vol = calculate_stake(p_over=0.70, edge=3.0, volatility_score=0.45)

        assert high_vol["volatility_mult"] < low_vol["volatility_mult"]
        assert high_vol["stake_units"] <= low_vol["stake_units"]

    def test_volatility_labels(self):
        """Test volatility label assignment."""
        from nba.config.thresholds import calculate_stake

        stable = calculate_stake(p_over=0.70, edge=3.0, volatility_score=0.10)
        moderate = calculate_stake(p_over=0.70, edge=3.0, volatility_score=0.25)
        volatile = calculate_stake(p_over=0.70, edge=3.0, volatility_score=0.50)

        assert stable["volatility_label"] == "STABLE"
        assert moderate["volatility_label"] == "MODERATE"
        assert volatile["volatility_label"] == "VOLATILE"

    def test_stake_labels(self):
        """Test stake action labels."""
        from nba.config.thresholds import calculate_stake

        # High confidence, low vol, high edge = PRESS
        press_result = calculate_stake(p_over=0.80, edge=4.5, volatility_score=0.10)
        # High vol, low edge = FADE or STANDARD
        fade_result = calculate_stake(p_over=0.56, edge=0.5, volatility_score=0.45)

        assert press_result["stake_label"] in ("PRESS", "STANDARD")
        assert fade_result["stake_label"] in ("FADE", "STANDARD")

    def test_stake_bounded(self):
        """Test stake is bounded by min/max units."""
        from nba.config.thresholds import STAKE_SIZING_CONFIG, calculate_stake

        # Extreme high case
        result_high = calculate_stake(p_over=0.95, edge=10.0, volatility_score=0.01)
        assert result_high["stake_units"] <= STAKE_SIZING_CONFIG.max_units

        # Extreme low case
        result_low = calculate_stake(p_over=0.55, edge=0.5, volatility_score=0.50)
        assert result_low["stake_units"] >= STAKE_SIZING_CONFIG.min_units

    def test_disabled_config(self):
        """Test stake calculation when disabled."""
        from nba.config.thresholds import StakeSizingConfig, calculate_stake

        disabled_config = StakeSizingConfig(enabled=False)
        result = calculate_stake(
            p_over=0.70,
            edge=3.0,
            volatility_score=0.20,
            config=disabled_config,
        )

        assert result["stake_units"] == disabled_config.base_units
        assert result["confidence_mult"] == 1.0
        assert result["volatility_mult"] == 1.0
        assert result["stake_label"] == "STANDARD"

    def test_market_volatility_scaling(self):
        """Test that market affects volatility scaling."""
        from nba.config.thresholds import calculate_stake

        points_result = calculate_stake(
            p_over=0.70, edge=3.0, volatility_score=0.25, market="POINTS"
        )
        rebounds_result = calculate_stake(
            p_over=0.70, edge=3.0, volatility_score=0.25, market="REBOUNDS"
        )

        # REBOUNDS has 1.2x vol scale, so should have lower volatility_mult
        assert rebounds_result["volatility_mult"] <= points_result["volatility_mult"]


# =============================================================================
# FEATURE PREPROCESSING TESTS
# =============================================================================


class TestFeaturePreprocessing:
    """Tests for FeaturePreprocessing."""

    def test_common_cols_to_drop(self):
        """Test common columns to drop."""
        from nba.config.thresholds import FEATURE_PREPROCESSING

        assert "injured_teammates_count" in FEATURE_PREPROCESSING.common_cols_to_drop
        assert "games_in_L7" in FEATURE_PREPROCESSING.common_cols_to_drop

    def test_get_cols_to_drop_points(self):
        """Test get_cols_to_drop for POINTS market."""
        from nba.config.thresholds import FEATURE_PREPROCESSING

        cols = FEATURE_PREPROCESSING.get_cols_to_drop("POINTS")

        # Should include common cols
        assert "injured_teammates_count" in cols
        # Should NOT include POINTS h2h cols (keep them)
        assert "h2h_avg_points" not in cols
        # Should include other market h2h cols (drop them)
        assert "h2h_avg_rebounds" in cols
        assert "h2h_avg_assists" in cols
        assert "h2h_avg_threes" in cols

    def test_get_cols_to_drop_rebounds(self):
        """Test get_cols_to_drop for REBOUNDS market."""
        from nba.config.thresholds import FEATURE_PREPROCESSING

        cols = FEATURE_PREPROCESSING.get_cols_to_drop("REBOUNDS")

        # Should include common cols
        assert "injured_teammates_count" in cols
        # Should NOT include REBOUNDS h2h cols (keep them)
        assert "h2h_avg_rebounds" not in cols
        # Should include other market h2h cols (drop them)
        assert "h2h_avg_points" in cols
        assert "h2h_avg_assists" in cols

    def test_get_cols_to_drop_case_insensitive(self):
        """Test get_cols_to_drop is case insensitive."""
        from nba.config.thresholds import FEATURE_PREPROCESSING

        cols_lower = FEATURE_PREPROCESSING.get_cols_to_drop("points")
        cols_upper = FEATURE_PREPROCESSING.get_cols_to_drop("POINTS")

        assert set(cols_lower) == set(cols_upper)


# =============================================================================
# CONDITIONAL THRESHOLDS TESTS
# =============================================================================


class TestConditionalThresholds:
    """Tests for ConditionalThresholds."""

    def test_default_values(self):
        """Test default conditional threshold values."""
        from nba.config.thresholds import CONDITIONAL_THRESHOLDS

        assert CONDITIONAL_THRESHOLDS.min_consensus_strength == 0.375
        assert CONDITIONAL_THRESHOLDS.max_line_delta == 0.0
        assert CONDITIONAL_THRESHOLDS.min_line_delta == -2.5


class TestCheckConditionalThresholds:
    """Tests for check_conditional_thresholds function."""

    def test_passes_all_checks(self):
        """Test when all conditional checks pass."""
        from nba.config.thresholds import check_conditional_thresholds

        features = {
            "consensus_strength": 0.5,  # 4 books
            "line_delta": -0.5,  # Line dropped
            "line_volatility": 2.0,
            "snapshot_count": 3,
            "game_date": "2025-12-15",
        }

        passes, reason = check_conditional_thresholds(features, "POINTS", 0.70, 3.0)
        assert passes is True
        assert reason == ""

    def test_fails_consensus_strength(self):
        """Test failure when consensus strength too low."""
        from nba.config.thresholds import check_conditional_thresholds

        features = {
            "consensus_strength": 0.2,  # Only ~2 books
            "line_delta": -0.5,
            "line_volatility": 2.0,
            "snapshot_count": 3,
        }

        passes, reason = check_conditional_thresholds(features, "POINTS", 0.70, 3.0)
        assert passes is False
        assert "consensus_strength" in reason

    def test_fails_line_delta_rose(self):
        """Test failure when line rose (delta > 0)."""
        from nba.config.thresholds import check_conditional_thresholds

        features = {
            "consensus_strength": 0.5,
            "line_delta": 1.0,  # Line rose
            "line_volatility": 2.0,
            "snapshot_count": 3,
            "game_date": "2025-12-15",  # Non-playoff
        }

        passes, reason = check_conditional_thresholds(features, "POINTS", 0.70, 3.0)
        assert passes is False
        assert "line_delta" in reason
        assert "rose" in reason

    def test_fails_extreme_line_drop(self):
        """Test failure when line dropped extremely (injury news)."""
        from nba.config.thresholds import check_conditional_thresholds

        features = {
            "consensus_strength": 0.5,
            "line_delta": -3.0,  # Extreme drop
            "line_volatility": 2.0,
            "snapshot_count": 3,
            "game_date": "2025-12-15",
        }

        passes, reason = check_conditional_thresholds(features, "POINTS", 0.70, 3.0)
        assert passes is False
        assert "extreme drop" in reason

    def test_fails_high_line_volatility(self):
        """Test failure when line volatility too high."""
        from nba.config.thresholds import check_conditional_thresholds

        features = {
            "consensus_strength": 0.5,
            "line_delta": -0.5,
            "line_volatility": 10.0,  # Very volatile
            "snapshot_count": 3,
            "game_date": "2025-12-15",
        }

        passes, reason = check_conditional_thresholds(features, "POINTS", 0.70, 3.0)
        assert passes is False
        assert "line_volatility" in reason

    def test_fails_low_snapshot_count(self):
        """Test failure when snapshot count too low."""
        from nba.config.thresholds import check_conditional_thresholds

        features = {
            "consensus_strength": 0.5,
            "line_delta": -0.5,
            "line_volatility": 2.0,
            "snapshot_count": 1,  # Only 1 snapshot
            "game_date": "2025-12-15",
        }

        passes, reason = check_conditional_thresholds(features, "POINTS", 0.70, 3.0)
        assert passes is False
        assert "snapshot_count" in reason

    def test_playoffs_skips_line_delta_check(self):
        """Test that playoffs skip line delta filter."""
        from nba.config.thresholds import check_conditional_thresholds

        features = {
            "consensus_strength": 0.5,
            "line_delta": 1.5,  # Line rose - would fail non-playoffs
            "line_volatility": 2.0,
            "snapshot_count": 3,
            "game_date": "2026-04-20",  # Playoffs
        }

        passes, reason = check_conditional_thresholds(features, "POINTS", 0.70, 3.0)
        # Should pass because line_delta check is skipped in playoffs
        assert passes is True

    def test_no_game_date_uses_mid_season(self):
        """Test that missing game_date uses mid season defaults."""
        from nba.config.thresholds import check_conditional_thresholds

        features = {
            "consensus_strength": 0.5,
            "line_delta": -0.5,
            "line_volatility": 2.0,
            "snapshot_count": 3,
            # No game_date
        }

        passes, reason = check_conditional_thresholds(features, "POINTS", 0.70, 3.0)
        assert passes is True


# =============================================================================
# ADDITIONAL MARKET CONFIG TESTS
# =============================================================================


class TestLineConstraints:
    """Tests for LineConstraints dataclass."""

    def test_default_values(self):
        """Test default line constraint values."""
        from nba.config.thresholds import LineConstraints

        constraints = LineConstraints()
        assert constraints.min_line == 0.0
        assert constraints.max_line == 999.0

    def test_custom_constraints(self):
        """Test custom line constraints."""
        from nba.config.thresholds import LineConstraints

        constraints = LineConstraints(min_line=10.0, max_line=50.0)
        assert constraints.min_line == 10.0
        assert constraints.max_line == 50.0


class TestTrapBookConfig:
    """Tests for trap book configuration."""

    def test_trap_books_defined(self):
        """Test trap books are properly defined."""
        from nba.config.thresholds import TRAP_BOOKS

        assert "DraftKings" in TRAP_BOOKS
        assert "BetMGM" in TRAP_BOOKS
        assert TRAP_BOOKS["DraftKings"].min_spread_required == 3.5

    def test_reliable_books_defined(self):
        """Test reliable books are properly defined."""
        from nba.config.thresholds import RELIABLE_BOOKS

        assert "Underdog" in RELIABLE_BOOKS
        assert "ESPNBet" in RELIABLE_BOOKS
        assert "DraftKings" not in RELIABLE_BOOKS
