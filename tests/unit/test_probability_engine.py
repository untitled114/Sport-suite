"""
Tests for ProbabilityEngine
============================
Tests distribution-based probability calculation and odds conversion.
Uses realistic but hardcoded test values — NO synthetic data.
"""

import math

import pytest

from nba.core.exceptions import DataValidationError
from nba.core.probability_engine import ProbabilityEngine


@pytest.fixture
def engine():
    return ProbabilityEngine()


# =============================================================================
# Normal Distribution (POINTS, REBOUNDS)
# =============================================================================


class TestNormalProbability:
    """Tests for high-count stats using Normal distribution."""

    def test_projection_above_line_gives_over_50(self, engine):
        """When projected > line, P(OVER) should be > 0.50."""
        p = engine.calculate_probability(
            projected_value=25.0, std_dev=5.0, line=22.5, stat_type="POINTS"
        )
        assert p > 0.5

    def test_projection_below_line_gives_under_50(self, engine):
        """When projected < line, P(OVER) should be < 0.50."""
        p = engine.calculate_probability(
            projected_value=20.0, std_dev=5.0, line=22.5, stat_type="POINTS"
        )
        assert p < 0.5

    def test_projection_equals_line_gives_50(self, engine):
        """When projected == line, P(OVER) should be ~0.50."""
        p = engine.calculate_probability(
            projected_value=22.5, std_dev=5.0, line=22.5, stat_type="POINTS"
        )
        assert abs(p - 0.5) < 0.001

    def test_high_confidence_projection(self, engine):
        """Large gap between projection and line should give high P(OVER)."""
        p = engine.calculate_probability(
            projected_value=30.0, std_dev=4.0, line=22.5, stat_type="POINTS"
        )
        # 7.5 points above line with std=4 => ~97% probability
        assert p > 0.95

    def test_rebounds_uses_normal(self, engine):
        """REBOUNDS should use Normal distribution."""
        p = engine.calculate_probability(
            projected_value=10.0, std_dev=3.0, line=8.5, stat_type="REBOUNDS"
        )
        assert 0.5 < p < 1.0

    def test_zero_std_projection_above(self, engine):
        """Zero std with projection above line => P(OVER) = 1.0."""
        p = engine.calculate_probability(
            projected_value=25.0, std_dev=0.0, line=22.5, stat_type="POINTS"
        )
        assert p == 1.0

    def test_zero_std_projection_below(self, engine):
        """Zero std with projection below line => P(OVER) = 0.0."""
        p = engine.calculate_probability(
            projected_value=20.0, std_dev=0.0, line=22.5, stat_type="POINTS"
        )
        assert p == 0.0

    def test_higher_std_widens_distribution(self, engine):
        """Higher std should make P(OVER) closer to 0.5."""
        p_tight = engine.calculate_probability(
            projected_value=25.0, std_dev=2.0, line=22.5, stat_type="POINTS"
        )
        p_wide = engine.calculate_probability(
            projected_value=25.0, std_dev=8.0, line=22.5, stat_type="POINTS"
        )
        # Tight std should be more confident
        assert p_tight > p_wide

    def test_assists_uses_normal(self, engine):
        """ASSISTS should use Normal distribution."""
        p = engine.calculate_probability(
            projected_value=8.0, std_dev=2.5, line=7.5, stat_type="ASSISTS"
        )
        assert 0.5 < p < 1.0


# =============================================================================
# Poisson Distribution (THREES, BLOCKS, STEALS)
# =============================================================================


class TestPoissonProbability:
    """Tests for low-count stats using Poisson distribution."""

    def test_threes_above_line(self, engine):
        """Higher lambda than line should give P(OVER) > 0.50."""
        p = engine.calculate_probability(
            projected_value=3.5, std_dev=1.5, line=2.5, stat_type="THREES"
        )
        assert p > 0.5

    def test_threes_below_line(self, engine):
        """Lower lambda than line should give P(OVER) < 0.50."""
        p = engine.calculate_probability(
            projected_value=1.5, std_dev=1.0, line=2.5, stat_type="THREES"
        )
        assert p < 0.5

    def test_half_point_line_threes(self, engine):
        """Over 2.5 means >= 3 three-pointers."""
        # Lambda = 2.5 => P(X >= 3) = 1 - P(X <= 2) ~ 0.456
        p = engine.calculate_probability(
            projected_value=2.5, std_dev=1.5, line=2.5, stat_type="THREES"
        )
        assert 0.3 < p < 0.6

    def test_blocks_uses_poisson(self, engine):
        """BLOCKS should use Poisson distribution."""
        p = engine.calculate_probability(
            projected_value=2.0, std_dev=1.0, line=1.5, stat_type="BLOCKS"
        )
        assert 0.0 < p < 1.0

    def test_steals_uses_poisson(self, engine):
        """STEALS should use Poisson distribution."""
        p = engine.calculate_probability(
            projected_value=1.5, std_dev=1.0, line=0.5, stat_type="STEALS"
        )
        assert p > 0.5

    def test_zero_lambda_returns_zero(self, engine):
        """Zero lambda (no expected threes) => P(OVER) = 0."""
        p = engine.calculate_probability(
            projected_value=0.0, std_dev=0.0, line=0.5, stat_type="THREES"
        )
        assert p == 0.0

    def test_whole_number_line_threes(self, engine):
        """Over 3.0 means >= 4 three-pointers (strict over)."""
        # Lambda = 3.0 => P(X >= 4) = 1 - P(X <= 3)
        p = engine.calculate_probability(
            projected_value=3.0, std_dev=1.5, line=3.0, stat_type="THREES"
        )
        # Should be less than if line were 2.5
        p_half = engine.calculate_probability(
            projected_value=3.0, std_dev=1.5, line=2.5, stat_type="THREES"
        )
        assert p < p_half


# =============================================================================
# Implied Probability (American Odds Conversion)
# =============================================================================


class TestImpliedProbability:
    """Tests for American odds to implied probability conversion."""

    def test_standard_minus_110(self, engine):
        """Standard -110 odds => ~52.4% implied probability."""
        p = engine.implied_probability(-110)
        assert abs(p - 0.524) < 0.001

    def test_heavy_favorite_minus_200(self, engine):
        """-200 odds => ~66.7% implied probability."""
        p = engine.implied_probability(-200)
        assert abs(p - 0.667) < 0.001

    def test_underdog_plus_150(self, engine):
        """+150 odds => 40% implied probability."""
        p = engine.implied_probability(150)
        assert abs(p - 0.400) < 0.001

    def test_even_money_plus_100(self, engine):
        """+100 odds => 50% implied probability."""
        p = engine.implied_probability(100)
        assert abs(p - 0.500) < 0.001

    def test_minus_100(self, engine):
        """-100 odds => 50% implied probability."""
        p = engine.implied_probability(-100)
        assert abs(p - 0.500) < 0.001

    def test_zero_odds_raises(self, engine):
        """Zero odds are invalid."""
        with pytest.raises(DataValidationError):
            engine.implied_probability(0)

    def test_heavy_underdog_plus_500(self, engine):
        """+500 odds => ~16.7% implied probability."""
        p = engine.implied_probability(500)
        assert abs(p - 0.167) < 0.01


# =============================================================================
# Decimal Odds Conversion
# =============================================================================


class TestDecimalOdds:
    """Tests for American to decimal odds conversion."""

    def test_minus_110_decimal(self, engine):
        """-110 American => ~1.909 decimal."""
        d = engine.american_to_decimal(-110)
        assert abs(d - 1.909) < 0.001

    def test_plus_150_decimal(self, engine):
        """+150 American => 2.50 decimal."""
        d = engine.american_to_decimal(150)
        assert abs(d - 2.50) < 0.001

    def test_even_money_decimal(self, engine):
        """+100 American => 2.00 decimal."""
        d = engine.american_to_decimal(100)
        assert abs(d - 2.00) < 0.001

    def test_zero_odds_raises(self, engine):
        with pytest.raises(DataValidationError):
            engine.american_to_decimal(0)


# =============================================================================
# Expected Value
# =============================================================================


class TestExpectedValue:
    """Tests for expected value calculation."""

    def test_positive_ev_with_edge(self, engine):
        """Model prob higher than implied => positive EV."""
        ev = engine.expected_value(model_prob=0.60, american_odds=-110)
        assert ev > 0

    def test_negative_ev_without_edge(self, engine):
        """Model prob lower than implied => negative EV."""
        ev = engine.expected_value(model_prob=0.50, american_odds=-110)
        assert ev < 0

    def test_break_even_at_implied(self, engine):
        """At exactly implied probability, EV should be ~0."""
        # -110 implied = 0.524
        ev = engine.expected_value(model_prob=0.524, american_odds=-110)
        assert abs(ev) < 0.01


# =============================================================================
# Input Validation
# =============================================================================


class TestInputValidation:
    """Tests for input validation."""

    def test_negative_projection_raises(self, engine):
        with pytest.raises(DataValidationError):
            engine.calculate_probability(-1.0, 5.0, 22.5, "POINTS")

    def test_negative_std_raises(self, engine):
        with pytest.raises(DataValidationError):
            engine.calculate_probability(25.0, -1.0, 22.5, "POINTS")

    def test_negative_line_raises(self, engine):
        with pytest.raises(DataValidationError):
            engine.calculate_probability(25.0, 5.0, -1.0, "POINTS")

    def test_output_bounded_0_1(self, engine):
        """Probability should always be between 0 and 1."""
        p = engine.calculate_probability(25.0, 5.0, 22.5, "POINTS")
        assert 0.0 <= p <= 1.0
