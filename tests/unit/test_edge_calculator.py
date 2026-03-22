"""
Tests for EdgeCalculator
=========================
Tests edge calculation, Kelly fraction, and edge thresholds.
Uses realistic but hardcoded test values — NO synthetic data.
"""

import pytest

from nba.core.edge_calculator import EdgeCalculator, EdgeConfig
from nba.core.exceptions import DataValidationError


@pytest.fixture
def calc():
    return EdgeCalculator(min_edge=0.05, max_edge=0.25)


# =============================================================================
# Basic Edge Calculation
# =============================================================================


class TestEdgeCalculation:
    """Tests for edge = model_prob - implied_prob."""

    def test_positive_edge(self, calc):
        """Model prob > implied prob => positive edge."""
        result = calc.calculate_edge(model_prob=0.62, book_odds=-110)
        assert result["edge"] > 0
        assert result["has_edge"] is True

    def test_no_edge(self, calc):
        """Model prob < implied prob + threshold => no edge."""
        result = calc.calculate_edge(model_prob=0.53, book_odds=-110)
        # 0.53 - 0.524 = 0.006, below 0.05 threshold
        assert result["has_edge"] is False

    def test_negative_edge(self, calc):
        """Model prob well below implied => negative edge."""
        result = calc.calculate_edge(model_prob=0.40, book_odds=-110)
        assert result["edge"] < 0
        assert result["has_edge"] is False

    def test_exact_threshold(self, calc):
        """Edge exactly at threshold should qualify."""
        # Need model_prob = 0.524 + 0.05 = 0.574
        result = calc.calculate_edge(model_prob=0.574, book_odds=-110)
        assert result["has_edge"] is True

    def test_edge_percentage(self, calc):
        """Edge percentage should be edge * 100."""
        result = calc.calculate_edge(model_prob=0.62, book_odds=-110)
        assert abs(result["edge_pct"] - result["edge"] * 100) < 0.1

    def test_suspicious_large_edge(self, calc):
        """Very large edge should be flagged as suspicious."""
        result = calc.calculate_edge(model_prob=0.90, book_odds=-110)
        # 0.90 - 0.524 = 0.376, above 0.25 threshold
        assert result["is_suspicious"] is True

    def test_standard_edge_not_suspicious(self, calc):
        """Normal edge should not be flagged."""
        result = calc.calculate_edge(model_prob=0.62, book_odds=-110)
        assert result["is_suspicious"] is False


# =============================================================================
# Kelly Criterion
# =============================================================================


class TestKellyFraction:
    """Tests for Kelly criterion calculation."""

    def test_kelly_with_edge(self, calc):
        """Positive edge should give positive Kelly fraction."""
        result = calc.calculate_edge(model_prob=0.62, book_odds=-110)
        assert result["kelly_fraction"] > 0

    def test_kelly_without_edge(self, calc):
        """No edge should give zero Kelly fraction."""
        result = calc.calculate_edge(model_prob=0.40, book_odds=-110)
        assert result["kelly_fraction"] == 0

    def test_kelly_capped(self, calc):
        """Kelly fraction should never exceed the cap (0.25)."""
        result = calc.calculate_edge(model_prob=0.95, book_odds=-110)
        assert result["kelly_fraction"] <= 0.25

    def test_kelly_scales_with_edge(self, calc):
        """Larger edge should give larger Kelly fraction."""
        r1 = calc.calculate_edge(model_prob=0.60, book_odds=-110)
        r2 = calc.calculate_edge(model_prob=0.70, book_odds=-110)
        assert r2["kelly_fraction"] >= r1["kelly_fraction"]


# =============================================================================
# Expected Value
# =============================================================================


class TestExpectedValue:
    """Tests for EV calculation within edge results."""

    def test_positive_ev_with_edge(self, calc):
        result = calc.calculate_edge(model_prob=0.62, book_odds=-110)
        assert result["expected_value"] > 0

    def test_negative_ev_without_edge(self, calc):
        result = calc.calculate_edge(model_prob=0.45, book_odds=-110)
        assert result["expected_value"] < 0


# =============================================================================
# Odds Handling
# =============================================================================


class TestOddsHandling:
    """Tests for different odds formats."""

    def test_favorite_odds(self, calc):
        """Favorite odds (-200) should have higher implied probability."""
        r1 = calc.calculate_edge(model_prob=0.62, book_odds=-110)
        r2 = calc.calculate_edge(model_prob=0.62, book_odds=-200)
        # Higher implied prob => lower edge
        assert r2["edge"] < r1["edge"]

    def test_underdog_odds(self, calc):
        """Underdog odds (+150) should have lower implied probability."""
        r1 = calc.calculate_edge(model_prob=0.62, book_odds=-110)
        r2 = calc.calculate_edge(model_prob=0.62, book_odds=150)
        # Lower implied prob => higher edge
        assert r2["edge"] > r1["edge"]

    def test_decimal_odds_returned(self, calc):
        result = calc.calculate_edge(model_prob=0.62, book_odds=-110)
        assert abs(result["decimal_odds"] - 1.909) < 0.01


# =============================================================================
# Custom Config
# =============================================================================


class TestCustomConfig:
    """Tests for custom edge thresholds."""

    def test_strict_threshold(self):
        """Stricter threshold should reject more bets."""
        strict = EdgeCalculator(min_edge=0.10)
        result = strict.calculate_edge(model_prob=0.60, book_odds=-110)
        # 0.60 - 0.524 = 0.076, below 0.10
        assert result["has_edge"] is False

    def test_loose_threshold(self):
        """Looser threshold should accept more bets."""
        loose = EdgeCalculator(min_edge=0.02)
        result = loose.calculate_edge(model_prob=0.55, book_odds=-110)
        # 0.55 - 0.524 = 0.026, above 0.02
        assert result["has_edge"] is True


# =============================================================================
# From Pre-computed Probabilities
# =============================================================================


class TestFromProbs:
    """Tests for calculate_edge_from_probs."""

    def test_from_probs_matches(self, calc):
        """Should give same results as calculate_edge for same inputs."""
        r1 = calc.calculate_edge(model_prob=0.62, book_odds=-110)
        r2 = calc.calculate_edge_from_probs(
            model_prob=0.62,
            implied_prob=r1["implied_prob"],
            decimal_odds=r1["decimal_odds"],
        )
        assert abs(r1["edge"] - r2["edge"]) < 0.001
        assert abs(r1["kelly_fraction"] - r2["kelly_fraction"]) < 0.001

    def test_from_probs_validates_model_prob(self, calc):
        with pytest.raises(DataValidationError):
            calc.calculate_edge_from_probs(model_prob=1.5, implied_prob=0.5, decimal_odds=2.0)

    def test_from_probs_validates_implied_prob(self, calc):
        """implied_prob=0 should raise."""
        with pytest.raises(DataValidationError):
            calc.calculate_edge_from_probs(model_prob=0.6, implied_prob=0.0, decimal_odds=2.0)

    def test_from_probs_no_edge(self, calc):
        """No edge case in calculate_edge_from_probs."""
        result = calc.calculate_edge_from_probs(
            model_prob=0.40, implied_prob=0.524, decimal_odds=1.909
        )
        assert result["kelly_fraction"] == 0.0
        assert result["has_edge"] is False


# =============================================================================
# Input Validation
# =============================================================================


class TestValidation:
    """Tests for input validation."""

    def test_model_prob_too_high(self, calc):
        with pytest.raises(DataValidationError):
            calc.calculate_edge(model_prob=1.1, book_odds=-110)

    def test_model_prob_too_low(self, calc):
        with pytest.raises(DataValidationError):
            calc.calculate_edge(model_prob=-0.1, book_odds=-110)

    def test_boundary_model_prob_zero(self, calc):
        """model_prob=0 should be valid (no edge)."""
        result = calc.calculate_edge(model_prob=0.0, book_odds=-110)
        assert result["has_edge"] is False

    def test_boundary_model_prob_one(self, calc):
        """model_prob=1.0 should be valid."""
        result = calc.calculate_edge(model_prob=1.0, book_odds=-110)
        assert result["has_edge"] is True
