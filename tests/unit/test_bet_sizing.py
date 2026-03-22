"""
Tests for KellySizer
=====================
Tests fractional Kelly criterion bet sizing.
Uses realistic but hardcoded test values — NO synthetic data.
"""

import pytest

from nba.core.bet_sizing import KellyConfig, KellySizer
from nba.core.exceptions import DataValidationError


@pytest.fixture
def sizer():
    return KellySizer(bankroll=1000.0, fraction=0.25, max_bet_pct=0.03)


# =============================================================================
# Basic Bet Sizing
# =============================================================================


class TestBasicSizing:
    """Tests for basic Kelly bet sizing."""

    def test_bet_with_positive_edge(self, sizer):
        """Positive edge should produce a non-zero bet."""
        result = sizer.size_bet(model_prob=0.62, odds=-110)
        assert result["bet_amount"] > 0
        assert result["has_edge"] is True

    def test_no_bet_without_edge(self, sizer):
        """No edge should produce zero bet."""
        result = sizer.size_bet(model_prob=0.45, odds=-110)
        assert result["bet_amount"] == 0
        assert result["has_edge"] is False

    def test_bet_scales_with_edge(self, sizer):
        """Larger edge should produce larger bet."""
        r1 = sizer.size_bet(model_prob=0.60, odds=-110)
        r2 = sizer.size_bet(model_prob=0.70, odds=-110)
        assert r2["bet_amount"] >= r1["bet_amount"]

    def test_fractional_kelly_smaller_than_full(self, sizer):
        """Quarter Kelly should be 25% of full Kelly."""
        result = sizer.size_bet(model_prob=0.62, odds=-110)
        assert result["scaled_kelly"] == pytest.approx(result["kelly_fraction"] * 0.25, abs=0.001)


# =============================================================================
# Hard Cap
# =============================================================================


class TestHardCap:
    """Tests for bankroll protection hard cap."""

    def test_never_exceeds_3_percent(self, sizer):
        """Bet should never exceed 3% of bankroll."""
        result = sizer.size_bet(model_prob=0.95, odds=-110)
        max_bet = 1000.0 * 0.03
        assert result["bet_amount"] <= max_bet

    def test_cap_applied_flag(self, sizer):
        """is_capped should be True when cap is applied."""
        result = sizer.size_bet(model_prob=0.95, odds=-110)
        assert result["is_capped"] is True

    def test_normal_bet_not_capped(self, sizer):
        """Small edge should not trigger cap."""
        result = sizer.size_bet(model_prob=0.56, odds=-110)
        assert result["is_capped"] is False

    def test_custom_cap(self):
        """Custom max_bet_pct should be respected."""
        sizer = KellySizer(bankroll=1000.0, fraction=0.25, max_bet_pct=0.01)
        result = sizer.size_bet(model_prob=0.95, odds=-110)
        assert result["bet_amount"] <= 10.0  # 1% of 1000


# =============================================================================
# Bankroll Management
# =============================================================================


class TestBankrollManagement:
    """Tests for bankroll-related behavior."""

    def test_bet_proportional_to_bankroll(self):
        """Bet amount should scale with bankroll."""
        s1 = KellySizer(bankroll=1000.0)
        s2 = KellySizer(bankroll=2000.0)
        r1 = s1.size_bet(model_prob=0.62, odds=-110)
        r2 = s2.size_bet(model_prob=0.62, odds=-110)
        # Double bankroll should double bet (unless capped)
        if not r1["is_capped"] and not r2["is_capped"]:
            assert abs(r2["bet_amount"] - r1["bet_amount"] * 2) < 1.0

    def test_override_bankroll(self, sizer):
        """Should be able to override bankroll per call."""
        r1 = sizer.size_bet(model_prob=0.62, odds=-110, bankroll=500.0)
        r2 = sizer.size_bet(model_prob=0.62, odds=-110, bankroll=2000.0)
        if not r1["is_capped"] and not r2["is_capped"]:
            assert r2["bet_amount"] > r1["bet_amount"]

    def test_override_fraction(self, sizer):
        """Should be able to override Kelly fraction per call."""
        r_quarter = sizer.size_bet(model_prob=0.62, odds=-110, fraction=0.25)
        r_half = sizer.size_bet(model_prob=0.62, odds=-110, fraction=0.50)
        assert r_half["scaled_kelly"] >= r_quarter["scaled_kelly"]

    def test_update_bankroll(self, sizer):
        """update_bankroll should change future bets."""
        r1 = sizer.size_bet(model_prob=0.62, odds=-110)
        sizer.update_bankroll(2000.0)
        r2 = sizer.size_bet(model_prob=0.62, odds=-110)
        assert sizer.bankroll == 2000.0
        if not r1["is_capped"] and not r2["is_capped"]:
            assert r2["bet_amount"] > r1["bet_amount"]


# =============================================================================
# Odds Handling
# =============================================================================


class TestOddsHandling:
    """Tests for different odds scenarios."""

    def test_favorite_odds(self, sizer):
        """-200 favorite should have smaller bet (less value)."""
        r_standard = sizer.size_bet(model_prob=0.70, odds=-110)
        r_favorite = sizer.size_bet(model_prob=0.70, odds=-200)
        # At -200 you need 66.7% just to break even, so less edge
        assert r_favorite["bet_amount"] <= r_standard["bet_amount"]

    def test_underdog_odds(self, sizer):
        """+150 underdog with same prob should have larger Kelly."""
        r_standard = sizer.size_bet(model_prob=0.55, odds=-110)
        r_underdog = sizer.size_bet(model_prob=0.55, odds=150)
        assert r_underdog["kelly_fraction"] >= r_standard["kelly_fraction"]

    def test_even_money(self, sizer):
        """+100 with 55% model prob should have positive bet."""
        result = sizer.size_bet(model_prob=0.55, odds=100)
        assert result["bet_amount"] > 0
        assert result["has_edge"] is True


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for boundary conditions."""

    def test_model_prob_below_implied(self, sizer):
        """Model prob below implied => zero Kelly, zero bet."""
        result = sizer.size_bet(model_prob=0.50, odds=-110)
        assert result["kelly_fraction"] == 0.0
        assert result["bet_amount"] == 0.0

    def test_minimum_bet_threshold(self):
        """Very small edge with small bankroll => skip (below min bet)."""
        sizer = KellySizer(bankroll=100.0, fraction=0.10)
        result = sizer.size_bet(model_prob=0.53, odds=-110)
        # Very tiny edge, small fraction, small bankroll => below $1 min
        assert result["bet_amount"] == 0.0

    def test_bet_units_calculation(self, sizer):
        """bet_units should be bet_pct * 100 (1 unit = 1% bankroll)."""
        result = sizer.size_bet(model_prob=0.62, odds=-110)
        assert abs(result["bet_units"] - result["bet_pct"] * 100) < 0.01


# =============================================================================
# Input Validation
# =============================================================================


class TestValidation:
    """Tests for input validation."""

    def test_zero_bankroll_raises(self):
        with pytest.raises(DataValidationError):
            KellySizer(bankroll=0.0)

    def test_negative_bankroll_raises(self):
        with pytest.raises(DataValidationError):
            KellySizer(bankroll=-100.0)

    def test_fraction_above_one_raises(self):
        with pytest.raises(DataValidationError):
            KellySizer(bankroll=1000.0, fraction=1.5)

    def test_fraction_zero_raises(self):
        with pytest.raises(DataValidationError):
            KellySizer(bankroll=1000.0, fraction=0.0)

    def test_model_prob_out_of_range(self, sizer):
        with pytest.raises(DataValidationError):
            sizer.size_bet(model_prob=1.5, odds=-110)

    def test_update_bankroll_zero_raises(self, sizer):
        with pytest.raises(DataValidationError):
            sizer.update_bankroll(0.0)

    def test_max_bet_pct_zero_raises(self):
        with pytest.raises(DataValidationError):
            KellySizer(bankroll=1000.0, max_bet_pct=0.0)

    def test_max_bet_pct_above_one_raises(self):
        with pytest.raises(DataValidationError):
            KellySizer(bankroll=1000.0, max_bet_pct=1.5)
