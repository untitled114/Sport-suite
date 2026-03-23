"""
Tests for Context Modifier (Model 3) — Rule-Based Projection Adjustments.

Tests multiplier computation, clamping, adjust(), and adjust_batch().
"""

import pytest

from nba.models.context_modifier import ContextCoefficients, ContextModifier


class TestComputeMultiplierHomeAway:
    """Test home/away effects on multiplier."""

    def test_compute_multiplier_home(self):
        """Home court increases multiplier for POINTS."""
        modifier = ContextModifier()
        ctx = {"is_home": True, "is_back_to_back": False, "days_rest": 2}
        m = modifier.compute_multiplier("POINTS", ctx)
        # Home adds +0.014, base 1.0, away subtracts nothing
        assert m > 1.0
        assert m == pytest.approx(1.0 + 0.014, abs=0.001)

    def test_compute_multiplier_away(self):
        """Away decreases multiplier for POINTS."""
        modifier = ContextModifier()
        ctx = {"is_home": False, "is_back_to_back": False, "days_rest": 2}
        m = modifier.compute_multiplier("POINTS", ctx)
        # Away subtracts -0.014 from base 1.0
        assert m < 1.0
        assert m == pytest.approx(1.0 - 0.014, abs=0.001)


class TestComputeMultiplierB2B:
    """Test back-to-back effects."""

    def test_compute_multiplier_b2b_points(self):
        """B2B adds positive coefficient for POINTS (empirically slightly positive)."""
        modifier = ContextModifier()
        ctx_no_b2b = {"is_home": True, "is_back_to_back": False, "days_rest": 1}
        ctx_b2b = {"is_home": True, "is_back_to_back": True, "days_rest": 1}
        m_no = modifier.compute_multiplier("POINTS", ctx_no_b2b)
        m_b2b = modifier.compute_multiplier("POINTS", ctx_b2b)
        # POINTS B2B coefficient is +0.0047, so B2B should increase
        assert m_b2b > m_no

    def test_compute_multiplier_b2b_rebounds(self):
        """B2B decreases multiplier for REBOUNDS (negative coefficient)."""
        modifier = ContextModifier()
        ctx_no_b2b = {"is_home": True, "is_back_to_back": False, "days_rest": 1}
        ctx_b2b = {"is_home": True, "is_back_to_back": True, "days_rest": 1}
        m_no = modifier.compute_multiplier("REBOUNDS", ctx_no_b2b)
        m_b2b = modifier.compute_multiplier("REBOUNDS", ctx_b2b)
        # REBOUNDS B2B coefficient is -0.01, so B2B should decrease
        assert m_b2b < m_no


class TestComputeMultiplierBlowout:
    """Test blowout risk effects."""

    def test_compute_multiplier_blowout(self):
        """Blowout spread (|spread| > 10) triggers blowout coefficient."""
        modifier = ContextModifier()
        ctx_normal = {"is_home": True, "vegas_spread": -3.0, "days_rest": 2}
        ctx_blowout = {"is_home": True, "vegas_spread": -12.0, "days_rest": 2}
        m_normal = modifier.compute_multiplier("POINTS", ctx_normal)
        m_blowout = modifier.compute_multiplier("POINTS", ctx_blowout)
        # Blowout adds the blowout coefficient (+0.0007 for POINTS)
        # but spread_per_pt effect also changes, so just check it's different
        assert m_blowout != m_normal

    def test_compute_multiplier_blowout_rebounds_negative(self):
        """REBOUNDS blowout coefficient is negative."""
        modifier = ContextModifier()
        ctx_normal = {"is_home": True, "vegas_spread": -3.0, "days_rest": 2}
        ctx_blowout = {"is_home": True, "vegas_spread": -12.0, "days_rest": 2}
        m_normal = modifier.compute_multiplier("REBOUNDS", ctx_normal)
        m_blowout = modifier.compute_multiplier("REBOUNDS", ctx_blowout)
        # REBOUNDS blowout = -0.0083, so blowout should be lower
        # (ignoring spread_per_pt which is 0 for rebounds)
        assert m_blowout < m_normal


class TestComputeMultiplierRest:
    """Test rest day effects."""

    def test_compute_multiplier_rest_3plus(self):
        """3+ rest days applies rest coefficient."""
        modifier = ContextModifier()
        ctx_2d = {"is_home": True, "days_rest": 2}
        ctx_3d = {"is_home": True, "days_rest": 3}
        m_2d = modifier.compute_multiplier("POINTS", ctx_2d)
        m_3d = modifier.compute_multiplier("POINTS", ctx_3d)
        # POINTS rest_3plus = -0.0099, so 3+ rest lowers multiplier
        assert m_3d < m_2d

    def test_rest_none_defaults_to_2(self):
        """When days_rest is None, it defaults to 2 (no rest effect)."""
        modifier = ContextModifier()
        ctx = {"is_home": True, "days_rest": None}
        m = modifier.compute_multiplier("POINTS", ctx)
        # days_rest=None defaults to 2, which is < 3, so no rest effect
        # Only home effect applies
        assert m == pytest.approx(1.0 + 0.014, abs=0.001)


class TestMultiplierClamping:
    """Test that extreme multipliers are clamped to [0.90, 1.10]."""

    def test_multiplier_clamped_upper(self):
        """Extreme positive context stays at 1.10 max."""
        # Create coefficients with large values to force clamping
        extreme_coeff = ContextCoefficients(
            points_home=0.20,
            points_b2b=0.20,
            points_rest_3plus=0.20,
            points_spread_per_pt=0.0,
            points_blowout=0.0,
        )
        modifier = ContextModifier(coefficients=extreme_coeff)
        ctx = {"is_home": True, "is_back_to_back": True, "days_rest": 5}
        m = modifier.compute_multiplier("POINTS", ctx)
        assert m <= 1.10

    def test_multiplier_clamped_lower(self):
        """Extreme negative context stays at 0.90 min."""
        extreme_coeff = ContextCoefficients(
            points_home=0.20,
            points_b2b=-0.20,
            points_rest_3plus=-0.20,
            points_spread_per_pt=0.0,
            points_blowout=-0.20,
        )
        modifier = ContextModifier(coefficients=extreme_coeff)
        ctx = {
            "is_home": False,
            "is_back_to_back": True,
            "days_rest": 5,
            "vegas_spread": -15,
        }
        m = modifier.compute_multiplier("POINTS", ctx)
        assert m >= 0.90


class TestAdjust:
    """Test adjust() and adjust_batch() methods."""

    def test_adjust_applies_multiplier(self):
        """adjust() returns projection * multiplier, rounded to 2 decimals."""
        modifier = ContextModifier()
        ctx = {"is_home": True, "days_rest": 2}
        result = modifier.adjust(25.0, "POINTS", ctx)
        multiplier = modifier.compute_multiplier("POINTS", ctx)
        assert result == round(25.0 * multiplier, 2)

    def test_adjust_batch(self):
        """adjust_batch() processes multiple projections correctly."""
        modifier = ContextModifier()
        projections = [20.0, 25.0, 30.0]
        contexts = [
            {"is_home": True, "days_rest": 2},
            {"is_home": False, "days_rest": 1},
            {"is_home": True, "is_back_to_back": True, "days_rest": 1},
        ]
        results = modifier.adjust_batch(projections, "POINTS", contexts)
        assert len(results) == 3
        # Each result should match individual adjust calls
        for i, (proj, ctx) in enumerate(zip(projections, contexts)):
            expected = modifier.adjust(proj, "POINTS", ctx)
            assert results[i] == expected


class TestDefaultContext:
    """Test handling of empty/missing context keys."""

    def test_default_context_empty_dict(self):
        """Empty context dict uses all defaults gracefully."""
        modifier = ContextModifier()
        m = modifier.compute_multiplier("POINTS", {})
        # is_home=False (subtract home), no b2b, days_rest=2 (< 3), no spread
        expected = 1.0 - 0.014
        assert m == pytest.approx(expected, abs=0.001)

    def test_default_context_missing_keys(self):
        """Missing context keys don't raise errors."""
        modifier = ContextModifier()
        m = modifier.compute_multiplier("REBOUNDS", {"is_home": True})
        # Only home effect, everything else defaults
        assert isinstance(m, float)
        assert 0.90 <= m <= 1.10

    def test_unknown_market_uses_zero_coefficients(self):
        """Unknown market gets zero coefficients via getattr defaults."""
        modifier = ContextModifier()
        m = modifier.compute_multiplier("STEALS", {"is_home": True})
        # All getattr calls return 0 for unknown market
        assert m == 1.0

    def test_no_spread_skips_spread_effect(self):
        """When vegas_spread is None, spread and blowout effects are skipped."""
        modifier = ContextModifier()
        ctx = {"is_home": True, "vegas_spread": None, "days_rest": 2}
        m = modifier.compute_multiplier("POINTS", ctx)
        # Only home effect
        expected = 1.0 + 0.014
        assert m == pytest.approx(expected, abs=0.001)
