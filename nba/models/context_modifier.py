#!/usr/bin/env python3
"""
Context Modifier (Model 3) — Rule-Based Projection Adjustments
================================================================
Applies empirically-derived adjustments to Model 1 projections based on
game context (home/away, back-to-back, blowout risk, rest days).

Coefficients derived from linear regression of (actual / season_avg) on
43,998 game logs (2023-2026, minutes >= 20). R² is near zero (~0.0004),
meaning these effects are small but directionally correct.

The modifier is multiplicative:
    adjusted_projection = projection * context_multiplier

Usage:
    from nba.models.context_modifier import ContextModifier
    modifier = ContextModifier()
    adjusted = modifier.adjust(projection=25.3, market="POINTS", context={
        "is_home": True,
        "is_back_to_back": False,
        "days_rest": 2,
        "vegas_spread": -3.5,
    })
"""

import logging
from dataclasses import dataclass
from typing import Any

log = logging.getLogger("nba.context_modifier")


@dataclass(frozen=True)
class ContextCoefficients:
    """Empirically derived from 43,998 game logs (2023-2026).

    Each coefficient represents the multiplicative effect on actual/season_avg.
    E.g., home=+0.014 means home players score ~1.4% more than their average.
    """

    # POINTS coefficients (R² = 0.0004)
    points_home: float = 0.0140
    points_b2b: float = 0.0047
    points_rest_3plus: float = -0.0099
    points_spread_per_pt: float = -0.0002
    points_blowout: float = 0.0007

    # REBOUNDS coefficients (R² = 0.0002)
    rebounds_home: float = 0.0070
    rebounds_b2b: float = -0.0100
    rebounds_rest_3plus: float = -0.0089
    rebounds_spread_per_pt: float = 0.0
    rebounds_blowout: float = -0.0083


CONTEXT_COEFFICIENTS = ContextCoefficients()


class ContextModifier:
    """Apply context-based adjustments to Model 1 projections.

    Effects are small (R² ~ 0.0004) but directionally correct.
    All multipliers are clamped to [0.90, 1.10] to prevent overcorrection.
    """

    def __init__(self, coefficients: ContextCoefficients = CONTEXT_COEFFICIENTS):
        self.coeff = coefficients

    def compute_multiplier(self, market: str, context: dict[str, Any]) -> float:
        """Compute the context multiplier for a given game context.

        Args:
            market: "POINTS" or "REBOUNDS"
            context: Dict with keys:
                is_home (bool), is_back_to_back (bool),
                days_rest (int), vegas_spread (float, from team's perspective)

        Returns:
            Multiplier in [0.90, 1.10] range.
        """
        c = self.coeff
        prefix = market.lower()

        home = getattr(c, f"{prefix}_home", 0)
        b2b = getattr(c, f"{prefix}_b2b", 0)
        rest_3 = getattr(c, f"{prefix}_rest_3plus", 0)
        spread_pt = getattr(c, f"{prefix}_spread_per_pt", 0)
        blowout = getattr(c, f"{prefix}_blowout", 0)

        multiplier = 1.0

        # Home/away
        is_home = context.get("is_home", False)
        if is_home:
            multiplier += home
        else:
            multiplier -= home

        # Back-to-back
        if context.get("is_back_to_back", False):
            multiplier += b2b

        # Rest advantage (3+ days)
        days_rest = context.get("days_rest", 2)
        if days_rest is not None and days_rest >= 3:
            multiplier += rest_3

        # Spread effect (continuous)
        spread = context.get("vegas_spread")
        if spread is not None:
            multiplier += spread_pt * spread

        # Blowout risk (|spread| > 10)
        if spread is not None and abs(spread) > 10:
            multiplier += blowout

        # Clamp to prevent overcorrection
        multiplier = max(0.90, min(1.10, multiplier))
        return round(multiplier, 4)

    def adjust(
        self,
        projection: float,
        market: str,
        context: dict[str, Any],
    ) -> float:
        """Apply context adjustment to a projection.

        Args:
            projection: Model 1's raw projection (e.g., 25.3 points)
            market: "POINTS" or "REBOUNDS"
            context: Game context dict

        Returns:
            Adjusted projection.
        """
        multiplier = self.compute_multiplier(market, context)
        adjusted = projection * multiplier
        return round(adjusted, 2)

    def adjust_batch(
        self,
        projections: list[float],
        market: str,
        contexts: list[dict[str, Any]],
    ) -> list[float]:
        """Apply context adjustments to a batch of projections."""
        return [self.adjust(proj, market, ctx) for proj, ctx in zip(projections, contexts)]
