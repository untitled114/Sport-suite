"""
Edge Calculator
================
Calculates betting edge as the difference between model probability
and sportsbook implied probability.

Edge = model_probability - implied_probability

This is the fundamental signal: if our model says P(OVER) = 0.62 and
the book's implied probability is 0.524 (-110 odds), edge = 0.096 (9.6%).

Usage:
    from nba.core.edge_calculator import EdgeCalculator

    calc = EdgeCalculator(min_edge=0.05)
    result = calc.calculate_edge(model_prob=0.62, book_odds=-110)
    # result = {edge: 0.096, model_prob: 0.62, implied_prob: 0.524, has_edge: True, ...}
"""

from dataclasses import dataclass
from typing import Dict

from nba.core.exceptions import DataValidationError
from nba.core.probability_engine import ProbabilityEngine


@dataclass(frozen=True)
class EdgeConfig:
    """Configuration for edge calculation thresholds.

    Attributes:
        min_edge: Minimum edge to consider a bet (default 5%)
        max_edge: Maximum edge before flagging as suspicious
        kelly_cap: Maximum Kelly fraction (full Kelly is too aggressive)
    """

    min_edge: float = 0.05
    max_edge: float = 0.25
    kelly_cap: float = 0.25


class EdgeCalculator:
    """Calculate betting edge and Kelly criterion fraction."""

    def __init__(self, min_edge: float = 0.05, max_edge: float = 0.25):
        self._engine = ProbabilityEngine()
        self.config = EdgeConfig(min_edge=min_edge, max_edge=max_edge)

    def calculate_edge(
        self,
        model_prob: float,
        book_odds: int,
    ) -> Dict:
        """
        Calculate edge between model probability and book implied probability.

        Args:
            model_prob: Model's probability of the outcome (0-1)
            book_odds: American odds from the sportsbook

        Returns:
            Dict with:
                - edge: Raw edge (model_prob - implied_prob)
                - edge_pct: Edge as percentage
                - model_prob: Model probability
                - implied_prob: Sportsbook implied probability
                - has_edge: Whether edge exceeds minimum threshold
                - is_suspicious: Whether edge exceeds maximum (data error?)
                - kelly_fraction: Optimal Kelly fraction (0 if no edge)
                - expected_value: EV per unit wagered
                - decimal_odds: Decimal odds equivalent

        Raises:
            DataValidationError: If inputs are out of range
        """
        if not 0.0 <= model_prob <= 1.0:
            raise DataValidationError("model_prob must be between 0.0 and 1.0")

        implied_prob = self._engine.implied_probability(book_odds)
        decimal_odds = self._engine.american_to_decimal(book_odds)

        edge = model_prob - implied_prob
        ev = model_prob * (decimal_odds - 1) - (1 - model_prob)

        # Kelly criterion: f* = (bp - q) / b
        # where b = decimal_odds - 1, p = model_prob, q = 1 - model_prob
        b = decimal_odds - 1
        if b > 0 and edge > 0:
            kelly = (model_prob * b - (1 - model_prob)) / b
            kelly = max(0.0, min(kelly, self.config.kelly_cap))
        else:
            kelly = 0.0

        return {
            "edge": round(edge, 4),
            "edge_pct": round(edge * 100, 2),
            "model_prob": round(model_prob, 4),
            "implied_prob": round(implied_prob, 4),
            "has_edge": edge >= self.config.min_edge,
            "is_suspicious": edge > self.config.max_edge,
            "kelly_fraction": round(kelly, 4),
            "expected_value": round(ev, 4),
            "decimal_odds": round(decimal_odds, 3),
        }

    def calculate_edge_from_probs(
        self,
        model_prob: float,
        implied_prob: float,
        decimal_odds: float,
    ) -> Dict:
        """
        Calculate edge when you already have implied probability and decimal odds.

        Useful when working with pre-computed data from the line optimizer.

        Args:
            model_prob: Model's probability
            implied_prob: Pre-computed implied probability
            decimal_odds: Pre-computed decimal odds

        Returns:
            Same dict as calculate_edge()
        """
        if not 0.0 <= model_prob <= 1.0:
            raise DataValidationError("model_prob must be between 0.0 and 1.0")
        if not 0.0 < implied_prob <= 1.0:
            raise DataValidationError("implied_prob must be between 0.0 and 1.0")

        edge = model_prob - implied_prob
        ev = model_prob * (decimal_odds - 1) - (1 - model_prob)

        b = decimal_odds - 1
        if b > 0 and edge > 0:
            kelly = (model_prob * b - (1 - model_prob)) / b
            kelly = max(0.0, min(kelly, self.config.kelly_cap))
        else:
            kelly = 0.0

        return {
            "edge": round(edge, 4),
            "edge_pct": round(edge * 100, 2),
            "model_prob": round(model_prob, 4),
            "implied_prob": round(implied_prob, 4),
            "has_edge": edge >= self.config.min_edge,
            "is_suspicious": edge > self.config.max_edge,
            "kelly_fraction": round(kelly, 4),
            "expected_value": round(ev, 4),
            "decimal_odds": round(decimal_odds, 3),
        }
