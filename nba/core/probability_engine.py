"""
Probability Distribution Engine
================================
Converts stat projections to probabilities using statistical distributions.

Uses Normal distribution for high-count stats (POINTS, REBOUNDS) and
Poisson distribution for low-count stats (THREES, BLOCKS, STEALS).

This is a first-principles approach: fit a distribution to the player's
recent performance, then calculate P(player > line) analytically.

Usage:
    from nba.core.probability_engine import ProbabilityEngine

    engine = ProbabilityEngine()
    p_over = engine.calculate_probability(
        projected_value=22.4, std_dev=5.1, line=21.5, stat_type="POINTS"
    )
    implied = engine.implied_probability(-110)
"""

import math
from typing import Dict

from scipy import stats

from nba.core.exceptions import DataValidationError

# Stats that follow approximately Poisson (low count, discrete)
POISSON_STATS = frozenset({"THREES", "BLOCKS", "STEALS", "3PM", "BLK", "STL"})

# Stats that follow approximately Normal (high count, continuous-ish)
NORMAL_STATS = frozenset({"POINTS", "REBOUNDS", "ASSISTS", "PTS", "REB", "AST"})


class ProbabilityEngine:
    """Convert stat projections into over/under probabilities using distributions."""

    def calculate_probability(
        self,
        projected_value: float,
        std_dev: float,
        line: float,
        stat_type: str,
    ) -> float:
        """
        Calculate P(player scores > line) from a fitted distribution.

        For Normal stats: P(X > line) = 1 - CDF(line)
        For Poisson stats: P(X > line) = 1 - CDF(floor(line))

        Uses a continuity correction for Poisson to handle the "over"
        threshold properly (over 2.5 threes means >= 3).

        Args:
            projected_value: Expected stat value (mean of distribution)
            std_dev: Standard deviation of player's recent performance
            line: Sportsbook prop line
            stat_type: Stat type (POINTS, REBOUNDS, THREES, etc.)

        Returns:
            P(OVER) probability between 0.0 and 1.0

        Raises:
            DataValidationError: If inputs are invalid
        """
        if projected_value < 0:
            raise DataValidationError("projected_value must be non-negative")
        if std_dev < 0:
            raise DataValidationError("std_dev must be non-negative")
        if line < 0:
            raise DataValidationError("line must be non-negative")

        stat_upper = stat_type.upper()

        if stat_upper in POISSON_STATS:
            return self._poisson_probability(projected_value, line)
        else:
            return self._normal_probability(projected_value, std_dev, line)

    def _normal_probability(self, mean: float, std_dev: float, line: float) -> float:
        """P(X > line) using Normal distribution."""
        if std_dev == 0:
            return 1.0 if mean > line else 0.0

        # P(X > line) = 1 - Phi((line - mean) / std)
        return 1.0 - stats.norm.cdf(line, loc=mean, scale=std_dev)

    def _poisson_probability(self, lam: float, line: float) -> float:
        """
        P(X > line) using Poisson distribution.

        Sportsbook lines for discrete stats use half-points (e.g., 2.5 threes).
        Over 2.5 means >= 3, so P(OVER) = P(X >= 3) = 1 - P(X <= 2).
        """
        if lam <= 0:
            return 0.0

        # For half-point lines (2.5), over means >= ceil(line)
        # For whole-point lines (3.0), over means > 3 = >= 4
        threshold = math.ceil(line) if line != int(line) else int(line) + 1

        # P(X >= threshold) = 1 - P(X <= threshold - 1)
        return 1.0 - stats.poisson.cdf(threshold - 1, mu=lam)

    @staticmethod
    def implied_probability(american_odds: int) -> float:
        """
        Convert American odds to implied probability.

        Negative odds (favorite): implied = |odds| / (|odds| + 100)
        Positive odds (underdog): implied = 100 / (odds + 100)

        Args:
            american_odds: American odds (e.g., -110, +150)

        Returns:
            Implied probability between 0.0 and 1.0

        Raises:
            DataValidationError: If odds are invalid (zero)
        """
        if american_odds == 0:
            raise DataValidationError("American odds cannot be zero")

        if american_odds < 0:
            return abs(american_odds) / (abs(american_odds) + 100)
        else:
            return 100 / (american_odds + 100)

    @staticmethod
    def american_to_decimal(american_odds: int) -> float:
        """
        Convert American odds to decimal odds.

        Args:
            american_odds: American odds (e.g., -110, +150)

        Returns:
            Decimal odds (e.g., 1.909, 2.50)
        """
        if american_odds == 0:
            raise DataValidationError("American odds cannot be zero")

        if american_odds < 0:
            return 1 + (100 / abs(american_odds))
        else:
            return 1 + (american_odds / 100)

    def expected_value(
        self,
        model_prob: float,
        american_odds: int,
    ) -> float:
        """
        Calculate expected value of a bet.

        EV = (prob * payout) - (1 - prob) * stake
        For a $1 bet: EV = prob * (decimal_odds - 1) - (1 - prob)

        Args:
            model_prob: Model's probability of winning
            american_odds: American odds for the bet

        Returns:
            Expected value per unit wagered
        """
        decimal_odds = self.american_to_decimal(american_odds)
        return model_prob * (decimal_odds - 1) - (1 - model_prob)
