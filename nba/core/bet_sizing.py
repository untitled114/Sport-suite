"""
Kelly Criterion Bet Sizing
============================
Fractional Kelly bet sizing for bankroll management.

Kelly criterion gives the optimal bet size to maximize long-term growth.
Full Kelly is too aggressive for sports betting, so we use fractional
Kelly (default 0.25x = quarter Kelly) with a hard cap.

Usage:
    from nba.core.bet_sizing import KellySizer

    sizer = KellySizer(bankroll=1000.0, fraction=0.25)
    result = sizer.size_bet(model_prob=0.62, odds=-110)
    # result = {bet_amount: 12.50, kelly_fraction: 0.05, ...}
"""

from dataclasses import dataclass
from typing import Dict

from nba.core.exceptions import DataValidationError
from nba.core.probability_engine import ProbabilityEngine


@dataclass(frozen=True)
class KellyConfig:
    """Configuration for Kelly bet sizing.

    Attributes:
        fraction: Kelly fraction (0.25 = quarter Kelly)
        max_bet_pct: Hard cap as percentage of bankroll (3%)
        min_bet_amount: Minimum bet amount (below this, skip)
    """

    fraction: float = 0.25
    max_bet_pct: float = 0.03
    min_bet_amount: float = 1.0


class KellySizer:
    """Fractional Kelly criterion bet sizing."""

    def __init__(
        self,
        bankroll: float = 1000.0,
        fraction: float = 0.25,
        max_bet_pct: float = 0.03,
    ):
        """
        Args:
            bankroll: Current bankroll in dollars
            fraction: Kelly fraction (0.25 = quarter Kelly)
            max_bet_pct: Maximum bet as % of bankroll (default 3%)
        """
        if bankroll <= 0:
            raise DataValidationError("bankroll must be positive")
        if not 0.0 < fraction <= 1.0:
            raise DataValidationError("fraction must be between 0 and 1")
        if not 0.0 < max_bet_pct <= 1.0:
            raise DataValidationError("max_bet_pct must be between 0 and 1")

        self.bankroll = bankroll
        self.config = KellyConfig(fraction=fraction, max_bet_pct=max_bet_pct)
        self._engine = ProbabilityEngine()

    def size_bet(
        self,
        model_prob: float,
        odds: int,
        bankroll: float = None,
        fraction: float = None,
    ) -> Dict:
        """
        Calculate recommended bet size using fractional Kelly criterion.

        Kelly formula: f* = (bp - q) / b
        where b = decimal_odds - 1, p = model_prob, q = 1 - model_prob
        Then apply fraction and cap.

        Args:
            model_prob: Model's probability of winning (0-1)
            odds: American odds (e.g., -110)
            bankroll: Override bankroll (uses instance default if None)
            fraction: Override Kelly fraction (uses instance default if None)

        Returns:
            Dict with:
                - bet_amount: Recommended bet in dollars
                - bet_units: Bet as units (1 unit = 1% of bankroll)
                - kelly_fraction: Raw Kelly fraction before scaling
                - scaled_kelly: Kelly * fraction
                - bet_pct: Bet as percentage of bankroll
                - is_capped: Whether hard cap was applied
                - edge: Model prob - implied prob
                - has_edge: Whether there's positive edge
                - expected_value: EV per dollar wagered

        Raises:
            DataValidationError: If inputs are out of range
        """
        if not 0.0 <= model_prob <= 1.0:
            raise DataValidationError("model_prob must be between 0.0 and 1.0")

        br = bankroll if bankroll is not None else self.bankroll
        frac = fraction if fraction is not None else self.config.fraction

        decimal_odds = self._engine.american_to_decimal(odds)
        implied_prob = self._engine.implied_probability(odds)
        edge = model_prob - implied_prob

        # Kelly criterion
        b = decimal_odds - 1
        if b > 0 and edge > 0:
            raw_kelly = (model_prob * b - (1 - model_prob)) / b
            raw_kelly = max(0.0, raw_kelly)
        else:
            raw_kelly = 0.0

        # Apply fraction
        scaled_kelly = raw_kelly * frac

        # Calculate bet amount
        bet_pct = scaled_kelly
        is_capped = False

        # Hard cap: never more than max_bet_pct of bankroll
        if bet_pct > self.config.max_bet_pct:
            bet_pct = self.config.max_bet_pct
            is_capped = True

        bet_amount = br * bet_pct

        # Skip tiny bets
        if bet_amount < self.config.min_bet_amount:
            bet_amount = 0.0
            bet_pct = 0.0

        # EV per dollar
        ev = model_prob * (decimal_odds - 1) - (1 - model_prob)

        return {
            "bet_amount": round(bet_amount, 2),
            "bet_units": round(bet_pct * 100, 2),
            "kelly_fraction": round(raw_kelly, 4),
            "scaled_kelly": round(scaled_kelly, 4),
            "bet_pct": round(bet_pct, 4),
            "is_capped": is_capped,
            "edge": round(edge, 4),
            "has_edge": edge > 0,
            "expected_value": round(ev, 4),
        }

    def update_bankroll(self, new_bankroll: float):
        """Update bankroll after wins/losses."""
        if new_bankroll <= 0:
            raise DataValidationError("bankroll must be positive")
        self.bankroll = new_bankroll
