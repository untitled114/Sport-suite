"""
Vegas Context Feature Extractor
===============================
Extracts 2 Vegas/game context features.

Features capture game-level betting context that affects
player prop outcomes (game totals, spreads, pace expectations).
"""

import logging
from typing import Any, Dict

from .base import BaseFeatureExtractor

logger = logging.getLogger(__name__)


class VegasContextExtractor(BaseFeatureExtractor):
    """
    Extracts 2 Vegas context features.

    Features:
    1. vegas_total: Expected game total from Vegas lines
    2. vegas_spread: Point spread (positive = underdog)
    """

    FEATURE_NAMES = (
        "vegas_total",
        "vegas_spread",
    )

    def __init__(self, conn: Any):
        """
        Initialize Vegas context extractor.

        Args:
            conn: Database connection to nba_games database
        """
        super().__init__(conn, name="VegasContext")

    @classmethod
    def get_defaults(cls) -> Dict[str, float]:
        """Get default values for Vegas features."""
        return {
            "vegas_total": 220.0,  # Typical NBA game total
            "vegas_spread": 0.0,
        }

    def extract(
        self,
        player_name: str,
        game_date: Any,
        stat_type: str,
        team_abbrev: str = None,
        opponent_team: str = None,
        **kwargs,
    ) -> Dict[str, float]:
        """
        Extract Vegas context features for a game.

        Args:
            player_name: Player's full name (used to find team if not provided)
            game_date: Game date
            stat_type: Stat type (not used, included for interface consistency)
            team_abbrev: Player's team abbreviation
            opponent_team: Opponent team abbreviation

        Returns:
            Dict with 2 Vegas context features
        """
        game_date_str = self._normalize_date(game_date)

        if not team_abbrev or not opponent_team:
            return self.get_defaults()

        # Query game context for Vegas lines
        query = """
        SELECT
            vegas_total,
            vegas_spread
        FROM game_context
        WHERE game_date = %s
          AND (
            (home_team = %s AND away_team = %s)
            OR (home_team = %s AND away_team = %s)
          )
        LIMIT 1
        """
        df = self._safe_query(
            query,
            (
                game_date_str,
                team_abbrev,
                opponent_team,
                opponent_team,
                team_abbrev,
            ),
        )

        if df is None or len(df) == 0:
            # Try alternate query on games table
            return self._extract_from_games_table(game_date_str, team_abbrev, opponent_team)

        features = self.get_defaults()
        features["vegas_total"] = self._safe_float(df["vegas_total"].iloc[0], default=220.0)
        features["vegas_spread"] = self._safe_float(df["vegas_spread"].iloc[0])

        return features

    def _extract_from_games_table(
        self,
        game_date_str: str,
        team_abbrev: str,
        opponent_team: str,
    ) -> Dict[str, float]:
        """
        Fallback extraction from games table.

        Args:
            game_date_str: Game date string
            team_abbrev: Player's team
            opponent_team: Opponent team

        Returns:
            Dict with Vegas features
        """
        query = """
        SELECT
            total_line,
            spread_line
        FROM games
        WHERE game_date = %s
          AND (
            (home_team = %s AND away_team = %s)
            OR (home_team = %s AND away_team = %s)
          )
        LIMIT 1
        """
        df = self._safe_query(
            query,
            (
                game_date_str,
                team_abbrev,
                opponent_team,
                opponent_team,
                team_abbrev,
            ),
        )

        features = self.get_defaults()

        if df is not None and len(df) > 0:
            features["vegas_total"] = self._safe_float(df["total_line"].iloc[0], default=220.0)
            features["vegas_spread"] = self._safe_float(df["spread_line"].iloc[0])

        return features
