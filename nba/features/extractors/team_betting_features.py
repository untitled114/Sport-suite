"""
Team Betting Performance Feature Extractor
==========================================
Extracts 5 team-level betting performance features.

Features capture how a team's players have historically performed
against props, providing context for player-level predictions.
"""

import logging
from typing import Any, Dict

from .base import BaseFeatureExtractor

logger = logging.getLogger(__name__)


class TeamBettingExtractor(BaseFeatureExtractor):
    """
    Extracts 5 team betting performance features.

    Features measure team-wide tendencies that affect player props:
    1. Team over rate: How often team players go OVER
    2. Opponent over rate: How often opponents go OVER vs this team's defense
    3. Points over rate: Team-specific points prop performance
    4. Rebounds over rate: Team-specific rebounds prop performance
    5. Assists over rate: Team-specific assists prop performance
    """

    FEATURE_NAMES = (
        "team_prop_over_rate",
        "opp_prop_over_rate",
        "team_points_over_rate",
        "team_rebounds_over_rate",
        "team_assists_over_rate",
    )

    def __init__(self, conn: Any):
        """
        Initialize team betting extractor.

        Args:
            conn: Database connection to nba_team database
        """
        super().__init__(conn, name="TeamBetting")
        self._cache = {}  # Cache for team stats

    @classmethod
    def get_defaults(cls) -> Dict[str, float]:
        """Get default values for team betting features."""
        return {
            "team_prop_over_rate": 0.5,
            "opp_prop_over_rate": 0.5,
            "team_points_over_rate": 0.5,
            "team_rebounds_over_rate": 0.5,
            "team_assists_over_rate": 0.5,
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
        Extract team betting performance features.

        Args:
            player_name: Player's full name
            game_date: Game date
            stat_type: Stat type
            team_abbrev: Player's team abbreviation
            opponent_team: Opponent team abbreviation

        Returns:
            Dict with 5 team betting features
        """
        if not team_abbrev:
            return self.get_defaults()

        game_date_str = self._normalize_date(game_date)

        features = self.get_defaults()

        # Get team betting stats
        team_stats = self._get_team_betting_stats(team_abbrev, game_date_str)
        if team_stats:
            features["team_prop_over_rate"] = team_stats.get("overall_over_rate", 0.5)
            features["team_points_over_rate"] = team_stats.get("points_over_rate", 0.5)
            features["team_rebounds_over_rate"] = team_stats.get("rebounds_over_rate", 0.5)
            features["team_assists_over_rate"] = team_stats.get("assists_over_rate", 0.5)

        # Get opponent defensive stats (how often opponents go OVER vs this defense)
        if opponent_team:
            opp_stats = self._get_opponent_allowed_stats(opponent_team, game_date_str)
            if opp_stats:
                features["opp_prop_over_rate"] = opp_stats.get("allowed_over_rate", 0.5)

        return features

    def _get_team_betting_stats(
        self,
        team_abbrev: str,
        game_date_str: str,
    ) -> Dict[str, float]:
        """
        Get team's historical prop betting performance.

        Args:
            team_abbrev: Team abbreviation
            game_date_str: Game date for temporal filtering

        Returns:
            Dict with over rates by stat type
        """
        cache_key = f"{team_abbrev}_{game_date_str[:7]}"  # Cache by team/month
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Query team betting stats
        query = """
        SELECT
            stat_type,
            COUNT(*) as total,
            SUM(CASE WHEN actual_result > over_line THEN 1 ELSE 0 END) as overs
        FROM (
            SELECT DISTINCT ON (p.player_name, p.game_date, p.stat_type)
                p.stat_type,
                p.over_line,
                p.actual_result
            FROM nba_prop_lines p
            JOIN player_profile pp ON p.player_name = pp.full_name
            WHERE pp.team_abbrev = %s
              AND p.game_date < %s
              AND p.game_date >= %s::date - interval '60 days'
              AND p.actual_result IS NOT NULL
        ) sub
        GROUP BY stat_type
        """
        try:
            df = self._safe_query(query, (team_abbrev, game_date_str, game_date_str))

            if df is None or len(df) == 0:
                return {}

            stats = {"overall_over_rate": 0.5}
            total_all = 0
            overs_all = 0

            for _, row in df.iterrows():
                stat_type = row["stat_type"].lower()
                total = row["total"]
                overs = row["overs"]
                rate = overs / total if total > 0 else 0.5

                stats[f"{stat_type}_over_rate"] = rate
                total_all += total
                overs_all += overs

            if total_all > 0:
                stats["overall_over_rate"] = overs_all / total_all

            self._cache[cache_key] = stats
            return stats

        except (KeyError, IndexError, TypeError, ZeroDivisionError) as e:
            logger.debug(f"Team betting stats query failed: {e}")
            return {}

    def _get_opponent_allowed_stats(
        self,
        opponent_team: str,
        game_date_str: str,
    ) -> Dict[str, float]:
        """
        Get how often opponents go OVER against this team's defense.

        Args:
            opponent_team: Opponent team abbreviation
            game_date_str: Game date

        Returns:
            Dict with allowed over rates
        """
        cache_key = f"opp_{opponent_team}_{game_date_str[:7]}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Query opponent's defense stats
        query = """
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN actual_result > over_line THEN 1 ELSE 0 END) as overs
        FROM (
            SELECT DISTINCT ON (p.player_name, p.game_date, p.stat_type)
                p.over_line,
                p.actual_result
            FROM nba_prop_lines p
            JOIN player_game_logs pgl ON p.player_name = pgl.player_name
                AND p.game_date = pgl.game_date
            WHERE pgl.opponent_team = %s
              AND p.game_date < %s
              AND p.game_date >= %s::date - interval '60 days'
              AND p.actual_result IS NOT NULL
        ) sub
        """
        try:
            df = self._safe_query(query, (opponent_team, game_date_str, game_date_str))

            if df is None or len(df) == 0:
                return {}

            total = df["total"].iloc[0]
            overs = df["overs"].iloc[0]
            rate = overs / total if total > 0 else 0.5

            stats = {"allowed_over_rate": rate}
            self._cache[cache_key] = stats
            return stats

        except (KeyError, IndexError, TypeError, ZeroDivisionError) as e:
            logger.debug(f"Opponent allowed stats query failed: {e}")
            return {}

    def clear_cache(self):
        """Clear the internal cache."""
        self._cache.clear()
