"""
Pace-Adjusted Projection Model
================================
First-principles stat projection using pace-adjusted rolling averages.

This is a simple, interpretable projection model that can run alongside
the existing two-head LightGBM system. It projects a player's expected
stat value by:

1. Fetching rolling averages (L5/L10/L20) from player_rolling_stats
2. Pace-adjusting based on matchup (team pace * opponent pace / league avg)
3. Adjusting for home/away splits
4. Adjusting for opponent defensive strength

The projection feeds into the ProbabilityEngine to get P(OVER).

Usage:
    from nba.models.projection_model import ProjectionModel

    model = ProjectionModel()
    projected = model.project("LeBron James", "POINTS", "BOS", is_home=True)
    # projected = 26.3
"""

import logging
from typing import Dict, Optional

import psycopg2

from nba.config.database import get_players_db_config, get_team_db_config
from nba.core.exceptions import DatabaseConnectionError, DataNotFoundError

logger = logging.getLogger(__name__)

# Stat type to column mapping in player_rolling_stats
STAT_COLUMN_MAP = {
    "POINTS": "points",
    "REBOUNDS": "rebounds",
    "ASSISTS": "assists",
    "THREES": "threes",
    "PTS": "points",
    "REB": "rebounds",
    "AST": "assists",
    "3PM": "threes",
}

# League average pace (updated seasonally, ~100 possessions/game for 2025-26)
LEAGUE_AVG_PACE = 100.0

# Home/away adjustment factors (from historical data analysis)
HOME_ADVANTAGE = {
    "POINTS": 0.015,  # +1.5% at home
    "REBOUNDS": 0.010,  # +1.0% at home
    "ASSISTS": 0.012,  # +1.2% at home
    "THREES": 0.008,  # +0.8% at home
}

# Rolling average weights (more recent = more weight)
ROLLING_WEIGHTS = {
    "L5": 0.50,
    "L10": 0.30,
    "L20": 0.20,
}


class ProjectionModel:
    """Pace-adjusted projection model using rolling averages."""

    def __init__(self):
        self._players_conn = None
        self._team_conn = None

    def _get_players_conn(self):
        """Get or create players database connection."""
        if self._players_conn is None or self._players_conn.closed:
            try:
                self._players_conn = psycopg2.connect(**get_players_db_config())
                self._players_conn.autocommit = True
            except Exception as e:
                raise DatabaseConnectionError("nba_players", "localhost", 5536) from e
        return self._players_conn

    def _get_team_conn(self):
        """Get or create team database connection."""
        if self._team_conn is None or self._team_conn.closed:
            try:
                self._team_conn = psycopg2.connect(**get_team_db_config())
                self._team_conn.autocommit = True
            except Exception as e:
                raise DatabaseConnectionError("nba_team", "localhost", 5538) from e
        return self._team_conn

    def project(
        self,
        player_name: str,
        stat_type: str,
        opponent_team: str,
        is_home: bool = True,
    ) -> float:
        """
        Project expected stat value for a player.

        Steps:
        1. Weighted average of L5/L10/L20 rolling stats
        2. Pace adjustment (matchup pace vs league average)
        3. Opponent defensive adjustment
        4. Home/away adjustment

        Args:
            player_name: Player full name (e.g., "LeBron James")
            stat_type: Stat type (POINTS, REBOUNDS, ASSISTS, THREES)
            opponent_team: Opponent team abbreviation (e.g., "BOS")
            is_home: Whether the player is at home

        Returns:
            Projected stat value (float)

        Raises:
            DataNotFoundError: If player or team data is missing
        """
        stat_type = stat_type.upper()
        stat_col = STAT_COLUMN_MAP.get(stat_type, stat_type.lower())

        # 1. Get player rolling averages
        rolling = self._get_player_rolling_stats(player_name, stat_col)

        # Weighted average of rolling windows
        base_projection = (
            rolling.get("L5", 0) * ROLLING_WEIGHTS["L5"]
            + rolling.get("L10", 0) * ROLLING_WEIGHTS["L10"]
            + rolling.get("L20", 0) * ROLLING_WEIGHTS["L20"]
        )

        if base_projection <= 0:
            logger.warning(f"Zero base projection for {player_name} {stat_type}")
            return 0.0

        # 2. Get minutes projection
        minutes_ratio = self._get_minutes_ratio(player_name, rolling)

        # 3. Pace adjustment
        pace_factor = self._get_pace_factor(
            rolling.get("team_abbrev", ""),
            opponent_team,
        )

        # 4. Opponent defensive adjustment
        def_factor = self._get_defensive_factor(opponent_team, stat_col)

        # 5. Home/away adjustment
        home_factor = 1.0
        if is_home:
            home_factor = 1.0 + HOME_ADVANTAGE.get(stat_type, 0.01)
        else:
            home_factor = 1.0 - HOME_ADVANTAGE.get(stat_type, 0.01)

        # Combine adjustments
        projected = base_projection * minutes_ratio * pace_factor * def_factor * home_factor

        logger.debug(
            f"Projection for {player_name} {stat_type}: "
            f"base={base_projection:.1f}, mins_ratio={minutes_ratio:.3f}, "
            f"pace={pace_factor:.3f}, def={def_factor:.3f}, "
            f"home={home_factor:.3f} => {projected:.1f}"
        )

        return round(projected, 1)

    def project_with_details(
        self,
        player_name: str,
        stat_type: str,
        opponent_team: str,
        is_home: bool = True,
    ) -> Dict:
        """
        Project with full detail breakdown for transparency.

        Returns dict with projection and all component factors.
        """
        stat_type = stat_type.upper()
        stat_col = STAT_COLUMN_MAP.get(stat_type, stat_type.lower())

        rolling = self._get_player_rolling_stats(player_name, stat_col)

        base_projection = (
            rolling.get("L5", 0) * ROLLING_WEIGHTS["L5"]
            + rolling.get("L10", 0) * ROLLING_WEIGHTS["L10"]
            + rolling.get("L20", 0) * ROLLING_WEIGHTS["L20"]
        )

        minutes_ratio = self._get_minutes_ratio(player_name, rolling)
        pace_factor = self._get_pace_factor(rolling.get("team_abbrev", ""), opponent_team)
        def_factor = self._get_defensive_factor(opponent_team, stat_col)

        if is_home:
            home_factor = 1.0 + HOME_ADVANTAGE.get(stat_type, 0.01)
        else:
            home_factor = 1.0 - HOME_ADVANTAGE.get(stat_type, 0.01)

        projected = base_projection * minutes_ratio * pace_factor * def_factor * home_factor

        return {
            "player_name": player_name,
            "stat_type": stat_type,
            "opponent_team": opponent_team,
            "is_home": is_home,
            "projection": round(projected, 1),
            "base_projection": round(base_projection, 1),
            "rolling_L5": rolling.get("L5", 0),
            "rolling_L10": rolling.get("L10", 0),
            "rolling_L20": rolling.get("L20", 0),
            "std_dev": rolling.get("std", 0),
            "minutes_ratio": round(minutes_ratio, 3),
            "pace_factor": round(pace_factor, 3),
            "defensive_factor": round(def_factor, 3),
            "home_factor": round(home_factor, 3),
            "team_abbrev": rolling.get("team_abbrev", ""),
        }

    def get_player_std_dev(self, player_name: str, stat_type: str) -> float:
        """
        Get standard deviation of a player's recent performance.

        Uses the L10 standard deviation from rolling stats as a stable estimate.

        Args:
            player_name: Player full name
            stat_type: Stat type (POINTS, REBOUNDS, etc.)

        Returns:
            Standard deviation of recent performance
        """
        stat_col = STAT_COLUMN_MAP.get(stat_type.upper(), stat_type.lower())
        rolling = self._get_player_rolling_stats(player_name, stat_col)
        return rolling.get("std", 5.0)  # Default to reasonable std if missing

    def _get_player_rolling_stats(self, player_name: str, stat_col: str) -> Dict:
        """Fetch rolling averages from player_rolling_stats table."""
        conn = self._get_players_conn()
        cur = conn.cursor()

        try:
            # Use unaccent for diacritics (Jokić, Dončić, etc.)
            cur.execute(
                """
                SELECT
                    prs.ema_{stat}_L3,
                    prs.ema_{stat}_L5,
                    prs.ema_{stat}_L10,
                    prs.ema_{stat}_L20,
                    prs.ema_minutes_L5,
                    prs.ema_minutes_L10,
                    pp.team_abbreviation
                FROM player_rolling_stats prs
                JOIN player_profile pp ON pp.player_id = prs.player_id
                WHERE unaccent(pp.full_name) = unaccent(%s)
                ORDER BY prs.last_updated DESC
                LIMIT 1
                """.format(
                    stat=stat_col
                ),
                (player_name,),
            )

            row = cur.fetchone()
            if not row:
                logger.warning(f"No rolling stats found for {player_name}")
                raise DataNotFoundError("player_rolling_stats", player_name)

            return {
                "L3": float(row[0] or 0),
                "L5": float(row[1] or 0),
                "L10": float(row[2] or 0),
                "L20": float(row[3] or 0),
                "minutes_L5": float(row[4] or 0),
                "minutes_L10": float(row[5] or 0),
                "team_abbrev": row[6] or "",
                "std": self._estimate_std_from_rolling(
                    float(row[1] or 0), float(row[2] or 0), float(row[3] or 0)
                ),
            }
        finally:
            cur.close()

    def _estimate_std_from_rolling(self, l5: float, l10: float, l20: float) -> float:
        """
        Estimate std dev from rolling averages.

        Uses the spread between rolling windows as a proxy for variance.
        Wider spread = more volatile player.
        """
        if l5 == 0 and l10 == 0 and l20 == 0:
            return 5.0

        avg = (l5 + l10 + l20) / 3.0
        if avg == 0:
            return 5.0

        # Coefficient of variation from rolling window spread
        spread = max(l5, l10, l20) - min(l5, l10, l20)
        # Typical CV for NBA stats is ~0.20-0.35
        estimated_cv = max(0.15, min(0.40, spread / avg + 0.15))
        return round(avg * estimated_cv, 2)

    def _get_minutes_ratio(self, player_name: str, rolling: Dict) -> float:
        """
        Calculate minutes adjustment ratio.

        If a player's recent minutes are trending up/down from their average,
        their stats should be adjusted proportionally.
        """
        minutes_l5 = rolling.get("minutes_L5", 0)
        minutes_l10 = rolling.get("minutes_L10", 0)

        if minutes_l10 <= 0 or minutes_l5 <= 0:
            return 1.0

        # Ratio of recent minutes to baseline
        ratio = minutes_l5 / minutes_l10

        # Clamp to reasonable range (0.85 to 1.15)
        return max(0.85, min(1.15, ratio))

    def _get_pace_factor(self, player_team: str, opponent_team: str) -> float:
        """
        Calculate pace adjustment factor.

        Pace factor = sqrt(team_pace * opponent_pace) / league_avg_pace
        Using geometric mean because pace is multiplicative.
        """
        if not player_team or not opponent_team:
            return 1.0

        conn = self._get_team_conn()
        cur = conn.cursor()

        try:
            cur.execute(
                """
                SELECT team_abbreviation, pace
                FROM team_season_stats
                WHERE team_abbreviation IN (%s, %s)
                AND season = (SELECT MAX(season) FROM team_season_stats)
                """,
                (player_team, opponent_team),
            )

            rows = cur.fetchall()
            if len(rows) < 2:
                return 1.0

            pace_map = {row[0]: float(row[1] or LEAGUE_AVG_PACE) for row in rows}
            team_pace = pace_map.get(player_team, LEAGUE_AVG_PACE)
            opp_pace = pace_map.get(opponent_team, LEAGUE_AVG_PACE)

            # Geometric mean of both teams' pace relative to league average
            matchup_pace = (team_pace * opp_pace) ** 0.5
            factor = matchup_pace / LEAGUE_AVG_PACE

            # Clamp to reasonable range
            return max(0.90, min(1.10, factor))
        finally:
            cur.close()

    def _get_defensive_factor(self, opponent_team: str, stat_col: str) -> float:
        """
        Calculate opponent defensive adjustment.

        Uses opponent's defensive rating relative to league average.
        Higher def_rating = worse defense = higher projection.
        """
        if not opponent_team:
            return 1.0

        conn = self._get_team_conn()
        cur = conn.cursor()

        try:
            cur.execute(
                """
                SELECT def_rating
                FROM team_season_stats
                WHERE team_abbreviation = %s
                AND season = (SELECT MAX(season) FROM team_season_stats)
                """,
                (opponent_team,),
            )

            row = cur.fetchone()
            if not row or not row[0]:
                return 1.0

            opp_def_rating = float(row[0])

            # League average defensive rating is ~112
            league_avg_def = 112.0

            # Higher def rating = worse defense = easier to score
            # Scale factor: 1% adjustment per 1 point of def rating difference
            adjustment = (opp_def_rating - league_avg_def) / league_avg_def
            factor = 1.0 + adjustment

            # Clamp to reasonable range
            return max(0.90, min(1.10, factor))
        finally:
            cur.close()

    def close(self):
        """Close database connections."""
        if self._players_conn and not self._players_conn.closed:
            self._players_conn.close()
        if self._team_conn and not self._team_conn.closed:
            self._team_conn.close()

    def __del__(self):
        self.close()
