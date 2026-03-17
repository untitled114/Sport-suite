"""
Temporal Milestone Feature Extractor (V4)
==========================================
Extracts ~10 features based on NBA calendar milestones and player
team continuity. These capture structural breaks in the season that
affect player performance in ways that rolling stats miss.

Key milestones:
- Trade deadline: players change teams, roles shift
- All-Star break: rest, rotation changes, playoff mentality starts
- Playoff push: tanking teams rest players, contenders play harder
- Season boundaries: cross-season data shouldn't blend seamlessly

Team continuity:
- Detects if player was recently traded (new system, new teammates)
- Counts games with current team (how settled is the player?)

No external API needed — uses game_date + player_game_logs only.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import pandas as pd

from .base import BaseFeatureExtractor

logger = logging.getLogger(__name__)

# NBA season milestone dates (season = year season STARTS)
# These are verified historical dates + best estimates for future seasons
NBA_MILESTONES = {
    2023: {
        "season_start": "2023-10-24",
        "trade_deadline": "2024-02-08",
        "allstar_start": "2024-02-16",
        "allstar_end": "2024-02-18",
        "regular_end": "2024-04-14",
        "playoffs_start": "2024-04-20",
    },
    2024: {
        "season_start": "2024-10-22",
        "trade_deadline": "2025-02-06",
        "allstar_start": "2025-02-14",
        "allstar_end": "2025-02-16",
        "regular_end": "2025-04-13",
        "playoffs_start": "2025-04-19",
    },
    2025: {
        "season_start": "2025-10-21",
        "trade_deadline": "2026-02-05",
        "allstar_start": "2026-02-13",
        "allstar_end": "2026-02-15",
        "regular_end": "2026-04-12",
        "playoffs_start": "2026-04-18",
    },
}

# Default milestones for unknown seasons (approximate)
DEFAULT_MILESTONES = {
    "trade_deadline_day": 107,  # ~Feb 6 (days from Oct 22)
    "allstar_day": 115,  # ~Feb 14
    "regular_end_day": 174,  # ~Apr 13
    "playoffs_start_day": 180,  # ~Apr 19
}


def _get_season(game_date_str: str) -> int:
    """Get season start year from game date. Oct+ = current year, else year-1."""
    year = int(game_date_str[:4])
    month = int(game_date_str[5:7])
    return year if month >= 10 else year - 1


def _parse_date(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%Y-%m-%d")


class TemporalFeatureExtractor(BaseFeatureExtractor):
    """V4 features from NBA calendar milestones and team continuity.

    Features (10 total):
    - is_post_trade_deadline: 1 if game is after trade deadline
    - days_since_trade_deadline: days since deadline (0 if before, capped at 60)
    - is_post_allstar: 1 if game is after All-Star break
    - days_since_allstar: days since ASB end (0 if before, capped at 60)
    - is_playoff_push: 1 if in final 30 days of regular season
    - is_regular_season: 1 if regular season (not playoffs)
    - season_pct: progress through season (0.0 = opening night, 1.0 = end)
    - player_games_with_team: consecutive games with current team (trade detection)
    - is_new_team: 1 if player changed teams within last 15 games
    - team_tenure_games: total games with current team this season
    """

    FEATURE_NAMES = (
        "is_post_trade_deadline",
        "days_since_trade_deadline",
        "is_post_allstar",
        "days_since_allstar",
        "is_playoff_push",
        "is_regular_season",
        "season_pct",
        "player_games_with_team",
        "is_new_team",
        "team_tenure_games",
    )

    def __init__(self, players_conn: Any):
        """
        Args:
            players_conn: Connection to nba_players database (port 5536)
        """
        super().__init__(players_conn, name="Temporal")

    @classmethod
    def get_defaults(cls) -> Dict[str, float]:
        return {
            "is_post_trade_deadline": 0.0,
            "days_since_trade_deadline": 0.0,
            "is_post_allstar": 0.0,
            "days_since_allstar": 0.0,
            "is_playoff_push": 0.0,
            "is_regular_season": 1.0,
            "season_pct": 0.5,
            "player_games_with_team": 30.0,  # Assume established
            "is_new_team": 0.0,
            "team_tenure_games": 40.0,  # Assume mid-season
        }

    def extract(
        self,
        player_name: str,
        game_date: Any,
        stat_type: str,
        **kwargs,
    ) -> Dict[str, float]:
        game_date_str = self._normalize_date(game_date)
        features = self.get_defaults()

        # 1. Calendar milestone features (no DB needed)
        self._compute_milestone_features(game_date_str, features)

        # 2. Team continuity features (needs player_game_logs)
        self._compute_team_continuity(player_name, game_date_str, features)

        return self.validate_features(features)

    def _compute_milestone_features(self, game_date_str: str, features: Dict[str, float]) -> None:
        """Compute calendar-based milestone features."""
        season = _get_season(game_date_str)
        game_date = _parse_date(game_date_str)

        milestones = NBA_MILESTONES.get(season)
        if milestones:
            season_start = _parse_date(milestones["season_start"])
            trade_deadline = _parse_date(milestones["trade_deadline"])
            allstar_end = _parse_date(milestones["allstar_end"])
            regular_end = _parse_date(milestones["regular_end"])
            playoffs_start = _parse_date(milestones["playoffs_start"])
        else:
            # Approximate for unknown seasons
            season_start = datetime(season, 10, 22)
            trade_deadline = season_start + timedelta(days=DEFAULT_MILESTONES["trade_deadline_day"])
            allstar_end = season_start + timedelta(days=DEFAULT_MILESTONES["allstar_day"])
            regular_end = season_start + timedelta(days=DEFAULT_MILESTONES["regular_end_day"])
            playoffs_start = season_start + timedelta(days=DEFAULT_MILESTONES["playoffs_start_day"])

        total_season_days = (regular_end - season_start).days or 174

        # Trade deadline
        days_since_td = (game_date - trade_deadline).days
        features["is_post_trade_deadline"] = 1.0 if days_since_td > 0 else 0.0
        features["days_since_trade_deadline"] = float(max(0, min(days_since_td, 60)))

        # All-Star break
        days_since_asb = (game_date - allstar_end).days
        features["is_post_allstar"] = 1.0 if days_since_asb > 0 else 0.0
        features["days_since_allstar"] = float(max(0, min(days_since_asb, 60)))

        # Playoff push (final 30 days of regular season)
        days_to_end = (regular_end - game_date).days
        features["is_playoff_push"] = 1.0 if 0 <= days_to_end <= 30 else 0.0

        # Regular season vs playoffs
        features["is_regular_season"] = 1.0 if game_date < playoffs_start else 0.0

        # Season progress (0.0 to 1.0)
        days_in = (game_date - season_start).days
        features["season_pct"] = float(max(0.0, min(1.0, days_in / total_season_days)))

    def _compute_team_continuity(
        self, player_name: str, game_date_str: str, features: Dict[str, float]
    ) -> None:
        """Detect if player recently changed teams."""
        query = """
            SELECT gl.team_abbrev, gl.game_date
            FROM player_game_logs gl
            JOIN player_profile pp ON gl.player_id = pp.player_id
            WHERE unaccent(pp.full_name) = %s
              AND gl.game_date < %s
              AND gl.minutes_played > 0
            ORDER BY gl.game_date DESC
            LIMIT 30
        """
        try:
            df = pd.read_sql_query(query, self.conn, params=(player_name, game_date_str))
        except Exception as e:
            logger.debug(f"Temporal: team continuity query failed: {e}")
            return

        if df is None or len(df) == 0:
            return

        teams = df["team_abbrev"].tolist()
        current_team = teams[0]

        # Count consecutive games with current team (from most recent)
        consecutive = 0
        for team in teams:
            if team == current_team:
                consecutive += 1
            else:
                break
        features["player_games_with_team"] = float(consecutive)

        # Total games with current team this season
        season_games = sum(1 for t in teams if t == current_team)
        features["team_tenure_games"] = float(season_games)

        # Is new team? (changed within last 15 games)
        unique_teams = set(teams[:15])
        features["is_new_team"] = 1.0 if len(unique_teams) > 1 else 0.0
