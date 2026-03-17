"""
Game Context Feature Extractor (V4)
====================================
Extracts ~8 features from games table and player_game_logs that capture
game-level context not available in player rolling stats.

Features derived from our own database — no external API needed.
Full historical coverage since Oct 2023.

Databases:
  - nba_games (port 5537): pace, blowout_flag, scores, possessions
  - nba_players (port 5536): player_game_logs for minutes, plus_minus
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .base import BaseFeatureExtractor

logger = logging.getLogger(__name__)


class GameContextFeatureExtractor(BaseFeatureExtractor):
    """V4 features from game-level and player-level context.

    Features (8 total):
    - game_pace: Pace of the upcoming game (from team averages)
    - opp_score_margin_avg: Opponent's avg margin of victory L10 (blowout proxy)
    - player_minutes_stability: Std dev of minutes L10 (lower = more stable)
    - player_plus_minus_L5: Average plus/minus over last 5 games
    - player_usage_proxy: FGA / minutes ratio L5 (scoring involvement)
    - player_scoring_efficiency: Points / FGA ratio L5
    - player_blowout_risk: Fraction of player's L10 games that were blowouts
    - player_minutes_vs_avg: Latest game minutes / season avg (load indicator)
    """

    FEATURE_NAMES = (
        "game_pace",
        "opp_score_margin_avg",
        "player_minutes_stability",
        "player_plus_minus_L5",
        "player_usage_proxy",
        "player_scoring_efficiency",
        "player_blowout_risk",
        "player_minutes_vs_avg",
    )

    def __init__(self, games_conn: Any, players_conn: Any):
        """
        Initialize with connections to games and players databases.

        Args:
            games_conn: Connection to nba_games (port 5537)
            players_conn: Connection to nba_players (port 5536)
        """
        # Use games_conn as primary for BaseFeatureExtractor
        super().__init__(games_conn, name="GameContext")
        self.players_conn = players_conn

    @classmethod
    def get_defaults(cls) -> Dict[str, float]:
        return {
            "game_pace": 100.0,
            "opp_score_margin_avg": 0.0,
            "player_minutes_stability": 5.0,
            "player_plus_minus_L5": 0.0,
            "player_usage_proxy": 0.0,
            "player_scoring_efficiency": 0.0,
            "player_blowout_risk": 0.0,
            "player_minutes_vs_avg": 1.0,
        }

    def extract(
        self,
        player_name: str,
        game_date: Any,
        stat_type: str,
        opponent_team: str = None,
        **kwargs,
    ) -> Dict[str, float]:
        game_date_str = self._normalize_date(game_date)
        features = self.get_defaults()

        # 1. Game-level features (pace, opponent margin)
        if opponent_team:
            self._compute_game_features(opponent_team, game_date_str, features)

        # 2. Player-level features (minutes stability, plus/minus, usage, efficiency)
        self._compute_player_features(player_name, game_date_str, features)

        return self.validate_features(features)

    def _compute_game_features(
        self, opponent_team: str, game_date_str: str, features: Dict[str, float]
    ) -> None:
        """Compute pace and opponent blowout tendency from games table."""
        # Opponent's avg pace from last 10 games
        query_pace = """
            SELECT AVG(pace) as avg_pace,
                   AVG(ABS(home_score - away_score)) as avg_margin,
                   COUNT(CASE WHEN blowout_flag THEN 1 END)::float
                       / NULLIF(COUNT(*), 0) as blowout_rate
            FROM games
            WHERE (home_team = %s OR away_team = %s)
              AND game_date < %s
              AND pace IS NOT NULL
            ORDER BY game_date DESC
            LIMIT 10
        """
        try:
            df = pd.read_sql_query(
                query_pace, self.conn, params=(opponent_team, opponent_team, game_date_str)
            )
            if df is not None and len(df) > 0:
                row = df.iloc[0]
                pace = row.get("avg_pace")
                if pace is not None and not pd.isna(pace):
                    features["game_pace"] = float(pace)
                margin = row.get("avg_margin")
                if margin is not None and not pd.isna(margin):
                    features["opp_score_margin_avg"] = float(margin)
        except Exception as e:
            logger.debug(f"GameContext: pace query failed: {e}")

    def _compute_player_features(
        self, player_name: str, game_date_str: str, features: Dict[str, float]
    ) -> None:
        """Compute player minutes stability, plus/minus, usage from game logs."""
        query = """
            SELECT minutes_played, plus_minus, points, fg_attempted
            FROM player_game_logs gl
            JOIN player_profile pp ON gl.player_id = pp.player_id
            WHERE pp.full_name = %s
              AND gl.game_date < %s
              AND gl.minutes_played > 0
            ORDER BY gl.game_date DESC
            LIMIT 10
        """
        try:
            df = pd.read_sql_query(query, self.players_conn, params=(player_name, game_date_str))
        except Exception as e:
            logger.debug(f"GameContext: player query failed: {e}")
            return

        if df is None or len(df) == 0:
            return

        minutes = df["minutes_played"].astype(float)
        plus_minus = df["plus_minus"].dropna().astype(float)
        points = df["points"].astype(float)
        fga = df["fg_attempted"].astype(float)

        # Minutes stability (std dev of L10 minutes)
        if len(minutes) >= 3:
            features["player_minutes_stability"] = float(minutes.std())

        # Plus/minus L5
        if len(plus_minus) >= 5:
            features["player_plus_minus_L5"] = float(plus_minus.head(5).mean())
        elif len(plus_minus) > 0:
            features["player_plus_minus_L5"] = float(plus_minus.mean())

        # Usage proxy (FGA / minutes) for L5
        l5_minutes = minutes.head(5)
        l5_fga = fga.head(5)
        total_min = l5_minutes.sum()
        if total_min > 0:
            features["player_usage_proxy"] = float(l5_fga.sum() / total_min)

        # Scoring efficiency (points / FGA) for L5
        l5_points = points.head(5)
        total_fga = l5_fga.sum()
        if total_fga > 0:
            features["player_scoring_efficiency"] = float(l5_points.sum() / total_fga)

        # Blowout risk: fraction of L10 games with plus_minus > 15 or < -15
        if len(plus_minus) >= 3:
            blowout_games = (plus_minus.abs() > 15).sum()
            features["player_blowout_risk"] = float(blowout_games / len(plus_minus))

        # Minutes vs season avg
        season_avg = minutes.mean()
        if season_avg > 0 and len(minutes) >= 2:
            latest = minutes.iloc[0]
            features["player_minutes_vs_avg"] = float(latest / season_avg)
