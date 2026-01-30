"""
Head-to-Head Feature Extractor
==============================
Extracts 36 historical matchup features for player vs opponent.

Features capture how a player historically performs against specific
opponents, providing valuable context for prop predictions.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .base import BaseFeatureExtractor

logger = logging.getLogger(__name__)


class H2HFeatureExtractor(BaseFeatureExtractor):
    """
    Extracts 36 head-to-head matchup features.

    Feature Groups:
    1. Career H2H (12 features): Career stats vs opponent
    2. Recent H2H (12 features): Last 5 games vs opponent
    3. Venue H2H (6 features): Home/away splits vs opponent
    4. Context (6 features): Games played, trends, etc.
    """

    FEATURE_NAMES = (
        # Career H2H stats (12)
        "h2h_career_games",
        "h2h_career_points_avg",
        "h2h_career_rebounds_avg",
        "h2h_career_assists_avg",
        "h2h_career_threes_avg",
        "h2h_career_minutes_avg",
        "h2h_career_points_std",
        "h2h_career_rebounds_std",
        "h2h_career_assists_std",
        "h2h_career_threes_std",
        "h2h_career_fg_pct",
        "h2h_career_three_pct",
        # Recent H2H (last 5 games) (12)
        "h2h_recent_games",
        "h2h_recent_points_avg",
        "h2h_recent_rebounds_avg",
        "h2h_recent_assists_avg",
        "h2h_recent_threes_avg",
        "h2h_recent_minutes_avg",
        "h2h_recent_points_std",
        "h2h_recent_rebounds_std",
        "h2h_recent_assists_std",
        "h2h_recent_threes_std",
        "h2h_recent_fg_pct",
        "h2h_recent_three_pct",
        # Venue splits vs opponent (6)
        "h2h_home_points_avg",
        "h2h_home_rebounds_avg",
        "h2h_home_assists_avg",
        "h2h_away_points_avg",
        "h2h_away_rebounds_avg",
        "h2h_away_assists_avg",
        # Context (6)
        "h2h_last_game_days_ago",
        "h2h_points_trend",
        "h2h_rebounds_trend",
        "h2h_assists_trend",
        "h2h_matchup_advantage",
        "h2h_consistency_score",
    )

    def __init__(self, conn: Any):
        """
        Initialize H2H feature extractor.

        Args:
            conn: Database connection to nba_players database
        """
        super().__init__(conn, name="H2HFeatures")

    @classmethod
    def get_defaults(cls) -> Dict[str, float]:
        """Get default values for all 36 H2H features."""
        return {name: 0.0 for name in cls.FEATURE_NAMES}

    def extract(
        self,
        player_name: str,
        game_date: Any,
        stat_type: str,
        opponent_team: str = None,
        **kwargs,
    ) -> Dict[str, float]:
        """
        Extract H2H features for player vs opponent.

        Args:
            player_name: Player's full name
            game_date: Game date
            stat_type: Stat type (for context, not used in extraction)
            opponent_team: Opponent team abbreviation (required)

        Returns:
            Dict with 36 H2H features
        """
        if not opponent_team:
            return self.get_defaults()

        game_date_str = self._normalize_date(game_date)

        # Query career history vs opponent
        query = """
        SELECT
            game_date,
            points,
            rebounds,
            assists,
            three_pointers_made,
            minutes,
            field_goals_made,
            field_goals_attempted,
            three_pointers_attempted,
            is_home
        FROM player_game_logs
        WHERE player_name = %s
          AND opponent_team = %s
          AND game_date < %s
        ORDER BY game_date DESC
        """
        df = self._safe_query(query, (player_name, opponent_team, game_date_str))

        if df is None or len(df) == 0:
            return self.get_defaults()

        return self._compute_features(df, game_date_str)

    def _compute_features(
        self,
        df: pd.DataFrame,
        game_date_str: str,
    ) -> Dict[str, float]:
        """
        Compute all H2H features from historical games.

        Args:
            df: DataFrame with game logs vs opponent
            game_date_str: Current game date for recency calculations

        Returns:
            Dict with computed features
        """
        features = self.get_defaults()

        # Career H2H stats
        features["h2h_career_games"] = float(len(df))
        features["h2h_career_points_avg"] = self._safe_float(df["points"].mean())
        features["h2h_career_rebounds_avg"] = self._safe_float(df["rebounds"].mean())
        features["h2h_career_assists_avg"] = self._safe_float(df["assists"].mean())
        features["h2h_career_threes_avg"] = self._safe_float(df["three_pointers_made"].mean())
        features["h2h_career_minutes_avg"] = self._safe_float(df["minutes"].mean())

        features["h2h_career_points_std"] = self._safe_float(df["points"].std())
        features["h2h_career_rebounds_std"] = self._safe_float(df["rebounds"].std())
        features["h2h_career_assists_std"] = self._safe_float(df["assists"].std())
        features["h2h_career_threes_std"] = self._safe_float(df["three_pointers_made"].std())

        # Career shooting percentages
        total_fgm = df["field_goals_made"].sum()
        total_fga = df["field_goals_attempted"].sum()
        total_3pm = df["three_pointers_made"].sum()
        total_3pa = df["three_pointers_attempted"].sum()

        features["h2h_career_fg_pct"] = total_fgm / total_fga if total_fga > 0 else 0.0
        features["h2h_career_three_pct"] = total_3pm / total_3pa if total_3pa > 0 else 0.0

        # Recent H2H (last 5 games)
        recent = df.head(5)
        features["h2h_recent_games"] = float(len(recent))
        features["h2h_recent_points_avg"] = self._safe_float(recent["points"].mean())
        features["h2h_recent_rebounds_avg"] = self._safe_float(recent["rebounds"].mean())
        features["h2h_recent_assists_avg"] = self._safe_float(recent["assists"].mean())
        features["h2h_recent_threes_avg"] = self._safe_float(recent["three_pointers_made"].mean())
        features["h2h_recent_minutes_avg"] = self._safe_float(recent["minutes"].mean())

        features["h2h_recent_points_std"] = self._safe_float(recent["points"].std())
        features["h2h_recent_rebounds_std"] = self._safe_float(recent["rebounds"].std())
        features["h2h_recent_assists_std"] = self._safe_float(recent["assists"].std())
        features["h2h_recent_threes_std"] = self._safe_float(recent["three_pointers_made"].std())

        recent_fgm = recent["field_goals_made"].sum()
        recent_fga = recent["field_goals_attempted"].sum()
        recent_3pm = recent["three_pointers_made"].sum()
        recent_3pa = recent["three_pointers_attempted"].sum()

        features["h2h_recent_fg_pct"] = recent_fgm / recent_fga if recent_fga > 0 else 0.0
        features["h2h_recent_three_pct"] = recent_3pm / recent_3pa if recent_3pa > 0 else 0.0

        # Venue splits
        if "is_home" in df.columns:
            home_games = df[df["is_home"].fillna(False)]
            away_games = df[~df["is_home"].fillna(True)]

            if len(home_games) > 0:
                features["h2h_home_points_avg"] = self._safe_float(home_games["points"].mean())
                features["h2h_home_rebounds_avg"] = self._safe_float(home_games["rebounds"].mean())
                features["h2h_home_assists_avg"] = self._safe_float(home_games["assists"].mean())

            if len(away_games) > 0:
                features["h2h_away_points_avg"] = self._safe_float(away_games["points"].mean())
                features["h2h_away_rebounds_avg"] = self._safe_float(away_games["rebounds"].mean())
                features["h2h_away_assists_avg"] = self._safe_float(away_games["assists"].mean())

        # Context features
        if len(df) > 0 and "game_date" in df.columns:
            last_game = pd.to_datetime(df["game_date"].iloc[0])
            current_date = pd.to_datetime(game_date_str)
            days_ago = (current_date - last_game).days
            features["h2h_last_game_days_ago"] = float(days_ago)

        # Trends (comparing recent to career)
        if len(df) >= 3:
            features["h2h_points_trend"] = (
                features["h2h_recent_points_avg"] - features["h2h_career_points_avg"]
            )
            features["h2h_rebounds_trend"] = (
                features["h2h_recent_rebounds_avg"] - features["h2h_career_rebounds_avg"]
            )
            features["h2h_assists_trend"] = (
                features["h2h_recent_assists_avg"] - features["h2h_career_assists_avg"]
            )

        # Matchup advantage (composite score)
        features["h2h_matchup_advantage"] = self._compute_matchup_advantage(features)

        # Consistency score (inverse of std/mean ratio)
        features["h2h_consistency_score"] = self._compute_consistency(features)

        return features

    def _compute_matchup_advantage(self, features: Dict[str, float]) -> float:
        """
        Compute matchup advantage score.

        Combines H2H performance trends and shooting percentages
        into a single advantage score.

        Args:
            features: Computed H2H features

        Returns:
            Advantage score (-1 to 1 range)
        """
        if features["h2h_career_games"] < 2:
            return 0.0

        # Weight factors
        points_factor = features["h2h_points_trend"] / 10.0  # Normalize
        shooting_factor = (features["h2h_recent_fg_pct"] - 0.45) * 2  # Above/below avg

        return float(np.clip(points_factor + shooting_factor, -1.0, 1.0))

    def _compute_consistency(self, features: Dict[str, float]) -> float:
        """
        Compute consistency score for H2H performance.

        Lower standard deviation relative to mean = higher consistency.

        Args:
            features: Computed H2H features

        Returns:
            Consistency score (0 to 1 range)
        """
        if features["h2h_career_points_avg"] < 1:
            return 0.5  # Neutral if insufficient data

        cv = features["h2h_career_points_std"] / features["h2h_career_points_avg"]
        # Convert coefficient of variation to consistency (inverse, capped)
        consistency = 1.0 - min(cv, 1.0)
        return float(consistency)
