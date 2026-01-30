"""
Prop History Feature Extractor
==============================
Extracts 12 features from player's historical prop betting outcomes.

Features capture how well a player has historically performed relative
to sportsbook lines, providing signal for model calibration.
"""

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

from .base import BaseFeatureExtractor

logger = logging.getLogger(__name__)


class PropHistoryExtractor(BaseFeatureExtractor):
    """
    Extracts 12 historical prop betting features.

    Feature Groups:
    1. Hit Rate (4 features): Over/under hit percentages
    2. Margin (4 features): Average margin above/below lines
    3. Trend (4 features): Recent performance vs historical
    """

    FEATURE_NAMES = (
        # Hit rates (4)
        "prop_over_hit_rate_l10",
        "prop_over_hit_rate_l20",
        "prop_over_hit_rate_season",
        "prop_over_hit_rate_vs_market",
        # Margins (4)
        "prop_avg_margin_l10",
        "prop_avg_margin_l20",
        "prop_avg_margin_season",
        "prop_margin_std",
        # Trends (4)
        "prop_trend_direction",
        "prop_hit_streak",
        "prop_miss_streak",
        "prop_recent_vs_season",
    )

    def __init__(self, conn: Any):
        """
        Initialize prop history extractor.

        Args:
            conn: Database connection to nba_intelligence database
        """
        super().__init__(conn, name="PropHistory")

    @classmethod
    def get_defaults(cls) -> Dict[str, float]:
        """Get default values for all 12 prop history features."""
        return {
            "prop_over_hit_rate_l10": 0.5,
            "prop_over_hit_rate_l20": 0.5,
            "prop_over_hit_rate_season": 0.5,
            "prop_over_hit_rate_vs_market": 0.0,
            "prop_avg_margin_l10": 0.0,
            "prop_avg_margin_l20": 0.0,
            "prop_avg_margin_season": 0.0,
            "prop_margin_std": 0.0,
            "prop_trend_direction": 0.0,
            "prop_hit_streak": 0.0,
            "prop_miss_streak": 0.0,
            "prop_recent_vs_season": 0.0,
        }

    def extract(
        self,
        player_name: str,
        game_date: Any,
        stat_type: str,
        **kwargs,
    ) -> Dict[str, float]:
        """
        Extract prop history features for player.

        Args:
            player_name: Player's full name
            game_date: Game date
            stat_type: POINTS, REBOUNDS, ASSISTS, or THREES

        Returns:
            Dict with 12 prop history features
        """
        game_date_str = self._normalize_date(game_date)

        # Query historical props with outcomes
        query = """
        SELECT
            game_date,
            over_line,
            actual_result,
            (actual_result - over_line) as margin
        FROM nba_prop_lines
        WHERE player_name = %s
          AND stat_type = %s
          AND game_date < %s
          AND actual_result IS NOT NULL
          AND over_line IS NOT NULL
        ORDER BY game_date DESC
        LIMIT 50
        """
        df = self._safe_query(query, (player_name, stat_type.upper(), game_date_str))

        if df is None or len(df) == 0:
            return self.get_defaults()

        return self._compute_features(df)

    def _compute_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute all prop history features.

        Args:
            df: DataFrame with historical prop results

        Returns:
            Dict with computed features
        """
        features = self.get_defaults()

        # Calculate hit (actual > line)
        df["hit"] = (df["actual_result"] > df["over_line"]).astype(int)

        # Last 10 games
        l10 = df.head(10)
        if len(l10) > 0:
            features["prop_over_hit_rate_l10"] = l10["hit"].mean()
            features["prop_avg_margin_l10"] = l10["margin"].mean()

        # Last 20 games
        l20 = df.head(20)
        if len(l20) > 0:
            features["prop_over_hit_rate_l20"] = l20["hit"].mean()
            features["prop_avg_margin_l20"] = l20["margin"].mean()

        # Season (all available)
        features["prop_over_hit_rate_season"] = df["hit"].mean()
        features["prop_avg_margin_season"] = df["margin"].mean()
        features["prop_margin_std"] = df["margin"].std() if len(df) > 1 else 0.0

        # Vs market (deviation from 50%)
        features["prop_over_hit_rate_vs_market"] = features["prop_over_hit_rate_season"] - 0.5

        # Trends
        features["prop_recent_vs_season"] = (
            features["prop_over_hit_rate_l10"] - features["prop_over_hit_rate_season"]
        )

        # Trend direction (-1 to 1)
        if len(df) >= 10:
            recent_5 = l10.head(5)["hit"].mean() if len(l10) >= 5 else l10["hit"].mean()
            older_5 = l10.tail(5)["hit"].mean() if len(l10) >= 5 else l10["hit"].mean()
            features["prop_trend_direction"] = recent_5 - older_5

        # Streaks
        features["prop_hit_streak"] = self._count_streak(df["hit"], target=1)
        features["prop_miss_streak"] = self._count_streak(df["hit"], target=0)

        return features

    def _count_streak(self, series: pd.Series, target: int) -> float:
        """
        Count current streak of target value from most recent.

        Args:
            series: Boolean/int series of outcomes
            target: Value to count streak of (0 or 1)

        Returns:
            Current streak length
        """
        streak = 0
        for val in series:
            if val == target:
                streak += 1
            else:
                break
        return float(streak)
