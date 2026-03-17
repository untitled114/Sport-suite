"""
BettingPros Analytics Feature Extractor (V4)
=============================================
Extracts ~15 features from BP historical analytics and DVP data.

For training: queries bp_historical_analytics + bp_dvp_historical tables
(backfilled from BP API /v3/props and DVP page).

For live prediction: queries the same tables (populated by daily fetcher)
or falls back to live cheatsheet data already in the pipeline.

Database: nba_intelligence (port 5539)
  - bp_historical_analytics: per-prop projection, EV, hit rates, opposition rank
  - bp_dvp_historical: per-team per-position defensive stats allowed

Name normalization: handles Jr/Jr./III/II/period mismatches between
props_xl and BP naming conventions.
"""

import logging
from typing import Any, Dict, Optional

import pandas as pd

from .base import BaseFeatureExtractor

logger = logging.getLogger(__name__)


def normalize_name(name: str) -> str:
    """Normalize player name for matching across sources.

    Must produce the same result as the SQL:
    REPLACE(REPLACE(REPLACE(REPLACE(LOWER(name), '.', ''), ' jr', ''), ' iii', ''), ' ii', '')
    """
    if not name:
        return ""
    n = name.lower().strip()
    n = n.replace(".", "")
    # Strip suffixes in order (iii before ii to avoid partial match)
    for suffix in (" iii", " ii", " iv", " jr", " sr"):
        n = n.replace(suffix, "")
    return n.strip()


class BPAnalyticsFeatureExtractor(BaseFeatureExtractor):
    """V4 features from BettingPros historical analytics.

    Features (15 total):
    - bp_analytics_projection_diff: BP projection minus line
    - bp_analytics_ev: BP expected value for the over
    - bp_analytics_bet_rating: 1-5 star rating
    - bp_analytics_probability: BP computed P(over)
    - bp_analytics_opp_rank: Opposition defensive rank (1-30)
    - bp_analytics_opp_value: Opposition defensive rating value
    - bp_analytics_hit_rate_L5: Over hit rate last 5 games
    - bp_analytics_hit_rate_L10: Over hit rate last 10 games
    - bp_analytics_hit_rate_L15: Over hit rate last 15 games
    - bp_analytics_hit_rate_season: Season over hit rate
    - bp_analytics_hit_rate_trend: L5 - L15 momentum
    - bp_analytics_recommended_over: 1 if BP recommends over, 0 otherwise
    - dvp_stat_allowed: Opponent DVP stat allowed for position
    - dvp_stat_rank: Opponent DVP rank for position (1-30, normalized)
    - bp_analytics_proj_vs_consensus: BP projection minus consensus line
    """

    FEATURE_NAMES = (
        "bp_analytics_projection_diff",
        "bp_analytics_ev",
        "bp_analytics_bet_rating",
        "bp_analytics_probability",
        "bp_analytics_opp_rank",
        "bp_analytics_opp_value",
        "bp_analytics_hit_rate_L5",
        "bp_analytics_hit_rate_L10",
        "bp_analytics_hit_rate_L15",
        "bp_analytics_hit_rate_season",
        "bp_analytics_hit_rate_trend",
        "bp_analytics_recommended_over",
        "dvp_stat_allowed",
        "dvp_stat_rank",
        "bp_analytics_proj_vs_consensus",
    )

    # Map stat_type to DVP stat name
    STAT_TO_DVP = {
        "POINTS": "points",
        "REBOUNDS": "rebounds",
        "ASSISTS": "assists",
        "THREES": "three_points_made",
    }

    def __init__(self, conn: Any):
        super().__init__(conn, name="BPAnalytics")
        self._dvp_cache: Dict[str, Dict] = {}

    @classmethod
    def get_defaults(cls) -> Dict[str, float]:
        return {
            "bp_analytics_projection_diff": 0.0,
            "bp_analytics_ev": 0.0,
            "bp_analytics_bet_rating": 3.0,
            "bp_analytics_probability": 0.5,
            "bp_analytics_opp_rank": 15.0,
            "bp_analytics_opp_value": 0.0,  # 0 = no data (imputer handles)
            "bp_analytics_hit_rate_L5": 0.5,
            "bp_analytics_hit_rate_L10": 0.5,
            "bp_analytics_hit_rate_L15": 0.5,
            "bp_analytics_hit_rate_season": 0.5,
            "bp_analytics_hit_rate_trend": 0.0,
            "bp_analytics_recommended_over": 0.0,
            "dvp_stat_allowed": 0.0,  # 0 = no data (imputer handles)
            "dvp_stat_rank": 15.0,
            "bp_analytics_proj_vs_consensus": 0.0,
        }

    def extract(
        self,
        player_name: str,
        game_date: Any,
        stat_type: str,
        line: float = None,
        opponent_team: str = None,
        player_position: str = None,
        **kwargs,
    ) -> Dict[str, float]:
        game_date_str = self._normalize_date(game_date)
        features = self.get_defaults()

        # 1. BP historical analytics (projection, EV, hit rates, opp rank)
        bp_data = self._fetch_bp_analytics(player_name, game_date_str, stat_type)
        if bp_data is not None and len(bp_data) > 0:
            self._compute_bp_features(bp_data.iloc[0], features, line)

        # 2. DVP features (opponent defensive stats for position)
        if opponent_team and stat_type in self.STAT_TO_DVP:
            self._compute_dvp_features(
                opponent_team, stat_type, game_date_str, player_position, features
            )

        return self.validate_features(features)

    def _fetch_bp_analytics(
        self, player_name: str, game_date_str: str, stat_type: str
    ) -> Optional[pd.DataFrame]:
        """Fetch from bp_historical_analytics with name normalization."""
        # Try exact match first
        query = """
            SELECT bp_projection, bp_projection_diff, bp_probability,
                   bp_expected_value, bp_bet_rating, bp_recommended_side,
                   bp_opposition_rank, bp_opposition_value,
                   bp_hit_rate_L5, bp_hit_rate_L10, bp_hit_rate_L15,
                   bp_hit_rate_season, bp_consensus_line
            FROM bp_historical_analytics
            WHERE player_name = %s
              AND game_date = %s
              AND stat_type = %s
        """
        df = self._safe_query(query, (player_name, game_date_str, stat_type.upper()))
        if df is not None:
            return df

        # Try normalized name match
        norm = normalize_name(player_name)
        query_fuzzy = """
            SELECT bp_projection, bp_projection_diff, bp_probability,
                   bp_expected_value, bp_bet_rating, bp_recommended_side,
                   bp_opposition_rank, bp_opposition_value,
                   bp_hit_rate_L5, bp_hit_rate_L10, bp_hit_rate_L15,
                   bp_hit_rate_season, bp_consensus_line
            FROM bp_historical_analytics
            WHERE REPLACE(REPLACE(REPLACE(REPLACE(LOWER(player_name), '.', ''),
                  ' jr', ''), ' iii', ''), ' ii', '') = %s
              AND game_date = %s
              AND stat_type = %s
        """
        return self._safe_query(query_fuzzy, (norm, game_date_str, stat_type.upper()))

    def _compute_bp_features(
        self, row: pd.Series, features: Dict[str, float], line: float = None
    ) -> None:
        proj = self._safe_float(row.get("bp_projection"))
        proj_diff = self._safe_float(row.get("bp_projection_diff"))
        consensus = self._safe_float(row.get("bp_consensus_line"))

        features["bp_analytics_projection_diff"] = proj_diff
        features["bp_analytics_ev"] = self._safe_float(row.get("bp_expected_value"))
        features["bp_analytics_bet_rating"] = self._safe_float(row.get("bp_bet_rating"), 3.0)
        features["bp_analytics_probability"] = self._safe_float(row.get("bp_probability"), 0.5)
        features["bp_analytics_opp_rank"] = self._safe_float(row.get("bp_opposition_rank"), 15.0)
        features["bp_analytics_opp_value"] = self._safe_float(row.get("bp_opposition_value"))

        # Hit rates
        hr_l5 = self._safe_float(row.get("bp_hit_rate_l5"), 0.5)
        hr_l10 = self._safe_float(row.get("bp_hit_rate_l10"), 0.5)
        hr_l15 = self._safe_float(row.get("bp_hit_rate_l15"), 0.5)
        hr_season = self._safe_float(row.get("bp_hit_rate_season"), 0.5)

        features["bp_analytics_hit_rate_L5"] = hr_l5
        features["bp_analytics_hit_rate_L10"] = hr_l10
        features["bp_analytics_hit_rate_L15"] = hr_l15
        features["bp_analytics_hit_rate_season"] = hr_season
        features["bp_analytics_hit_rate_trend"] = hr_l5 - hr_l15

        # Recommended side
        rec_side = row.get("bp_recommended_side")
        features["bp_analytics_recommended_over"] = 1.0 if rec_side == "over" else 0.0

        # Projection vs consensus
        if proj > 0 and consensus > 0:
            features["bp_analytics_proj_vs_consensus"] = proj - consensus

    def _compute_dvp_features(
        self,
        opponent_team: str,
        stat_type: str,
        game_date_str: str,
        player_position: str,
        features: Dict[str, float],
    ) -> None:
        """Look up opponent DVP for the relevant stat and position."""
        dvp_stat = self.STAT_TO_DVP.get(stat_type.upper())
        if not dvp_stat:
            return

        # Determine season from game_date
        year = int(game_date_str[:4])
        month = int(game_date_str[5:7])
        season = year if month >= 10 else year - 1

        position = player_position or "ALL"

        cache_key = f"{season}:{opponent_team}:{position}:{dvp_stat}"
        if cache_key in self._dvp_cache:
            cached = self._dvp_cache[cache_key]
            features["dvp_stat_allowed"] = cached.get("value", 0.0)
            features["dvp_stat_rank"] = cached.get("rank", 15.0)
            return

        # Query DVP value
        query_val = """
            SELECT value FROM bp_dvp_historical
            WHERE season = %s AND team = %s AND position = %s AND stat_name = %s
        """
        df_val = self._safe_query(query_val, (season, opponent_team, position, dvp_stat))
        if df_val is not None and len(df_val) > 0:
            val = self._safe_float(df_val.iloc[0]["value"])
            features["dvp_stat_allowed"] = val

            # Compute rank: how does this team compare to all teams?
            query_rank = """
                SELECT team, value FROM bp_dvp_historical
                WHERE season = %s AND position = %s AND stat_name = %s
                ORDER BY value DESC
            """
            df_rank = self._safe_query(query_rank, (season, position, dvp_stat))
            if df_rank is not None and len(df_rank) > 0:
                rank_list = df_rank["team"].tolist()
                try:
                    rank = rank_list.index(opponent_team) + 1
                except ValueError:
                    rank = 15
                features["dvp_stat_rank"] = float(rank)

            self._dvp_cache[cache_key] = {
                "value": features["dvp_stat_allowed"],
                "rank": features["dvp_stat_rank"],
            }
