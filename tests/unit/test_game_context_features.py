"""
Unit tests for nba.features.extractors.game_context_features

Tests cover:
- GameContextFeatureExtractor.get_defaults()
- GameContextFeatureExtractor.extract() with mocked DB
- _compute_game_features — pace, opponent margin
- _compute_player_features — minutes stability, plus/minus, usage, efficiency,
  blowout risk, minutes vs avg
- Edge cases: empty DataFrames, None values, insufficient games
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from nba.features.extractors.game_context_features import GameContextFeatureExtractor

# ─────────────────────────────────────────────────────────────────
# get_defaults
# ─────────────────────────────────────────────────────────────────


class TestGetDefaults:
    def test_returns_8_features(self):
        defaults = GameContextFeatureExtractor.get_defaults()
        assert len(defaults) == 8

    def test_all_feature_names_present(self):
        defaults = GameContextFeatureExtractor.get_defaults()
        for name in GameContextFeatureExtractor.FEATURE_NAMES:
            assert name in defaults

    def test_default_pace(self):
        assert GameContextFeatureExtractor.get_defaults()["game_pace"] == 100.0

    def test_default_minutes_vs_avg(self):
        assert GameContextFeatureExtractor.get_defaults()["player_minutes_vs_avg"] == 1.0

    def test_all_values_are_floats(self):
        for k, v in GameContextFeatureExtractor.get_defaults().items():
            assert isinstance(v, (int, float)), f"{k} is {type(v)}"


# ─────────────────────────────────────────────────────────────────
# FEATURE_NAMES
# ─────────────────────────────────────────────────────────────────


class TestFeatureNames:
    def test_feature_names_count(self):
        assert len(GameContextFeatureExtractor.FEATURE_NAMES) == 8

    def test_no_duplicates(self):
        names = GameContextFeatureExtractor.FEATURE_NAMES
        assert len(names) == len(set(names))

    def test_feature_names_match_defaults(self):
        defaults = GameContextFeatureExtractor.get_defaults()
        for name in GameContextFeatureExtractor.FEATURE_NAMES:
            assert name in defaults


# ─────────────────────────────────────────────────────────────────
# __init__
# ─────────────────────────────────────────────────────────────────


class TestInit:
    def test_stores_both_connections(self):
        games_conn = MagicMock()
        players_conn = MagicMock()
        ext = GameContextFeatureExtractor(games_conn, players_conn)
        assert ext.conn is games_conn
        assert ext.players_conn is players_conn
        assert ext.name == "GameContext"


# ─────────────────────────────────────────────────────────────────
# extract — full pipeline
# ─────────────────────────────────────────────────────────────────


class TestExtract:
    @pytest.fixture
    def extractor(self):
        return GameContextFeatureExtractor(MagicMock(), MagicMock())

    def test_returns_defaults_when_no_data(self, extractor):
        with patch("pandas.read_sql_query", side_effect=Exception("no db")):
            result = extractor.extract("Test Player", "2025-01-15", "POINTS")
        assert len(result) == 8
        assert result["game_pace"] == 100.0

    def test_returns_8_features(self, extractor):
        with patch("pandas.read_sql_query", return_value=pd.DataFrame()):
            result = extractor.extract("Test", "2025-01-15", "POINTS")
        assert len(result) == 8

    def test_no_opponent_skips_game_features(self, extractor):
        with patch("pandas.read_sql_query", return_value=pd.DataFrame()):
            result = extractor.extract("Test", "2025-01-15", "POINTS", opponent_team=None)
        assert result["game_pace"] == 100.0
        assert result["opp_score_margin_avg"] == 0.0


# ─────────────────────────────────────────────────────────────────
# _compute_game_features
# ─────────────────────────────────────────────────────────────────


class TestComputeGameFeatures:
    @pytest.fixture
    def extractor(self):
        return GameContextFeatureExtractor(MagicMock(), MagicMock())

    def test_pace_and_margin(self, extractor):
        game_df = pd.DataFrame([{"avg_pace": 102.5, "avg_margin": 8.3, "blowout_rate": 0.2}])
        with patch("pandas.read_sql_query", return_value=game_df):
            features = GameContextFeatureExtractor.get_defaults()
            extractor._compute_game_features("MIA", "2025-01-15", features)
        assert features["game_pace"] == 102.5
        assert features["opp_score_margin_avg"] == 8.3

    def test_null_pace_keeps_default(self, extractor):
        game_df = pd.DataFrame([{"avg_pace": None, "avg_margin": None, "blowout_rate": None}])
        with patch("pandas.read_sql_query", return_value=game_df):
            features = GameContextFeatureExtractor.get_defaults()
            extractor._compute_game_features("MIA", "2025-01-15", features)
        assert features["game_pace"] == 100.0

    def test_empty_result_keeps_defaults(self, extractor):
        with patch("pandas.read_sql_query", return_value=pd.DataFrame()):
            features = GameContextFeatureExtractor.get_defaults()
            extractor._compute_game_features("MIA", "2025-01-15", features)
        assert features["game_pace"] == 100.0

    def test_query_error_keeps_defaults(self, extractor):
        with patch("pandas.read_sql_query", side_effect=Exception("db error")):
            features = GameContextFeatureExtractor.get_defaults()
            extractor._compute_game_features("MIA", "2025-01-15", features)
        assert features["game_pace"] == 100.0


# ─────────────────────────────────────────────────────────────────
# _compute_player_features
# ─────────────────────────────────────────────────────────────────


class TestComputePlayerFeatures:
    @pytest.fixture
    def extractor(self):
        return GameContextFeatureExtractor(MagicMock(), MagicMock())

    def _make_game_logs(self, n=10, minutes=30, pm=5, pts=25, fga=18):
        """Create a mock game logs DataFrame."""
        return pd.DataFrame(
            {
                "minutes_played": [minutes + (i % 5) for i in range(n)],
                "plus_minus": [pm - (i % 3) for i in range(n)],
                "points": [pts + (i % 4) for i in range(n)],
                "fg_attempted": [fga + (i % 3) for i in range(n)],
            }
        )

    def test_full_features_with_10_games(self, extractor):
        df = self._make_game_logs(10)
        with patch("pandas.read_sql_query", return_value=df):
            features = GameContextFeatureExtractor.get_defaults()
            extractor._compute_player_features("LeBron James", "2025-01-15", features)

        assert features["player_minutes_stability"] > 0
        assert features["player_plus_minus_L5"] != 0
        assert features["player_usage_proxy"] > 0
        assert features["player_scoring_efficiency"] > 0
        assert features["player_minutes_vs_avg"] > 0

    def test_plus_minus_L5_with_5_games(self, extractor):
        df = pd.DataFrame(
            {
                "minutes_played": [30, 32, 28, 35, 31],
                "plus_minus": [10, -5, 8, 3, -2],
                "points": [25, 20, 30, 22, 18],
                "fg_attempted": [18, 15, 22, 16, 14],
            }
        )
        with patch("pandas.read_sql_query", return_value=df):
            features = GameContextFeatureExtractor.get_defaults()
            extractor._compute_player_features("Test", "2025-01-15", features)
        # mean of [10, -5, 8, 3, -2] = 2.8
        assert features["player_plus_minus_L5"] == pytest.approx(2.8)

    def test_plus_minus_with_fewer_than_5_games(self, extractor):
        df = pd.DataFrame(
            {
                "minutes_played": [30, 32],
                "plus_minus": [10, -4],
                "points": [25, 20],
                "fg_attempted": [18, 15],
            }
        )
        with patch("pandas.read_sql_query", return_value=df):
            features = GameContextFeatureExtractor.get_defaults()
            extractor._compute_player_features("Test", "2025-01-15", features)
        # mean of [10, -4] = 3.0
        assert features["player_plus_minus_L5"] == pytest.approx(3.0)

    def test_usage_proxy_computation(self, extractor):
        df = pd.DataFrame(
            {
                "minutes_played": [30, 30, 30, 30, 30],
                "plus_minus": [0, 0, 0, 0, 0],
                "points": [20, 20, 20, 20, 20],
                "fg_attempted": [15, 15, 15, 15, 15],
            }
        )
        with patch("pandas.read_sql_query", return_value=df):
            features = GameContextFeatureExtractor.get_defaults()
            extractor._compute_player_features("Test", "2025-01-15", features)
        # 75 fga / 150 min = 0.5
        assert features["player_usage_proxy"] == pytest.approx(0.5)

    def test_scoring_efficiency(self, extractor):
        df = pd.DataFrame(
            {
                "minutes_played": [30, 30, 30, 30, 30],
                "plus_minus": [0, 0, 0, 0, 0],
                "points": [20, 20, 20, 20, 20],
                "fg_attempted": [10, 10, 10, 10, 10],
            }
        )
        with patch("pandas.read_sql_query", return_value=df):
            features = GameContextFeatureExtractor.get_defaults()
            extractor._compute_player_features("Test", "2025-01-15", features)
        # 100 pts / 50 fga = 2.0
        assert features["player_scoring_efficiency"] == pytest.approx(2.0)

    def test_blowout_risk(self, extractor):
        df = pd.DataFrame(
            {
                "minutes_played": [30] * 10,
                "plus_minus": [20, -18, 5, 3, -1, 25, -2, 4, 16, -20],
                "points": [25] * 10,
                "fg_attempted": [18] * 10,
            }
        )
        with patch("pandas.read_sql_query", return_value=df):
            features = GameContextFeatureExtractor.get_defaults()
            extractor._compute_player_features("Test", "2025-01-15", features)
        # |20|, |18|, |25|, |16|, |20| > 15 = 5 out of 10 = 0.5
        assert features["player_blowout_risk"] == pytest.approx(0.5)

    def test_minutes_vs_avg(self, extractor):
        df = pd.DataFrame(
            {
                "minutes_played": [36, 30, 30, 30, 30, 30, 30, 30, 30, 30],
                "plus_minus": [0] * 10,
                "points": [25] * 10,
                "fg_attempted": [18] * 10,
            }
        )
        with patch("pandas.read_sql_query", return_value=df):
            features = GameContextFeatureExtractor.get_defaults()
            extractor._compute_player_features("Test", "2025-01-15", features)
        # latest (36) / avg (30.6) ≈ 1.176
        assert features["player_minutes_vs_avg"] > 1.0

    def test_empty_game_logs_keeps_defaults(self, extractor):
        with patch("pandas.read_sql_query", return_value=pd.DataFrame()):
            features = GameContextFeatureExtractor.get_defaults()
            extractor._compute_player_features("Ghost", "2025-01-15", features)
        assert features["player_plus_minus_L5"] == 0.0
        assert features["player_usage_proxy"] == 0.0

    def test_query_error_keeps_defaults(self, extractor):
        with patch("pandas.read_sql_query", side_effect=Exception("db error")):
            features = GameContextFeatureExtractor.get_defaults()
            extractor._compute_player_features("Test", "2025-01-15", features)
        assert features["player_plus_minus_L5"] == 0.0

    def test_minutes_stability_needs_3_games(self, extractor):
        df = pd.DataFrame(
            {
                "minutes_played": [30, 32],
                "plus_minus": [5, -3],
                "points": [25, 20],
                "fg_attempted": [18, 15],
            }
        )
        with patch("pandas.read_sql_query", return_value=df):
            features = GameContextFeatureExtractor.get_defaults()
            extractor._compute_player_features("Test", "2025-01-15", features)
        # Only 2 games, should keep default 5.0
        assert features["player_minutes_stability"] == 5.0

    def test_zero_fga_skips_efficiency(self, extractor):
        df = pd.DataFrame(
            {
                "minutes_played": [30, 30, 30, 30, 30],
                "plus_minus": [0, 0, 0, 0, 0],
                "points": [0, 0, 0, 0, 0],
                "fg_attempted": [0, 0, 0, 0, 0],
            }
        )
        with patch("pandas.read_sql_query", return_value=df):
            features = GameContextFeatureExtractor.get_defaults()
            extractor._compute_player_features("Test", "2025-01-15", features)
        assert features["player_scoring_efficiency"] == 0.0

    def test_null_plus_minus_handled(self, extractor):
        df = pd.DataFrame(
            {
                "minutes_played": [30, 30, 30, 30, 30],
                "plus_minus": [None, None, None, 5, -3],
                "points": [20, 20, 20, 20, 20],
                "fg_attempted": [15, 15, 15, 15, 15],
            }
        )
        with patch("pandas.read_sql_query", return_value=df):
            features = GameContextFeatureExtractor.get_defaults()
            extractor._compute_player_features("Test", "2025-01-15", features)
        # Only 2 non-null plus_minus values, should still compute mean
        assert features["player_plus_minus_L5"] == pytest.approx(1.0)
