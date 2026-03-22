"""
Unit tests for nba.features.extractors.bp_analytics_features

Tests cover:
- normalize_name() — suffix/period stripping
- BPAnalyticsFeatureExtractor.get_defaults()
- BPAnalyticsFeatureExtractor.extract() with mocked DB
- _fetch_bp_analytics exact and fuzzy match
- _compute_bp_features — projection, EV, hit rates, recommended side
- _compute_dvp_features — DVP value, rank, caching
- Edge cases: None values, empty DataFrames, missing opponent
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from nba.features.extractors.bp_analytics_features import (
    BPAnalyticsFeatureExtractor,
    normalize_name,
)

# ─────────────────────────────────────────────────────────────────
# normalize_name
# ─────────────────────────────────────────────────────────────────


class TestNormalizeName:
    def test_empty_string(self):
        assert normalize_name("") == ""

    def test_none(self):
        assert normalize_name(None) == ""

    def test_simple_name(self):
        assert normalize_name("LeBron James") == "lebron james"

    def test_strips_periods(self):
        assert normalize_name("P.J. Washington") == "pj washington"
        assert normalize_name("A.J. Lawson") == "aj lawson"
        assert normalize_name("T.J. McConnell") == "tj mcconnell"

    def test_strips_jr(self):
        assert normalize_name("Jabari Smith Jr") == "jabari smith"
        assert normalize_name("Jabari Smith Jr.") == "jabari smith"

    def test_strips_iii(self):
        assert normalize_name("Robert Williams III") == "robert williams"
        assert normalize_name("Russell Westbrook III") == "russell westbrook"

    def test_strips_ii(self):
        assert normalize_name("Ronald Holland II") == "ronald holland"

    def test_strips_sr(self):
        assert normalize_name("Xavier Tillman Sr") == "xavier tillman"
        assert normalize_name("Xavier Tillman Sr.") == "xavier tillman"

    def test_combined_period_and_suffix(self):
        assert normalize_name("P.J. Washington Jr.") == "pj washington"
        assert normalize_name("PJ Washington Jr") == "pj washington"

    def test_matching_across_formats(self):
        """Both formats should produce the same normalized name."""
        pairs = [
            ("PJ Washington Jr", "P.J. Washington Jr."),
            ("Michael Porter", "Michael Porter Jr."),
            ("Bobby Portis", "Bobby Portis Jr."),
            ("AJ Lawson", "A.J. Lawson"),
            ("Robert Williams", "Robert Williams III"),
            ("Tim Hardaway Jr", "Tim Hardaway Jr."),
        ]
        for a, b in pairs:
            assert normalize_name(a) == normalize_name(b), f"{a!r} != {b!r}"

    def test_preserves_normal_names(self):
        assert normalize_name("Nikola Jokic") == "nikola jokic"
        assert normalize_name("Stephen Curry") == "stephen curry"

    def test_strips_whitespace(self):
        assert normalize_name("  LeBron James  ") == "lebron james"


# ─────────────────────────────────────────────────────────────────
# get_defaults
# ─────────────────────────────────────────────────────────────────


class TestGetDefaults:
    def test_returns_15_features(self):
        defaults = BPAnalyticsFeatureExtractor.get_defaults()
        assert len(defaults) == 15

    def test_all_feature_names_present(self):
        defaults = BPAnalyticsFeatureExtractor.get_defaults()
        for name in BPAnalyticsFeatureExtractor.FEATURE_NAMES:
            assert name in defaults

    def test_default_values_are_neutral(self):
        defaults = BPAnalyticsFeatureExtractor.get_defaults()
        assert defaults["bp_analytics_bet_rating"] == 3.0
        assert defaults["bp_analytics_probability"] == 0.5
        assert defaults["bp_analytics_opp_rank"] == 15.0
        assert defaults["bp_analytics_hit_rate_L5"] == 0.5
        assert defaults["dvp_stat_rank"] == 15.0

    def test_all_values_are_floats(self):
        defaults = BPAnalyticsFeatureExtractor.get_defaults()
        for k, v in defaults.items():
            assert isinstance(v, float), f"{k} is {type(v)}"


# ─────────────────────────────────────────────────────────────────
# STAT_TO_DVP mapping
# ─────────────────────────────────────────────────────────────────


class TestStatToDvp:
    def test_points(self):
        assert BPAnalyticsFeatureExtractor.STAT_TO_DVP["POINTS"] == "points"

    def test_rebounds(self):
        assert BPAnalyticsFeatureExtractor.STAT_TO_DVP["REBOUNDS"] == "rebounds"

    def test_assists(self):
        assert BPAnalyticsFeatureExtractor.STAT_TO_DVP["ASSISTS"] == "assists"

    def test_threes(self):
        assert BPAnalyticsFeatureExtractor.STAT_TO_DVP["THREES"] == "three_points_made"

    def test_unknown_stat_not_in_map(self):
        assert "STEALS" not in BPAnalyticsFeatureExtractor.STAT_TO_DVP


# ─────────────────────────────────────────────────────────────────
# extract — full pipeline with mocked DB
# ─────────────────────────────────────────────────────────────────


class TestExtract:
    @pytest.fixture
    def mock_conn(self):
        return MagicMock()

    @pytest.fixture
    def extractor(self, mock_conn):
        return BPAnalyticsFeatureExtractor(mock_conn)

    def test_returns_defaults_when_no_data(self, extractor):
        with patch.object(extractor, "_safe_query", return_value=None):
            result = extractor.extract("Unknown Player", "2025-01-15", "POINTS")
        defaults = BPAnalyticsFeatureExtractor.get_defaults()
        assert result == defaults

    def test_returns_15_features(self, extractor):
        with patch.object(extractor, "_safe_query", return_value=None):
            result = extractor.extract("Test Player", "2025-01-15", "POINTS")
        assert len(result) == 15

    def test_exact_match_returns_features(self, extractor):
        bp_df = pd.DataFrame(
            [
                {
                    "bp_projection": 28.5,
                    "bp_projection_diff": 2.1,
                    "bp_probability": 0.72,
                    "bp_expected_value": 0.15,
                    "bp_bet_rating": 5,
                    "bp_recommended_side": "over",
                    "bp_opposition_rank": 3,
                    "bp_opposition_value": 118.5,
                    "bp_hit_rate_l5": 0.8,
                    "bp_hit_rate_l10": 0.7,
                    "bp_hit_rate_l15": 0.6,
                    "bp_hit_rate_season": 0.65,
                    "bp_consensus_line": 26.5,
                }
            ]
        )
        with patch.object(extractor, "_safe_query", return_value=bp_df):
            result = extractor.extract("Nikola Jokic", "2025-01-15", "POINTS", line=26.5)

        assert result["bp_analytics_projection_diff"] == 2.1
        assert result["bp_analytics_ev"] == 0.15
        assert result["bp_analytics_bet_rating"] == 5.0
        assert result["bp_analytics_probability"] == 0.72
        assert result["bp_analytics_opp_rank"] == 3.0
        assert result["bp_analytics_opp_value"] == 118.5
        assert result["bp_analytics_hit_rate_L5"] == 0.8
        assert result["bp_analytics_hit_rate_L10"] == 0.7
        assert result["bp_analytics_hit_rate_L15"] == 0.6
        assert result["bp_analytics_hit_rate_season"] == 0.65
        assert result["bp_analytics_recommended_over"] == 1.0
        assert result["bp_analytics_proj_vs_consensus"] == pytest.approx(2.0, abs=0.01)

    def test_hit_rate_trend_computed(self, extractor):
        bp_df = pd.DataFrame(
            [
                {
                    "bp_projection": 20.0,
                    "bp_projection_diff": 0.0,
                    "bp_probability": 0.5,
                    "bp_expected_value": 0.0,
                    "bp_bet_rating": 3,
                    "bp_recommended_side": "under",
                    "bp_opposition_rank": 15,
                    "bp_opposition_value": 110.0,
                    "bp_hit_rate_l5": 0.8,
                    "bp_hit_rate_l10": 0.6,
                    "bp_hit_rate_l15": 0.4,
                    "bp_hit_rate_season": 0.5,
                    "bp_consensus_line": 20.0,
                }
            ]
        )
        with patch.object(extractor, "_safe_query", return_value=bp_df):
            result = extractor.extract("Test", "2025-01-15", "POINTS")

        # trend = L5 - L15 = 0.8 - 0.4 = 0.4
        assert result["bp_analytics_hit_rate_trend"] == pytest.approx(0.4)

    def test_under_recommendation(self, extractor):
        bp_df = pd.DataFrame(
            [
                {
                    "bp_projection": 18.0,
                    "bp_projection_diff": -2.0,
                    "bp_probability": 0.35,
                    "bp_expected_value": -0.1,
                    "bp_bet_rating": 1,
                    "bp_recommended_side": "under",
                    "bp_opposition_rank": 28,
                    "bp_opposition_value": 105.0,
                    "bp_hit_rate_l5": 0.2,
                    "bp_hit_rate_l10": 0.3,
                    "bp_hit_rate_l15": 0.35,
                    "bp_hit_rate_season": 0.4,
                    "bp_consensus_line": 20.0,
                }
            ]
        )
        with patch.object(extractor, "_safe_query", return_value=bp_df):
            result = extractor.extract("Test", "2025-01-15", "REBOUNDS")

        assert result["bp_analytics_recommended_over"] == 0.0
        assert result["bp_analytics_bet_rating"] == 1.0

    def test_no_opponent_skips_dvp(self, extractor):
        defaults = BPAnalyticsFeatureExtractor.get_defaults()
        with patch.object(extractor, "_safe_query", return_value=None):
            result = extractor.extract("Test", "2025-01-15", "POINTS", opponent_team=None)
        assert result["dvp_stat_allowed"] == defaults["dvp_stat_allowed"]
        assert result["dvp_stat_rank"] == defaults["dvp_stat_rank"]

    def test_unknown_stat_type_skips_dvp(self, extractor):
        defaults = BPAnalyticsFeatureExtractor.get_defaults()
        with patch.object(extractor, "_safe_query", return_value=None):
            result = extractor.extract("Test", "2025-01-15", "STEALS", opponent_team="BOS")
        assert result["dvp_stat_allowed"] == defaults["dvp_stat_allowed"]


# ─────────────────────────────────────────────────────────────────
# _fetch_bp_analytics — exact vs fuzzy match
# ─────────────────────────────────────────────────────────────────


class TestFetchBpAnalytics:
    @pytest.fixture
    def extractor(self):
        return BPAnalyticsFeatureExtractor(MagicMock())

    def test_exact_match_returns_first(self, extractor):
        exact_df = pd.DataFrame([{"bp_projection": 25.0}])
        extractor._safe_query = MagicMock(return_value=exact_df)

        result = extractor._fetch_bp_analytics("Nikola Jokic", "2025-01-15", "POINTS")
        assert result is not None
        # Only called once (exact match succeeded)
        assert extractor._safe_query.call_count == 1

    def test_falls_back_to_fuzzy_on_no_exact(self, extractor):
        fuzzy_df = pd.DataFrame([{"bp_projection": 25.0}])
        extractor._safe_query = MagicMock(side_effect=[None, fuzzy_df])

        result = extractor._fetch_bp_analytics("PJ Washington Jr", "2025-01-22", "REBOUNDS")
        assert result is not None
        assert extractor._safe_query.call_count == 2

    def test_returns_none_when_both_fail(self, extractor):
        extractor._safe_query = MagicMock(return_value=None)
        result = extractor._fetch_bp_analytics("Ghost Player", "2025-01-15", "POINTS")
        assert result is None


# ─────────────────────────────────────────────────────────────────
# _compute_bp_features — edge cases
# ─────────────────────────────────────────────────────────────────


class TestComputeBpFeatures:
    @pytest.fixture
    def extractor(self):
        return BPAnalyticsFeatureExtractor(MagicMock())

    def test_none_values_get_defaults(self, extractor):
        row = pd.Series(
            {
                "bp_projection": None,
                "bp_projection_diff": None,
                "bp_probability": None,
                "bp_expected_value": None,
                "bp_bet_rating": None,
                "bp_recommended_side": None,
                "bp_opposition_rank": None,
                "bp_opposition_value": None,
                "bp_hit_rate_l5": None,
                "bp_hit_rate_l10": None,
                "bp_hit_rate_l15": None,
                "bp_hit_rate_season": None,
                "bp_consensus_line": None,
            }
        )
        features = BPAnalyticsFeatureExtractor.get_defaults()
        extractor._compute_bp_features(row, features)

        assert features["bp_analytics_bet_rating"] == 3.0
        assert features["bp_analytics_probability"] == 0.5

    def test_proj_vs_consensus_zero_when_no_projection(self, extractor):
        row = pd.Series(
            {
                "bp_projection": 0,
                "bp_projection_diff": 0,
                "bp_probability": 0.5,
                "bp_expected_value": 0,
                "bp_bet_rating": 3,
                "bp_recommended_side": "over",
                "bp_opposition_rank": 15,
                "bp_opposition_value": 110,
                "bp_hit_rate_l5": 0.5,
                "bp_hit_rate_l10": 0.5,
                "bp_hit_rate_l15": 0.5,
                "bp_hit_rate_season": 0.5,
                "bp_consensus_line": 0,
            }
        )
        features = BPAnalyticsFeatureExtractor.get_defaults()
        extractor._compute_bp_features(row, features)
        assert features["bp_analytics_proj_vs_consensus"] == 0.0


# ─────────────────────────────────────────────────────────────────
# _compute_dvp_features
# ─────────────────────────────────────────────────────────────────


class TestComputeDvpFeatures:
    @pytest.fixture
    def extractor(self):
        return BPAnalyticsFeatureExtractor(MagicMock())

    def test_dvp_value_and_rank(self, extractor):
        val_df = pd.DataFrame([{"value": 24.5}])
        rank_df = pd.DataFrame(
            [{"team": t, "value": 25 - i} for i, t in enumerate(["MIA", "WAS", "CHA"])]
        )
        extractor._safe_query = MagicMock(side_effect=[val_df, rank_df])

        features = BPAnalyticsFeatureExtractor.get_defaults()
        extractor._compute_dvp_features("MIA", "POINTS", "2025-01-15", None, features)

        assert features["dvp_stat_allowed"] == 24.5
        assert features["dvp_stat_rank"] == 1.0  # MIA is first in desc order

    def test_dvp_caching(self, extractor):
        val_df = pd.DataFrame([{"value": 22.0}])
        rank_df = pd.DataFrame([{"team": "BOS", "value": 22.0}])
        extractor._safe_query = MagicMock(side_effect=[val_df, rank_df])

        features = BPAnalyticsFeatureExtractor.get_defaults()
        extractor._compute_dvp_features("BOS", "POINTS", "2025-01-15", None, features)
        assert features["dvp_stat_allowed"] == 22.0

        # Second call should use cache — no new queries
        features2 = BPAnalyticsFeatureExtractor.get_defaults()
        extractor._safe_query = MagicMock()  # Reset mock
        extractor._compute_dvp_features("BOS", "POINTS", "2025-01-15", None, features2)
        assert features2["dvp_stat_allowed"] == 22.0
        extractor._safe_query.assert_not_called()

    def test_dvp_no_data_keeps_defaults(self, extractor):
        defaults = BPAnalyticsFeatureExtractor.get_defaults()
        extractor._safe_query = MagicMock(return_value=None)
        features = BPAnalyticsFeatureExtractor.get_defaults()
        extractor._compute_dvp_features("XXX", "POINTS", "2025-01-15", None, features)
        assert features["dvp_stat_allowed"] == defaults["dvp_stat_allowed"]
        assert features["dvp_stat_rank"] == defaults["dvp_stat_rank"]

    def test_season_detection_october(self, extractor):
        """October = start of new season, season = year."""
        extractor._safe_query = MagicMock(return_value=None)
        features = BPAnalyticsFeatureExtractor.get_defaults()
        extractor._compute_dvp_features("BOS", "POINTS", "2025-10-22", None, features)
        # Should query season=2025
        call_args = extractor._safe_query.call_args_list[0][0]
        assert call_args[1][0] == 2025  # season param

    def test_season_detection_march(self, extractor):
        """March = mid-season, season = year - 1."""
        extractor._safe_query = MagicMock(return_value=None)
        features = BPAnalyticsFeatureExtractor.get_defaults()
        extractor._compute_dvp_features("BOS", "POINTS", "2025-03-15", None, features)
        call_args = extractor._safe_query.call_args_list[0][0]
        assert call_args[1][0] == 2024  # season param

    def test_dvp_uses_player_position_when_provided(self, extractor):
        extractor._safe_query = MagicMock(return_value=None)
        features = BPAnalyticsFeatureExtractor.get_defaults()
        extractor._compute_dvp_features("BOS", "POINTS", "2025-01-15", "PG", features)
        call_args = extractor._safe_query.call_args_list[0][0]
        assert call_args[1][2] == "PG"  # position param

    def test_dvp_defaults_to_all_position(self, extractor):
        extractor._safe_query = MagicMock(return_value=None)
        features = BPAnalyticsFeatureExtractor.get_defaults()
        extractor._compute_dvp_features("BOS", "POINTS", "2025-01-15", None, features)
        call_args = extractor._safe_query.call_args_list[0][0]
        assert call_args[1][2] == "ALL"

    def test_dvp_team_not_in_rank_list(self, extractor):
        val_df = pd.DataFrame([{"value": 20.0}])
        rank_df = pd.DataFrame([{"team": "OTHER", "value": 20.0}])
        extractor._safe_query = MagicMock(side_effect=[val_df, rank_df])

        features = BPAnalyticsFeatureExtractor.get_defaults()
        extractor._compute_dvp_features("XXX", "REBOUNDS", "2025-01-15", None, features)
        assert features["dvp_stat_rank"] == 15.0  # Fallback


# ─────────────────────────────────────────────────────────────────
# FEATURE_NAMES tuple
# ─────────────────────────────────────────────────────────────────


class TestFeatureNames:
    def test_feature_names_count(self):
        assert len(BPAnalyticsFeatureExtractor.FEATURE_NAMES) == 15

    def test_feature_names_match_defaults(self):
        defaults = BPAnalyticsFeatureExtractor.get_defaults()
        for name in BPAnalyticsFeatureExtractor.FEATURE_NAMES:
            assert name in defaults, f"{name} missing from defaults"

    def test_no_duplicate_feature_names(self):
        names = BPAnalyticsFeatureExtractor.FEATURE_NAMES
        assert len(names) == len(set(names))


# ─────────────────────────────────────────────────────────────────
# Coverage: missing lines/branches
# ─────────────────────────────────────────────────────────────────


class TestCoverageMissingPaths:
    @pytest.fixture
    def extractor(self):
        return BPAnalyticsFeatureExtractor(MagicMock())

    def test_stat_type_not_in_dvp_map_skips_dvp(self, extractor):
        """Line 137 (and branch): stat_type 'BLOCKS' not in STAT_TO_DVP returns early."""
        defaults = BPAnalyticsFeatureExtractor.get_defaults()
        with patch.object(extractor, "_safe_query", return_value=None):
            result = extractor.extract(
                "Test Player",
                "2025-01-15",
                "BLOCKS",
                opponent_team="BOS",
            )
        # DVP features should remain at defaults since BLOCKS is not in STAT_TO_DVP
        assert result["dvp_stat_allowed"] == defaults["dvp_stat_allowed"]
        assert result["dvp_stat_rank"] == defaults["dvp_stat_rank"]

    def test_dvp_stat_is_none_returns_early(self, extractor):
        """Line 224: dvp_stat is None when stat_type not in STAT_TO_DVP."""
        features = BPAnalyticsFeatureExtractor.get_defaults()
        extractor._compute_dvp_features("BOS", "STEALS", "2025-01-15", None, features)
        # Should return early without querying
        assert features["dvp_stat_allowed"] == 0.0
        assert features["dvp_stat_rank"] == 15.0

    def test_dvp_rank_df_empty(self, extractor):
        """Branch 257->265: val_df has data but rank_df is empty/None."""
        val_df = pd.DataFrame([{"value": 22.5}])
        # First call returns val_df, second call returns None (empty rank df)
        extractor._safe_query = MagicMock(side_effect=[val_df, None])

        features = BPAnalyticsFeatureExtractor.get_defaults()
        extractor._compute_dvp_features("LAL", "POINTS", "2025-01-15", None, features)

        # Value should be set from val_df
        assert features["dvp_stat_allowed"] == 22.5
        # Rank should remain default since rank_df is empty
        assert features["dvp_stat_rank"] == 15.0
        # Should still cache the result
        assert len(extractor._dvp_cache) == 1

    def test_dvp_rank_df_empty_dataframe(self, extractor):
        """Branch 257->265: rank_df is empty DataFrame (len == 0)."""
        val_df = pd.DataFrame([{"value": 18.0}])
        # _safe_query returns None for empty DataFrames (see base.py line 99)
        extractor._safe_query = MagicMock(side_effect=[val_df, None])

        features = BPAnalyticsFeatureExtractor.get_defaults()
        extractor._compute_dvp_features("CHI", "REBOUNDS", "2025-02-10", "C", features)

        assert features["dvp_stat_allowed"] == 18.0
        assert features["dvp_stat_rank"] == 15.0

    def test_extract_with_opponent_and_valid_stat_calls_dvp(self, extractor):
        """Line 137: extract() with opponent_team + stat_type in STAT_TO_DVP hits DVP path."""
        bp_df = pd.DataFrame(
            [
                {
                    "bp_projection": 25.0,
                    "bp_projection_diff": 1.0,
                    "bp_probability": 0.6,
                    "bp_expected_value": 0.05,
                    "bp_bet_rating": 4,
                    "bp_recommended_side": "over",
                    "bp_opposition_rank": 5,
                    "bp_opposition_value": 112.0,
                    "bp_hit_rate_l5": 0.7,
                    "bp_hit_rate_l10": 0.6,
                    "bp_hit_rate_l15": 0.55,
                    "bp_hit_rate_season": 0.58,
                    "bp_consensus_line": 24.0,
                }
            ]
        )
        # First call from _fetch_bp_analytics (exact match returns bp_df),
        # then two calls from _compute_dvp_features (val query + rank query)
        extractor._safe_query = MagicMock(side_effect=[bp_df, None, None])

        result = extractor.extract(
            "Nikola Jokic",
            "2025-01-15",
            "POINTS",
            line=24.0,
            opponent_team="BOS",
            player_position="C",
        )
        assert result["bp_analytics_projection_diff"] == 1.0
        # DVP defaults since val query returned None
        assert result["dvp_stat_rank"] == 15.0
