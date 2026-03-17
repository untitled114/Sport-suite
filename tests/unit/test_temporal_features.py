"""
Unit tests for nba.features.extractors.temporal_features

Tests cover:
- NBA_MILESTONES data integrity
- _get_season() — season detection from game date
- TemporalFeatureExtractor.get_defaults()
- _compute_milestone_features — trade deadline, all-star, playoff push, season pct
- _compute_team_continuity — trade detection, team tenure, consecutive games
- Edge cases: unknown season, pre-deadline, post-deadline, playoff dates
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from nba.features.extractors.temporal_features import (
    NBA_MILESTONES,
    TemporalFeatureExtractor,
    _get_season,
    _parse_date,
)


class TestGetSeason:
    def test_october_is_current_year(self):
        assert _get_season("2025-10-22") == 2025

    def test_november_is_current_year(self):
        assert _get_season("2025-11-15") == 2025

    def test_january_is_previous_year(self):
        assert _get_season("2026-01-15") == 2025

    def test_march_is_previous_year(self):
        assert _get_season("2026-03-17") == 2025

    def test_april_is_previous_year(self):
        assert _get_season("2026-04-10") == 2025

    def test_june_is_previous_year(self):
        assert _get_season("2026-06-15") == 2025


class TestNbaMilestones:
    def test_has_three_seasons(self):
        assert len(NBA_MILESTONES) >= 3

    def test_each_season_has_required_keys(self):
        required = {
            "season_start",
            "trade_deadline",
            "allstar_start",
            "allstar_end",
            "regular_end",
            "playoffs_start",
        }
        for season, data in NBA_MILESTONES.items():
            assert required.issubset(data.keys()), f"Season {season} missing keys"

    def test_dates_are_chronological(self):
        for season, data in NBA_MILESTONES.items():
            dates = [
                _parse_date(data[k])
                for k in [
                    "season_start",
                    "trade_deadline",
                    "allstar_start",
                    "allstar_end",
                    "regular_end",
                    "playoffs_start",
                ]
            ]
            for i in range(len(dates) - 1):
                assert dates[i] < dates[i + 1], f"Season {season}: dates not chronological"


class TestGetDefaults:
    def test_returns_10_features(self):
        assert len(TemporalFeatureExtractor.get_defaults()) == 10

    def test_all_feature_names_present(self):
        defaults = TemporalFeatureExtractor.get_defaults()
        for name in TemporalFeatureExtractor.FEATURE_NAMES:
            assert name in defaults

    def test_no_duplicates(self):
        names = TemporalFeatureExtractor.FEATURE_NAMES
        assert len(names) == len(set(names))

    def test_defaults_are_neutral(self):
        d = TemporalFeatureExtractor.get_defaults()
        assert d["is_post_trade_deadline"] == 0.0
        assert d["is_new_team"] == 0.0
        assert d["is_regular_season"] == 1.0
        assert d["season_pct"] == 0.5


class TestMilestoneFeatures:
    @pytest.fixture
    def extractor(self):
        return TemporalFeatureExtractor(MagicMock())

    def test_pre_trade_deadline(self, extractor):
        features = TemporalFeatureExtractor.get_defaults()
        extractor._compute_milestone_features("2026-01-15", features)
        assert features["is_post_trade_deadline"] == 0.0
        assert features["days_since_trade_deadline"] == 0.0
        assert features["is_post_allstar"] == 0.0

    def test_post_trade_deadline(self, extractor):
        features = TemporalFeatureExtractor.get_defaults()
        extractor._compute_milestone_features("2026-02-10", features)
        assert features["is_post_trade_deadline"] == 1.0
        assert features["days_since_trade_deadline"] == 5.0  # Feb 10 - Feb 5

    def test_post_allstar(self, extractor):
        features = TemporalFeatureExtractor.get_defaults()
        extractor._compute_milestone_features("2026-02-20", features)
        assert features["is_post_trade_deadline"] == 1.0
        assert features["is_post_allstar"] == 1.0
        assert features["days_since_allstar"] == 5.0  # Feb 20 - Feb 15

    def test_playoff_push(self, extractor):
        features = TemporalFeatureExtractor.get_defaults()
        extractor._compute_milestone_features("2026-03-25", features)
        assert features["is_playoff_push"] == 1.0
        assert features["is_regular_season"] == 1.0

    def test_playoffs(self, extractor):
        features = TemporalFeatureExtractor.get_defaults()
        extractor._compute_milestone_features("2026-04-25", features)
        assert features["is_regular_season"] == 0.0

    def test_early_season_pct(self, extractor):
        features = TemporalFeatureExtractor.get_defaults()
        extractor._compute_milestone_features("2025-11-01", features)
        assert features["season_pct"] < 0.1

    def test_mid_season_pct(self, extractor):
        features = TemporalFeatureExtractor.get_defaults()
        extractor._compute_milestone_features("2026-01-15", features)
        assert 0.4 < features["season_pct"] < 0.6

    def test_late_season_pct(self, extractor):
        features = TemporalFeatureExtractor.get_defaults()
        extractor._compute_milestone_features("2026-04-05", features)
        assert features["season_pct"] > 0.9

    def test_days_since_capped_at_60(self, extractor):
        features = TemporalFeatureExtractor.get_defaults()
        extractor._compute_milestone_features("2026-04-10", features)
        assert features["days_since_trade_deadline"] == 60.0
        assert features["days_since_allstar"] == 54.0  # Apr 10 - Feb 15

    def test_unknown_season_uses_defaults(self, extractor):
        features = TemporalFeatureExtractor.get_defaults()
        extractor._compute_milestone_features("2028-01-15", features)
        # Should not crash, uses approximate dates
        assert features["season_pct"] > 0

    def test_not_playoff_push_early(self, extractor):
        features = TemporalFeatureExtractor.get_defaults()
        extractor._compute_milestone_features("2025-12-01", features)
        assert features["is_playoff_push"] == 0.0


class TestTeamContinuity:
    @pytest.fixture
    def extractor(self):
        return TemporalFeatureExtractor(MagicMock())

    def test_stable_team(self, extractor):
        df = pd.DataFrame(
            {
                "team_abbrev": ["BOS"] * 20,
                "game_date": pd.date_range("2026-01-01", periods=20, freq="2D"),
            }
        )
        with patch("pandas.read_sql_query", return_value=df):
            features = TemporalFeatureExtractor.get_defaults()
            extractor._compute_team_continuity("Jaylen Brown", "2026-02-15", features)
        assert features["player_games_with_team"] == 20.0
        assert features["is_new_team"] == 0.0
        assert features["team_tenure_games"] == 20.0

    def test_recently_traded(self, extractor):
        # Last 5 games on new team, before that different team
        teams = ["SAS"] * 5 + ["SAC"] * 10
        df = pd.DataFrame(
            {
                "team_abbrev": teams,
                "game_date": pd.date_range("2026-01-01", periods=15, freq="2D"),
            }
        )
        with patch("pandas.read_sql_query", return_value=df):
            features = TemporalFeatureExtractor.get_defaults()
            extractor._compute_team_continuity("De'Aaron Fox", "2026-02-15", features)
        assert features["player_games_with_team"] == 5.0
        assert features["is_new_team"] == 1.0
        assert features["team_tenure_games"] == 5.0

    def test_no_game_logs_keeps_defaults(self, extractor):
        with patch("pandas.read_sql_query", return_value=pd.DataFrame()):
            features = TemporalFeatureExtractor.get_defaults()
            extractor._compute_team_continuity("Ghost", "2026-01-15", features)
        assert features["player_games_with_team"] == 30.0  # Default
        assert features["is_new_team"] == 0.0

    def test_query_error_keeps_defaults(self, extractor):
        with patch("pandas.read_sql_query", side_effect=Exception("db error")):
            features = TemporalFeatureExtractor.get_defaults()
            extractor._compute_team_continuity("Test", "2026-01-15", features)
        assert features["player_games_with_team"] == 30.0

    def test_single_game(self, extractor):
        df = pd.DataFrame(
            {
                "team_abbrev": ["LAL"],
                "game_date": ["2026-01-10"],
            }
        )
        with patch("pandas.read_sql_query", return_value=df):
            features = TemporalFeatureExtractor.get_defaults()
            extractor._compute_team_continuity("Test", "2026-01-15", features)
        assert features["player_games_with_team"] == 1.0
        assert features["is_new_team"] == 0.0


class TestExtractFull:
    def test_returns_10_features(self):
        ext = TemporalFeatureExtractor(MagicMock())
        with patch("pandas.read_sql_query", return_value=pd.DataFrame()):
            result = ext.extract("Test Player", "2026-01-15", "POINTS")
        assert len(result) == 10

    def test_all_features_are_floats(self):
        ext = TemporalFeatureExtractor(MagicMock())
        with patch("pandas.read_sql_query", return_value=pd.DataFrame()):
            result = ext.extract("Test", "2026-01-15", "POINTS")
        for k, v in result.items():
            assert isinstance(v, float), f"{k} is {type(v)}"
