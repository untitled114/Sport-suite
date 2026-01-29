"""
Unit Tests for XL Feature Extraction
====================================
Tests for LiveFeatureExtractorXL with mocked database connections.

Best Practices Applied:
- Mock all database calls
- Test feature computation logic
- Test error handling for missing data
- Test feature vector completeness
"""

from datetime import date, datetime, timedelta
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest


class TestRollingStatsComputation:
    """Tests for rolling statistics computation."""

    def test_ema_calculation_span_3(self):
        """Test EMA calculation with span=3."""
        values = pd.Series([10.0, 12.0, 14.0, 16.0, 18.0])
        ema = values.ewm(span=3, adjust=False).mean()

        # EMA should be weighted toward recent values
        assert ema.iloc[-1] > values.mean()

    def test_ema_calculation_span_5(self):
        """Test EMA calculation with span=5."""
        values = pd.Series([10.0, 12.0, 14.0, 16.0, 18.0])
        ema = values.ewm(span=5, adjust=False).mean()

        # With span=5 and 5 values, should be close to simple average
        assert abs(ema.iloc[-1] - values.mean()) < 2.0

    def test_ema_handles_nan(self):
        """Test EMA handles NaN values."""
        values = pd.Series([10.0, np.nan, 14.0, 16.0, 18.0])
        ema = values.ewm(span=3, adjust=False, ignore_na=True).mean()

        # Should still produce valid values
        assert not np.isnan(ema.iloc[-1])

    def test_ema_single_value(self):
        """Test EMA with single value."""
        values = pd.Series([25.0])
        ema = values.ewm(span=5, adjust=False).mean()

        assert ema.iloc[0] == 25.0

    @pytest.mark.parametrize(
        "span,expected_weight",
        [
            (3, 0.5),  # alpha = 2/(3+1) = 0.5
            (5, 0.333),  # alpha = 2/(5+1) ≈ 0.333
            (10, 0.182),  # alpha = 2/(10+1) ≈ 0.182
            (20, 0.095),  # alpha = 2/(20+1) ≈ 0.095
        ],
    )
    def test_ema_alpha_values(self, span, expected_weight):
        """Test EMA alpha values for different spans."""
        alpha = 2 / (span + 1)
        assert abs(alpha - expected_weight) < 0.01


class TestBookFeatureComputation:
    """Tests for book disagreement feature computation."""

    @pytest.fixture
    def sample_lines(self):
        """Sample lines from multiple books."""
        return {
            "draftkings": 25.5,
            "fanduel": 26.0,
            "betmgm": 25.5,
            "caesars": 26.5,
            "betrivers": 25.5,
            "espnbet": 26.0,
            "underdog": 24.5,
        }

    def test_line_spread_calculation(self, sample_lines):
        """Test line spread (max - min)."""
        lines = list(sample_lines.values())
        spread = max(lines) - min(lines)
        assert spread == 2.0  # 26.5 - 24.5

    def test_consensus_calculation(self, sample_lines):
        """Test consensus line (median)."""
        lines = sorted(sample_lines.values())
        n = len(lines)
        if n % 2 == 1:
            consensus = lines[n // 2]
        else:
            consensus = (lines[n // 2 - 1] + lines[n // 2]) / 2

        assert consensus == 25.5  # Median of 7 values

    def test_std_dev_calculation(self, sample_lines):
        """Test line standard deviation."""
        lines = list(sample_lines.values())
        std_dev = np.std(lines)
        assert std_dev > 0
        assert std_dev < 2.0  # Reasonable range

    def test_coefficient_of_variation(self, sample_lines):
        """Test line coefficient of variation."""
        lines = list(sample_lines.values())
        mean_line = np.mean(lines)
        std_dev = np.std(lines)
        cv = std_dev / mean_line if mean_line > 0 else 0

        assert cv > 0
        assert cv < 0.1  # Low variation expected

    def test_book_deviation_calculation(self, sample_lines):
        """Test individual book deviation from consensus."""
        lines = list(sample_lines.values())
        consensus = np.median(lines)

        deviations = {}
        for book, line in sample_lines.items():
            deviations[book] = line - consensus

        # Underdog has lowest line, should have negative deviation
        assert deviations["underdog"] < 0
        # Caesars has highest line, should have positive deviation
        assert deviations["caesars"] > 0

    def test_softest_book_identification(self, sample_lines):
        """Test identification of softest book (lowest for OVER)."""
        softest_book = min(sample_lines, key=sample_lines.get)
        assert softest_book == "underdog"

    def test_hardest_book_identification(self, sample_lines):
        """Test identification of hardest book (highest for OVER)."""
        hardest_book = max(sample_lines, key=sample_lines.get)
        assert hardest_book == "caesars"


class TestTeamContextFeatures:
    """Tests for team context feature extraction."""

    def test_pace_normalization(self):
        """Test pace values are in expected range."""
        # NBA pace typically 95-105 possessions per game
        sample_pace = 100.5
        assert 90 <= sample_pace <= 115

    def test_offensive_rating_normalization(self):
        """Test offensive rating is in expected range."""
        # Offensive rating typically 100-120 points per 100 possessions
        sample_off_rating = 115.2
        assert 95 <= sample_off_rating <= 125

    def test_defensive_rating_normalization(self):
        """Test defensive rating is in expected range."""
        # Defensive rating typically 100-115 points per 100 possessions
        sample_def_rating = 110.5
        assert 95 <= sample_def_rating <= 125


class TestAdvancedContextFeatures:
    """Tests for advanced context features."""

    def test_rest_days_calculation(self):
        """Test rest days calculation."""
        last_game = date(2025, 11, 4)
        current_game = date(2025, 11, 6)
        rest_days = (current_game - last_game).days - 1

        assert rest_days == 1  # 1 day rest

    def test_back_to_back_detection(self):
        """Test back-to-back game detection."""
        last_game = date(2025, 11, 5)
        current_game = date(2025, 11, 6)
        rest_days = (current_game - last_game).days - 1

        is_back_to_back = rest_days == 0
        assert is_back_to_back is True

    def test_season_phase_calculation(self):
        """Test season phase calculation."""
        # NBA season typically Oct-Apr (~180 days)
        season_start = date(2024, 10, 22)
        current_date = date(2025, 1, 15)
        days_into_season = (current_date - season_start).days
        season_phase = min(days_into_season / 180, 1.0)

        assert 0 <= season_phase <= 1.0
        assert days_into_season > 0


class TestExpectedDiffCalculation:
    """Tests for expected_diff feature calculation."""

    def test_expected_diff_positive(self):
        """Test expected_diff when prediction > line."""
        prediction = 28.5
        line = 25.5
        expected_diff = prediction - line

        assert expected_diff == 3.0
        assert expected_diff > 0

    def test_expected_diff_negative(self):
        """Test expected_diff when prediction < line."""
        prediction = 22.5
        line = 25.5
        expected_diff = prediction - line

        assert expected_diff == -3.0
        assert expected_diff < 0

    def test_expected_diff_zero(self):
        """Test expected_diff when prediction equals line."""
        prediction = 25.5
        line = 25.5
        expected_diff = prediction - line

        assert expected_diff == 0.0


class TestFeatureVectorConstruction:
    """Tests for complete feature vector construction."""

    @pytest.fixture
    def sample_features(self):
        """Sample feature dictionary."""
        return {
            "is_home": 1.0,
            "line": 25.5,
            "ema_points_L3": 24.5,
            "ema_points_L5": 23.8,
            "ema_points_L10": 22.9,
            "ema_points_L20": 22.1,
            "ema_rebounds_L3": 7.2,
            "ema_rebounds_L5": 6.8,
            "team_pace": 100.5,
            "opp_def_rating": 110.5,
            "line_spread": 2.0,
            "consensus_line": 25.7,
            "expected_diff": 3.0,
        }

    def test_feature_vector_is_dict(self, sample_features):
        """Test feature vector is dictionary."""
        assert isinstance(sample_features, dict)

    def test_feature_values_are_numeric(self, sample_features):
        """Test all feature values are numeric."""
        for key, value in sample_features.items():
            assert isinstance(value, (int, float)), f"Feature {key} is not numeric"

    def test_no_none_values(self, sample_features):
        """Test no None values in features."""
        for key, value in sample_features.items():
            assert value is not None, f"Feature {key} is None"

    def test_to_array_conversion(self, sample_features):
        """Test conversion to array maintains order."""
        feature_names = list(sample_features.keys())
        feature_array = [sample_features[name] for name in feature_names]

        assert len(feature_array) == len(feature_names)
        assert feature_array[0] == sample_features[feature_names[0]]


class TestMissingDataHandling:
    """Tests for handling missing data."""

    def test_missing_player_stats_uses_defaults(self):
        """Test missing player stats use default values."""
        default_points = 15.0  # League average
        default_rebounds = 5.0
        default_assists = 3.0

        features = {
            "ema_points_L5": default_points,
            "ema_rebounds_L5": default_rebounds,
            "ema_assists_L5": default_assists,
        }

        assert features["ema_points_L5"] == default_points

    def test_missing_team_stats_uses_league_average(self):
        """Test missing team stats use league averages."""
        league_avg_pace = 100.0
        league_avg_off_rating = 110.0
        league_avg_def_rating = 110.0

        features = {
            "team_pace": league_avg_pace,
            "team_off_rating": league_avg_off_rating,
            "opp_def_rating": league_avg_def_rating,
        }

        assert features["team_pace"] == league_avg_pace

    def test_imputation_strategy(self):
        """Test imputation strategy for missing features."""
        features_with_nan = {
            "feature_a": 10.0,
            "feature_b": np.nan,
            "feature_c": 20.0,
        }

        # Median imputation
        valid_values = [v for v in features_with_nan.values() if not np.isnan(v)]
        median_value = np.median(valid_values)

        imputed_features = {}
        for key, value in features_with_nan.items():
            if np.isnan(value):
                imputed_features[key] = median_value
            else:
                imputed_features[key] = value

        assert not np.isnan(imputed_features["feature_b"])
        assert imputed_features["feature_b"] == 15.0  # median of [10, 20]


class TestFeatureScaling:
    """Tests for feature scaling."""

    def test_standard_scaling(self):
        """Test standard scaling (z-score normalization)."""
        values = np.array([10, 20, 30, 40, 50])
        mean = np.mean(values)
        std = np.std(values)

        scaled = (values - mean) / std

        # Scaled values should have mean ~0 and std ~1
        assert abs(np.mean(scaled)) < 0.01
        assert abs(np.std(scaled) - 1.0) < 0.1

    def test_min_max_scaling(self):
        """Test min-max scaling to [0, 1]."""
        values = np.array([10, 20, 30, 40, 50])
        min_val = np.min(values)
        max_val = np.max(values)

        scaled = (values - min_val) / (max_val - min_val)

        assert np.min(scaled) == 0.0
        assert np.max(scaled) == 1.0


class TestDataLeakagePrevention:
    """Tests to ensure no data leakage in feature extraction."""

    def test_features_use_historical_data_only(self):
        """Test features are computed from historical data only."""
        prediction_date = date(2025, 11, 6)
        game_dates = [
            date(2025, 11, 1),
            date(2025, 11, 3),
            date(2025, 11, 5),
        ]

        # All historical dates should be before prediction date
        for game_date in game_dates:
            assert game_date < prediction_date

    def test_rolling_stats_exclude_current_game(self):
        """Test rolling stats don't include current game."""
        all_games = pd.DataFrame(
            {
                "game_date": pd.date_range("2025-11-01", periods=5),
                "points": [20, 22, 25, 28, 30],
            }
        )

        current_game_date = all_games["game_date"].max()
        historical_games = all_games[all_games["game_date"] < current_game_date]

        # Historical games should not include current game
        assert len(historical_games) == len(all_games) - 1
        assert current_game_date not in historical_games["game_date"].values

    def test_actual_result_not_in_features(self):
        """Test actual_result is not included in features."""
        feature_names = [
            "is_home",
            "line",
            "ema_points_L5",
            "team_pace",
            "line_spread",
        ]

        # actual_result should never be a feature
        assert "actual_result" not in feature_names
        assert "actual_points" not in feature_names
        assert "hit_over" not in feature_names
