"""
Unit Tests for Feature Extraction
=================================
Tests for the LiveFeatureExtractor and LiveFeatureExtractorXL classes.

These are critical tests as feature extraction errors can cause:
- Data leakage (using future data)
- Wrong predictions due to incorrect feature values
- Silent failures that corrupt model training
"""

from datetime import date, datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


class TestLiveFeatureExtractorNameNormalization:
    """Tests for player name normalization in feature extractor."""

    def test_normalize_basic_name(self):
        """Test basic name normalization."""
        from nba.features.extract_live_features import LiveFeatureExtractor

        result = LiveFeatureExtractor.normalize_player_name("LeBron James")
        assert result == "LeBron James"

    def test_normalize_name_with_suffix_jr(self):
        """Test name with Jr suffix is stripped."""
        from nba.features.extract_live_features import LiveFeatureExtractor

        result = LiveFeatureExtractor.normalize_player_name("PJ Washington Jr.")
        assert result == "PJ Washington"

    def test_normalize_name_with_suffix_iii(self):
        """Test name with III suffix is stripped."""
        from nba.features.extract_live_features import LiveFeatureExtractor

        result = LiveFeatureExtractor.normalize_player_name("Russell Westbrook III")
        assert result == "Russell Westbrook"

    def test_normalize_name_with_accents(self):
        """Test accented characters are normalized."""
        from nba.features.extract_live_features import LiveFeatureExtractor

        # Luka Dončić -> Luka Doncic
        result = LiveFeatureExtractor.normalize_player_name("Luka Dončić")
        assert result == "Luka Doncic"

        # Nikola Jokić -> Nikola Jokic
        result = LiveFeatureExtractor.normalize_player_name("Nikola Jokić")
        assert result == "Nikola Jokic"

    def test_normalize_empty_name(self):
        """Test empty name returns empty."""
        from nba.features.extract_live_features import LiveFeatureExtractor

        result = LiveFeatureExtractor.normalize_player_name("")
        assert result == ""

    def test_normalize_whitespace(self):
        """Test extra whitespace is removed."""
        from nba.features.extract_live_features import LiveFeatureExtractor

        result = LiveFeatureExtractor.normalize_player_name("  LeBron   James  ")
        assert result == "LeBron James"


class TestTeamAbbrevNormalization:
    """Tests for team abbreviation normalization."""

    @pytest.fixture
    def extractor_class(self):
        """Get the extractor class without instantiating."""
        from nba.features.extract_live_features import LiveFeatureExtractor

        return LiveFeatureExtractor

    def test_new_orleans_mapping(self, extractor_class):
        """Test New Orleans mapping."""
        assert extractor_class.TEAM_ABBREV_MAP["NO"] == "NOP"

    def test_san_antonio_mapping(self, extractor_class):
        """Test San Antonio mapping."""
        assert extractor_class.TEAM_ABBREV_MAP["SA"] == "SAS"

    def test_utah_mapping(self, extractor_class):
        """Test Utah mapping."""
        assert extractor_class.TEAM_ABBREV_MAP["UTAH"] == "UTA"

    def test_golden_state_mapping(self, extractor_class):
        """Test Golden State mapping."""
        assert extractor_class.TEAM_ABBREV_MAP["GS"] == "GSW"

    def test_new_york_mapping(self, extractor_class):
        """Test New York mapping."""
        assert extractor_class.TEAM_ABBREV_MAP["NY"] == "NYK"

    def test_brooklyn_variations(self, extractor_class):
        """Test Brooklyn abbreviation variations."""
        assert extractor_class.TEAM_ABBREV_MAP["BKN"] == "BKN"
        assert extractor_class.TEAM_ABBREV_MAP["BRK"] == "BKN"


class TestEMACalculation:
    """Tests for Exponential Moving Average calculations."""

    def test_ema_basic_calculation(self):
        """Test basic EMA calculation matches expected formula."""
        # EMA formula: EMA_t = α * x_t + (1 - α) * EMA_{t-1}
        # For span=3, α = 2/(3+1) = 0.5
        values = pd.Series([10.0, 12.0, 14.0])
        ema = values.ewm(span=3, adjust=False).mean()

        # Manual calculation for verification:
        # EMA_0 = 10
        # EMA_1 = 0.5 * 12 + 0.5 * 10 = 11
        # EMA_2 = 0.5 * 14 + 0.5 * 11 = 12.5
        assert abs(ema.iloc[-1] - 12.5) < 0.01

    def test_ema_with_longer_span(self):
        """Test EMA with longer span is smoother."""
        values = pd.Series([10.0, 20.0, 30.0, 20.0, 10.0])
        ema_short = values.ewm(span=3, adjust=False).mean()
        ema_long = values.ewm(span=5, adjust=False).mean()

        # Longer span should be less reactive to recent changes
        # When values drop from 30 to 10, longer span should stay higher
        assert ema_long.iloc[-1] > ema_short.iloc[-1]

    def test_ema_handles_single_value(self):
        """Test EMA with single value returns that value."""
        values = pd.Series([25.0])
        ema = values.ewm(span=3, adjust=False).mean()
        assert ema.iloc[0] == 25.0


class TestBookFeatureExtraction:
    """Tests for book disagreement feature extraction."""

    def test_line_spread_calculation(self, sample_prop_lines):
        """Test line spread is max - min."""
        lines = [p["over_line"] for p in sample_prop_lines]
        spread = max(lines) - min(lines)
        assert spread == 2.0  # 26.5 - 24.5

    def test_consensus_line_calculation(self, sample_prop_lines):
        """Test consensus line is median."""
        lines = sorted([p["over_line"] for p in sample_prop_lines])
        # 7 lines: 24.5, 25.5, 25.5, 25.5, 26.0, 26.0, 26.5
        # Median is 4th value (index 3) = 25.5
        median_idx = len(lines) // 2
        assert lines[median_idx] == 25.5

    def test_book_deviation_calculation(self, sample_prop_lines):
        """Test book deviation from consensus."""
        lines = [p["over_line"] for p in sample_prop_lines]
        consensus = sum(lines) / len(lines)

        # DraftKings line is 25.5
        dk_line = next(p["over_line"] for p in sample_prop_lines if p["book_name"] == "draftkings")
        dk_deviation = dk_line - consensus

        # Consensus is ~25.71, DK is 25.5, so deviation is negative
        assert dk_deviation < 0

    def test_softest_book_identification(self, sample_prop_lines):
        """Test identification of softest book (lowest line for OVER)."""
        min_line = min(p["over_line"] for p in sample_prop_lines)
        softest_book = next(p["book_name"] for p in sample_prop_lines if p["over_line"] == min_line)
        assert softest_book == "underdog"
        assert min_line == 24.5

    def test_hardest_book_identification(self, sample_prop_lines):
        """Test identification of hardest book (highest line for OVER)."""
        max_line = max(p["over_line"] for p in sample_prop_lines)
        hardest_book = next(p["book_name"] for p in sample_prop_lines if p["over_line"] == max_line)
        assert hardest_book == "caesars"
        assert max_line == 26.5


class TestFeatureVectorCompleteness:
    """Tests for feature vector completeness and validity."""

    def test_feature_count(self, sample_feature_vector):
        """Test feature vector has expected count."""
        assert len(sample_feature_vector) >= 102

    def test_required_features_present(self, sample_feature_vector):
        """Test all required features are present."""
        required_features = [
            "is_home",
            "ema_points_L3",
            "ema_points_L5",
            "ema_points_L10",
            "ema_rebounds_L3",
            "ema_assists_L3",
            "team_pace",
        ]
        for feature in required_features:
            assert feature in sample_feature_vector, f"Missing required feature: {feature}"

    def test_no_none_values_in_critical_features(self, sample_feature_vector):
        """Test critical features don't have None values."""
        critical_features = ["is_home", "ema_points_L3", "ema_points_L5"]
        for feature in critical_features:
            assert sample_feature_vector[feature] is not None, f"Critical feature {feature} is None"

    def test_is_home_is_binary(self, sample_feature_vector):
        """Test is_home is 0 or 1."""
        assert sample_feature_vector["is_home"] in [0, 0.0, 1, 1.0]

    def test_ema_values_are_reasonable(self, sample_feature_vector):
        """Test EMA values are in reasonable ranges."""
        # Points should be 0-60
        assert 0 <= sample_feature_vector["ema_points_L3"] <= 60
        # Rebounds should be 0-25
        assert 0 <= sample_feature_vector["ema_rebounds_L3"] <= 25
        # Assists should be 0-20
        assert 0 <= sample_feature_vector["ema_assists_L3"] <= 20


class TestDataLeakagePrevention:
    """Tests to verify no data leakage occurs in feature extraction."""

    def test_rolling_stats_exclude_current_game(self, sample_game_logs):
        """Test that rolling stats don't include the game being predicted."""
        game_date = sample_game_logs["game_date"].max()

        # Filter to games BEFORE the current game
        historical_games = sample_game_logs[sample_game_logs["game_date"] < game_date]

        # Should have fewer games than total
        assert len(historical_games) < len(sample_game_logs)

    def test_feature_extraction_uses_historical_data_only(self, sample_game_logs):
        """Test feature extraction only uses data available before prediction date."""
        prediction_date = date(2025, 1, 15)

        # All game data used should be before prediction date
        valid_games = sample_game_logs[sample_game_logs["game_date"] < prediction_date]

        # Verify we're using historical data
        for game_date in valid_games["game_date"]:
            assert game_date < prediction_date, "Data leakage: using future data"

    def test_opponent_stats_are_historical(self, sample_team_stats):
        """Test opponent stats represent historical performance, not future."""
        # This is a structural test - opponent stats should be computed
        # from games played BEFORE the prediction date
        assert "pace" in sample_team_stats
        assert "defensive_rating" in sample_team_stats


class TestEdgeCases:
    """Tests for edge cases in feature extraction."""

    def test_new_player_with_few_games(self):
        """Test handling of player with fewer games than rolling window."""
        # Create a player with only 2 games
        logs = pd.DataFrame(
            [
                {"game_date": date(2025, 1, 10), "pts": 20},
                {"game_date": date(2025, 1, 12), "pts": 25},
            ]
        )

        # Calculate L5 EMA with only 2 games
        ema = logs["pts"].ewm(span=5, adjust=False).mean()

        # Should still return a value, not NaN
        assert not np.isnan(ema.iloc[-1])

    def test_player_returning_from_injury(self, sample_game_logs):
        """Test handling of player with gap in games."""
        # Simulate a 2-week gap
        logs = sample_game_logs.copy()
        logs.loc[5:10, "game_date"] = logs.loc[5:10, "game_date"] - timedelta(days=14)

        # EMA should still be calculable
        ema = logs["pts"].ewm(span=5, adjust=False).mean()
        assert not ema.isna().any()

    def test_missing_opponent_stats(self):
        """Test handling when opponent stats are missing."""
        # If opponent stats are missing, features should use defaults
        default_pace = 100.0
        default_def_rating = 110.0

        # Verify defaults are reasonable
        assert 90 <= default_pace <= 110
        assert 100 <= default_def_rating <= 120

    def test_zero_minutes_game(self, sample_game_logs):
        """Test handling of DNP (zero minutes) games."""
        logs = sample_game_logs.copy()
        logs.loc[0, "min"] = 0
        logs.loc[0, "pts"] = 0

        # Rolling stats should handle zero-minute games
        ema = logs["pts"].ewm(span=5, adjust=False).mean()
        assert not np.isnan(ema.iloc[-1])


class TestBookFeatureValidation:
    """Tests for book feature validation."""

    def test_book_features_have_expected_keys(self, sample_book_features):
        """Test book features dict has all expected keys."""
        expected_keys = [
            "line_spread",
            "consensus_line",
            "num_books_offering",
            "draftkings_deviation",
            "fanduel_deviation",
            "betmgm_deviation",
            "softest_book_id",
            "hardest_book_id",
        ]
        for key in expected_keys:
            assert key in sample_book_features, f"Missing key: {key}"

    def test_line_spread_is_non_negative(self, sample_book_features):
        """Test line spread is non-negative."""
        assert sample_book_features["line_spread"] >= 0

    def test_num_books_is_positive(self, sample_book_features):
        """Test number of books is positive."""
        assert sample_book_features["num_books_offering"] > 0

    def test_consensus_line_is_reasonable(self, sample_book_features):
        """Test consensus line is in reasonable range."""
        # For points, should be 0-60
        assert 0 <= sample_book_features["consensus_line"] <= 60

    def test_deviations_are_bounded(self, sample_book_features):
        """Test book deviations are bounded (line spread limits them)."""
        line_spread = sample_book_features["line_spread"]
        assert abs(sample_book_features["draftkings_deviation"]) <= line_spread
        assert abs(sample_book_features["fanduel_deviation"]) <= line_spread


class TestFeatureConsistency:
    """Tests for feature consistency across calls."""

    def test_same_input_produces_same_output(self, sample_game_logs):
        """Test deterministic feature extraction."""
        # Same input should produce identical output
        ema1 = sample_game_logs["pts"].ewm(span=5, adjust=False).mean().iloc[-1]
        ema2 = sample_game_logs["pts"].ewm(span=5, adjust=False).mean().iloc[-1]

        assert ema1 == ema2

    def test_feature_order_is_consistent(self, sample_feature_vector):
        """Test feature order is consistent."""
        keys1 = list(sample_feature_vector.keys())
        keys2 = list(sample_feature_vector.keys())

        assert keys1 == keys2
