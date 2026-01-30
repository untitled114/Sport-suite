"""
Unit tests for nba/features/extractors/

Tests feature extractor initialization, defaults, and helper methods.
"""

from datetime import datetime

import pytest

from nba.features.extractors import (
    BaseFeatureExtractor,
    BookFeatureExtractor,
    CheatsheetExtractor,
    H2HFeatureExtractor,
    PropHistoryExtractor,
    TeamBettingExtractor,
    VegasContextExtractor,
)


class TestBaseFeatureExtractor:
    """Tests for BaseFeatureExtractor helper methods."""

    def test_normalize_date_string(self):
        """Test _normalize_date with string input."""

        # Create a concrete subclass for testing
        class ConcreteExtractor(BaseFeatureExtractor):
            FEATURE_NAMES = ("test_feature",)

            def extract(self, player_name, game_date, stat_type, **kwargs):
                return {}

            @classmethod
            def get_defaults(cls):
                return {"test_feature": 0.0}

        ext = ConcreteExtractor(conn=None)

        # Test various string formats
        assert ext._normalize_date("2024-01-15") == "2024-01-15"
        assert ext._normalize_date("2024-01-15 10:30:00") == "2024-01-15"

    def test_normalize_date_datetime(self):
        """Test _normalize_date with datetime input."""

        class ConcreteExtractor(BaseFeatureExtractor):
            FEATURE_NAMES = ("test_feature",)

            def extract(self, player_name, game_date, stat_type, **kwargs):
                return {}

            @classmethod
            def get_defaults(cls):
                return {"test_feature": 0.0}

        ext = ConcreteExtractor(conn=None)
        dt = datetime(2024, 1, 15, 10, 30, 0)
        assert ext._normalize_date(dt) == "2024-01-15"

    def test_safe_float_valid(self):
        """Test _safe_float with valid inputs."""

        class ConcreteExtractor(BaseFeatureExtractor):
            FEATURE_NAMES = ("test_feature",)

            def extract(self, player_name, game_date, stat_type, **kwargs):
                return {}

            @classmethod
            def get_defaults(cls):
                return {"test_feature": 0.0}

        ext = ConcreteExtractor(conn=None)

        assert ext._safe_float(1.5) == 1.5
        assert ext._safe_float(10) == 10.0
        assert ext._safe_float("3.14") == 3.14
        assert ext._safe_float("100") == 100.0

    def test_safe_float_invalid(self):
        """Test _safe_float with invalid inputs."""

        class ConcreteExtractor(BaseFeatureExtractor):
            FEATURE_NAMES = ("test_feature",)

            def extract(self, player_name, game_date, stat_type, **kwargs):
                return {}

            @classmethod
            def get_defaults(cls):
                return {"test_feature": 0.0}

        ext = ConcreteExtractor(conn=None)

        assert ext._safe_float(None) == 0.0
        assert ext._safe_float(None, default=5.0) == 5.0
        assert ext._safe_float("invalid") == 0.0
        assert ext._safe_float("not a number", default=-1.0) == -1.0

    def test_safe_float_nan(self):
        """Test _safe_float with NaN."""
        import math

        class ConcreteExtractor(BaseFeatureExtractor):
            FEATURE_NAMES = ("test_feature",)

            def extract(self, player_name, game_date, stat_type, **kwargs):
                return {}

            @classmethod
            def get_defaults(cls):
                return {"test_feature": 0.0}

        ext = ConcreteExtractor(conn=None)
        assert ext._safe_float(float("nan"), default=0.0) == 0.0

    def test_encode_book_name(self):
        """Test _encode_book_name for known books."""

        class ConcreteExtractor(BaseFeatureExtractor):
            FEATURE_NAMES = ("test_feature",)

            def extract(self, player_name, game_date, stat_type, **kwargs):
                return {}

            @classmethod
            def get_defaults(cls):
                return {"test_feature": 0.0}

        ext = ConcreteExtractor(conn=None)

        assert ext._encode_book_name("DraftKings") == 1.0
        assert ext._encode_book_name("fanduel") == 2.0
        assert ext._encode_book_name("BetMGM") == 3.0
        assert ext._encode_book_name("caesars") == 4.0
        assert ext._encode_book_name("bet365") == 5.0
        assert ext._encode_book_name("BetRivers") == 6.0
        assert ext._encode_book_name("ESPNBet") == 7.0
        assert ext._encode_book_name("fanatics") == 8.0
        assert ext._encode_book_name("underdog") == 9.0
        assert ext._encode_book_name("prizepicks") == 10.0

    def test_encode_book_name_unknown(self):
        """Test _encode_book_name with unknown book."""

        class ConcreteExtractor(BaseFeatureExtractor):
            FEATURE_NAMES = ("test_feature",)

            def extract(self, player_name, game_date, stat_type, **kwargs):
                return {}

            @classmethod
            def get_defaults(cls):
                return {"test_feature": 0.0}

        ext = ConcreteExtractor(conn=None)

        assert ext._encode_book_name("UnknownBook") == 0.0
        assert ext._encode_book_name("") == 0.0
        assert ext._encode_book_name(None) == 0.0

    def test_validate_features(self):
        """Test validate_features fills missing with defaults."""

        class ConcreteExtractor(BaseFeatureExtractor):
            FEATURE_NAMES = ("feature_a", "feature_b", "feature_c")

            def extract(self, player_name, game_date, stat_type, **kwargs):
                return {}

            @classmethod
            def get_defaults(cls):
                return {"feature_a": 1.0, "feature_b": 2.0, "feature_c": 3.0}

        ext = ConcreteExtractor(conn=None)

        # Partial features
        result = ext.validate_features({"feature_a": 10.0})
        assert result["feature_a"] == 10.0
        assert result["feature_b"] == 2.0  # default
        assert result["feature_c"] == 3.0  # default

    def test_extractor_name_default(self):
        """Test extractor name defaults to class name."""

        class ConcreteExtractor(BaseFeatureExtractor):
            FEATURE_NAMES = ("test_feature",)

            def extract(self, player_name, game_date, stat_type, **kwargs):
                return {}

            @classmethod
            def get_defaults(cls):
                return {"test_feature": 0.0}

        ext = ConcreteExtractor(conn=None)
        assert ext.name == "ConcreteExtractor"

    def test_extractor_name_custom(self):
        """Test extractor with custom name."""

        class ConcreteExtractor(BaseFeatureExtractor):
            FEATURE_NAMES = ("test_feature",)

            def extract(self, player_name, game_date, stat_type, **kwargs):
                return {}

            @classmethod
            def get_defaults(cls):
                return {"test_feature": 0.0}

        ext = ConcreteExtractor(conn=None, name="CustomName")
        assert ext.name == "CustomName"


class TestBookFeatureExtractor:
    """Tests for BookFeatureExtractor."""

    def test_feature_names_defined(self):
        """Test FEATURE_NAMES is defined and non-empty."""
        assert hasattr(BookFeatureExtractor, "FEATURE_NAMES")
        assert len(BookFeatureExtractor.FEATURE_NAMES) == 23

    def test_feature_names_are_strings(self):
        """Test all feature names are strings."""
        for name in BookFeatureExtractor.FEATURE_NAMES:
            assert isinstance(name, str)
            assert len(name) > 0

    def test_get_defaults_returns_all_features(self):
        """Test get_defaults returns all features."""
        defaults = BookFeatureExtractor.get_defaults()
        assert isinstance(defaults, dict)
        assert len(defaults) == len(BookFeatureExtractor.FEATURE_NAMES)
        for name in BookFeatureExtractor.FEATURE_NAMES:
            assert name in defaults

    def test_get_defaults_values_are_floats(self):
        """Test default values are floats."""
        defaults = BookFeatureExtractor.get_defaults()
        for value in defaults.values():
            assert isinstance(value, (int, float))

    def test_initialization(self):
        """Test BookFeatureExtractor can be initialized."""
        ext = BookFeatureExtractor(conn=None)
        assert ext.conn is None
        assert ext.name == "BookFeatures"

    def test_expected_features(self):
        """Test expected feature names exist."""
        expected = [
            "line_spread",
            "consensus_line",
            "line_std",
            "num_books_offering",
            "softest_book_id",
            "hardest_book_id",
        ]
        for feat in expected:
            assert feat in BookFeatureExtractor.FEATURE_NAMES


class TestH2HFeatureExtractor:
    """Tests for H2HFeatureExtractor."""

    def test_feature_names_defined(self):
        """Test FEATURE_NAMES is defined."""
        assert hasattr(H2HFeatureExtractor, "FEATURE_NAMES")
        assert len(H2HFeatureExtractor.FEATURE_NAMES) == 36

    def test_get_defaults_returns_all_features(self):
        """Test get_defaults returns all features."""
        defaults = H2HFeatureExtractor.get_defaults()
        assert len(defaults) == len(H2HFeatureExtractor.FEATURE_NAMES)

    def test_initialization(self):
        """Test H2HFeatureExtractor can be initialized."""
        ext = H2HFeatureExtractor(conn=None)
        assert ext.name == "H2HFeatures"


class TestPropHistoryExtractor:
    """Tests for PropHistoryExtractor."""

    def test_feature_names_defined(self):
        """Test FEATURE_NAMES is defined."""
        assert hasattr(PropHistoryExtractor, "FEATURE_NAMES")
        assert len(PropHistoryExtractor.FEATURE_NAMES) == 12

    def test_get_defaults_returns_all_features(self):
        """Test get_defaults returns all features."""
        defaults = PropHistoryExtractor.get_defaults()
        assert len(defaults) == len(PropHistoryExtractor.FEATURE_NAMES)

    def test_initialization(self):
        """Test PropHistoryExtractor can be initialized."""
        ext = PropHistoryExtractor(conn=None)
        assert ext.name == "PropHistory"

    def test_expected_features(self):
        """Test expected feature names exist."""
        expected = [
            "prop_over_hit_rate_l10",
            "prop_over_hit_rate_l20",
            "prop_avg_margin_l10",
        ]
        for feat in expected:
            assert feat in PropHistoryExtractor.FEATURE_NAMES


class TestVegasContextExtractor:
    """Tests for VegasContextExtractor."""

    def test_feature_names_defined(self):
        """Test FEATURE_NAMES is defined."""
        assert hasattr(VegasContextExtractor, "FEATURE_NAMES")
        assert len(VegasContextExtractor.FEATURE_NAMES) == 2

    def test_get_defaults_returns_all_features(self):
        """Test get_defaults returns all features."""
        defaults = VegasContextExtractor.get_defaults()
        assert len(defaults) == len(VegasContextExtractor.FEATURE_NAMES)

    def test_initialization(self):
        """Test VegasContextExtractor can be initialized."""
        ext = VegasContextExtractor(conn=None)
        assert ext.name == "VegasContext"

    def test_expected_features(self):
        """Test expected feature names exist."""
        assert "vegas_total" in VegasContextExtractor.FEATURE_NAMES
        assert "vegas_spread" in VegasContextExtractor.FEATURE_NAMES


class TestTeamBettingExtractor:
    """Tests for TeamBettingExtractor."""

    def test_feature_names_defined(self):
        """Test FEATURE_NAMES is defined."""
        assert hasattr(TeamBettingExtractor, "FEATURE_NAMES")
        assert len(TeamBettingExtractor.FEATURE_NAMES) == 5

    def test_get_defaults_returns_all_features(self):
        """Test get_defaults returns all features."""
        defaults = TeamBettingExtractor.get_defaults()
        assert len(defaults) == len(TeamBettingExtractor.FEATURE_NAMES)

    def test_initialization(self):
        """Test TeamBettingExtractor can be initialized."""
        ext = TeamBettingExtractor(conn=None)
        assert ext.name == "TeamBetting"


class TestCheatsheetExtractor:
    """Tests for CheatsheetExtractor."""

    def test_feature_names_defined(self):
        """Test FEATURE_NAMES is defined."""
        assert hasattr(CheatsheetExtractor, "FEATURE_NAMES")
        assert len(CheatsheetExtractor.FEATURE_NAMES) == 8

    def test_get_defaults_returns_all_features(self):
        """Test get_defaults returns all features."""
        defaults = CheatsheetExtractor.get_defaults()
        assert len(defaults) == len(CheatsheetExtractor.FEATURE_NAMES)

    def test_initialization(self):
        """Test CheatsheetExtractor can be initialized."""
        ext = CheatsheetExtractor(conn=None)
        assert ext.name == "Cheatsheet"


class TestExtractorConsistency:
    """Tests for consistency across all extractors."""

    def test_all_extractors_have_feature_names(self):
        """Test all extractors define FEATURE_NAMES."""
        extractors = [
            BookFeatureExtractor,
            H2HFeatureExtractor,
            PropHistoryExtractor,
            VegasContextExtractor,
            TeamBettingExtractor,
            CheatsheetExtractor,
        ]
        for ext_cls in extractors:
            assert hasattr(ext_cls, "FEATURE_NAMES"), f"{ext_cls.__name__} missing FEATURE_NAMES"
            assert len(ext_cls.FEATURE_NAMES) > 0, f"{ext_cls.__name__} has empty FEATURE_NAMES"

    def test_all_extractors_have_get_defaults(self):
        """Test all extractors implement get_defaults."""
        extractors = [
            BookFeatureExtractor,
            H2HFeatureExtractor,
            PropHistoryExtractor,
            VegasContextExtractor,
            TeamBettingExtractor,
            CheatsheetExtractor,
        ]
        for ext_cls in extractors:
            defaults = ext_cls.get_defaults()
            assert isinstance(defaults, dict), f"{ext_cls.__name__}.get_defaults() not dict"
            assert len(defaults) == len(
                ext_cls.FEATURE_NAMES
            ), f"{ext_cls.__name__} defaults count mismatch"

    def test_no_duplicate_feature_names(self):
        """Test no duplicate feature names within extractors."""
        extractors = [
            BookFeatureExtractor,
            H2HFeatureExtractor,
            PropHistoryExtractor,
            VegasContextExtractor,
            TeamBettingExtractor,
            CheatsheetExtractor,
        ]
        for ext_cls in extractors:
            names = ext_cls.FEATURE_NAMES
            assert len(names) == len(set(names)), f"{ext_cls.__name__} has duplicate feature names"

    def test_total_feature_count(self):
        """Test total features across all extractors."""
        extractors = [
            BookFeatureExtractor,  # 23
            H2HFeatureExtractor,  # 36
            PropHistoryExtractor,  # 12
            VegasContextExtractor,  # 2
            TeamBettingExtractor,  # 5
            CheatsheetExtractor,  # 8
        ]
        total = sum(len(ext_cls.FEATURE_NAMES) for ext_cls in extractors)
        # Should have 86 features from extractors
        assert total == 86
