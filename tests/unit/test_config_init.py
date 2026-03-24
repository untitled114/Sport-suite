"""
Unit Tests for nba/config/__init__.py
======================================
Tests for the NBA config module initialization, environment detection,
and re-exports from submodules.
"""

import importlib
import os
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Environment Detection
# ---------------------------------------------------------------------------
class TestEnvironmentDetection:
    """Tests for environment detection logic."""

    def test_default_environment_is_development(self):
        """Test that default environment is 'development'."""
        with patch.dict(os.environ, {}, clear=True):
            # Force reimport to pick up patched environment
            import nba.config as config_mod

            mod = importlib.reload(config_mod)
            assert mod.ENVIRONMENT == "development"
            assert mod.IS_DEVELOPMENT is True
            assert mod.IS_PRODUCTION is False

    def test_production_environment(self):
        """Test detection of production environment."""
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}, clear=False):
            import nba.config as config_mod

            mod = importlib.reload(config_mod)
            assert mod.ENVIRONMENT == "production"
            assert mod.IS_PRODUCTION is True
            assert mod.IS_DEVELOPMENT is False

    def test_environment_case_insensitive(self):
        """Test that ENVIRONMENT value is lowercased."""
        with patch.dict(os.environ, {"ENVIRONMENT": "PRODUCTION"}, clear=False):
            import nba.config as config_mod

            mod = importlib.reload(config_mod)
            assert mod.ENVIRONMENT == "production"
            assert mod.IS_PRODUCTION is True

    def test_custom_environment(self):
        """Test a custom (non-standard) environment value."""
        with patch.dict(os.environ, {"ENVIRONMENT": "staging"}, clear=False):
            import nba.config as config_mod

            mod = importlib.reload(config_mod)
            assert mod.ENVIRONMENT == "staging"
            assert mod.IS_PRODUCTION is False
            assert mod.IS_DEVELOPMENT is False


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
class TestGetEnvironment:
    """Tests for get_environment helper."""

    def test_get_environment_returns_string(self):
        """Test that get_environment returns a string."""
        from nba.config import get_environment

        result = get_environment()
        assert isinstance(result, str)

    def test_get_environment_matches_module_var(self):
        """Test that get_environment matches ENVIRONMENT module variable."""
        import nba.config as config_mod

        assert config_mod.get_environment() == config_mod.ENVIRONMENT


class TestIsProduction:
    """Tests for is_production helper."""

    def test_is_production_returns_bool(self):
        """Test that is_production returns a boolean."""
        from nba.config import is_production

        result = is_production()
        assert isinstance(result, bool)

    def test_is_production_matches_module_var(self):
        """Test that is_production matches IS_PRODUCTION module variable."""
        import nba.config as config_mod

        assert config_mod.is_production() == config_mod.IS_PRODUCTION


class TestIsDevelopment:
    """Tests for is_development helper."""

    def test_is_development_returns_bool(self):
        """Test that is_development returns a boolean."""
        from nba.config import is_development

        result = is_development()
        assert isinstance(result, bool)

    def test_is_development_matches_module_var(self):
        """Test that is_development matches IS_DEVELOPMENT module variable."""
        import nba.config as config_mod

        assert config_mod.is_development() == config_mod.IS_DEVELOPMENT


# ---------------------------------------------------------------------------
# Re-exports from constants
# ---------------------------------------------------------------------------
class TestConstantsReexports:
    """Tests that constants are properly re-exported."""

    def test_active_markets_exported(self):
        """Test ACTIVE_MARKETS is accessible from config."""
        from nba.config import ACTIVE_MARKETS

        assert "POINTS" in ACTIVE_MARKETS
        assert "REBOUNDS" in ACTIVE_MARKETS

    def test_all_books_exported(self):
        """Test ALL_BOOKS is accessible from config."""
        from nba.config import ALL_BOOKS

        assert isinstance(ALL_BOOKS, dict)
        assert len(ALL_BOOKS) > 0

    def test_valid_stat_types_exported(self):
        """Test VALID_STAT_TYPES is accessible from config."""
        from nba.config import VALID_STAT_TYPES

        assert "POINTS" in VALID_STAT_TYPES
        assert "REBOUNDS" in VALID_STAT_TYPES
        assert "ASSISTS" in VALID_STAT_TYPES
        assert "THREES" in VALID_STAT_TYPES

    def test_bettingpros_market_ids_exported(self):
        """Test BETTINGPROS_MARKET_IDS is accessible from config."""
        from nba.config import BETTINGPROS_MARKET_IDS

        assert "points" in BETTINGPROS_MARKET_IDS
        assert BETTINGPROS_MARKET_IDS["points"] == 156

    def test_stat_column_map_exported(self):
        """Test STAT_COLUMN_MAP is accessible from config."""
        from nba.config import STAT_COLUMN_MAP

        assert STAT_COLUMN_MAP["POINTS"] == "points"
        assert STAT_COLUMN_MAP["REBOUNDS"] == "rebounds"

    def test_team_abbrev_map_exported(self):
        """Test TEAM_ABBREV_MAP is accessible from config."""
        from nba.config import TEAM_ABBREV_MAP

        assert "WSH" in TEAM_ABBREV_MAP
        assert TEAM_ABBREV_MAP["WSH"] == "WAS"

    def test_priority_books_exported(self):
        """Test PRIORITY_BOOKS is accessible from config."""
        from nba.config import PRIORITY_BOOKS

        assert isinstance(PRIORITY_BOOKS, dict)
        assert 12 in PRIORITY_BOOKS  # DraftKings

    def test_disabled_markets_exported(self):
        """Test DISABLED_MARKETS is accessible from config."""
        from nba.config import DISABLED_MARKETS

        assert "ASSISTS" in DISABLED_MARKETS
        assert "THREES" in DISABLED_MARKETS

    def test_extended_stat_types_exported(self):
        """Test EXTENDED_STAT_TYPES is accessible from config."""
        from nba.config import EXTENDED_STAT_TYPES

        assert "STEALS" in EXTENDED_STAT_TYPES
        assert "BLOCKS" in EXTENDED_STAT_TYPES

    def test_combo_stat_map_exported(self):
        """Test COMBO_STAT_MAP is accessible from config."""
        from nba.config import COMBO_STAT_MAP

        assert "PRA" in COMBO_STAT_MAP

    def test_book_market_exclusions_exported(self):
        """Test BOOK_MARKET_EXCLUSIONS is accessible from config."""
        from nba.config import BOOK_MARKET_EXCLUSIONS

        assert isinstance(BOOK_MARKET_EXCLUSIONS, dict)


# ---------------------------------------------------------------------------
# Re-exports from database
# ---------------------------------------------------------------------------
class TestDatabaseReexports:
    """Tests that database config functions are properly re-exported."""

    def test_get_db_config_exported(self):
        """Test get_db_config is accessible from config."""
        from nba.config import get_db_config

        assert callable(get_db_config)

    def test_get_players_db_config_exported(self):
        """Test get_players_db_config routes to consolidated DB."""
        from nba.config import get_players_db_config

        config = get_players_db_config()
        assert config["port"] == 5500
        assert config["database"] == "sportsuite"

    def test_get_games_db_config_exported(self):
        """Test get_games_db_config routes to consolidated DB."""
        from nba.config import get_games_db_config

        config = get_games_db_config()
        assert config["port"] == 5500
        assert config["database"] == "sportsuite"

    def test_get_team_db_config_exported(self):
        """Test get_team_db_config routes to consolidated DB."""
        from nba.config import get_team_db_config

        config = get_team_db_config()
        assert config["port"] == 5500
        assert config["database"] == "sportsuite"

    def test_get_intelligence_db_config_exported(self):
        """Test get_intelligence_db_config routes to consolidated DB."""
        from nba.config import get_intelligence_db_config

        config = get_intelligence_db_config()
        assert config["port"] == 5500
        assert config["database"] == "sportsuite"

    def test_get_mongo_config_exported(self):
        """Test get_mongo_config is accessible from config."""
        from nba.config import get_mongo_config

        config = get_mongo_config()
        assert "uri" in config
        assert "database" in config

    def test_db_default_user_exported(self):
        """Test DB_DEFAULT_USER is accessible from config."""
        from nba.config import DB_DEFAULT_USER

        assert isinstance(DB_DEFAULT_USER, str)

    def test_db_default_password_exported(self):
        """Test DB_DEFAULT_PASSWORD is accessible from config."""
        from nba.config import DB_DEFAULT_PASSWORD

        # May be None if DB_PASSWORD not set, but the import should work
        assert DB_DEFAULT_PASSWORD is None or isinstance(DB_DEFAULT_PASSWORD, str)


# ---------------------------------------------------------------------------
# Re-exports from thresholds
# ---------------------------------------------------------------------------
class TestThresholdsReexports:
    """Tests that threshold configs are properly re-exported."""

    def test_probability_thresholds_exported(self):
        """Test PROBABILITY_THRESHOLDS is accessible from config."""
        from nba.config import PROBABILITY_THRESHOLDS

        assert PROBABILITY_THRESHOLDS.min_p_over == 0.55

    def test_edge_thresholds_exported(self):
        """Test EDGE_THRESHOLDS is accessible from config."""
        from nba.config import EDGE_THRESHOLDS

        assert EDGE_THRESHOLDS.min_edge_points == 2.5

    def test_spread_thresholds_exported(self):
        """Test SPREAD_THRESHOLDS is accessible from config."""
        from nba.config import SPREAD_THRESHOLDS

        assert SPREAD_THRESHOLDS.min_spread == 2.0

    def test_blend_weights_exported(self):
        """Test BLEND_WEIGHTS is accessible from config."""
        from nba.config import BLEND_WEIGHTS

        assert BLEND_WEIGHTS.classifier_weight == 1.0

    def test_training_hyperparameters_exported(self):
        """Test TRAINING_HYPERPARAMETERS is accessible from config."""
        from nba.config import TRAINING_HYPERPARAMETERS

        assert TRAINING_HYPERPARAMETERS.n_estimators == 500

    def test_points_config_exported(self):
        """Test POINTS_CONFIG is accessible from config."""
        from nba.config import POINTS_CONFIG

        assert POINTS_CONFIG.market == "POINTS"
        assert POINTS_CONFIG.enabled is True

    def test_rebounds_config_exported(self):
        """Test REBOUNDS_CONFIG is accessible from config."""
        from nba.config import REBOUNDS_CONFIG

        assert REBOUNDS_CONFIG.market == "REBOUNDS"

    def test_assists_config_exported(self):
        """Test ASSISTS_CONFIG is accessible from config."""
        from nba.config import ASSISTS_CONFIG

        assert ASSISTS_CONFIG.enabled is False

    def test_threes_config_exported(self):
        """Test THREES_CONFIG is accessible from config."""
        from nba.config import THREES_CONFIG

        assert THREES_CONFIG.enabled is False

    def test_star_points_config_exported(self):
        """Test STAR_POINTS_CONFIG is accessible from config."""
        from nba.config import STAR_POINTS_CONFIG

        assert STAR_POINTS_CONFIG.min_p_over == 0.70

    def test_star_rebounds_config_exported(self):
        """Test STAR_REBOUNDS_CONFIG is accessible from config."""
        from nba.config import STAR_REBOUNDS_CONFIG

        assert STAR_REBOUNDS_CONFIG.min_p_over == 0.55

    def test_trap_books_exported(self):
        """Test TRAP_BOOKS is accessible from config."""
        from nba.config import TRAP_BOOKS

        assert "DraftKings" in TRAP_BOOKS

    def test_reliable_books_exported(self):
        """Test RELIABLE_BOOKS is accessible from config."""
        from nba.config import RELIABLE_BOOKS

        assert "Underdog" in RELIABLE_BOOKS


# ---------------------------------------------------------------------------
# Threshold helper functions re-exported
# ---------------------------------------------------------------------------
class TestThresholdHelperReexports:
    """Tests that threshold helper functions are properly re-exported."""

    def test_get_market_config_exported(self):
        """Test get_market_config is callable from config."""
        from nba.config import get_market_config

        config = get_market_config("POINTS")
        assert config.market == "POINTS"

    def test_get_tier_config_exported(self):
        """Test get_tier_config is callable from config."""
        from nba.config import get_tier_config

        tier = get_tier_config("POINTS", "X")
        assert tier is not None
        assert tier.name == "X"

    def test_is_trap_book_exported(self):
        """Test is_trap_book is callable from config."""
        from nba.config import is_trap_book

        assert is_trap_book("DraftKings") is True
        assert is_trap_book("Underdog") is False

    def test_get_star_config_exported(self):
        """Test get_star_config is callable from config."""
        from nba.config import get_star_config

        config = get_star_config("POINTS")
        assert config is not None

    def test_is_reliable_book_exported(self):
        """Test is_reliable_book is callable from config."""
        from nba.config import is_reliable_book

        assert is_reliable_book("Underdog") is True


# ---------------------------------------------------------------------------
# Threshold dataclass types re-exported
# ---------------------------------------------------------------------------
class TestThresholdTypesReexports:
    """Tests that threshold dataclass types are properly re-exported."""

    def test_probability_thresholds_type(self):
        """Test ProbabilityThresholds type is importable."""
        from nba.config import ProbabilityThresholds

        instance = ProbabilityThresholds(min_p_over=0.60, max_p_over=0.90)
        assert instance.min_p_over == 0.60

    def test_edge_thresholds_type(self):
        """Test EdgeThresholds type is importable."""
        from nba.config import EdgeThresholds

        instance = EdgeThresholds(min_edge_points=3.0, max_edge_points=6.0)
        assert instance.min_edge_points == 3.0

    def test_spread_thresholds_type(self):
        """Test SpreadThresholds type is importable."""
        from nba.config import SpreadThresholds

        instance = SpreadThresholds(min_spread=2.5)
        assert instance.min_spread == 2.5

    def test_blend_weights_type(self):
        """Test BlendWeights type is importable."""
        from nba.config import BlendWeights

        instance = BlendWeights(classifier_weight=0.5, residual_weight=0.5)
        assert instance.classifier_weight == 0.5

    def test_market_config_type(self):
        """Test MarketConfig type is importable."""
        from nba.config import MarketConfig

        assert MarketConfig is not None

    def test_tier_config_type(self):
        """Test TierConfig type is importable."""
        from nba.config import TierConfig

        tier = TierConfig(name="TEST", direction="OVER", min_p_over=0.70)
        assert tier.name == "TEST"

    def test_line_constraints_type(self):
        """Test LineConstraints type is importable."""
        from nba.config import LineConstraints

        constraints = LineConstraints(min_line=10.0, max_line=50.0)
        assert constraints.min_line == 10.0

    def test_star_player_config_type(self):
        """Test StarPlayerConfig type is importable."""
        from nba.config import StarPlayerConfig

        assert StarPlayerConfig is not None

    def test_training_hyperparameters_type(self):
        """Test TrainingHyperparameters type is importable."""
        from nba.config import TrainingHyperparameters

        hp = TrainingHyperparameters(n_estimators=1000, learning_rate=0.01)
        assert hp.n_estimators == 1000

    def test_trap_book_config_type(self):
        """Test TrapBookConfig type is importable."""
        from nba.config import TrapBookConfig

        assert TrapBookConfig is not None


# ---------------------------------------------------------------------------
# __all__ validation
# ---------------------------------------------------------------------------
class TestAllExports:
    """Tests that __all__ contains expected exports."""

    def test_all_is_defined(self):
        """Test that __all__ is defined."""
        import nba.config as config_mod

        assert hasattr(config_mod, "__all__")
        assert isinstance(config_mod.__all__, list)

    def test_all_contains_environment_exports(self):
        """Test __all__ contains environment-related exports."""
        import nba.config as config_mod

        assert "ENVIRONMENT" in config_mod.__all__
        assert "IS_PRODUCTION" in config_mod.__all__
        assert "IS_DEVELOPMENT" in config_mod.__all__
        assert "get_environment" in config_mod.__all__
        assert "is_production" in config_mod.__all__
        assert "is_development" in config_mod.__all__

    def test_all_contains_constants_exports(self):
        """Test __all__ contains constants exports."""
        import nba.config as config_mod

        assert "ACTIVE_MARKETS" in config_mod.__all__
        assert "ALL_BOOKS" in config_mod.__all__
        assert "VALID_STAT_TYPES" in config_mod.__all__

    def test_all_contains_database_exports(self):
        """Test __all__ contains database exports."""
        import nba.config as config_mod

        assert "get_db_config" in config_mod.__all__
        assert "get_players_db_config" in config_mod.__all__
        assert "get_intelligence_db_config" in config_mod.__all__

    def test_all_contains_threshold_exports(self):
        """Test __all__ contains threshold exports."""
        import nba.config as config_mod

        assert "get_market_config" in config_mod.__all__
        assert "get_tier_config" in config_mod.__all__
        assert "is_trap_book" in config_mod.__all__

    def test_all_exports_are_accessible(self):
        """Test that every name in __all__ is actually accessible."""
        import nba.config as config_mod

        for name in config_mod.__all__:
            assert hasattr(config_mod, name), f"{name} in __all__ but not accessible"
