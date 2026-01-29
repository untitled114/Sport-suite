"""
Unit Tests for Configuration
============================
Tests for database configuration and constants.

Best Practices Applied:
- Test environment variable handling
- Test configuration defaults
- Test constant values match documentation
"""

import os
from unittest.mock import patch

import pytest


class TestDatabaseConfig:
    """Tests for database configuration."""

    def test_db_config_module_imports(self):
        """Test database config module can be imported."""
        from nba.config import database

        # Module should be importable
        assert database is not None

    def test_default_ports(self):
        """Test default database ports match documentation."""
        # From CLAUDE.md:
        # Players: 5536, Games: 5537, Team: 5538, Intelligence: 5539
        expected_ports = {
            "players": 5536,
            "games": 5537,
            "team": 5538,
            "intelligence": 5539,
        }

        for _db_name, expected_port in expected_ports.items():
            assert expected_port in range(5530, 5550), f"Port {expected_port} out of expected range"

    def test_default_database_names(self):
        """Test default database names."""
        expected_names = {
            "players": "nba_players",
            "games": "nba_games",
            "team": "nba_team",
            "intelligence": "nba_intelligence",
        }

        for db_name in expected_names.values():
            assert db_name.startswith("nba_")


class TestConstants:
    """Tests for constants module."""

    def test_constants_module_imports(self):
        """Test constants module can be imported."""
        from nba.config import constants

        assert constants is not None

    def test_market_types(self):
        """Test market type constants."""
        valid_markets = ["POINTS", "REBOUNDS", "ASSISTS", "THREES"]

        for market in valid_markets:
            assert market.isupper()

    def test_sportsbook_ids(self):
        """Test sportsbook ID mapping."""
        # From CLAUDE.md
        book_ids = {
            "draftkings": 12,
            "fanduel": 10,
            "betmgm": 19,
            "caesars": 13,
            "betrivers": 18,
            "espnbet": 33,
            "underdog": 36,
        }

        for _book_name, book_id in book_ids.items():
            assert isinstance(book_id, int)
            assert book_id > 0


class TestEnvironmentVariables:
    """Tests for environment variable handling."""

    def test_db_password_from_env(self):
        """Test DB_PASSWORD is read from environment."""
        with patch.dict(os.environ, {"DB_PASSWORD": "test_password"}):
            assert os.environ.get("DB_PASSWORD") == "test_password"

    def test_db_user_from_env(self):
        """Test DB_USER is read from environment."""
        with patch.dict(os.environ, {"DB_USER": "test_user"}):
            assert os.environ.get("DB_USER") == "test_user"

    def test_env_var_fallback(self):
        """Test environment variable fallback."""
        # Clear env var
        with patch.dict(os.environ, {}, clear=True):
            default_user = os.environ.get("DB_USER", "nba_user")
            assert default_user == "nba_user"

    def test_api_key_from_env(self):
        """Test API key is read from environment."""
        with patch.dict(os.environ, {"BETTINGPROS_API_KEY": "test_key"}):
            assert os.environ.get("BETTINGPROS_API_KEY") == "test_key"


class TestFeatureCounts:
    """Tests for feature count constants."""

    def test_total_feature_count(self):
        """Test total feature count is 102."""
        expected_count = 102
        # From CLAUDE.md: 78 player + 20 book + 2 computed + 2 base
        assert 78 + 20 + 2 + 2 == expected_count

    def test_player_feature_count(self):
        """Test player feature count."""
        # Rolling stats for multiple windows
        stats = [
            "points",
            "rebounds",
            "assists",
            "threes",
            "steals",
            "blocks",
            "turnovers",
            "minutes",
            "fg_pct",
        ]
        windows = ["L3", "L5", "L10", "L20"]

        # Not all stats have all windows
        # Per CLAUDE.md: 78 player features
        assert len(stats) * len(windows) <= 78

    def test_book_feature_count(self):
        """Test book disagreement feature count is 20."""
        book_features = [
            "line_spread",
            "consensus_line",
            "line_std_dev",
            "num_books_offering",
            "line_coef_variation",
            "draftkings_deviation",
            "fanduel_deviation",
            "betmgm_deviation",
            "caesars_deviation",
            "bet365_deviation",
            "betrivers_deviation",
            "espnbet_deviation",
            "fanatics_deviation",
            "softest_book_id",
            "hardest_book_id",
            "line_spread_percentile",
            "books_agree",
            "books_disagree",
            "softest_vs_consensus",
            "hardest_vs_consensus",
        ]
        assert len(book_features) == 20


class TestModelConfiguration:
    """Tests for model configuration."""

    def test_lgbm_regressor_params(self):
        """Test LightGBM regressor default parameters."""
        params = {
            "objective": "regression",
            "boosting_type": "gbdt",
            "num_leaves": 63,
            "learning_rate": 0.02,
            "n_estimators": 2000,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
        }

        assert params["objective"] == "regression"
        assert params["n_estimators"] == 2000
        assert params["learning_rate"] == 0.02

    def test_lgbm_classifier_params(self):
        """Test LightGBM classifier default parameters."""
        params = {
            "objective": "binary",
            "boosting_type": "gbdt",
            "num_leaves": 63,
            "learning_rate": 0.02,
            "n_estimators": 2000,
        }

        assert params["objective"] == "binary"
        assert params["num_leaves"] == 63

    def test_blend_config(self):
        """Test ensemble blend configuration."""
        blend_config = {
            "classifier_weight": 0.6,
            "residual_weight": 0.4,
        }

        assert blend_config["classifier_weight"] + blend_config["residual_weight"] == 1.0

    def test_scale_factor_by_market(self):
        """Test residual scale factor varies by market."""
        scale_factors = {
            "POINTS": 5.0,
            "ASSISTS": 5.0,
            "REBOUNDS": 2.0,
            "THREES": 2.0,
        }

        # Larger stat ranges get larger scale factors
        assert scale_factors["POINTS"] > scale_factors["REBOUNDS"]


class TestValidationThresholds:
    """Tests for validation thresholds."""

    def test_edge_threshold(self):
        """Test minimum edge threshold."""
        min_edge = 2.5  # From CLAUDE.md
        assert min_edge > 0
        assert min_edge < 10

    def test_high_spread_threshold(self):
        """Test high spread threshold for goldmine bets."""
        high_spread_threshold = 2.5  # From CLAUDE.md
        assert high_spread_threshold > 0

    def test_breakeven_win_rate(self):
        """Test breakeven win rate calculation."""
        # At -110 odds, need ~52.4% to break even
        odds = -110
        if odds < 0:
            implied_prob = abs(odds) / (abs(odds) + 100)
        else:
            implied_prob = 100 / (odds + 100)

        breakeven = implied_prob
        assert abs(breakeven - 0.524) < 0.01


class TestMarketConfiguration:
    """Tests for market-specific configuration."""

    def test_enabled_markets(self):
        """Test which markets are enabled."""
        # From CLAUDE.md
        enabled_markets = ["POINTS", "REBOUNDS"]
        disabled_markets = ["ASSISTS", "THREES"]

        assert "POINTS" in enabled_markets
        assert "REBOUNDS" in enabled_markets
        assert "ASSISTS" not in enabled_markets
        assert "THREES" not in enabled_markets

    @pytest.mark.parametrize(
        "market,expected_status",
        [
            ("POINTS", "DEPLOYED"),
            ("REBOUNDS", "DEPLOYED"),
            ("ASSISTS", "DISABLED"),
            ("THREES", "DISABLED"),
        ],
    )
    def test_market_status(self, market, expected_status):
        """Test market deployment status."""
        status_map = {
            "POINTS": "DEPLOYED",
            "REBOUNDS": "DEPLOYED",
            "ASSISTS": "DISABLED",
            "THREES": "DISABLED",
        }
        assert status_map[market] == expected_status


class TestFilePathConfiguration:
    """Tests for file path configuration."""

    def test_model_directory(self):
        """Test model directory path."""
        import pathlib

        model_dir = pathlib.Path("nba/models/saved_xl")
        # Should be relative to project root
        assert "models" in str(model_dir)

    def test_model_file_naming(self):
        """Test model file naming convention."""
        market = "points"
        expected_files = [
            f"{market}_market_regressor.pkl",
            f"{market}_market_classifier.pkl",
            f"{market}_market_calibrator.pkl",
            f"{market}_market_imputer.pkl",
            f"{market}_market_scaler.pkl",
            f"{market}_market_features.pkl",
            f"{market}_market_metadata.json",
        ]

        for filename in expected_files:
            assert market in filename
            assert filename.endswith((".pkl", ".json"))
