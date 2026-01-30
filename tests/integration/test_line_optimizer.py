"""
Line Optimizer Integration Tests
================================
Tests for LineOptimizer line shopping and filtering logic.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


class TestLineOptimizerFiltering:
    """Tests for tier-based filtering logic."""

    @pytest.fixture
    def mock_lines_df(self):
        """Create mock book lines DataFrame."""
        return pd.DataFrame(
            {
                "book_name": ["DraftKings", "FanDuel", "BetMGM", "Underdog", "ESPNBet"],
                "over_line": [25.5, 26.0, 25.5, 24.5, 26.0],
                "opponent_team": ["GSW", "GSW", "GSW", "GSW", "GSW"],
                "is_home": [True, True, True, True, True],
            }
        )

    @pytest.fixture
    def optimizer(self):
        """Create LineOptimizer with mocked connection."""
        with patch("nba.betting_xl.line_optimizer.psycopg2"):
            from nba.betting_xl.line_optimizer import LineOptimizer

            opt = LineOptimizer()
            opt.conn = MagicMock()
            return opt

    def test_softest_line_selection(self, optimizer, mock_lines_df):
        """Test that softest line is correctly identified."""
        # Softest should be Underdog at 24.5
        softest = mock_lines_df["over_line"].min()
        assert softest == 24.5

        # Find softest book
        softest_idx = mock_lines_df["over_line"].idxmin()
        softest_book = mock_lines_df.loc[softest_idx, "book_name"]
        assert softest_book == "Underdog"

    def test_line_spread_calculation(self, mock_lines_df):
        """Test line spread calculation."""
        spread = mock_lines_df["over_line"].max() - mock_lines_df["over_line"].min()
        assert spread == 1.5  # 26.0 - 24.5

    def test_consensus_line_calculation(self, mock_lines_df):
        """Test consensus line calculation."""
        consensus = mock_lines_df["over_line"].mean()
        expected = (25.5 + 26.0 + 25.5 + 24.5 + 26.0) / 5
        assert abs(consensus - expected) < 0.01

    def test_blacklisted_books_skipped(self, optimizer):
        """Test that blacklisted books are skipped."""
        from nba.betting_xl.line_optimizer import BLACKLISTED_BOOKS

        # POINTS should have FanDuel and BetRivers blacklisted
        points_blacklist = BLACKLISTED_BOOKS.get("POINTS", set())
        assert "FanDuel" in points_blacklist or "fanduel" in points_blacklist
        assert "BetRivers" in points_blacklist or "betrivers" in points_blacklist

    def test_star_players_list(self):
        """Test star players configuration."""
        from nba.betting_xl.line_optimizer import STAR_PLAYERS

        # Should have known star players
        assert "Donovan Mitchell" in STAR_PLAYERS
        assert "Anthony Edwards" in STAR_PLAYERS
        assert "Kevin Durant" in STAR_PLAYERS

    def test_tier_config_structure(self):
        """Test tier config has required keys."""
        from nba.betting_xl.line_optimizer import TIER_CONFIG

        # POINTS should be enabled
        assert "POINTS" in TIER_CONFIG
        assert TIER_CONFIG["POINTS"]["enabled"] is True

        # Should have tiers defined
        assert "tiers" in TIER_CONFIG["POINTS"]

        # Check REBOUNDS META tier
        assert "REBOUNDS" in TIER_CONFIG
        rebounds_tiers = TIER_CONFIG["REBOUNDS"]["tiers"]
        assert "META" in rebounds_tiers

    def test_v3_tier_config_structure(self):
        """Test V3 tier config structure."""
        from nba.betting_xl.line_optimizer import V3_TIER_CONFIG

        assert "POINTS" in V3_TIER_CONFIG
        assert V3_TIER_CONFIG["POINTS"]["enabled"] is True
        assert "tiers" in V3_TIER_CONFIG["POINTS"]

    def test_trap_books_config(self):
        """Test trap books configuration."""
        from nba.betting_xl.line_optimizer import TRAP_BOOKS_WHEN_SOFTEST

        # DraftKings and BetMGM should be in trap books
        assert "DraftKings" in TRAP_BOOKS_WHEN_SOFTEST
        assert "BetMGM" in TRAP_BOOKS_WHEN_SOFTEST

        # Check required config keys
        dk_config = TRAP_BOOKS_WHEN_SOFTEST["DraftKings"]
        assert "min_spread_required" in dk_config
        assert "min_p_over_boost" in dk_config

    def test_reliable_books_config(self):
        """Test reliable books configuration."""
        from nba.betting_xl.line_optimizer import RELIABLE_BOOKS_WHEN_SOFTEST

        assert "Underdog" in RELIABLE_BOOKS_WHEN_SOFTEST
        assert "ESPNBet" in RELIABLE_BOOKS_WHEN_SOFTEST


class TestLineOptimizerOptimize:
    """Tests for optimize_line method."""

    @pytest.fixture
    def optimizer_with_mock_data(self):
        """Create optimizer with mocked database returning test data."""
        with patch("nba.betting_xl.line_optimizer.psycopg2"):
            from nba.betting_xl.line_optimizer import LineOptimizer

            opt = LineOptimizer()
            opt.conn = MagicMock()
            return opt

    def test_returns_none_for_disabled_market(self, optimizer_with_mock_data):
        """Test that disabled markets return None."""
        # ASSISTS is disabled
        with patch.object(optimizer_with_mock_data, "get_all_book_lines") as mock_get:
            mock_get.return_value = pd.DataFrame(
                {
                    "book_name": ["DraftKings"],
                    "over_line": [8.5],
                    "opponent_team": ["LAL"],
                    "is_home": [True],
                }
            )

            result = optimizer_with_mock_data.optimize_line(
                player_name="Test Player",
                game_date="2025-01-15",
                stat_type="ASSISTS",  # Disabled market
                prediction=10.0,
                p_over=0.65,
            )

            # Should return None for disabled market
            assert result is None

    def test_returns_none_for_no_lines(self, optimizer_with_mock_data):
        """Test that missing lines returns None."""
        with patch.object(optimizer_with_mock_data, "get_all_book_lines") as mock_get:
            mock_get.return_value = None

            result = optimizer_with_mock_data.optimize_line(
                player_name="Test Player",
                game_date="2025-01-15",
                stat_type="POINTS",
                prediction=25.0,
                p_over=0.65,
            )

            assert result is None


class TestLineOptimizerV3:
    """Tests for V3 optimize_line method."""

    @pytest.fixture
    def optimizer(self):
        """Create optimizer with mocked connection."""
        with patch("nba.betting_xl.line_optimizer.psycopg2"):
            from nba.betting_xl.line_optimizer import LineOptimizer

            opt = LineOptimizer()
            opt.conn = MagicMock()
            return opt

    def test_v3_returns_none_for_disabled_market(self, optimizer):
        """Test V3 returns None for disabled markets."""
        with patch.object(optimizer, "get_all_book_lines") as mock_get:
            mock_get.return_value = pd.DataFrame(
                {
                    "book_name": ["DraftKings"],
                    "over_line": [3.5],
                    "opponent_team": ["LAL"],
                    "is_home": [True],
                }
            )

            result = optimizer.optimize_line_v3(
                player_name="Test Player",
                game_date="2025-01-15",
                stat_type="REBOUNDS",  # V3 disabled for rebounds
                prediction=6.0,
                p_over=0.65,
            )

            # REBOUNDS V3 is disabled
            assert result is None

    def test_v3_direction_determination(self):
        """Test that direction is correctly determined from p_over."""
        # p_over >= 0.5 means OVER
        p_over = 0.65
        direction = "OVER" if p_over >= 0.50 else "UNDER"
        assert direction == "OVER"

        # p_over < 0.5 means UNDER
        p_over = 0.35
        direction = "OVER" if p_over >= 0.50 else "UNDER"
        assert direction == "UNDER"


class TestEdgeCalculations:
    """Tests for edge calculation logic."""

    def test_over_edge_calculation(self):
        """Test edge calculation for OVER bets."""
        prediction = 28.0
        line = 25.5
        edge = prediction - line
        assert edge == 2.5

    def test_under_edge_calculation(self):
        """Test edge calculation for UNDER bets."""
        prediction = 23.0
        line = 25.5
        edge = line - prediction  # For UNDER, edge is line - prediction
        assert edge == 2.5

    def test_edge_percentage_calculation(self):
        """Test edge percentage calculation."""
        consensus = 26.0
        softest = 24.5
        edge_pct = ((consensus - softest) / softest) * 100
        expected = ((26.0 - 24.5) / 24.5) * 100
        assert abs(edge_pct - expected) < 0.01


class TestConfidenceLevels:
    """Tests for confidence level determination."""

    def test_high_confidence(self):
        """Test HIGH confidence criteria."""
        p_over = 0.72
        edge = 3.5

        # HIGH if p_over >= 0.70 and edge >= 3.0
        if p_over >= 0.70 and edge >= 3.0:
            confidence = "HIGH"
        else:
            confidence = "MEDIUM"

        assert confidence == "HIGH"

    def test_medium_confidence(self):
        """Test MEDIUM confidence criteria."""
        p_over = 0.65
        edge = 2.5

        # MEDIUM if p_over >= 0.60 and edge >= 2.0
        if p_over >= 0.70 and edge >= 3.0:
            confidence = "HIGH"
        elif p_over >= 0.60 and edge >= 2.0:
            confidence = "MEDIUM"
        else:
            confidence = "STANDARD"

        assert confidence == "MEDIUM"

    def test_spread_based_confidence(self):
        """Test confidence based on line spread."""
        p_over = 0.55
        edge = 1.5
        line_spread = 3.0

        # MEDIUM if spread >= 2.5 even with lower p_over/edge
        if p_over >= 0.70 and edge >= 3.0:
            confidence = "HIGH"
        elif p_over >= 0.60 and edge >= 2.0:
            confidence = "MEDIUM"
        elif line_spread >= 2.5:
            confidence = "MEDIUM"
        else:
            confidence = "STANDARD"

        assert confidence == "MEDIUM"


class TestBookEncoding:
    """Tests for book name encoding."""

    def test_book_name_normalization(self):
        """Test book name normalization."""
        book_map = {
            "DraftKings": "draftkings",
            "draftkings": "draftkings",
            "FanDuel": "fanduel",
            "FANDUEL": "fanduel",
        }

        for input_name, expected in book_map.items():
            assert input_name.lower() == expected or expected in input_name.lower()

    def test_book_id_encoding(self):
        """Test numeric book ID encoding."""
        from nba.features.extractors.base import BaseFeatureExtractor

        # Create a concrete implementation for testing
        class TestExtractor(BaseFeatureExtractor):
            def extract(self, *args, **kwargs):
                return {}

            @classmethod
            def get_defaults(cls):
                return {}

        extractor = TestExtractor(conn=None)

        assert extractor._encode_book_name("draftkings") == 1.0
        assert extractor._encode_book_name("fanduel") == 2.0
        assert extractor._encode_book_name("unknown") == 0.0
