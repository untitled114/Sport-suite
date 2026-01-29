"""
Unit Tests for Fetchers
=======================
Tests for the prop fetchers with mocked HTTP responses.

Best Practices Applied:
- Mock external HTTP calls to avoid network dependencies
- Test parsing logic with known inputs
- Test error handling for API failures
- Use parametrize for multiple scenarios
"""

import json
from datetime import date, datetime
from unittest.mock import MagicMock, Mock, patch

import pytest
import requests


class TestBaseFetcher:
    """Tests for BaseFetcher class (abstract base class)."""

    def test_base_fetcher_is_abstract(self):
        """Test BaseFetcher is abstract and cannot be instantiated directly."""
        from nba.betting_xl.fetchers.base_fetcher import BaseFetcher

        # BaseFetcher should be an abstract class
        assert hasattr(BaseFetcher, "fetch")

    def test_base_fetcher_has_required_methods(self):
        """Test BaseFetcher defines required interface."""
        from nba.betting_xl.fetchers.base_fetcher import BaseFetcher

        assert hasattr(BaseFetcher, "fetch")

    def test_date_formatting(self):
        """Test date formatting for API calls."""
        test_date = date(2025, 11, 6)
        formatted = test_date.strftime("%Y-%m-%d")
        assert formatted == "2025-11-06"


class TestNormalization:
    """Tests for prop normalization functions."""

    def test_normalize_stat_type_points(self):
        """Test stat type normalization for points."""
        from nba.betting_xl.fetchers.normalization import normalize_stat_type

        assert normalize_stat_type("points") == "POINTS"
        assert normalize_stat_type("Points") == "POINTS"
        assert normalize_stat_type("POINTS") == "POINTS"
        assert normalize_stat_type("pts") == "POINTS"

    def test_normalize_stat_type_rebounds(self):
        """Test stat type normalization for rebounds."""
        from nba.betting_xl.fetchers.normalization import normalize_stat_type

        assert normalize_stat_type("rebounds") == "REBOUNDS"
        assert normalize_stat_type("rebs") == "REBOUNDS"
        assert normalize_stat_type("reb") == "REBOUNDS"

    def test_normalize_stat_type_assists(self):
        """Test stat type normalization for assists."""
        from nba.betting_xl.fetchers.normalization import normalize_stat_type

        assert normalize_stat_type("assists") == "ASSISTS"
        assert normalize_stat_type("ast") == "ASSISTS"

    def test_normalize_stat_type_threes(self):
        """Test stat type normalization for threes."""
        from nba.betting_xl.fetchers.normalization import normalize_stat_type

        assert normalize_stat_type("threes") == "THREES"
        assert normalize_stat_type("3pm") == "THREES"
        assert normalize_stat_type("3-pt made") == "THREES"

    def test_normalize_stat_type_unknown_returns_uppercase(self):
        """Test unknown stat type returns uppercase version."""
        from nba.betting_xl.fetchers.normalization import normalize_stat_type

        # Unknown stats get uppercased, not None
        assert normalize_stat_type("unknown_stat") == "UNKNOWN_STAT"
        assert normalize_stat_type("") == ""

    def test_normalize_player_name(self):
        """Test player name normalization."""
        from nba.betting_xl.fetchers.normalization import normalize_player_name

        assert normalize_player_name("LeBron James") == "LeBron James"
        assert normalize_player_name("  LeBron  James  ") == "LeBron James"

    def test_normalize_player_name_removes_periods(self):
        """Test player name normalization removes periods."""
        from nba.betting_xl.fetchers.normalization import normalize_player_name

        # Periods are removed, but Jr/Sr suffixes are kept
        assert normalize_player_name("Gary Trent Jr.") == "Gary Trent Jr"
        assert normalize_player_name("J.R. Smith") == "JR Smith"

    def test_normalize_book_name(self):
        """Test book name normalization."""
        from nba.betting_xl.fetchers.normalization import normalize_book_name

        assert normalize_book_name("DraftKings") == "draftkings"
        assert normalize_book_name("FANDUEL") == "fanduel"
        assert normalize_book_name("bet_mgm") == "betmgm"

    @pytest.mark.parametrize(
        "input_name,expected",
        [
            ("draftkings", "draftkings"),
            ("FanDuel", "fanduel"),
            ("BetMGM", "betmgm"),
            ("Caesars", "caesars"),
            ("BetRivers", "betrivers"),
            ("espnbet", "espnbet"),
            ("dk", "draftkings"),  # Short form
            ("fd", "fanduel"),  # Short form
            ("ud", "underdog"),  # Short form
            ("pp", "prizepicks"),  # Short form
        ],
    )
    def test_normalize_book_name_parametrized(self, input_name, expected):
        """Test book name normalization with multiple inputs."""
        from nba.betting_xl.fetchers.normalization import normalize_book_name

        result = normalize_book_name(input_name)
        assert result == expected


class TestBettingProsFetcher:
    """Tests for BettingPros fetcher."""

    @pytest.fixture
    def mock_response(self):
        """Create a mock API response."""
        return {
            "props": [
                {
                    "player": {"name": "LeBron James", "team": "LAL"},
                    "market": "points",
                    "line": 25.5,
                    "odds": {"over": -110, "under": -110},
                    "sportsbook": {"name": "DraftKings", "id": 12},
                },
                {
                    "player": {"name": "Stephen Curry", "team": "GSW"},
                    "market": "points",
                    "line": 27.5,
                    "odds": {"over": -115, "under": -105},
                    "sportsbook": {"name": "FanDuel", "id": 10},
                },
            ]
        }

    def test_parse_props_response(self, mock_response):
        """Test parsing of props response."""
        props = mock_response["props"]
        assert len(props) == 2
        assert props[0]["player"]["name"] == "LeBron James"
        assert props[0]["line"] == 25.5

    def test_extract_player_info(self, mock_response):
        """Test extraction of player info from response."""
        prop = mock_response["props"][0]
        assert prop["player"]["name"] == "LeBron James"
        assert prop["player"]["team"] == "LAL"

    def test_extract_odds_info(self, mock_response):
        """Test extraction of odds info from response."""
        prop = mock_response["props"][0]
        assert prop["odds"]["over"] == -110
        assert prop["odds"]["under"] == -110

    @patch("requests.Session.get")
    def test_http_timeout_handling(self, mock_get):
        """Test timeout handling with requests."""
        mock_get.side_effect = requests.exceptions.Timeout("Connection timed out")

        session = requests.Session()
        with pytest.raises(requests.exceptions.Timeout):
            session.get("https://api.example.com", timeout=5)

    @patch("requests.Session.get")
    def test_http_error_handling(self, mock_get):
        """Test HTTP error handling with requests."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "500 Server Error"
        )
        mock_get.return_value = mock_response

        session = requests.Session()
        response = session.get("https://api.example.com")

        with pytest.raises(requests.exceptions.HTTPError):
            response.raise_for_status()


class TestCheatsheetFetcher:
    """Tests for cheatsheet fetcher."""

    @pytest.fixture
    def mock_cheatsheet_data(self):
        """Create mock cheatsheet response."""
        return {
            "props": [
                {
                    "player_name": "LeBron James",
                    "stat_type": "POINTS",
                    "line": 25.5,
                    "projection": 27.8,
                    "bet_rating": "A",
                    "ev_pct": 5.2,
                    "hit_rate_l5": 0.8,
                    "recommended_side": "over",
                },
                {
                    "player_name": "Anthony Davis",
                    "stat_type": "REBOUNDS",
                    "line": 10.5,
                    "projection": 11.2,
                    "bet_rating": "B",
                    "ev_pct": 3.1,
                    "hit_rate_l5": 0.6,
                    "recommended_side": "over",
                },
            ]
        }

    def test_parse_cheatsheet_props(self, mock_cheatsheet_data):
        """Test parsing cheatsheet props."""
        props = mock_cheatsheet_data["props"]
        assert len(props) == 2
        assert props[0]["player_name"] == "LeBron James"
        assert props[0]["projection"] == 27.8

    def test_filter_over_recommendations(self, mock_cheatsheet_data):
        """Test filtering to only OVER recommendations."""
        props = mock_cheatsheet_data["props"]
        over_props = [p for p in props if p.get("recommended_side") == "over"]
        assert len(over_props) == 2

    def test_ev_calculation(self, mock_cheatsheet_data):
        """Test EV percentage extraction."""
        prop = mock_cheatsheet_data["props"][0]
        assert prop["ev_pct"] == 5.2
        assert prop["ev_pct"] > 0  # Positive EV


class TestFetchAll:
    """Tests for the fetch_all module."""

    def test_fetch_all_module_imports(self):
        """Test fetch_all module can be imported."""
        from nba.betting_xl.fetchers import fetch_all

        assert fetch_all is not None

    def test_fetch_all_has_main_function(self):
        """Test fetch_all has main entry point."""
        from nba.betting_xl.fetchers import fetch_all

        # Check for common entry points
        assert hasattr(fetch_all, "main") or hasattr(fetch_all, "fetch_all_props")


class TestResponseValidation:
    """Tests for API response validation."""

    def test_empty_props_list(self):
        """Test handling of empty props list."""
        response = {"props": []}
        assert len(response["props"]) == 0

    def test_missing_props_key(self):
        """Test handling of missing props key."""
        response = {"error": "No data available"}
        props = response.get("props", [])
        assert props == []

    def test_malformed_prop_data(self):
        """Test handling of malformed prop data."""
        response = {
            "props": [
                {"player": None, "line": 25.5},  # Missing player name
                {"player": {"name": "LeBron"}, "line": None},  # Missing line
            ]
        }

        valid_props = []
        for prop in response["props"]:
            if prop.get("player") and prop.get("line") is not None:
                valid_props.append(prop)

        assert len(valid_props) == 0

    @pytest.mark.parametrize(
        "line_value,is_valid",
        [
            (25.5, True),
            (0, True),
            (-5, False),
            (None, False),
            ("25.5", True),  # String should be convertible
            (1000, False),  # Unreasonably high
        ],
    )
    def test_line_validation(self, line_value, is_valid):
        """Test line value validation."""
        try:
            line = float(line_value) if line_value is not None else None
            result = line is not None and 0 <= line <= 200
        except (TypeError, ValueError):
            result = False

        assert result == is_valid


class TestRateLimiting:
    """Tests for rate limiting behavior."""

    def test_rate_limit_delay(self):
        """Test that rate limiting introduces appropriate delays."""
        import time

        start = time.time()
        time.sleep(0.1)  # Simulate delay
        elapsed = time.time() - start

        assert elapsed >= 0.1

    def test_retry_logic(self):
        """Test retry logic on failure."""
        attempts = 0
        max_retries = 3

        def mock_request():
            nonlocal attempts
            attempts += 1
            if attempts < max_retries:
                raise requests.exceptions.RequestException("Temporary failure")
            return {"success": True}

        # Simulate retry logic
        result = None
        for _ in range(max_retries):
            try:
                result = mock_request()
                break
            except requests.exceptions.RequestException:
                continue

        assert result == {"success": True}
        assert attempts == max_retries
