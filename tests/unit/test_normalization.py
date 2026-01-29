"""
Unit Tests for Normalization Utilities
======================================
Tests for player name, stat type, and book name normalization functions.
"""

import pytest

from nba.betting_xl.fetchers.normalization import (
    calculate_consensus,
    calculate_line_spread,
    normalize_book_name,
    normalize_player_name,
    normalize_stat_type,
    parse_game_date,
    validate_line,
)


class TestNormalizePlayerName:
    """Tests for player name normalization."""

    def test_basic_name(self):
        """Test basic name passes through."""
        assert normalize_player_name("LeBron James") == "LeBron James"

    def test_uppercase_name(self):
        """Test uppercase name is converted to title case."""
        assert normalize_player_name("LEBRON JAMES") == "Lebron James"

    def test_lowercase_name(self):
        """Test lowercase name is converted to title case."""
        assert normalize_player_name("lebron james") == "Lebron James"

    def test_mixed_case_preserved(self):
        """Test mixed case names are preserved."""
        assert normalize_player_name("De'Aaron Fox") == "De'Aaron Fox"

    def test_whitespace_trimmed(self):
        """Test leading/trailing whitespace is trimmed."""
        assert normalize_player_name("  LeBron James  ") == "LeBron James"

    def test_extra_whitespace_collapsed(self):
        """Test extra internal whitespace is collapsed."""
        assert normalize_player_name("LeBron   James") == "LeBron James"

    def test_periods_removed(self):
        """Test periods are removed from names."""
        assert normalize_player_name("J.R. Smith") == "JR Smith"
        assert normalize_player_name("P.J. Washington") == "PJ Washington"

    def test_apostrophe_standardized(self):
        """Test different apostrophe characters are standardized."""
        assert normalize_player_name("De'Aaron Fox") == "De'Aaron Fox"
        assert normalize_player_name("De`Aaron Fox") == "De'Aaron Fox"

    def test_empty_string(self):
        """Test empty string returns empty."""
        assert normalize_player_name("") == ""

    def test_none_returns_empty(self):
        """Test None returns empty string."""
        assert normalize_player_name(None) == ""


class TestNormalizeStatType:
    """Tests for stat type normalization."""

    def test_points_variations(self):
        """Test various points representations."""
        assert normalize_stat_type("points") == "POINTS"
        assert normalize_stat_type("POINTS") == "POINTS"
        assert normalize_stat_type("pts") == "POINTS"
        assert normalize_stat_type("point") == "POINTS"

    def test_rebounds_variations(self):
        """Test various rebounds representations."""
        assert normalize_stat_type("rebounds") == "REBOUNDS"
        assert normalize_stat_type("rebound") == "REBOUNDS"
        assert normalize_stat_type("reb") == "REBOUNDS"
        assert normalize_stat_type("rebs") == "REBOUNDS"

    def test_assists_variations(self):
        """Test various assists representations."""
        assert normalize_stat_type("assists") == "ASSISTS"
        assert normalize_stat_type("assist") == "ASSISTS"
        assert normalize_stat_type("ast") == "ASSISTS"
        assert normalize_stat_type("asts") == "ASSISTS"

    def test_threes_variations(self):
        """Test various threes representations."""
        assert normalize_stat_type("threes") == "THREES"
        assert normalize_stat_type("three") == "THREES"
        assert normalize_stat_type("3-pt made") == "THREES"
        assert normalize_stat_type("3pm") == "THREES"
        assert normalize_stat_type("3 pointers made") == "THREES"

    def test_steals_variations(self):
        """Test steals variations."""
        assert normalize_stat_type("steals") == "STEALS"
        assert normalize_stat_type("stl") == "STEALS"

    def test_blocks_variations(self):
        """Test blocks variations."""
        assert normalize_stat_type("blocks") == "BLOCKS"
        assert normalize_stat_type("blk") == "BLOCKS"

    def test_unknown_stat_uppercased(self):
        """Test unknown stat types are uppercased."""
        assert normalize_stat_type("unknown") == "UNKNOWN"

    def test_whitespace_handled(self):
        """Test whitespace is handled."""
        assert normalize_stat_type("  points  ") == "POINTS"


class TestNormalizeBookName:
    """Tests for book name normalization."""

    def test_draftkings_variations(self):
        """Test DraftKings variations."""
        assert normalize_book_name("DraftKings") == "draftkings"
        assert normalize_book_name("draftkings") == "draftkings"
        assert normalize_book_name("dk") == "draftkings"
        assert normalize_book_name("Draft Kings") == "draftkings"

    def test_fanduel_variations(self):
        """Test FanDuel variations."""
        assert normalize_book_name("FanDuel") == "fanduel"
        assert normalize_book_name("fanduel") == "fanduel"
        assert normalize_book_name("fd") == "fanduel"

    def test_betmgm_variations(self):
        """Test BetMGM variations."""
        assert normalize_book_name("BetMGM") == "betmgm"
        assert normalize_book_name("Bet MGM") == "betmgm"
        assert normalize_book_name("mgm") == "betmgm"

    def test_caesars_variations(self):
        """Test Caesars variations."""
        assert normalize_book_name("Caesars") == "caesars"
        assert normalize_book_name("czr") == "caesars"
        assert normalize_book_name("cz") == "caesars"

    def test_betrivers_variations(self):
        """Test BetRivers variations."""
        assert normalize_book_name("BetRivers") == "betrivers"
        assert normalize_book_name("br") == "betrivers"

    def test_underdog_variations(self):
        """Test Underdog variations."""
        assert normalize_book_name("Underdog") == "underdog"
        assert normalize_book_name("ud") == "underdog"

    def test_prizepicks_variations(self):
        """Test PrizePicks variations."""
        assert normalize_book_name("Prize Picks") == "prizepicks"
        assert normalize_book_name("pp") == "prizepicks"


class TestParseGameDate:
    """Tests for game date parsing."""

    def test_iso_format(self):
        """Test ISO date format."""
        assert parse_game_date("2025-11-05") == "2025-11-05"

    def test_iso_with_time(self):
        """Test ISO format with time."""
        assert parse_game_date("2025-11-05T19:30:00Z") == "2025-11-05"

    def test_us_format(self):
        """Test US date format (MM/DD/YYYY)."""
        assert parse_game_date("11/5/2025") == "2025-11-05"

    def test_written_format(self):
        """Test written date format."""
        assert parse_game_date("Nov 5, 2025") == "2025-11-05"

    def test_compact_format(self):
        """Test compact format (YYYYMMDD)."""
        assert parse_game_date("20251105") == "2025-11-05"

    def test_empty_returns_none(self):
        """Test empty string returns None."""
        assert parse_game_date("") is None

    def test_none_returns_none(self):
        """Test None returns None."""
        assert parse_game_date(None) is None

    def test_invalid_returns_none(self):
        """Test invalid format returns None."""
        assert parse_game_date("not-a-date") is None


class TestValidateLine:
    """Tests for line validation."""

    def test_valid_float(self):
        """Test valid float line."""
        assert validate_line(25.5) is True

    def test_valid_int(self):
        """Test valid integer line."""
        assert validate_line(25) is True

    def test_valid_string_number(self):
        """Test valid string number."""
        assert validate_line("25.5") is True

    def test_zero_is_valid(self):
        """Test zero is valid."""
        assert validate_line(0) is True

    def test_none_is_invalid(self):
        """Test None is invalid."""
        assert validate_line(None) is False

    def test_negative_is_invalid(self):
        """Test negative is invalid."""
        assert validate_line(-5) is False

    def test_too_large_is_invalid(self):
        """Test value > 200 is invalid."""
        assert validate_line(250) is False

    def test_non_numeric_string_invalid(self):
        """Test non-numeric string is invalid."""
        assert validate_line("invalid") is False


class TestCalculateConsensus:
    """Tests for consensus calculation."""

    def test_single_line(self):
        """Test single line returns itself."""
        assert calculate_consensus([25.5]) == 25.5

    def test_odd_count_returns_median(self):
        """Test odd count returns median."""
        assert calculate_consensus([25.5, 26.0, 26.5]) == 26.0

    def test_even_count_returns_average_of_middle(self):
        """Test even count returns average of middle two."""
        assert calculate_consensus([25.5, 26.0, 26.5, 27.0]) == 26.25

    def test_empty_returns_none(self):
        """Test empty list returns None."""
        assert calculate_consensus([]) is None

    def test_unsorted_input(self):
        """Test unsorted input still works."""
        assert calculate_consensus([27.0, 25.5, 26.5, 26.0]) == 26.25


class TestCalculateLineSpread:
    """Tests for line spread calculation."""

    def test_basic_spread(self):
        """Test basic spread calculation."""
        assert calculate_line_spread([25.5, 26.5, 27.5]) == 2.0

    def test_single_line_returns_zero(self):
        """Test single line returns zero."""
        assert calculate_line_spread([25.5]) == 0.0

    def test_empty_returns_zero(self):
        """Test empty list returns zero."""
        assert calculate_line_spread([]) == 0.0

    def test_identical_lines_return_zero(self):
        """Test identical lines return zero spread."""
        assert calculate_line_spread([25.5, 25.5, 25.5]) == 0.0
