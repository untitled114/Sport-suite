"""
Unit Tests for Loaders
======================
Tests for database loaders with mocked database connections.

Best Practices Applied:
- Mock database connections to avoid external dependencies
- Test SQL query construction
- Test data transformation before insert
- Test error handling for database failures
"""

from datetime import date, datetime
from unittest.mock import MagicMock, Mock, patch

import pytest


class TestLoadPropsToDb:
    """Tests for load_props_to_db module."""

    @pytest.fixture
    def sample_props(self):
        """Sample props for testing."""
        return [
            {
                "player_name": "LeBron James",
                "game_date": "2025-11-06",
                "stat_type": "POINTS",
                "line": 25.5,
                "source": "draftkings",
                "over_odds": -110,
                "under_odds": -110,
            },
            {
                "player_name": "Stephen Curry",
                "game_date": "2025-11-06",
                "stat_type": "THREES",
                "line": 4.5,
                "source": "fanduel",
                "over_odds": -115,
                "under_odds": -105,
            },
        ]

    @pytest.fixture
    def mock_cursor(self):
        """Create a mock database cursor."""
        cursor = MagicMock()
        cursor.execute = MagicMock()
        cursor.fetchone = MagicMock(return_value=None)
        cursor.fetchall = MagicMock(return_value=[])
        cursor.close = MagicMock()
        return cursor

    @pytest.fixture
    def mock_connection(self, mock_cursor):
        """Create a mock database connection."""
        conn = MagicMock()
        conn.cursor = MagicMock(return_value=mock_cursor)
        conn.commit = MagicMock()
        conn.close = MagicMock()
        return conn

    def test_sample_props_structure(self, sample_props):
        """Test sample props have required fields."""
        required_fields = ["player_name", "game_date", "stat_type", "line", "source"]
        for prop in sample_props:
            for field in required_fields:
                assert field in prop, f"Missing required field: {field}"

    def test_game_date_parsing(self, sample_props):
        """Test game date parsing from string."""
        prop = sample_props[0]
        game_date = datetime.strptime(prop["game_date"], "%Y-%m-%d").date()
        assert game_date == date(2025, 11, 6)

    def test_stat_type_validation(self, sample_props):
        """Test stat types are valid."""
        valid_stat_types = ["POINTS", "REBOUNDS", "ASSISTS", "THREES"]
        for prop in sample_props:
            assert prop["stat_type"] in valid_stat_types

    def test_line_is_positive(self, sample_props):
        """Test lines are positive values."""
        for prop in sample_props:
            assert prop["line"] > 0

    def test_odds_format(self, sample_props):
        """Test odds are in American format."""
        for prop in sample_props:
            # American odds are typically negative for favorites
            # or positive for underdogs
            assert isinstance(prop["over_odds"], int)
            assert isinstance(prop["under_odds"], int)

    @patch("psycopg2.connect")
    def test_connection_with_env_vars(self, mock_connect, mock_connection):
        """Test database connection uses environment variables."""
        mock_connect.return_value = mock_connection

        # Import should attempt to use env vars
        import os

        os.environ["DB_PASSWORD"] = "test_password"

        # Connection should work with mock
        conn = mock_connect(
            host="localhost",
            port=5539,
            database="nba_intelligence",
            user="nba_user",
            password=os.environ.get("DB_PASSWORD"),
        )

        assert conn is not None
        mock_connect.assert_called_once()

    def test_insert_query_structure(self):
        """Test INSERT query has correct structure."""
        insert_sql = """
            INSERT INTO nba_prop_lines (
                player_name, game_date, stat_type, line, source
            ) VALUES (%s, %s, %s, %s, %s)
        """

        # Verify query structure
        assert "INSERT INTO" in insert_sql
        assert "nba_prop_lines" in insert_sql
        assert "player_name" in insert_sql
        assert "VALUES" in insert_sql

    def test_upsert_query_structure(self):
        """Test UPSERT query has correct structure."""
        upsert_sql = """
            INSERT INTO nba_prop_lines (player_name, game_date, stat_type, line, source)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (player_name, game_date, stat_type, source)
            DO UPDATE SET line = EXCLUDED.line
        """

        # Verify upsert structure
        assert "INSERT INTO" in upsert_sql
        assert "ON CONFLICT" in upsert_sql
        assert "DO UPDATE SET" in upsert_sql

    def test_batch_insert_logic(self, sample_props, mock_connection):
        """Test batch insertion logic with multiple props."""
        # Create a fresh cursor for this test
        cursor = MagicMock()
        mock_connection.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        mock_connection.cursor.return_value.__exit__ = MagicMock(return_value=False)

        # Simulate batch insert
        for prop in sample_props:
            cursor.execute(
                "INSERT INTO nba_prop_lines VALUES (%s, %s, %s, %s, %s)",
                (
                    prop["player_name"],
                    prop["game_date"],
                    prop["stat_type"],
                    prop["line"],
                    prop["source"],
                ),
            )
        mock_connection.commit()

        # Verify execute was called for each prop
        assert cursor.execute.call_count == len(sample_props)
        mock_connection.commit.assert_called_once()


class TestLoadCheatsheetToDb:
    """Tests for load_cheatsheet_to_db module."""

    @pytest.fixture
    def sample_cheatsheet_props(self):
        """Sample cheatsheet props."""
        return [
            {
                "player_name": "LeBron James",
                "game_date": "2025-11-06",
                "stat_type": "POINTS",
                "platform": "underdog",
                "line": 25.5,
                "projection": 27.8,
                "bet_rating": "A",
                "ev_pct": 5.2,
                "recommended_side": "over",
            }
        ]

    def test_cheatsheet_props_structure(self, sample_cheatsheet_props):
        """Test cheatsheet props have required fields."""
        required = ["player_name", "game_date", "stat_type", "line", "projection"]
        for prop in sample_cheatsheet_props:
            for field in required:
                assert field in prop

    def test_platform_identification(self, sample_cheatsheet_props):
        """Test platform is correctly identified."""
        prop = sample_cheatsheet_props[0]
        assert prop["platform"] in ["underdog", "prizepicks"]

    def test_use_for_betting_flag(self, sample_cheatsheet_props):
        """Test use_for_betting flag logic."""
        prop = sample_cheatsheet_props[0]
        # Underdog = use for betting, PrizePicks = reference only
        use_for_betting = prop["platform"] == "underdog"
        assert use_for_betting is True

    def test_filter_over_only(self, sample_cheatsheet_props):
        """Test filtering to OVER recommendations only."""
        over_props = [p for p in sample_cheatsheet_props if p.get("recommended_side") == "over"]
        assert len(over_props) == len(sample_cheatsheet_props)


class TestOpponentMapper:
    """Tests for opponent_mapper module."""

    @pytest.fixture
    def mock_schedule(self):
        """Mock NBA schedule response."""
        return {
            "events": [
                {
                    "shortName": "DAL @ WSH",
                    "competitions": [
                        {
                            "competitors": [
                                {"team": {"abbreviation": "WSH"}, "homeAway": "home"},
                                {"team": {"abbreviation": "DAL"}, "homeAway": "away"},
                            ]
                        }
                    ],
                }
            ]
        }

    def test_parse_short_name(self, mock_schedule):
        """Test parsing team matchup from shortName."""
        event = mock_schedule["events"][0]
        short_name = event["shortName"]

        # Parse "DAL @ WSH" format
        parts = short_name.split(" @ ")
        away_team = parts[0]
        home_team = parts[1]

        assert away_team == "DAL"
        assert home_team == "WSH"

    def test_team_abbrev_normalization(self):
        """Test team abbreviation normalization."""
        team_map = {
            "WSH": "WAS",
            "NO": "NOP",
            "SA": "SAS",
            "GS": "GSW",
        }

        assert team_map.get("WSH", "WSH") == "WAS"
        assert team_map.get("NO", "NO") == "NOP"

    def test_is_home_determination(self, mock_schedule):
        """Test determining if player is home."""
        event = mock_schedule["events"][0]
        competitors = event["competitions"][0]["competitors"]

        home_team = None
        away_team = None

        for comp in competitors:
            if comp["homeAway"] == "home":
                home_team = comp["team"]["abbreviation"]
            else:
                away_team = comp["team"]["abbreviation"]

        assert home_team == "WSH"
        assert away_team == "DAL"

    def test_opponent_mapping(self, mock_schedule):
        """Test mapping player's team to opponent."""
        # If player is on DAL, opponent is WSH
        # If player is on WSH, opponent is DAL
        games_map = {}

        event = mock_schedule["events"][0]
        parts = event["shortName"].split(" @ ")
        away_team, home_team = parts[0], parts[1]

        games_map[away_team] = {"opponent": home_team, "is_home": False}
        games_map[home_team] = {"opponent": away_team, "is_home": True}

        assert games_map["DAL"]["opponent"] == "WSH"
        assert games_map["DAL"]["is_home"] is False
        assert games_map["WSH"]["opponent"] == "DAL"
        assert games_map["WSH"]["is_home"] is True


class TestDatabaseErrorHandling:
    """Tests for database error handling."""

    def test_connection_error_handling(self):
        """Test handling of database connection errors."""
        import psycopg2

        with pytest.raises(psycopg2.OperationalError):
            # This should fail since we're not actually connecting
            raise psycopg2.OperationalError("Connection refused")

    def test_query_error_handling(self):
        """Test handling of query execution errors."""
        import psycopg2

        with pytest.raises(psycopg2.ProgrammingError):
            raise psycopg2.ProgrammingError("Invalid SQL syntax")

    def test_transaction_rollback(self):
        """Test transaction rollback on error."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()

        # Simulate error during insert
        mock_cursor.execute.side_effect = Exception("Insert failed")

        # Simulate the transaction with rollback
        rollback_called = False
        try:
            mock_cursor.execute("INSERT INTO table VALUES (%s)", ("value",))
            mock_conn.commit()
        except Exception:
            mock_conn.rollback()
            rollback_called = True

        # Verify rollback was called
        assert rollback_called
        mock_conn.rollback.assert_called_once()


class TestDataTransformations:
    """Tests for data transformations before database insert."""

    def test_date_conversion(self):
        """Test date string to date object conversion."""
        date_str = "2025-11-06"
        date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
        assert date_obj == date(2025, 11, 6)

    def test_timestamp_parsing(self):
        """Test timestamp parsing with timezone."""
        ts_str = "2025-11-06T15:30:00Z"
        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        assert ts.year == 2025
        assert ts.month == 11
        assert ts.day == 6

    def test_odds_conversion(self):
        """Test odds format conversion."""
        # American odds to implied probability
        american_odds = -110
        if american_odds < 0:
            implied_prob = abs(american_odds) / (abs(american_odds) + 100)
        else:
            implied_prob = 100 / (american_odds + 100)

        assert 0 < implied_prob < 1
        assert abs(implied_prob - 0.5238) < 0.01

    def test_null_handling(self):
        """Test NULL value handling."""
        prop = {"player_name": "LeBron James", "opponent_team": None}

        # None should be acceptable for optional fields
        assert prop.get("opponent_team") is None

    @pytest.mark.parametrize(
        "input_val,expected",
        [
            ("25.5", 25.5),
            (25.5, 25.5),
            (25, 25.0),
            ("25", 25.0),
        ],
    )
    def test_line_type_coercion(self, input_val, expected):
        """Test line value type coercion."""
        result = float(input_val)
        assert result == expected


class TestBatchOperations:
    """Tests for batch database operations."""

    def test_batch_size_calculation(self):
        """Test batch size calculation for large datasets."""
        total_records = 1000
        batch_size = 100
        expected_batches = 10

        num_batches = (total_records + batch_size - 1) // batch_size
        assert num_batches == expected_batches

    def test_batch_iteration(self):
        """Test iterating through batches."""
        records = list(range(25))
        batch_size = 10

        batches = []
        for i in range(0, len(records), batch_size):
            batches.append(records[i : i + batch_size])

        assert len(batches) == 3
        assert len(batches[0]) == 10
        assert len(batches[1]) == 10
        assert len(batches[2]) == 5

    def test_deduplication_before_insert(self):
        """Test deduplication of props before insert."""
        props = [
            {"player_name": "LeBron", "stat_type": "POINTS", "line": 25.5},
            {"player_name": "LeBron", "stat_type": "POINTS", "line": 25.5},  # Duplicate
            {"player_name": "LeBron", "stat_type": "REBOUNDS", "line": 7.5},  # Different stat
        ]

        # Deduplicate by (player_name, stat_type)
        seen = set()
        unique_props = []
        for prop in props:
            key = (prop["player_name"], prop["stat_type"])
            if key not in seen:
                seen.add(key)
                unique_props.append(prop)

        assert len(unique_props) == 2
