#!/usr/bin/env python3
"""
Unit tests for nba.core.data_quality_checks module.

Tests the CheckResult dataclass and DataQualityChecker class for
data quality validation.
"""

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pytest

from nba.core.data_quality_checks import CheckResult, DataQualityChecker

# =============================================================================
# CheckResult Dataclass Tests
# =============================================================================


class TestCheckResult:
    """Tests for CheckResult dataclass."""

    def test_default_values(self):
        """Test default initialization."""
        result = CheckResult(name="test_check", passed=True, message="All good")
        assert result.name == "test_check"
        assert result.passed is True
        assert result.message == "All good"
        assert result.details is None

    def test_with_details(self):
        """Test initialization with details."""
        details = {"count": 100, "threshold": 50}
        result = CheckResult(
            name="volume_check",
            passed=True,
            message="100 samples found",
            details=details,
        )
        assert result.details == details
        assert result.details["count"] == 100

    def test_str_passed(self):
        """Test string representation for passed check."""
        result = CheckResult(name="test_check", passed=True, message="Success")
        result_str = str(result)
        assert "test_check" in result_str
        assert "Success" in result_str
        # Should contain checkmark emoji
        assert "\u2705" in result_str

    def test_str_failed(self):
        """Test string representation for failed check."""
        result = CheckResult(name="test_check", passed=False, message="Failed")
        result_str = str(result)
        assert "test_check" in result_str
        assert "Failed" in result_str
        # Should contain X emoji
        assert "\u274c" in result_str


# =============================================================================
# DataQualityChecker Tests
# =============================================================================


class TestDataQualityCheckerInit:
    """Tests for DataQualityChecker initialization."""

    def test_init_defaults(self):
        """Test default initialization."""
        checker = DataQualityChecker()
        assert checker.results == []
        assert checker.conn_players is None
        assert checker.conn_games is None
        assert checker.conn_intel is None


class TestDataQualityCheckerAddResult:
    """Tests for DataQualityChecker.add_result method."""

    def test_add_result(self):
        """Test adding a result."""
        checker = DataQualityChecker()
        result = CheckResult(name="test", passed=True, message="OK")
        checker.add_result(result)
        assert len(checker.results) == 1
        assert checker.results[0] == result

    def test_add_multiple_results(self):
        """Test adding multiple results."""
        checker = DataQualityChecker()
        checker.add_result(CheckResult(name="test1", passed=True, message="OK"))
        checker.add_result(CheckResult(name="test2", passed=False, message="Failed"))
        assert len(checker.results) == 2


class TestDataQualityCheckerSummarize:
    """Tests for DataQualityChecker.summarize method."""

    def test_all_passed(self):
        """Test summary when all checks pass."""
        checker = DataQualityChecker()
        checker.add_result(CheckResult(name="test1", passed=True, message="OK"))
        checker.add_result(CheckResult(name="test2", passed=True, message="OK"))
        result = checker.summarize()
        assert result is True

    def test_some_failed(self):
        """Test summary when some checks fail."""
        checker = DataQualityChecker()
        checker.add_result(CheckResult(name="test1", passed=True, message="OK"))
        checker.add_result(CheckResult(name="test2", passed=False, message="Failed"))
        result = checker.summarize()
        assert result is False

    def test_all_failed(self):
        """Test summary when all checks fail."""
        checker = DataQualityChecker()
        checker.add_result(CheckResult(name="test1", passed=False, message="Failed"))
        checker.add_result(CheckResult(name="test2", passed=False, message="Failed"))
        result = checker.summarize()
        assert result is False


# =============================================================================
# Individual Check Method Tests (with mocked database)
# =============================================================================


class TestCheckPlayerGameLogsFreshness:
    """Tests for check_player_game_logs_freshness."""

    def test_fresh_logs(self):
        """Test with fresh game logs."""
        checker = DataQualityChecker()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (date.today() - timedelta(days=1), 500)
        mock_conn.cursor.return_value = mock_cursor
        checker.conn_players = mock_conn

        result = checker.check_player_game_logs_freshness()
        assert result.passed is True
        assert "1 days old" in result.message

    def test_stale_logs(self):
        """Test with stale game logs."""
        checker = DataQualityChecker()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (date.today() - timedelta(days=5), 500)
        mock_conn.cursor.return_value = mock_cursor
        checker.conn_players = mock_conn

        result = checker.check_player_game_logs_freshness()
        assert result.passed is False
        assert "5 days old" in result.message

    def test_no_logs(self):
        """Test with no recent game logs."""
        checker = DataQualityChecker()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (None, 0)
        mock_conn.cursor.return_value = mock_cursor
        checker.conn_players = mock_conn

        result = checker.check_player_game_logs_freshness()
        assert result.passed is False
        assert "No recent game logs" in result.message


class TestCheckPlayerRollingStatsCoverage:
    """Tests for check_player_rolling_stats_coverage."""

    def test_good_coverage(self):
        """Test with good rolling stats coverage."""
        checker = DataQualityChecker()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (100, 90)  # 90% coverage
        mock_conn.cursor.return_value = mock_cursor
        checker.conn_players = mock_conn

        result = checker.check_player_rolling_stats_coverage()
        assert result.passed is True
        assert "90/100" in result.message

    def test_poor_coverage(self):
        """Test with poor rolling stats coverage."""
        checker = DataQualityChecker()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (100, 50)  # 50% coverage
        mock_conn.cursor.return_value = mock_cursor
        checker.conn_players = mock_conn

        result = checker.check_player_rolling_stats_coverage()
        assert result.passed is False


class TestCheckNoNullCriticalFields:
    """Tests for check_no_null_critical_fields."""

    def test_no_nulls(self):
        """Test with no null values."""
        checker = DataQualityChecker()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (0, 0, 0, 10000)
        mock_conn.cursor.return_value = mock_cursor
        checker.conn_players = mock_conn

        result = checker.check_no_null_critical_fields()
        assert result.passed is True
        assert "Null counts" in result.message

    def test_has_nulls(self):
        """Test with null values present."""
        checker = DataQualityChecker()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (5, 10, 0, 10000)
        mock_conn.cursor.return_value = mock_cursor
        checker.conn_players = mock_conn

        result = checker.check_no_null_critical_fields()
        assert result.passed is False
        assert "points: 5" in result.message


class TestCheckPropsFreshness:
    """Tests for check_props_freshness."""

    def test_fresh_props(self):
        """Test with fresh props data."""
        checker = DataQualityChecker()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (date.today(), 200)
        mock_conn.cursor.return_value = mock_cursor
        checker.conn_intel = mock_conn

        result = checker.check_props_freshness()
        assert result.passed is True
        assert "200 props" in result.message

    def test_stale_props(self):
        """Test with stale props data."""
        checker = DataQualityChecker()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (date.today() - timedelta(days=3), 30)
        mock_conn.cursor.return_value = mock_cursor
        checker.conn_intel = mock_conn

        result = checker.check_props_freshness()
        assert result.passed is False

    def test_no_props(self):
        """Test with no props data."""
        checker = DataQualityChecker()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (None, 0)
        mock_conn.cursor.return_value = mock_cursor
        checker.conn_intel = mock_conn

        result = checker.check_props_freshness()
        assert result.passed is False
        assert "No props found" in result.message


class TestCheckPropsMultiBookCoverage:
    """Tests for check_props_multi_book_coverage."""

    def test_good_multi_book(self):
        """Test with good multi-book coverage."""
        checker = DataQualityChecker()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (4.5, 0.85)  # 85% have 3+ books
        mock_conn.cursor.return_value = mock_cursor
        checker.conn_intel = mock_conn

        result = checker.check_props_multi_book_coverage()
        assert result.passed is True
        assert "4.5 books/prop" in result.message

    def test_poor_multi_book(self):
        """Test with poor multi-book coverage."""
        checker = DataQualityChecker()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (2.0, 0.4)  # Only 40% have 3+ books
        mock_conn.cursor.return_value = mock_cursor
        checker.conn_intel = mock_conn

        result = checker.check_props_multi_book_coverage()
        assert result.passed is False


class TestCheckPropsStatTypeDistribution:
    """Tests for check_props_stat_type_distribution."""

    def test_all_stat_types(self):
        """Test with all stat types present."""
        checker = DataQualityChecker()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            ("POINTS", 1000),
            ("REBOUNDS", 900),
            ("ASSISTS", 800),
            ("THREES", 700),
        ]
        mock_conn.cursor.return_value = mock_cursor
        checker.conn_intel = mock_conn

        result = checker.check_props_stat_type_distribution()
        assert result.passed is True

    def test_missing_stat_types(self):
        """Test with missing stat types."""
        checker = DataQualityChecker()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            ("POINTS", 1000),
            ("REBOUNDS", 900),
            # Missing ASSISTS and THREES
        ]
        mock_conn.cursor.return_value = mock_cursor
        checker.conn_intel = mock_conn

        result = checker.check_props_stat_type_distribution()
        assert result.passed is False


class TestCheckTrainingDataVolume:
    """Tests for check_training_data_volume."""

    def test_sufficient_volume(self):
        """Test with sufficient training data."""
        checker = DataQualityChecker()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            ("POINTS", 25000),
            ("REBOUNDS", 24000),
            ("ASSISTS", 22000),
            ("THREES", 21000),
        ]
        mock_conn.cursor.return_value = mock_cursor
        checker.conn_intel = mock_conn

        result = checker.check_training_data_volume()
        assert result.passed is True

    def test_insufficient_main_markets(self):
        """Test with insufficient data for main markets."""
        checker = DataQualityChecker()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            ("POINTS", 10000),  # Below 20000 threshold
            ("REBOUNDS", 24000),
        ]
        mock_conn.cursor.return_value = mock_cursor
        checker.conn_intel = mock_conn

        result = checker.check_training_data_volume()
        assert result.passed is False


class TestCheckHomeAwayBalance:
    """Tests for check_home_away_balance."""

    def test_balanced(self):
        """Test with balanced home/away distribution."""
        checker = DataQualityChecker()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (0.52,)  # 52% home
        mock_conn.cursor.return_value = mock_cursor
        checker.conn_players = mock_conn

        result = checker.check_home_away_balance()
        assert result.passed is True
        assert "52.0%" in result.message

    def test_imbalanced_too_many_home(self):
        """Test with too many home games."""
        checker = DataQualityChecker()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (0.75,)  # 75% home
        mock_conn.cursor.return_value = mock_cursor
        checker.conn_players = mock_conn

        result = checker.check_home_away_balance()
        assert result.passed is False

    def test_imbalanced_too_few_home(self):
        """Test with too few home games."""
        checker = DataQualityChecker()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (0.25,)  # 25% home
        mock_conn.cursor.return_value = mock_cursor
        checker.conn_players = mock_conn

        result = checker.check_home_away_balance()
        assert result.passed is False


class TestCheckNoFutureDataLeakage:
    """Tests for check_no_future_data_leakage."""

    def test_no_future_data(self):
        """Test with no future data."""
        checker = DataQualityChecker()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (0,)
        mock_conn.cursor.return_value = mock_cursor
        checker.conn_intel = mock_conn

        result = checker.check_no_future_data_leakage()
        assert result.passed is True
        assert "0" in result.message

    def test_has_future_data(self):
        """Test with future data present."""
        checker = DataQualityChecker()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (15,)
        mock_conn.cursor.return_value = mock_cursor
        checker.conn_intel = mock_conn

        result = checker.check_no_future_data_leakage()
        assert result.passed is False
        assert "15" in result.message


# =============================================================================
# Run Check Suite Tests
# =============================================================================


class TestRunPreTrainingChecks:
    """Tests for run_pre_training_checks."""

    @patch.object(DataQualityChecker, "check_player_game_logs_freshness")
    @patch.object(DataQualityChecker, "check_player_rolling_stats_coverage")
    @patch.object(DataQualityChecker, "check_no_null_critical_fields")
    @patch.object(DataQualityChecker, "check_training_data_volume")
    @patch.object(DataQualityChecker, "check_home_away_balance")
    @patch.object(DataQualityChecker, "check_no_future_data_leakage")
    def test_all_pass(self, mock6, mock5, mock4, mock3, mock2, mock1):
        """Test when all pre-training checks pass."""
        for mock in [mock1, mock2, mock3, mock4, mock5, mock6]:
            mock.return_value = CheckResult(name="test", passed=True, message="OK")

        checker = DataQualityChecker()
        result = checker.run_pre_training_checks()
        assert result is True

    @patch.object(DataQualityChecker, "check_player_game_logs_freshness")
    @patch.object(DataQualityChecker, "check_player_rolling_stats_coverage")
    @patch.object(DataQualityChecker, "check_no_null_critical_fields")
    @patch.object(DataQualityChecker, "check_training_data_volume")
    @patch.object(DataQualityChecker, "check_home_away_balance")
    @patch.object(DataQualityChecker, "check_no_future_data_leakage")
    def test_some_fail(self, mock6, mock5, mock4, mock3, mock2, mock1):
        """Test when some pre-training checks fail."""
        mock1.return_value = CheckResult(name="test1", passed=True, message="OK")
        mock2.return_value = CheckResult(name="test2", passed=False, message="Failed")
        mock3.return_value = CheckResult(name="test3", passed=True, message="OK")
        mock4.return_value = CheckResult(name="test4", passed=True, message="OK")
        mock5.return_value = CheckResult(name="test5", passed=True, message="OK")
        mock6.return_value = CheckResult(name="test6", passed=True, message="OK")

        checker = DataQualityChecker()
        result = checker.run_pre_training_checks()
        assert result is False


class TestRunDailyChecks:
    """Tests for run_daily_checks."""

    @patch.object(DataQualityChecker, "check_props_freshness")
    @patch.object(DataQualityChecker, "check_props_multi_book_coverage")
    @patch.object(DataQualityChecker, "check_props_stat_type_distribution")
    @patch.object(DataQualityChecker, "check_player_game_logs_freshness")
    def test_all_pass(self, mock4, mock3, mock2, mock1):
        """Test when all daily checks pass."""
        for mock in [mock1, mock2, mock3, mock4]:
            mock.return_value = CheckResult(name="test", passed=True, message="OK")

        checker = DataQualityChecker()
        result = checker.run_daily_checks()
        assert result is True

    @patch.object(DataQualityChecker, "check_props_freshness")
    @patch.object(DataQualityChecker, "check_props_multi_book_coverage")
    @patch.object(DataQualityChecker, "check_props_stat_type_distribution")
    @patch.object(DataQualityChecker, "check_player_game_logs_freshness")
    def test_some_fail(self, mock4, mock3, mock2, mock1):
        """Test when some daily checks fail."""
        mock1.return_value = CheckResult(name="test1", passed=False, message="Failed")
        mock2.return_value = CheckResult(name="test2", passed=True, message="OK")
        mock3.return_value = CheckResult(name="test3", passed=True, message="OK")
        mock4.return_value = CheckResult(name="test4", passed=True, message="OK")

        checker = DataQualityChecker()
        result = checker.run_daily_checks()
        assert result is False


# =============================================================================
# Coverage: connect() and close() (lines 64-66, 71-73)
# =============================================================================


class TestConnectAndClose:
    """Tests for connect() and close() methods."""

    @patch("nba.core.data_quality_checks.get_intelligence_db_config")
    @patch("nba.core.data_quality_checks.get_games_db_config")
    @patch("nba.core.data_quality_checks.get_players_db_config")
    @patch("nba.core.data_quality_checks.psycopg2.connect")
    def test_connect_creates_all_connections(
        self, mock_connect, mock_players_cfg, mock_games_cfg, mock_intel_cfg
    ):
        """Lines 64-66: connect() calls psycopg2.connect three times."""
        mock_players_cfg.return_value = {"host": "localhost", "port": 5536}
        mock_games_cfg.return_value = {"host": "localhost", "port": 5537}
        mock_intel_cfg.return_value = {"host": "localhost", "port": 5539}
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        checker = DataQualityChecker()
        checker.connect()
        assert mock_connect.call_count == 3
        assert checker.conn_players is mock_conn
        assert checker.conn_games is mock_conn
        assert checker.conn_intel is mock_conn

    def test_close_closes_all_connections(self):
        """Lines 71-73: close() calls conn.close() on each connection."""
        checker = DataQualityChecker()
        mock_p = MagicMock()
        mock_g = MagicMock()
        mock_i = MagicMock()
        checker.conn_players = mock_p
        checker.conn_games = mock_g
        checker.conn_intel = mock_i

        checker.close()
        mock_p.close.assert_called_once()
        mock_g.close.assert_called_once()
        mock_i.close.assert_called_once()

    def test_close_handles_none_connections(self):
        """close() should not raise when connections are None."""
        checker = DataQualityChecker()
        checker.close()  # Should not raise


# =============================================================================
# Coverage: exception handlers in run_pre_training_checks (lines 376-377)
# and run_daily_checks (lines 398-399)
# =============================================================================


class TestRunChecksExceptionHandling:
    """Tests for exception handling in run checks."""

    def test_pre_training_check_exception(self):
        """Lines 376-377: check that raises exception gets caught."""
        import psycopg2

        checker = DataQualityChecker()

        def failing_freshness():
            raise psycopg2.Error("connection lost")

        failing_freshness.__name__ = "check_player_game_logs_freshness"
        ok_result = CheckResult(name="test", passed=True, message="OK")

        with (
            patch.object(
                checker,
                "check_player_game_logs_freshness",
                failing_freshness,
            ),
            patch.object(
                checker,
                "check_player_rolling_stats_coverage",
                return_value=ok_result,
            ),
            patch.object(
                checker,
                "check_no_null_critical_fields",
                return_value=ok_result,
            ),
            patch.object(
                checker,
                "check_training_data_volume",
                return_value=ok_result,
            ),
            patch.object(
                checker,
                "check_home_away_balance",
                return_value=ok_result,
            ),
            patch.object(
                checker,
                "check_no_future_data_leakage",
                return_value=ok_result,
            ),
        ):
            result = checker.run_pre_training_checks()

        # Should have a failed result from the caught exception
        assert result is False
        failed = [r for r in checker.results if not r.passed]
        assert len(failed) == 1
        assert "connection lost" in failed[0].message

    def test_daily_check_exception(self):
        """Lines 398-399: daily check that raises exception gets caught."""
        checker = DataQualityChecker()

        def failing_props():
            raise TypeError("type error in check")

        failing_props.__name__ = "check_props_freshness"
        ok_result = CheckResult(name="test", passed=True, message="OK")

        with (
            patch.object(
                checker,
                "check_props_freshness",
                failing_props,
            ),
            patch.object(
                checker,
                "check_props_multi_book_coverage",
                return_value=ok_result,
            ),
            patch.object(
                checker,
                "check_props_stat_type_distribution",
                return_value=ok_result,
            ),
            patch.object(
                checker,
                "check_player_game_logs_freshness",
                return_value=ok_result,
            ),
        ):
            result = checker.run_daily_checks()

        assert result is False
        failed = [r for r in checker.results if not r.passed]
        assert len(failed) == 1
        assert "type error in check" in failed[0].message
