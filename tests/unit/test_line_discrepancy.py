"""
Unit tests for nba.betting_xl.analysis.line_discrepancy

Tests cover:
- _strip_direct_suffix helper
- DIRECT_TO_BP_BOOK_MAP / BP_TO_DIRECT_BOOK_MAP constants
- DiscrepancyAnalyzer.compute_discrepancies (mocked DB)
- DiscrepancyAnalyzer.flag_stale_lines (mocked DB, including error paths)
- DiscrepancyAnalyzer.generate_daily_report (mocked DB)
- DiscrepancyAnalyzer.compute_cross_source_consensus (mocked DB)
- DiscrepancyAnalyzer.close()
- DiscrepancyAnalyzer._print_report
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from nba.betting_xl.analysis.line_discrepancy import (
    BP_TO_DIRECT_BOOK_MAP,
    DIRECT_TO_BP_BOOK_MAP,
    DiscrepancyAnalyzer,
    _strip_direct_suffix,
)

# ─────────────────────────────────────────────────────────────────
# _strip_direct_suffix
# ─────────────────────────────────────────────────────────────────


class TestStripDirectSuffix:
    def test_strips_direct_suffix(self):
        assert _strip_direct_suffix("draftkings_direct") == "draftkings"

    def test_no_suffix_unchanged(self):
        assert _strip_direct_suffix("draftkings") == "draftkings"

    def test_empty_string(self):
        assert _strip_direct_suffix("") == ""

    def test_only_direct(self):
        assert _strip_direct_suffix("_direct") == ""

    def test_multiple_direct_strips_only_last(self):
        assert _strip_direct_suffix("test_direct_direct") == "test_direct"


# ─────────────────────────────────────────────────────────────────
# Book mapping constants
# ─────────────────────────────────────────────────────────────────


class TestBookMappings:
    def test_direct_to_bp_map_has_entries(self):
        assert len(DIRECT_TO_BP_BOOK_MAP) >= 8

    def test_bp_to_direct_is_reverse(self):
        for direct, bp in DIRECT_TO_BP_BOOK_MAP.items():
            assert BP_TO_DIRECT_BOOK_MAP[bp] == direct

    def test_draftkings_mapping(self):
        assert DIRECT_TO_BP_BOOK_MAP["draftkings_direct"] == "draftkings"
        assert BP_TO_DIRECT_BOOK_MAP["draftkings"] == "draftkings_direct"


# ─────────────────────────────────────────────────────────────────
# Helper: build mock analyzer with mocked connection
# ─────────────────────────────────────────────────────────────────


def _make_analyzer(verbose=False):
    """Create a DiscrepancyAnalyzer with a mocked DB connection."""
    analyzer = DiscrepancyAnalyzer(verbose=verbose)
    mock_conn = MagicMock()
    mock_conn.closed = False
    analyzer._conn = mock_conn
    return analyzer, mock_conn


# ─────────────────────────────────────────────────────────────────
# DiscrepancyAnalyzer.close
# ─────────────────────────────────────────────────────────────────


class TestAnalyzerClose:
    def test_close_calls_conn_close(self):
        analyzer, mock_conn = _make_analyzer()
        analyzer.close()
        mock_conn.close.assert_called_once()
        assert analyzer._conn is None

    def test_close_noop_when_no_conn(self):
        analyzer = DiscrepancyAnalyzer(verbose=False)
        analyzer._conn = None
        analyzer.close()  # Should not raise

    def test_close_noop_when_already_closed(self):
        analyzer = DiscrepancyAnalyzer(verbose=False)
        mock_conn = MagicMock()
        mock_conn.closed = True
        analyzer._conn = mock_conn
        analyzer.close()
        mock_conn.close.assert_not_called()


# ─────────────────────────────────────────────────────────────────
# DiscrepancyAnalyzer._get_connection
# ─────────────────────────────────────────────────────────────────


class TestGetConnection:
    @patch("nba.betting_xl.analysis.line_discrepancy.get_intelligence_db_config")
    @patch("nba.betting_xl.analysis.line_discrepancy.psycopg2.connect")
    def test_creates_new_connection(self, mock_connect, mock_config):
        mock_config.return_value = {"host": "localhost", "port": 5539}
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        analyzer = DiscrepancyAnalyzer(verbose=False)
        conn = analyzer._get_connection()
        assert conn is mock_conn
        mock_connect.assert_called_once()

    def test_reuses_existing_connection(self):
        analyzer, mock_conn = _make_analyzer()
        conn = analyzer._get_connection()
        assert conn is mock_conn


# ─────────────────────────────────────────────────────────────────
# compute_discrepancies
# ─────────────────────────────────────────────────────────────────


class TestComputeDiscrepancies:
    def _setup_cursor(self, mock_conn, direct_rows, bp_rows):
        """Set up cursor to return direct rows on first call, BP rows on second."""
        mock_cur = MagicMock()
        mock_cur.fetchall.side_effect = [direct_rows, bp_rows]
        mock_conn.cursor.return_value = mock_cur
        return mock_cur

    def test_empty_data_returns_empty_df(self):
        analyzer, mock_conn = _make_analyzer()
        self._setup_cursor(mock_conn, [], [])
        df = analyzer.compute_discrepancies("2026-03-16")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_matched_pair_computes_diff(self):
        analyzer, mock_conn = _make_analyzer()
        ts = datetime(2026, 3, 16, 12, 0, 0)
        direct_rows = [
            ("LeBron James", "POINTS", "draftkings_direct", 25.5, -110, 100, ts),
        ]
        bp_rows = [
            ("LeBron James", "POINTS", "draftkings", 26.0, -115, 105, ts - timedelta(minutes=10)),
        ]
        self._setup_cursor(mock_conn, direct_rows, bp_rows)

        df = analyzer.compute_discrepancies("2026-03-16")
        assert len(df) == 1
        row = df.iloc[0]
        assert row["player_name"] == "LeBron James"
        assert row["stat_type"] == "POINTS"
        assert row["book_base"] == "draftkings"
        assert row["direct_line"] == 25.5
        assert row["bp_line"] == 26.0
        assert row["line_diff"] == -0.5
        assert row["odds_diff"] == -110 - (-115)
        assert row["latency_seconds"] == 600  # 10 minutes

    def test_direct_only_row_has_null_bp_fields(self):
        analyzer, mock_conn = _make_analyzer()
        ts = datetime(2026, 3, 16, 12, 0, 0)
        direct_rows = [
            ("Player A", "REBOUNDS", "fanduel_direct", 8.5, -110, 100, ts),
        ]
        bp_rows = []
        self._setup_cursor(mock_conn, direct_rows, bp_rows)

        df = analyzer.compute_discrepancies("2026-03-16")
        assert len(df) == 1
        row = df.iloc[0]
        assert row["direct_line"] == 8.5
        assert pd.isna(row["bp_line"]) or row["bp_line"] is None
        assert row["line_diff"] is None

    def test_bp_only_row_has_null_direct_fields(self):
        analyzer, mock_conn = _make_analyzer()
        ts = datetime(2026, 3, 16, 12, 0, 0)
        direct_rows = []
        bp_rows = [
            ("Player B", "POINTS", "betmgm", 22.0, -110, 100, ts),
        ]
        self._setup_cursor(mock_conn, direct_rows, bp_rows)

        df = analyzer.compute_discrepancies("2026-03-16")
        assert len(df) == 1
        row = df.iloc[0]
        assert row["bp_line"] == 22.0
        assert row["direct_line"] is None
        assert row["line_diff"] is None

    def test_book_name_filter(self):
        """When a book_name is provided, the SQL should include book filter."""
        analyzer, mock_conn = _make_analyzer()
        mock_cur = MagicMock()
        mock_cur.fetchall.side_effect = [[], []]
        mock_conn.cursor.return_value = mock_cur

        analyzer.compute_discrepancies("2026-03-16", book_name="draftkings")
        # Verify execute was called with the direct book name param
        calls = mock_cur.execute.call_args_list
        assert len(calls) == 2
        # Direct query should have book_name param "draftkings_direct"
        direct_params = calls[0][0][1]
        assert "draftkings_direct" in direct_params
        # BP query should have "draftkings"
        bp_params = calls[1][0][1]
        assert "draftkings" in bp_params

    def test_deduplication_keeps_first_row_per_key(self):
        """Most recent row (first in DESC order) should be kept."""
        analyzer, mock_conn = _make_analyzer()
        ts1 = datetime(2026, 3, 16, 14, 0, 0)
        ts2 = datetime(2026, 3, 16, 12, 0, 0)
        direct_rows = [
            ("Player C", "POINTS", "espnbet_direct", 30.0, -110, 100, ts1),
            ("Player C", "POINTS", "espnbet_direct", 29.0, -110, 100, ts2),
        ]
        self._setup_cursor(mock_conn, direct_rows, [])
        df = analyzer.compute_discrepancies("2026-03-16")
        assert len(df) == 1
        assert df.iloc[0]["direct_line"] == 30.0

    def test_none_line_handled(self):
        analyzer, mock_conn = _make_analyzer()
        ts = datetime(2026, 3, 16, 12, 0, 0)
        direct_rows = [("Player D", "POINTS", "caesars_direct", None, None, None, ts)]
        self._setup_cursor(mock_conn, direct_rows, [])
        df = analyzer.compute_discrepancies("2026-03-16")
        assert len(df) == 1
        assert df.iloc[0]["direct_line"] is None

    def test_verbose_output(self, capsys):
        analyzer, mock_conn = _make_analyzer(verbose=True)
        ts = datetime(2026, 3, 16, 12, 0, 0)
        direct_rows = [("X", "POINTS", "dk_direct", 25.0, -110, 100, ts)]
        bp_rows = [("X", "POINTS", "dk", 25.0, -110, 100, ts)]
        self._setup_cursor(mock_conn, direct_rows, bp_rows)
        analyzer.compute_discrepancies("2026-03-16")
        out = capsys.readouterr().out
        assert "matched pairs" in out


# ─────────────────────────────────────────────────────────────────
# flag_stale_lines
# ─────────────────────────────────────────────────────────────────


class TestFlagStaleLines:
    def test_returns_stale_lines(self):
        analyzer, mock_conn = _make_analyzer()
        mock_cur = MagicMock()
        ts_direct = datetime(2026, 3, 16, 14, 0, 0)
        ts_bp = datetime(2026, 3, 16, 12, 30, 0)
        lag_sec = (ts_direct - ts_bp).total_seconds()

        mock_cur.fetchall.return_value = [
            ("LeBron", "POINTS", "dk_direct", "dk", 25.5, 26.0, ts_direct, ts_bp, lag_sec),
        ]
        mock_conn.cursor.return_value = mock_cur
        result = analyzer.flag_stale_lines("2026-03-16", staleness_minutes=30)
        assert len(result) == 1
        assert result[0]["player_name"] == "LeBron"
        assert result[0]["lag_minutes"] == pytest.approx(90.0, abs=0.1)
        assert result[0]["line_diff"] == -0.5

    def test_undefined_table_returns_empty(self):
        import psycopg2.errors

        analyzer, mock_conn = _make_analyzer(verbose=True)
        mock_cur = MagicMock()
        mock_cur.execute.side_effect = psycopg2.errors.UndefinedTable("no table")
        mock_conn.cursor.return_value = mock_cur

        result = analyzer.flag_stale_lines("2026-03-16")
        assert result == []
        mock_conn.rollback.assert_called_once()

    def test_generic_psycopg2_error_returns_empty(self):
        import psycopg2

        analyzer, mock_conn = _make_analyzer(verbose=True)
        mock_cur = MagicMock()
        mock_cur.execute.side_effect = psycopg2.Error("some error")
        mock_conn.cursor.return_value = mock_cur

        result = analyzer.flag_stale_lines("2026-03-16")
        assert result == []
        mock_conn.rollback.assert_called_once()

    def test_null_line_values_handled(self):
        analyzer, mock_conn = _make_analyzer()
        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = [
            ("Player", "REBOUNDS", "dk_direct", "dk", None, None, None, None, None),
        ]
        mock_conn.cursor.return_value = mock_cur
        result = analyzer.flag_stale_lines("2026-03-16")
        assert len(result) == 1
        assert result[0]["direct_line"] is None
        assert result[0]["bp_line"] is None
        assert result[0]["lag_minutes"] is None


# ─────────────────────────────────────────────────────────────────
# generate_daily_report
# ─────────────────────────────────────────────────────────────────


class TestGenerateDailyReport:
    def test_empty_discrepancies_returns_zero_report(self):
        analyzer, mock_conn = _make_analyzer()
        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = []
        mock_conn.cursor.return_value = mock_cur

        report = analyzer.generate_daily_report("2026-03-16")
        assert report["total_comparisons"] == 0
        assert report["avg_line_diff"] == 0.0
        assert report["max_line_diff"] == 0.0
        assert report["missing_from_bp"] == []
        assert report["missing_from_direct"] == []

    @patch.object(DiscrepancyAnalyzer, "flag_stale_lines", return_value=[])
    @patch.object(DiscrepancyAnalyzer, "compute_discrepancies")
    def test_report_with_matched_pairs(self, mock_compute, mock_stale):
        analyzer, mock_conn = _make_analyzer()
        ts = datetime(2026, 3, 16, 12, 0, 0)

        df = pd.DataFrame(
            [
                {
                    "player_name": "P1",
                    "stat_type": "POINTS",
                    "book_base": "dk",
                    "direct_line": 25.0,
                    "bp_line": 25.5,
                    "line_diff": -0.5,
                    "direct_odds_over": -110,
                    "bp_odds_over": -115,
                    "odds_diff": 5,
                    "direct_timestamp": ts,
                    "bp_timestamp": ts,
                    "latency_seconds": 0,
                },
                {
                    "player_name": "P2",
                    "stat_type": "REBOUNDS",
                    "book_base": "fd",
                    "direct_line": 8.0,
                    "bp_line": 7.5,
                    "line_diff": 0.5,
                    "direct_odds_over": -110,
                    "bp_odds_over": -110,
                    "odds_diff": 0,
                    "direct_timestamp": ts,
                    "bp_timestamp": ts,
                    "latency_seconds": 0,
                },
            ]
        )
        mock_compute.return_value = df

        report = analyzer.generate_daily_report("2026-03-16")
        assert report["total_comparisons"] == 2
        assert report["avg_line_diff"] == 0.0  # (-0.5 + 0.5) / 2
        assert report["max_line_diff"] == 0.5
        assert "dk" in report["per_book_summary"]
        assert "fd" in report["per_book_summary"]

    @patch.object(DiscrepancyAnalyzer, "flag_stale_lines", return_value=[])
    @patch.object(DiscrepancyAnalyzer, "compute_discrepancies")
    def test_report_with_missing_from_bp(self, mock_compute, mock_stale):
        analyzer, _ = _make_analyzer()
        df = pd.DataFrame(
            [
                {
                    "player_name": "P3",
                    "stat_type": "POINTS",
                    "book_base": "dk",
                    "direct_line": 22.0,
                    "bp_line": None,
                    "line_diff": None,
                    "direct_odds_over": -110,
                    "bp_odds_over": None,
                    "odds_diff": None,
                    "direct_timestamp": None,
                    "bp_timestamp": None,
                    "latency_seconds": None,
                },
            ]
        )
        mock_compute.return_value = df

        report = analyzer.generate_daily_report("2026-03-16")
        assert len(report["missing_from_bp"]) == 1
        assert report["missing_from_bp"][0]["player_name"] == "P3"

    @patch.object(DiscrepancyAnalyzer, "flag_stale_lines", return_value=[])
    @patch.object(DiscrepancyAnalyzer, "compute_discrepancies")
    def test_report_with_missing_from_direct(self, mock_compute, mock_stale):
        analyzer, _ = _make_analyzer()
        df = pd.DataFrame(
            [
                {
                    "player_name": "P4",
                    "stat_type": "REBOUNDS",
                    "book_base": "fd",
                    "direct_line": None,
                    "bp_line": 9.0,
                    "line_diff": None,
                    "direct_odds_over": None,
                    "bp_odds_over": -110,
                    "odds_diff": None,
                    "direct_timestamp": None,
                    "bp_timestamp": None,
                    "latency_seconds": None,
                },
            ]
        )
        mock_compute.return_value = df
        report = analyzer.generate_daily_report("2026-03-16")
        assert len(report["missing_from_direct"]) == 1


# ─────────────────────────────────────────────────────────────────
# compute_cross_source_consensus
# ─────────────────────────────────────────────────────────────────


class TestComputeCrossSourceConsensus:
    def test_empty_returns_empty_df(self):
        analyzer, mock_conn = _make_analyzer(verbose=True)
        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = []
        mock_conn.cursor.return_value = mock_cur

        df = analyzer.compute_cross_source_consensus("2026-03-16")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_single_book_zero_spread(self):
        analyzer, mock_conn = _make_analyzer()
        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = [
            ("LeBron James", "POINTS", "draftkings_direct", 25.5),
        ]
        mock_conn.cursor.return_value = mock_cur

        df = analyzer.compute_cross_source_consensus("2026-03-16")
        assert len(df) == 1
        assert df.iloc[0]["direct_spread"] == 0.0
        assert df.iloc[0]["direct_num_books"] == 1

    def test_multi_book_consensus(self):
        analyzer, mock_conn = _make_analyzer()
        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = [
            ("LeBron James", "POINTS", "draftkings_direct", 25.0),
            ("LeBron James", "POINTS", "fanduel_direct", 26.0),
            ("LeBron James", "POINTS", "betmgm_direct", 25.5),
        ]
        mock_conn.cursor.return_value = mock_cur

        df = analyzer.compute_cross_source_consensus("2026-03-16")
        assert len(df) == 1
        row = df.iloc[0]
        assert row["direct_consensus"] == pytest.approx(25.5, abs=0.01)
        assert row["direct_spread"] == 1.0
        assert row["direct_num_books"] == 3
        assert row["softest_direct_book"] == "draftkings"
        assert row["hardest_direct_book"] == "fanduel"
        assert row["direct_min_line"] == 25.0
        assert row["direct_max_line"] == 26.0


# ─────────────────────────────────────────────────────────────────
# _print_report
# ─────────────────────────────────────────────────────────────────


class TestPrintReport:
    def test_print_report_no_data(self, capsys):
        analyzer = DiscrepancyAnalyzer(verbose=True)
        report = {
            "game_date": "2026-03-16",
            "total_comparisons": 0,
            "avg_line_diff": 0.0,
            "avg_abs_line_diff": 0.0,
            "max_line_diff": 0.0,
            "stale_lines_count": 0,
            "missing_from_bp": [],
            "missing_from_direct": [],
            "per_book_summary": {},
        }
        analyzer._print_report(report)
        out = capsys.readouterr().out
        assert "LINE DISCREPANCY REPORT" in out
        assert "2026-03-16" in out

    def test_print_report_with_per_book(self, capsys):
        analyzer = DiscrepancyAnalyzer(verbose=True)
        report = {
            "game_date": "2026-03-16",
            "total_comparisons": 50,
            "avg_line_diff": 0.1,
            "avg_abs_line_diff": 0.3,
            "max_line_diff": 1.5,
            "stale_lines_count": 2,
            "missing_from_bp": [{"player_name": "P1", "stat_type": "POINTS", "book_base": "dk"}],
            "missing_from_direct": [],
            "per_book_summary": {
                "draftkings": {
                    "count": 25,
                    "avg_diff": 0.05,
                    "avg_abs_diff": 0.2,
                    "max_diff": 1.0,
                    "zero_diff_pct": 40.0,
                },
            },
        }
        analyzer._print_report(report)
        out = capsys.readouterr().out
        assert "draftkings" in out
        assert "Props in Direct but NOT in BP" in out

    def test_print_report_truncates_missing(self, capsys):
        """When > 10 missing, shows '... and N more'."""
        analyzer = DiscrepancyAnalyzer(verbose=True)
        missing = [
            {"player_name": f"P{i}", "stat_type": "POINTS", "book_base": "dk"} for i in range(15)
        ]
        report = {
            "game_date": "2026-03-16",
            "total_comparisons": 10,
            "avg_line_diff": 0.0,
            "avg_abs_line_diff": 0.0,
            "max_line_diff": 0.0,
            "stale_lines_count": 0,
            "missing_from_bp": missing,
            "missing_from_direct": [],
            "per_book_summary": {},
        }
        analyzer._print_report(report)
        out = capsys.readouterr().out
        assert "and 5 more" in out

    def test_print_report_missing_from_direct_truncated(self, capsys):
        """When > 10 missing_from_direct, shows '... and N more'."""
        analyzer = DiscrepancyAnalyzer(verbose=True)
        missing_direct = [
            {"player_name": f"P{i}", "stat_type": "POINTS", "book_base": "fd"} for i in range(12)
        ]
        report = {
            "game_date": "2026-03-16",
            "total_comparisons": 10,
            "avg_line_diff": 0.0,
            "avg_abs_line_diff": 0.0,
            "max_line_diff": 0.0,
            "stale_lines_count": 0,
            "missing_from_bp": [],
            "missing_from_direct": missing_direct,
            "per_book_summary": {},
        }
        analyzer._print_report(report)
        out = capsys.readouterr().out
        assert "Props in BP but NOT in Direct" in out
        assert "and 2 more" in out


# ─────────────────────────────────────────────────────────────────
# Verbose output paths (coverage for lines 358-362, 455-456, 537-543, 585-590)
# ─────────────────────────────────────────────────────────────────


class TestVerboseOutputPaths:
    def test_flag_stale_lines_verbose_prints_count(self, capsys):
        """Line 358-362: verbose=True in flag_stale_lines prints count."""
        analyzer, mock_conn = _make_analyzer(verbose=True)
        mock_cur = MagicMock()
        ts_direct = datetime(2026, 3, 16, 14, 0, 0)
        ts_bp = datetime(2026, 3, 16, 12, 30, 0)
        lag_sec = (ts_direct - ts_bp).total_seconds()
        mock_cur.fetchall.return_value = [
            ("LeBron", "POINTS", "dk_direct", "dk", 25.5, 26.0, ts_direct, ts_bp, lag_sec),
        ]
        mock_conn.cursor.return_value = mock_cur
        result = analyzer.flag_stale_lines("2026-03-16", staleness_minutes=30)
        out = capsys.readouterr().out
        assert "1 stale lines" in out
        assert ">30 min behind direct" in out

    @patch.object(DiscrepancyAnalyzer, "flag_stale_lines", return_value=[])
    @patch.object(DiscrepancyAnalyzer, "compute_discrepancies")
    def test_generate_daily_report_verbose_prints_report(self, mock_compute, mock_stale, capsys):
        """Line 455-456: verbose=True in generate_daily_report calls _print_report."""
        analyzer, _ = _make_analyzer(verbose=True)
        ts = datetime(2026, 3, 16, 12, 0, 0)
        df = pd.DataFrame(
            [
                {
                    "player_name": "P1",
                    "stat_type": "POINTS",
                    "book_base": "dk",
                    "direct_line": 25.0,
                    "bp_line": 25.5,
                    "line_diff": -0.5,
                    "direct_odds_over": -110,
                    "bp_odds_over": -115,
                    "odds_diff": 5,
                    "direct_timestamp": ts,
                    "bp_timestamp": ts,
                    "latency_seconds": 0,
                },
            ]
        )
        mock_compute.return_value = df
        report = analyzer.generate_daily_report("2026-03-16")
        out = capsys.readouterr().out
        assert "LINE DISCREPANCY REPORT" in out

    def test_compute_cross_source_consensus_verbose(self, capsys):
        """Lines 537-543: verbose=True in compute_cross_source_consensus prints stats."""
        analyzer, mock_conn = _make_analyzer(verbose=True)
        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = [
            ("LeBron James", "POINTS", "draftkings_direct", 25.0),
            ("LeBron James", "POINTS", "fanduel_direct", 26.0),
        ]
        mock_conn.cursor.return_value = mock_cur
        df = analyzer.compute_cross_source_consensus("2026-03-16")
        out = capsys.readouterr().out
        assert "Direct consensus:" in out
        assert "1 props" in out
        assert "with cross-book spread" in out


# ─────────────────────────────────────────────────────────────────
# Branch 185→181: empty bp_map (bp_rows empty, all keys from direct_map)
# ─────────────────────────────────────────────────────────────────


class TestEmptyBpMap:
    def test_compute_discrepancies_with_no_bp_rows(self):
        """Branch 185→181: bp_rows is empty so bp_map stays empty."""
        analyzer, mock_conn = _make_analyzer()
        ts = datetime(2026, 3, 16, 12, 0, 0)
        direct_rows = [
            ("Player A", "POINTS", "draftkings_direct", 25.0, -110, 100, ts),
            ("Player B", "REBOUNDS", "fanduel_direct", 8.5, -110, 100, ts),
        ]
        bp_rows = []  # empty bp_map, branch 185→181 never entered
        mock_cur = MagicMock()
        mock_cur.fetchall.side_effect = [direct_rows, bp_rows]
        mock_conn.cursor.return_value = mock_cur

        df = analyzer.compute_discrepancies("2026-03-16")
        assert len(df) == 2
        # All bp fields should be None
        assert df.iloc[0]["bp_line"] is None
        assert df.iloc[1]["bp_line"] is None
