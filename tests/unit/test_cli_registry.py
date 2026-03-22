"""Tests for nba.core.cli_registry — CLI for Atlas Data Registry."""

from unittest.mock import MagicMock, patch

import pytest


class TestConnect:
    def test_connect_returns_connection(self):
        mock_conn = MagicMock()
        with patch("psycopg2.connect", return_value=mock_conn) as mock_pg:
            from nba.core.cli_registry import _connect

            conn = _connect()
            assert conn == mock_conn
            mock_pg.assert_called_once()


class TestCmdHealth:
    def test_health_with_sources(self):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = lambda s: mock_cursor
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchall.return_value = [
            {
                "source_name": "bettingpros_props",
                "last_success": "2026-03-16 10:00:00",
                "last_status": "success",
                "total_runs_24h": 5,
                "success_rate_24h": 100.0,
                "avg_records": 2500,
                "avg_duration_sec": 45.0,
                "last_error": None,
            }
        ]
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.close = MagicMock()

        with patch("nba.core.cli_registry._connect", return_value=mock_conn):
            from nba.core.cli_registry import cmd_health

            cmd_health(None)  # Should not raise

    def test_health_empty(self):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = lambda s: mock_cursor
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchall.return_value = []
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.close = MagicMock()

        with patch("nba.core.cli_registry._connect", return_value=mock_conn):
            from nba.core.cli_registry import cmd_health

            cmd_health(None)


class TestCmdIngestions:
    def test_ingestions(self):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = lambda s: mock_cursor
        mock_cursor.__exit__ = MagicMock(return_value=False)
        from datetime import datetime

        mock_cursor.fetchall.return_value = [
            (
                "draftkings_direct",
                "fetch",
                "success",
                340,
                4,
                189000,
                0,
                8.5,
                datetime(2026, 3, 16, 10, 0, 0),
                None,
            )
        ]
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.close = MagicMock()

        with patch("nba.core.cli_registry._connect", return_value=mock_conn):
            from nba.core.cli_registry import cmd_ingestions

            args = MagicMock()
            args.limit = 10
            args.source = None
            cmd_ingestions(args)


class TestCmdAlerts:
    def test_alerts(self):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = lambda s: mock_cursor
        mock_cursor.__exit__ = MagicMock(return_value=False)
        from datetime import datetime

        mock_cursor.fetchall.return_value = [
            (
                1,
                "warning",
                "fetch_failure",
                "fanduel_direct",
                "Fetch Failed",
                "HTTP 429",
                datetime(2026, 3, 16, 10, 0, 0),
            )
        ]
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.close = MagicMock()

        with patch("nba.core.cli_registry._connect", return_value=mock_conn):
            from nba.core.cli_registry import cmd_alerts

            cmd_alerts(None)

    def test_no_alerts(self):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = lambda s: mock_cursor
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchall.return_value = []
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.close = MagicMock()

        with patch("nba.core.cli_registry._connect", return_value=mock_conn):
            from nba.core.cli_registry import cmd_alerts

            cmd_alerts(None)


class TestCmdCoverage:
    def test_coverage(self):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = lambda s: mock_cursor
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchall.return_value = [
            {
                "run_date": "2026-03-16",
                "total_runs": 3,
                "total_records": 7500,
                "avg_records": 2500,
                "sources_active": 5,
            }
        ]
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.close = MagicMock()

        with patch("nba.core.cli_registry._connect", return_value=mock_conn):
            from nba.core.cli_registry import cmd_coverage

            args = MagicMock()
            args.source = "bettingpros_props"
            args.start = "2026-03-01"
            args.end = "2026-03-16"
            cmd_coverage(args)


class TestCmdStatus:
    def test_status(self):
        with (
            patch("nba.core.cli_registry.cmd_health"),
            patch("nba.core.cli_registry.cmd_alerts"),
            patch("nba.core.cli_registry.cmd_ingestions"),
        ):
            from nba.core.cli_registry import cmd_status

            args = MagicMock()
            args.limit = 10
            args.source = None
            cmd_status(args)


# ─────────────────────────────────────────────────────────────────
# Coverage: cmd_health with empty/disabled sources (lines 48-49, 57-58)
# ─────────────────────────────────────────────────────────────────


class TestCmdHealthEdgeCases:
    def test_health_empty_sources(self, capsys):
        """Line 48-49: get_source_health returns empty list."""
        with patch("nba.core.data_registry.get_source_health", return_value=[]):
            from nba.core.cli_registry import cmd_health

            cmd_health(None)
        out = capsys.readouterr().out
        assert "No data sources found" in out

    def test_health_disabled_source_skipped(self, capsys):
        """Line 57-58: disabled source is skipped (continue)."""
        sources = [
            {
                "source": "bettingpros_props",
                "enabled": False,
                "sla_ok": True,
                "age_hours": 1.0,
                "last_records": 100,
                "success_rate_24h": 100.0,
                "total_ingestions_24h": 5,
                "total_errors_24h": 0,
            },
            {
                "source": "draftkings_direct",
                "enabled": True,
                "sla_ok": True,
                "age_hours": 0.5,
                "last_records": 300,
                "success_rate_24h": 95.0,
                "total_ingestions_24h": 4,
                "total_errors_24h": 1,
            },
        ]
        with patch("nba.core.data_registry.get_source_health", return_value=sources):
            from nba.core.cli_registry import cmd_health

            cmd_health(None)
        out = capsys.readouterr().out
        # Disabled source should not appear in output
        assert "bettingpros_props" not in out
        assert "draftkings_direct" in out


# ─────────────────────────────────────────────────────────────────
# Coverage: cmd_coverage empty/many missing (lines 79-81, 100-101)
# ─────────────────────────────────────────────────────────────────


class TestCmdCoverageEdgeCases:
    def test_coverage_empty_report(self, capsys):
        """Lines 79-81: get_coverage_report returns empty/None."""
        with patch("nba.core.data_registry.get_coverage_report", return_value=None):
            from nba.core.cli_registry import cmd_coverage

            args = MagicMock()
            args.source = "nonexistent"
            args.start = "2026-03-01"
            args.end = "2026-03-16"
            cmd_coverage(args)
        out = capsys.readouterr().out
        assert "No coverage data" in out

    def test_coverage_more_than_20_missing_days(self, capsys):
        """Lines 100-101: more than 20 missing days triggers '... and N more'."""
        report = {
            "source": "bettingpros_props",
            "date_range": ["2026-01-01", "2026-03-16"],
            "total_days": 75,
            "days_with_data": 50,
            "days_missing": [f"2026-01-{i:02d}" for i in range(1, 26)],
            "total_records": 50000,
            "avg_records_per_day": 1000,
            "days_with_actuals": 40,
            "days_with_enrichment": 45,
            "coverage_pct": 66.7,
            "books_breakdown": {"draftkings": 50, "fanduel": 48},
        }
        with patch("nba.core.data_registry.get_coverage_report", return_value=report):
            from nba.core.cli_registry import cmd_coverage

            args = MagicMock()
            args.source = "bettingpros_props"
            args.start = "2026-01-01"
            args.end = "2026-03-16"
            cmd_coverage(args)
        out = capsys.readouterr().out
        assert "and 5 more" in out

    def test_coverage_no_missing_no_books(self, capsys):
        """Branches 96->103, 103->108: no missing days and no books breakdown."""
        report = {
            "source": "bettingpros_props",
            "date_range": ["2026-03-01", "2026-03-16"],
            "total_days": 16,
            "days_with_data": 16,
            "days_missing": [],
            "total_records": 40000,
            "avg_records_per_day": 2500,
            "days_with_actuals": 14,
            "days_with_enrichment": 16,
            "coverage_pct": 100.0,
            "books_breakdown": {},
        }
        with patch("nba.core.data_registry.get_coverage_report", return_value=report):
            from nba.core.cli_registry import cmd_coverage

            args = MagicMock()
            args.source = "bettingpros_props"
            args.start = "2026-03-01"
            args.end = "2026-03-16"
            cmd_coverage(args)
        out = capsys.readouterr().out
        assert "Days with data" in out
        assert "Missing dates" not in out
        assert "Books breakdown" not in out


# ─────────────────────────────────────────────────────────────────
# Coverage: cmd_ingestions empty rows (lines 130-131)
# ─────────────────────────────────────────────────────────────────


class TestCmdIngestionsEdgeCases:
    def test_ingestions_empty(self, capsys):
        """Lines 129-131: no rows returns 'No ingestion records found'."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = lambda s: mock_cursor
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchall.return_value = []
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.close = MagicMock()

        with patch("nba.core.cli_registry._connect", return_value=mock_conn):
            from nba.core.cli_registry import cmd_ingestions

            args = MagicMock()
            args.limit = 10
            cmd_ingestions(args)
        out = capsys.readouterr().out
        assert "No ingestion records found" in out
