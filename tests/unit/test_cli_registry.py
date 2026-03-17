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
