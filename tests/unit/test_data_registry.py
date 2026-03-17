"""
Unit tests for nba.core.data_registry

Tests cover:
- IngestionTracker context manager (success, failure, partial, add_error, add_bytes)
- _log_ingestion_start (mocked DB)
- _log_ingestion_complete (mocked DB, None ID case)
- log_ingestion (one-shot)
- update_coverage (mocked DB)
- get_coverage_report (mocked DB)
- get_source_health (mocked DB, with and without source_name)
- raise_alert (mocked DB)
- resolve_alert (mocked DB)
- heartbeat (mocked DB)
- Error handling: all functions are fire-and-forget (never raise)
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from nba.core.data_registry import (
    IngestionTracker,
    _log_ingestion_complete,
    _log_ingestion_start,
    get_coverage_report,
    get_source_health,
    heartbeat,
    log_ingestion,
    raise_alert,
    resolve_alert,
    update_coverage,
)

# ─────────────────────────────────────────────────────────────────
# Helper: mock _connect to return a mock connection
# ─────────────────────────────────────────────────────────────────


def _mock_conn_with_cursor(fetchone_val=None, fetchall_val=None):
    """Create a mock connection with a cursor that supports context manager."""
    mock_cur = MagicMock()
    mock_cur.__enter__ = lambda s: mock_cur
    mock_cur.__exit__ = MagicMock(return_value=False)
    if fetchone_val is not None:
        mock_cur.fetchone.return_value = fetchone_val
    if fetchall_val is not None:
        mock_cur.fetchall.return_value = fetchall_val

    mock_conn = MagicMock()
    mock_conn.__enter__ = lambda s: mock_conn
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_conn.cursor.return_value = mock_cur
    return mock_conn, mock_cur


# ─────────────────────────────────────────────────────────────────
# IngestionTracker
# ─────────────────────────────────────────────────────────────────


class TestIngestionTracker:
    @patch("nba.core.data_registry._log_ingestion_complete")
    @patch("nba.core.data_registry._log_ingestion_start", return_value=42)
    def test_success_flow(self, mock_start, mock_complete):
        with IngestionTracker("test_source", "fetch") as t:
            t.records_fetched = 100
            t.records_new = 80
            t.api_calls_made = 5

        mock_start.assert_called_once()
        assert t.ingestion_id == 42
        mock_complete.assert_called_once()
        call_kwargs = mock_complete.call_args
        assert call_kwargs[1]["status"] == "success"
        assert call_kwargs[1]["records_fetched"] == 100
        assert call_kwargs[1]["records_new"] == 80

    @patch("nba.core.data_registry._log_ingestion_complete")
    @patch("nba.core.data_registry._log_ingestion_start", return_value=43)
    def test_failure_flow(self, mock_start, mock_complete):
        with pytest.raises(ValueError):
            with IngestionTracker("test_source", "fetch") as t:
                t.records_fetched = 50
                raise ValueError("something broke")

        call_kwargs = mock_complete.call_args
        assert call_kwargs[1]["status"] == "failed"
        assert "something broke" in call_kwargs[1]["error_message"]
        assert call_kwargs[1]["error_count"] >= 1

    @patch("nba.core.data_registry._log_ingestion_complete")
    @patch("nba.core.data_registry._log_ingestion_start", return_value=44)
    def test_partial_flow(self, mock_start, mock_complete):
        with IngestionTracker("test_source", "load") as t:
            t.records_fetched = 100
            t.error_count = 3

        call_kwargs = mock_complete.call_args
        assert call_kwargs[1]["status"] == "partial"

    @patch("nba.core.data_registry._log_ingestion_complete")
    @patch("nba.core.data_registry._log_ingestion_start", return_value=45)
    def test_zero_records_with_errors_is_failed(self, mock_start, mock_complete):
        with IngestionTracker("test_source", "load") as t:
            t.records_fetched = 0
            t.error_count = 2

        call_kwargs = mock_complete.call_args
        assert call_kwargs[1]["status"] == "failed"

    @patch("nba.core.data_registry._log_ingestion_complete")
    @patch("nba.core.data_registry._log_ingestion_start", return_value=46)
    def test_zero_records_zero_errors_is_success(self, mock_start, mock_complete):
        with IngestionTracker("test_source", "load") as t:
            t.records_fetched = 0
            t.error_count = 0

        call_kwargs = mock_complete.call_args
        assert call_kwargs[1]["status"] == "success"

    def test_add_error(self):
        tracker = IngestionTracker("src", "op")
        tracker.add_error("bad data")
        assert tracker.error_count == 1
        assert tracker.error_message == "bad data"
        tracker.add_error("worse data")
        assert tracker.error_count == 2
        assert tracker.error_message == "worse data"

    def test_add_bytes(self):
        tracker = IngestionTracker("src", "op")
        tracker.add_bytes(1024)
        tracker.add_bytes(2048)
        assert tracker.bytes_transferred == 3072

    def test_metadata_defaults_to_empty_dict(self):
        tracker = IngestionTracker("src", "op")
        assert tracker.metadata == {}

    def test_metadata_set(self):
        tracker = IngestionTracker("src", "op", metadata={"game_date": "2026-03-16"})
        assert tracker.metadata["game_date"] == "2026-03-16"

    @patch("nba.core.data_registry._log_ingestion_complete")
    @patch("nba.core.data_registry._log_ingestion_start", return_value=47)
    def test_does_not_suppress_exceptions(self, mock_start, mock_complete):
        """__exit__ returns False → exceptions propagate."""
        with pytest.raises(RuntimeError):
            with IngestionTracker("src", "op"):
                raise RuntimeError("oops")


# ─────────────────────────────────────────────────────────────────
# _log_ingestion_start
# ─────────────────────────────────────────────────────────────────


class TestLogIngestionStart:
    @patch("nba.core.data_registry._connect")
    def test_returns_id(self, mock_connect):
        mock_conn, mock_cur = _mock_conn_with_cursor(fetchone_val=(99,))
        mock_connect.return_value = mock_conn
        result = _log_ingestion_start("src", "fetch", {"key": "val"})
        assert result == 99
        mock_conn.close.assert_called_once()

    @patch("nba.core.data_registry._connect")
    def test_returns_none_on_error(self, mock_connect):
        mock_connect.side_effect = Exception("connect failed")
        result = _log_ingestion_start("src", "fetch")
        assert result is None

    @patch("nba.core.data_registry._connect")
    def test_returns_none_on_empty_fetchone(self, mock_connect):
        mock_conn, mock_cur = _mock_conn_with_cursor()
        # Explicitly set fetchone to return None (no row from RETURNING)
        mock_cur.fetchone.return_value = None
        mock_connect.return_value = mock_conn
        result = _log_ingestion_start("src", "fetch")
        assert result is None


# ─────────────────────────────────────────────────────────────────
# _log_ingestion_complete
# ─────────────────────────────────────────────────────────────────


class TestLogIngestionComplete:
    def test_none_id_returns_false(self):
        result = _log_ingestion_complete(
            ingestion_id=None,
            status="success",
            duration_seconds=1.0,
        )
        assert result is False

    @patch("nba.core.data_registry._connect")
    def test_success_returns_true(self, mock_connect):
        mock_conn, _ = _mock_conn_with_cursor()
        mock_connect.return_value = mock_conn
        result = _log_ingestion_complete(
            ingestion_id=42,
            status="success",
            duration_seconds=5.5,
            records_fetched=100,
            records_new=80,
        )
        assert result is True
        mock_conn.close.assert_called_once()

    @patch("nba.core.data_registry._connect")
    def test_error_returns_false(self, mock_connect):
        mock_connect.side_effect = Exception("db down")
        result = _log_ingestion_complete(ingestion_id=42, status="success", duration_seconds=1.0)
        assert result is False


# ─────────────────────────────────────────────────────────────────
# log_ingestion (one-shot)
# ─────────────────────────────────────────────────────────────────


class TestLogIngestion:
    @patch("nba.core.data_registry._connect")
    def test_logs_successfully(self, mock_connect):
        mock_conn, _ = _mock_conn_with_cursor()
        mock_connect.return_value = mock_conn
        result = log_ingestion(
            "test_source",
            "fetch",
            "success",
            records_fetched=50,
            records_new=40,
            metadata={"game_date": "2026-03-16"},
        )
        assert result is True

    @patch("nba.core.data_registry._connect")
    def test_error_returns_false(self, mock_connect):
        mock_connect.side_effect = Exception("db down")
        result = log_ingestion("test_source", "fetch", "success")
        assert result is False


# ─────────────────────────────────────────────────────────────────
# update_coverage
# ─────────────────────────────────────────────────────────────────


class TestUpdateCoverage:
    @patch("nba.core.data_registry._connect")
    def test_upserts_coverage(self, mock_connect):
        mock_conn, _ = _mock_conn_with_cursor()
        mock_connect.return_value = mock_conn
        result = update_coverage(
            "bettingpros_props",
            "2026-03-16",
            market="POINTS",
            book_name="draftkings",
            record_count=127,
            player_count=50,
            has_actuals=True,
        )
        assert result is True

    @patch("nba.core.data_registry._connect")
    def test_error_returns_false(self, mock_connect):
        mock_connect.side_effect = Exception("db error")
        result = update_coverage("src", "2026-03-16", record_count=10)
        assert result is False


# ─────────────────────────────────────────────────────────────────
# get_coverage_report
# ─────────────────────────────────────────────────────────────────


class TestGetCoverageReport:
    @patch("nba.core.data_registry._connect")
    def test_returns_report(self, mock_connect):
        from datetime import date

        mock_conn, mock_cur = _mock_conn_with_cursor(
            fetchall_val=[
                (date(2026, 3, 14), "draftkings", "POINTS", 100, 30, True, True),
                (date(2026, 3, 14), "fanduel", "POINTS", 90, 28, False, True),
                (date(2026, 3, 15), "draftkings", "POINTS", 110, 32, True, False),
            ]
        )
        mock_connect.return_value = mock_conn

        result = get_coverage_report("bettingpros_props", "2026-03-14", "2026-03-15")
        assert result is not None
        assert result["source"] == "bettingpros_props"
        assert result["total_days"] == 2
        assert result["days_with_data"] == 2
        assert result["total_records"] == 300
        assert result["days_with_actuals"] >= 1
        assert result["coverage_pct"] == 100.0
        assert "draftkings" in result["books_breakdown"]

    @patch("nba.core.data_registry._connect")
    def test_missing_days_detected(self, mock_connect):
        from datetime import date

        # Only 1 day of data but range is 3 days
        mock_conn, mock_cur = _mock_conn_with_cursor(
            fetchall_val=[
                (date(2026, 3, 14), "draftkings", "POINTS", 100, 30, True, True),
            ]
        )
        mock_connect.return_value = mock_conn

        result = get_coverage_report("src", "2026-03-14", "2026-03-16")
        assert result is not None
        assert result["total_days"] == 3
        assert result["days_with_data"] == 1
        assert len(result["days_missing"]) == 2

    @patch("nba.core.data_registry._connect")
    def test_no_data_returns_all_missing(self, mock_connect):
        mock_conn, mock_cur = _mock_conn_with_cursor(fetchall_val=[])
        mock_connect.return_value = mock_conn

        result = get_coverage_report("src", "2026-03-14", "2026-03-15")
        assert result is not None
        assert result["days_with_data"] == 0
        assert result["avg_records_per_day"] == 0
        assert len(result["days_missing"]) == 2

    @patch("nba.core.data_registry._connect")
    def test_error_returns_none(self, mock_connect):
        mock_connect.side_effect = Exception("db error")
        result = get_coverage_report("src", "2026-03-14", "2026-03-16")
        assert result is None


# ─────────────────────────────────────────────────────────────────
# get_source_health
# ─────────────────────────────────────────────────────────────────


class TestGetSourceHealth:
    @patch("nba.core.data_registry._connect")
    def test_returns_health_list(self, mock_connect):
        from datetime import datetime

        completed = datetime(2026, 3, 16, 14, 0, 0)
        mock_conn, mock_cur = _mock_conn_with_cursor(
            fetchall_val=[
                # name, provider, sla_hours, enabled, status, records, completed,
                # error, total_24h, successes_24h, errors_24h, age_hours
                (
                    "bettingpros_props",
                    "BettingPros",
                    4,
                    True,
                    "success",
                    2684,
                    completed,
                    None,
                    6,
                    5,
                    1,
                    2.3,
                ),
            ]
        )
        mock_connect.return_value = mock_conn

        result = get_source_health()
        assert result is not None
        assert len(result) == 1
        src = result[0]
        assert src["source"] == "bettingpros_props"
        assert src["enabled"] is True
        assert src["sla_ok"] is True  # 2.3 < 4
        assert src["success_rate_24h"] == pytest.approx(83.3, abs=0.1)
        assert src["total_ingestions_24h"] == 6
        assert src["total_errors_24h"] == 1
        assert src["age_hours"] == 2.3

    @patch("nba.core.data_registry._connect")
    def test_sla_breach(self, mock_connect):
        from datetime import datetime

        completed = datetime(2026, 3, 16, 10, 0, 0)
        mock_conn, mock_cur = _mock_conn_with_cursor(
            fetchall_val=[
                ("src", "provider", 4, True, "success", 100, completed, None, 6, 6, 0, 5.0),
            ]
        )
        mock_connect.return_value = mock_conn

        result = get_source_health()
        assert result[0]["sla_ok"] is False  # 5.0 > 4

    @patch("nba.core.data_registry._connect")
    def test_null_completion_time(self, mock_connect):
        mock_conn, mock_cur = _mock_conn_with_cursor(
            fetchall_val=[
                ("src", "provider", 4, True, None, None, None, None, None, None, None, None),
            ]
        )
        mock_connect.return_value = mock_conn

        result = get_source_health()
        assert result is not None
        assert result[0]["age_hours"] is None
        assert result[0]["last_ingestion"] is None
        assert result[0]["sla_ok"] is True  # sla_hours set but age_hours is None

    @patch("nba.core.data_registry._connect")
    def test_with_source_name_filter(self, mock_connect):
        mock_conn, mock_cur = _mock_conn_with_cursor(
            fetchall_val=[
                ("specific_src", "p", 4, True, "success", 50, None, None, 3, 3, 0, None),
            ]
        )
        mock_connect.return_value = mock_conn

        result = get_source_health(source_name="specific_src")
        assert result is not None
        assert len(result) == 1
        assert result[0]["source"] == "specific_src"

    @patch("nba.core.data_registry._connect")
    def test_error_returns_none(self, mock_connect):
        mock_connect.side_effect = Exception("db error")
        result = get_source_health()
        assert result is None


# ─────────────────────────────────────────────────────────────────
# raise_alert
# ─────────────────────────────────────────────────────────────────


class TestRaiseAlert:
    @patch("nba.core.data_registry._connect")
    def test_returns_alert_id(self, mock_connect):
        mock_conn, mock_cur = _mock_conn_with_cursor(fetchone_val=(101,))
        mock_connect.return_value = mock_conn
        result = raise_alert(
            "sla_breach",
            "warning",
            "prizepicks",
            "PrizePicks stale",
            "No data in 8 hours",
            metadata={"source": "prizepicks"},
        )
        assert result == 101

    @patch("nba.core.data_registry._connect")
    def test_error_returns_none(self, mock_connect):
        mock_connect.side_effect = Exception("db error")
        result = raise_alert("type", "sev", "src", "title", "msg")
        assert result is None


# ─────────────────────────────────────────────────────────────────
# resolve_alert
# ─────────────────────────────────────────────────────────────────


class TestResolveAlert:
    @patch("nba.core.data_registry._connect")
    def test_resolves_successfully(self, mock_connect):
        mock_conn, _ = _mock_conn_with_cursor()
        mock_connect.return_value = mock_conn
        result = resolve_alert(101)
        assert result is True

    @patch("nba.core.data_registry._connect")
    def test_error_returns_false(self, mock_connect):
        mock_connect.side_effect = Exception("db error")
        result = resolve_alert(101)
        assert result is False


# ─────────────────────────────────────────────────────────────────
# heartbeat
# ─────────────────────────────────────────────────────────────────


class TestHeartbeat:
    @patch("nba.core.data_registry._connect")
    def test_sends_heartbeat(self, mock_connect):
        mock_conn, _ = _mock_conn_with_cursor()
        mock_connect.return_value = mock_conn
        result = heartbeat(
            "axiom_pipeline",
            "healthy",
            uptime_seconds=3600,
            version="1.2.3",
            metadata={"last_run": "ok"},
        )
        assert result is True

    @patch("nba.core.data_registry._connect")
    def test_minimal_heartbeat(self, mock_connect):
        mock_conn, _ = _mock_conn_with_cursor()
        mock_connect.return_value = mock_conn
        result = heartbeat("test_service")
        assert result is True

    @patch("nba.core.data_registry._connect")
    def test_error_returns_false(self, mock_connect):
        mock_connect.side_effect = Exception("db error")
        result = heartbeat("test_service")
        assert result is False
