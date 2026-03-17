"""
Data Registry — ingestion tracking and coverage reporting for Atlas.

Every fetcher, loader, and pipeline step logs its work here.
Atlas reads this to know: what data do we have, how fresh is it,
and where are the gaps. Claude Code reads it on escalation.

All functions are fire-and-forget: they log on failure but never raise,
so a registry failure never blocks a production pipeline.

Usage in fetchers:
    from nba.core.data_registry import IngestionTracker

    with IngestionTracker("bettingpros_props", "fetch", metadata={"game_date": "2026-03-14"}) as tracker:
        props = fetch_props()
        tracker.records_fetched = len(props)
        tracker.records_new = new_count

Usage for coverage:
    from nba.core.data_registry import update_coverage, get_coverage_report

    update_coverage("bettingpros_props", "2026-03-14", market="POINTS", book="draftkings", count=127)
    gaps = get_coverage_report("bettingpros_props", "2026-03-01", "2026-03-14")

Usage for alerts:
    from nba.core.data_registry import raise_alert

    raise_alert("sla_breach", "warning", "prizepicks", "PrizePicks stale",
                "No data in 8 hours (SLA: 6 hours)")
"""

import json
import logging
import os
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Optional
from zoneinfo import ZoneInfo

log = logging.getLogger("nba.data_registry")

_AXIOM_PORT = 5541
_AXIOM_DB = "cephalon_axiom"
_CONNECT_TIMEOUT = 5
_EST = ZoneInfo("America/New_York")


# ─────────────────────────────────────────────────────────────────
# Connection (same pattern as axiom_writer.py)
# ─────────────────────────────────────────────────────────────────


def _connect():
    """Open a connection to cephalon_axiom. Caller must close."""
    import psycopg2

    return psycopg2.connect(
        host=os.environ.get("DB_HOST", "localhost"),
        port=_AXIOM_PORT,
        dbname=_AXIOM_DB,
        user=os.environ.get("DB_USER", "mlb_user"),
        password=os.environ.get("DB_PASSWORD", ""),
        connect_timeout=_CONNECT_TIMEOUT,
    )


# ─────────────────────────────────────────────────────────────────
# Ingestion Tracking
# ─────────────────────────────────────────────────────────────────


class IngestionTracker:
    """Context manager that logs an ingestion operation start-to-finish.

    Usage:
        with IngestionTracker("bettingpros_props", "fetch") as t:
            props = do_fetch()
            t.records_fetched = len(props)
            t.records_new = count_new(props)
            t.api_calls_made = 28

    On __exit__, writes completion status to ingestion_log.
    If an exception occurs, status is set to 'failed' with the error message.
    """

    def __init__(
        self,
        source_name: str,
        operation: str,
        *,
        metadata: Optional[dict] = None,
    ):
        self.source_name = source_name
        self.operation = operation
        self.metadata = metadata or {}
        self.ingestion_id: Optional[int] = None
        self._start_time = time.time()

        # Counters — set these during the operation
        self.records_fetched: int = 0
        self.records_new: int = 0
        self.records_duplicate: int = 0
        self.records_failed: int = 0
        self.api_calls_made: int = 0
        self.bytes_transferred: int = 0
        self.error_count: int = 0
        self.error_message: Optional[str] = None

    def __enter__(self):
        self.ingestion_id = _log_ingestion_start(self.source_name, self.operation, self.metadata)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self._start_time

        if exc_type is not None:
            status = "failed"
            self.error_message = str(exc_val)
            self.error_count += 1
        elif self.error_count > 0 and self.records_fetched > 0:
            status = "partial"
        elif self.records_fetched == 0 and self.error_count > 0:
            status = "failed"
        else:
            status = "success"

        _log_ingestion_complete(
            ingestion_id=self.ingestion_id,
            status=status,
            duration_seconds=round(duration, 2),
            records_fetched=self.records_fetched,
            records_new=self.records_new,
            records_duplicate=self.records_duplicate,
            records_failed=self.records_failed,
            api_calls_made=self.api_calls_made,
            bytes_transferred=self.bytes_transferred,
            error_count=self.error_count,
            error_message=self.error_message,
            metadata=self.metadata,
        )

        # Don't suppress exceptions
        return False

    def add_error(self, msg: str):
        """Increment error count and set latest error message."""
        self.error_count += 1
        self.error_message = msg

    def add_bytes(self, n: int):
        """Track bytes received from API."""
        self.bytes_transferred += n


def _log_ingestion_start(
    source_name: str,
    operation: str,
    metadata: Optional[dict] = None,
) -> Optional[int]:
    """Insert a running ingestion_log row. Returns the row ID or None."""
    try:
        conn = _connect()
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO ingestion_log
                        (source_name, operation, started_at, status, metadata)
                    VALUES (%s, %s, NOW(), 'running', %s)
                    RETURNING id
                    """,
                    (
                        source_name,
                        operation,
                        json.dumps(metadata) if metadata else None,
                    ),
                )
                row = cur.fetchone()
                ingestion_id = row[0] if row else None
        conn.close()
        log.debug(f"Ingestion started: {source_name}/{operation} (id={ingestion_id})")
        return ingestion_id
    except Exception as e:
        log.warning(f"data_registry: log_ingestion_start failed (non-critical): {e}")
        return None


def _log_ingestion_complete(
    ingestion_id: Optional[int],
    status: str,
    duration_seconds: float,
    records_fetched: int = 0,
    records_new: int = 0,
    records_duplicate: int = 0,
    records_failed: int = 0,
    api_calls_made: int = 0,
    bytes_transferred: int = 0,
    error_count: int = 0,
    error_message: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> bool:
    """Update the ingestion_log row with final results."""
    if ingestion_id is None:
        return False

    try:
        conn = _connect()
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE ingestion_log SET
                        completed_at      = NOW(),
                        duration_seconds  = %s,
                        status            = %s,
                        records_fetched   = %s,
                        records_new       = %s,
                        records_duplicate = %s,
                        records_failed    = %s,
                        api_calls_made    = %s,
                        bytes_transferred = %s,
                        error_count       = %s,
                        error_message     = %s,
                        metadata          = %s
                    WHERE id = %s
                    """,
                    (
                        duration_seconds,
                        status,
                        records_fetched,
                        records_new,
                        records_duplicate,
                        records_failed,
                        api_calls_made,
                        bytes_transferred,
                        error_count,
                        error_message,
                        json.dumps(metadata) if metadata else None,
                        ingestion_id,
                    ),
                )
        conn.close()
        log.info(
            f"Ingestion complete: id={ingestion_id} status={status} "
            f"fetched={records_fetched} new={records_new} errors={error_count} "
            f"duration={duration_seconds:.1f}s"
        )
        return True
    except Exception as e:
        log.warning(f"data_registry: log_ingestion_complete failed (non-critical): {e}")
        return False


# ─────────────────────────────────────────────────────────────────
# Standalone logging (for scripts that don't use the context manager)
# ─────────────────────────────────────────────────────────────────


def log_ingestion(
    source_name: str,
    operation: str,
    status: str,
    *,
    records_fetched: int = 0,
    records_new: int = 0,
    records_duplicate: int = 0,
    records_failed: int = 0,
    api_calls_made: int = 0,
    bytes_transferred: int = 0,
    error_count: int = 0,
    error_message: Optional[str] = None,
    duration_seconds: Optional[float] = None,
    metadata: Optional[dict] = None,
) -> bool:
    """One-shot ingestion log — writes a complete row in a single call.

    Use this when you don't want the context manager pattern.
    """
    try:
        conn = _connect()
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO ingestion_log
                        (source_name, operation, started_at, completed_at,
                         duration_seconds, status, records_fetched, records_new,
                         records_duplicate, records_failed, api_calls_made,
                         bytes_transferred, error_count, error_message, metadata)
                    VALUES (%s, %s, NOW(), NOW(), %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        source_name,
                        operation,
                        duration_seconds,
                        status,
                        records_fetched,
                        records_new,
                        records_duplicate,
                        records_failed,
                        api_calls_made,
                        bytes_transferred,
                        error_count,
                        error_message,
                        json.dumps(metadata) if metadata else None,
                    ),
                )
        conn.close()
        log.info(
            f"Ingestion logged: {source_name}/{operation} → {status} ({records_fetched} records)"
        )
        return True
    except Exception as e:
        log.warning(f"data_registry: log_ingestion failed (non-critical): {e}")
        return False


# ─────────────────────────────────────────────────────────────────
# Coverage Tracking
# ─────────────────────────────────────────────────────────────────


def update_coverage(
    source_name: str,
    game_date: str,
    *,
    market: Optional[str] = None,
    book_name: Optional[str] = None,
    record_count: int = 0,
    player_count: int = 0,
    has_actuals: bool = False,
    has_enrichment: bool = False,
    quality_score: Optional[float] = None,
) -> bool:
    """Upsert a coverage record for a source/date/market/book combination."""
    try:
        conn = _connect()
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO data_coverage
                        (source_name, game_date, market, book_name,
                         record_count, player_count, has_actuals,
                         has_enrichment, quality_score, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (source_name, game_date, COALESCE(market, ''), COALESCE(book_name, ''))
                    DO UPDATE SET  -- uses idx_coverage_unique
                        record_count   = EXCLUDED.record_count,
                        player_count   = EXCLUDED.player_count,
                        has_actuals    = EXCLUDED.has_actuals OR data_coverage.has_actuals,
                        has_enrichment = EXCLUDED.has_enrichment OR data_coverage.has_enrichment,
                        quality_score  = COALESCE(EXCLUDED.quality_score, data_coverage.quality_score),
                        updated_at     = NOW()
                    """,
                    (
                        source_name,
                        game_date,
                        market,
                        book_name,
                        record_count,
                        player_count,
                        has_actuals,
                        has_enrichment,
                        quality_score,
                    ),
                )
        conn.close()
        return True
    except Exception as e:
        log.warning(f"data_registry: update_coverage failed (non-critical): {e}")
        return False


def get_coverage_report(
    source_name: str,
    start_date: str,
    end_date: str,
) -> Optional[dict]:
    """Get coverage summary for a source over a date range.

    Returns:
        {
            "source": "bettingpros_props",
            "date_range": ["2026-03-01", "2026-03-14"],
            "total_days": 14,
            "days_with_data": 12,
            "days_missing": ["2026-03-05", "2026-03-09"],
            "total_records": 34521,
            "avg_records_per_day": 2877,
            "days_with_actuals": 10,
            "days_with_enrichment": 12,
            "coverage_pct": 85.7,
            "books_breakdown": {"draftkings": 12, "fanduel": 11, ...}
        }
    """
    try:
        conn = _connect()
        with conn.cursor() as cur:
            # Get coverage data
            cur.execute(
                """
                SELECT game_date, book_name, market,
                       SUM(record_count) as records,
                       SUM(player_count) as players,
                       BOOL_OR(has_actuals) as has_actuals,
                       BOOL_OR(has_enrichment) as has_enrichment
                FROM data_coverage
                WHERE source_name = %s
                  AND game_date BETWEEN %s AND %s
                GROUP BY game_date, book_name, market
                ORDER BY game_date
                """,
                (source_name, start_date, end_date),
            )
            rows = cur.fetchall()

            # Generate all dates in range
            from datetime import timedelta

            start = datetime.strptime(start_date, "%Y-%m-%d").date()
            end = datetime.strptime(end_date, "%Y-%m-%d").date()
            all_dates = set()
            d = start
            while d <= end:
                all_dates.add(d)
                d += timedelta(days=1)

            # Process results
            dates_with_data = set()
            dates_with_actuals = set()
            dates_with_enrichment = set()
            total_records = 0
            books_breakdown = {}

            for game_date, book_name, _market, records, _players, has_act, has_enr in rows:
                dates_with_data.add(game_date)
                total_records += records or 0
                if has_act:
                    dates_with_actuals.add(game_date)
                if has_enr:
                    dates_with_enrichment.add(game_date)
                if book_name:
                    books_breakdown[book_name] = books_breakdown.get(book_name, 0) + 1

            missing = sorted(str(d) for d in (all_dates - dates_with_data))
            days_with = len(dates_with_data)
            total_days = len(all_dates)

        conn.close()

        return {
            "source": source_name,
            "date_range": [start_date, end_date],
            "total_days": total_days,
            "days_with_data": days_with,
            "days_missing": missing,
            "total_records": total_records,
            "avg_records_per_day": round(total_records / days_with, 1) if days_with > 0 else 0,
            "days_with_actuals": len(dates_with_actuals),
            "days_with_enrichment": len(dates_with_enrichment),
            "coverage_pct": round(days_with / total_days * 100, 1) if total_days > 0 else 0,
            "books_breakdown": books_breakdown,
        }

    except Exception as e:
        log.warning(f"data_registry: get_coverage_report failed: {e}")
        return None


def get_source_health(source_name: Optional[str] = None) -> Optional[list]:
    """Get health status for one or all data sources.

    Returns list of dicts:
        [{
            "source": "bettingpros_props",
            "last_ingestion": "2026-03-14T14:20:00",
            "last_status": "success",
            "last_records": 2684,
            "age_hours": 2.3,
            "sla_max_age_hours": 4,
            "sla_ok": True,
            "success_rate_24h": 95.0,
            "total_ingestions_24h": 6,
            "total_errors_24h": 1,
        }]
    """
    try:
        conn = _connect()
        with conn.cursor() as cur:
            if source_name:
                cur.execute(
                    """
                    WITH latest AS (
                        SELECT DISTINCT ON (source_name)
                            source_name, status, records_fetched,
                            completed_at, error_message
                        FROM ingestion_log
                        WHERE source_name = %s
                        ORDER BY source_name, started_at DESC
                    ),
                    daily AS (
                        SELECT source_name,
                               COUNT(*) as total,
                               COUNT(*) FILTER (WHERE status = 'success') as successes,
                               SUM(error_count) as errors
                        FROM ingestion_log
                        WHERE started_at >= NOW() - INTERVAL '24 hours'
                          AND source_name = %s
                        GROUP BY source_name
                    )
                    SELECT ds.name, ds.provider, ds.sla_max_age_hours, ds.enabled,
                           l.status, l.records_fetched, l.completed_at, l.error_message,
                           d.total, d.successes, d.errors,
                           EXTRACT(EPOCH FROM (NOW() - l.completed_at)) / 3600.0 as age_hours
                    FROM data_sources ds
                    LEFT JOIN latest l ON ds.name = l.source_name
                    LEFT JOIN daily d ON ds.name = d.source_name
                    WHERE ds.name = %s
                    ORDER BY ds.name
                    """,
                    (source_name, source_name, source_name),
                )
            else:
                cur.execute(
                    """
                    WITH latest AS (
                        SELECT DISTINCT ON (source_name)
                            source_name, status, records_fetched,
                            completed_at, error_message
                        FROM ingestion_log
                        ORDER BY source_name, started_at DESC
                    ),
                    daily AS (
                        SELECT source_name,
                               COUNT(*) as total,
                               COUNT(*) FILTER (WHERE status = 'success') as successes,
                               SUM(error_count) as errors
                        FROM ingestion_log
                        WHERE started_at >= NOW() - INTERVAL '24 hours'
                        GROUP BY source_name
                    )
                    SELECT ds.name, ds.provider, ds.sla_max_age_hours, ds.enabled,
                           l.status, l.records_fetched, l.completed_at, l.error_message,
                           d.total, d.successes, d.errors,
                           EXTRACT(EPOCH FROM (NOW() - l.completed_at)) / 3600.0 as age_hours
                    FROM data_sources ds
                    LEFT JOIN latest l ON ds.name = l.source_name
                    LEFT JOIN daily d ON ds.name = d.source_name
                    ORDER BY ds.name
                    """,
                )
            rows = cur.fetchall()
        conn.close()

        results = []
        for row in rows:
            (
                name,
                provider,
                sla_hours,
                enabled,
                last_status,
                last_records,
                completed_at,
                last_error,
                total_24h,
                successes_24h,
                errors_24h,
                age_hours,
            ) = row

            sla_ok = True
            if sla_hours and age_hours:
                sla_ok = age_hours <= sla_hours

            results.append(
                {
                    "source": name,
                    "provider": provider,
                    "enabled": enabled,
                    "last_ingestion": completed_at.isoformat() if completed_at else None,
                    "last_status": last_status,
                    "last_records": last_records,
                    "last_error": last_error,
                    "age_hours": round(age_hours, 1) if age_hours else None,
                    "sla_max_age_hours": sla_hours,
                    "sla_ok": sla_ok,
                    "success_rate_24h": round(
                        (successes_24h / total_24h * 100) if total_24h else 0, 1
                    ),
                    "total_ingestions_24h": total_24h or 0,
                    "total_errors_24h": errors_24h or 0,
                }
            )

        return results

    except Exception as e:
        log.warning(f"data_registry: get_source_health failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────
# Alerts
# ─────────────────────────────────────────────────────────────────


def raise_alert(
    alert_type: str,
    severity: str,
    source: str,
    title: str,
    message: str,
    *,
    metadata: Optional[dict] = None,
) -> Optional[int]:
    """Raise an alert for Atlas to consume.

    Returns alert ID or None.
    """
    try:
        conn = _connect()
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO atlas_alerts
                        (alert_type, severity, source, title, message, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        alert_type,
                        severity,
                        source,
                        title,
                        message,
                        json.dumps(metadata) if metadata else None,
                    ),
                )
                row = cur.fetchone()
                alert_id = row[0] if row else None
        conn.close()
        log.info(f"Atlas alert raised: [{severity}] {title} (id={alert_id})")
        return alert_id
    except Exception as e:
        log.warning(f"data_registry: raise_alert failed (non-critical): {e}")
        return None


def resolve_alert(alert_id: int) -> bool:
    """Mark an alert as resolved."""
    try:
        conn = _connect()
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE atlas_alerts SET resolved = TRUE, resolved_at = NOW() WHERE id = %s",
                    (alert_id,),
                )
        conn.close()
        return True
    except Exception as e:
        log.warning(f"data_registry: resolve_alert failed: {e}")
        return False


# ─────────────────────────────────────────────────────────────────
# Heartbeats
# ─────────────────────────────────────────────────────────────────


def heartbeat(
    service_name: str,
    status: str = "healthy",
    *,
    uptime_seconds: Optional[int] = None,
    version: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> bool:
    """Send a heartbeat from a service. Upserts the row."""
    try:
        conn = _connect()
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO system_heartbeats
                        (service_name, status, heartbeat_at, uptime_seconds, version, metadata)
                    VALUES (%s, %s, NOW(), %s, %s, %s)
                    ON CONFLICT (service_name) DO UPDATE SET
                        status         = EXCLUDED.status,
                        heartbeat_at   = NOW(),
                        uptime_seconds = EXCLUDED.uptime_seconds,
                        version        = EXCLUDED.version,
                        metadata       = EXCLUDED.metadata
                    """,
                    (
                        service_name,
                        status,
                        uptime_seconds,
                        version,
                        json.dumps(metadata) if metadata else None,
                    ),
                )
        conn.close()
        return True
    except Exception as e:
        log.warning(f"data_registry: heartbeat failed (non-critical): {e}")
        return False
