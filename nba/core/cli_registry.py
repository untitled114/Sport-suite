#!/usr/bin/env python3
"""
CLI for querying the Atlas Data Registry.

Usage:
    # Source health overview
    python -m nba.core.cli_registry --health

    # Coverage report for a source
    python -m nba.core.cli_registry --coverage bettingpros_props --start 2026-03-01 --end 2026-03-14

    # Recent ingestion log
    python -m nba.core.cli_registry --ingestions --limit 20

    # Unresolved alerts
    python -m nba.core.cli_registry --alerts

    # Full system status (what Atlas would show)
    python -m nba.core.cli_registry --status
"""

import argparse
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

EST = ZoneInfo("America/New_York")

_EST = ZoneInfo("America/New_York")


def _connect():
    import os

    import psycopg2

    return psycopg2.connect(
        host=os.environ.get("DB_HOST", "localhost"),
        port=5541,
        dbname="cephalon_axiom",
        user=os.environ.get("DB_USER", "mlb_user"),
        password=os.environ.get("DB_PASSWORD", ""),
        connect_timeout=5,
    )


def cmd_health(args):
    """Show source health overview."""
    from nba.core.data_registry import get_source_health

    sources = get_source_health()
    if not sources:
        print("No data sources found. Run the schema migration first.")
        return

    print(f"\n{'='*80}")
    print(f"  ATLAS DATA REGISTRY — Source Health")
    print(f"  {datetime.now(_EST).strftime('%Y-%m-%d %I:%M %p EST')}")
    print(f"{'='*80}\n")

    for s in sources:
        if not s["enabled"]:
            continue

        status_icon = "OK" if s["sla_ok"] else "BREACH"
        age_str = f"{float(s['age_hours']):.1f}h ago" if s["age_hours"] is not None else "never"

        print(f"  {s['source']:<30} ", end="")
        print(f"[{status_icon:>6}]  ", end="")
        print(f"last: {age_str:<12} ", end="")
        print(f"records: {s['last_records'] or 0:<6} ", end="")
        print(
            f"24h: {s['success_rate_24h']:.0f}% ({s['total_ingestions_24h']} runs, {s['total_errors_24h']} errors)"
        )

    print()


def cmd_coverage(args):
    """Show coverage report for a source."""
    from nba.core.data_registry import get_coverage_report

    report = get_coverage_report(args.source, args.start, args.end)
    if not report:
        print(f"No coverage data for '{args.source}'.")
        return

    print(f"\n{'='*70}")
    print(f"  Coverage: {report['source']}")
    print(f"  {report['date_range'][0]} to {report['date_range'][1]}")
    print(f"{'='*70}\n")

    print(
        f"  Days with data:      {report['days_with_data']}/{report['total_days']} ({report['coverage_pct']}%)"
    )
    print(f"  Total records:       {report['total_records']:,}")
    print(f"  Avg records/day:     {report['avg_records_per_day']:,.0f}")
    print(f"  Days with actuals:   {report['days_with_actuals']}")
    print(f"  Days with enrichment:{report['days_with_enrichment']}")

    if report["days_missing"]:
        print(f"\n  Missing dates ({len(report['days_missing'])}):")
        for d in report["days_missing"][:20]:
            print(f"    - {d}")
        if len(report["days_missing"]) > 20:
            print(f"    ... and {len(report['days_missing']) - 20} more")

    if report["books_breakdown"]:
        print(f"\n  Books breakdown (days present):")
        for book, count in sorted(report["books_breakdown"].items(), key=lambda x: -x[1]):
            print(f"    {book:<20} {count} days")

    print()


def cmd_ingestions(args):
    """Show recent ingestion log."""
    conn = _connect()
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT source_name, operation, status, records_fetched,
                   api_calls_made, bytes_transferred, error_count,
                   duration_seconds, started_at, error_message
            FROM ingestion_log
            ORDER BY started_at DESC
            LIMIT %s
            """,
            (args.limit,),
        )
        rows = cur.fetchall()
    conn.close()

    if not rows:
        print("No ingestion records found.")
        return

    print(f"\n{'='*100}")
    print(f"  Recent Ingestions (last {args.limit})")
    print(f"{'='*100}\n")

    print(
        f"  {'Source':<28} {'Op':<10} {'Status':<8} {'Records':>8} {'API':>5} {'Bytes':>10} {'Err':>4} {'Time':>7} {'When'}"
    )
    print(f"  {'-'*28} {'-'*10} {'-'*8} {'-'*8} {'-'*5} {'-'*10} {'-'*4} {'-'*7} {'-'*20}")

    for row in rows:
        (source, op, status, records, api_calls, bytes_tx, errors, duration, started, error_msg) = (
            row
        )
        bytes_str = (
            f"{bytes_tx / 1024 / 1024:.1f}MB"
            if bytes_tx and bytes_tx > 1024 * 1024
            else f"{(bytes_tx or 0) / 1024:.0f}KB"
        )
        dur_str = f"{duration:.0f}s" if duration else "-"
        when = started.astimezone(_EST).strftime("%m/%d %I:%M%p") if started else "-"

        print(
            f"  {source:<28} {op:<10} {status:<8} {records or 0:>8} {api_calls or 0:>5} {bytes_str:>10} {errors or 0:>4} {dur_str:>7} {when}"
        )

    print()


def cmd_alerts(args):
    """Show unresolved alerts."""
    conn = _connect()
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, severity, alert_type, source, title, message, created_at
            FROM atlas_alerts
            WHERE resolved = FALSE
            ORDER BY
                CASE severity WHEN 'critical' THEN 1 WHEN 'warning' THEN 2 ELSE 3 END,
                created_at DESC
            LIMIT 50
            """,
        )
        rows = cur.fetchall()
    conn.close()

    if not rows:
        print("\n  No unresolved alerts.\n")
        return

    print(f"\n{'='*80}")
    print(f"  Atlas Alerts — {len(rows)} unresolved")
    print(f"{'='*80}\n")

    for alert_id, severity, atype, source, title, message, created in rows:
        when = created.astimezone(_EST).strftime("%m/%d %I:%M%p") if created else "-"
        print(f"  [{severity.upper():>8}] #{alert_id} {title}")
        print(f"           Source: {source}  Type: {atype}  When: {when}")
        print(f"           {message}")
        print()


def cmd_status(args):
    """Full system status — what Atlas morning brief would show."""
    print(f"\n{'='*80}")
    print(f"  ATLAS SYSTEM STATUS")
    print(f"  {datetime.now(_EST).strftime('%A, %B %d, %Y — %I:%M %p EST')}")
    print(f"{'='*80}")

    # Source health
    cmd_health(args)

    # Recent alerts
    cmd_alerts(args)

    # Last 10 ingestions
    args.limit = 10
    cmd_ingestions(args)


def main():
    parser = argparse.ArgumentParser(description="Atlas Data Registry CLI")
    parser.add_argument("--health", action="store_true", help="Source health overview")
    parser.add_argument("--coverage", metavar="SOURCE", help="Coverage report for a source")
    parser.add_argument(
        "--start", default=(datetime.now(EST) - timedelta(days=14)).strftime("%Y-%m-%d")
    )
    parser.add_argument("--end", default=datetime.now(EST).strftime("%Y-%m-%d"))
    parser.add_argument("--ingestions", action="store_true", help="Recent ingestion log")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--alerts", action="store_true", help="Unresolved alerts")
    parser.add_argument("--status", action="store_true", help="Full system status")

    args = parser.parse_args()

    if args.status:
        cmd_status(args)
    elif args.health:
        cmd_health(args)
    elif args.coverage:
        args.source = args.coverage
        cmd_coverage(args)
    elif args.ingestions:
        cmd_ingestions(args)
    elif args.alerts:
        cmd_alerts(args)
    else:
        # Default: show full status
        cmd_status(args)


if __name__ == "__main__":
    main()
