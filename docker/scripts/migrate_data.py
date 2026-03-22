#!/usr/bin/env python3
"""
Migrate data from 6 legacy PostgreSQL containers into 1 consolidated instance.

Reads from old containers (ports 5536-5541), writes to new sportsuite_db (port 5500).
Uses psycopg2 COPY for speed — handles hypertables correctly by copying row-by-row
through Python rather than relying on pg_dump/sed.

Usage:
    source .env && python3 docker/scripts/migrate_data.py
"""

import io
import os
import sys
import time

import psycopg2

NEW_DB = {
    "host": "localhost",
    "port": int(os.getenv("DB_PORT", 5500)),
    "database": os.getenv("DB_NAME", "sportsuite"),
    "user": os.getenv("DB_USER", "mlb_user"),
    "password": os.getenv("DB_PASSWORD", ""),
}

SOURCES = [
    {
        "name": "players",
        "port": 5536,
        "database": "nba_players",
        "tables": [
            "player_profile",
            "player_season_stats",
            "player_game_logs",
            "player_rolling_stats",
            "player_minutes_projections",
        ],
    },
    {
        "name": "games",
        "port": 5537,
        "database": "nba_games",
        "tables": [
            "games",
            "team_game_logs",
        ],
    },
    {
        "name": "teams",
        "port": 5538,
        "database": "nba_team",
        "tables": [
            "teams",
            "team_season_stats",
            "team_rolling_stats",
            "team_betting_performance",
        ],
    },
    {
        "name": "intelligence",
        "port": 5539,
        "database": "nba_intelligence",
        "tables": [
            "injury_report",
            "lineup_intel",
            "nba_prop_lines",
            "nba_props_xl",
            "nba_line_snapshots",
            "nba_picks_placed",
            "player_form",
            "matchup_history",
            "cheatsheet_data",
            "bp_historical_analytics",
            "bp_dvp_historical",
            "book_historical_accuracy",
            "prop_performance_history",
        ],
    },
    {
        "name": "axiom",
        "port": 5541,
        "database": "cephalon_axiom",
        "tables": [
            "axiom_pipeline_audit",
            "nba_prediction_history",
            "axiom_conviction",
            "axiom_posts",
            "axiom_memory",
            "data_sources",
            "ingestion_log",
            "data_coverage",
            "system_heartbeats",
            "atlas_alerts",
        ],
    },
]


def get_columns(conn, schema, table):
    """Get ordered column names for a table, excluding generated columns."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = %s AND table_name = %s "
            "AND is_generated = 'NEVER' AND generation_expression IS NULL "
            "ORDER BY ordinal_position",
            (schema, table),
        )
        return [r[0] for r in cur.fetchall()]


def copy_table(src_conn, dst_conn, src_table, dst_schema, dst_table):
    """Copy a table using COPY with explicit column lists to handle schema drift."""
    # Find columns that exist in BOTH source and destination
    src_cols = get_columns(src_conn, "public", src_table)
    dst_cols = get_columns(dst_conn, dst_schema, dst_table)

    # Use only columns present in both, preserving destination order
    common = [c for c in dst_cols if c in src_cols]
    if not common:
        return 0

    col_list = ", ".join(common)
    buf = io.BytesIO()

    with src_conn.cursor() as src_cur:
        src_cur.copy_expert(f"COPY (SELECT {col_list} FROM public.{src_table}) TO STDOUT", buf)

    row_bytes = buf.tell()
    if row_bytes == 0:
        return 0

    buf.seek(0)

    with dst_conn.cursor() as dst_cur:
        dst_cur.copy_expert(f"COPY {dst_schema}.{dst_table} ({col_list}) FROM STDIN", buf)
    dst_conn.commit()

    return row_bytes


def count_rows(conn, schema, table):
    """Quick row count."""
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {schema}.{table}")
        return cur.fetchone()[0]


def migrate():
    print("=" * 60)
    print("Sport-Suite Data Migration")
    print("=" * 60)
    print(f"Target: localhost:{NEW_DB['port']}/{NEW_DB['database']}")
    print()

    # Connect to destination
    dst = psycopg2.connect(**NEW_DB)
    dst.autocommit = False  # We'll commit per-table

    total_tables = 0
    total_bytes = 0
    errors = []

    for source in SOURCES:
        schema = source["name"]
        print(f"── {schema} (port {source['port']}) ──")

        try:
            src = psycopg2.connect(
                host="localhost",
                port=source["port"],
                database=source["database"],
                user=os.getenv("DB_USER", "mlb_user"),
                password=os.getenv("DB_PASSWORD", ""),
                connect_timeout=10,
            )
            src.autocommit = True
        except Exception as e:
            print(f"  SKIP: Cannot connect to {source['database']} — {e}")
            continue

        for table in source["tables"]:
            t0 = time.time()
            try:
                # Check if source table exists
                with src.cursor() as cur:
                    cur.execute(
                        "SELECT 1 FROM information_schema.tables "
                        "WHERE table_schema='public' AND table_name=%s",
                        (table,),
                    )
                    if not cur.fetchone():
                        print(f"  {table}: not found in source — skip")
                        continue

                # Truncate destination before loading
                with dst.cursor() as cur:
                    cur.execute(f"TRUNCATE {schema}.{table} CASCADE")
                dst.commit()

                nbytes = copy_table(src, dst, table, schema, table)
                elapsed = time.time() - t0

                rows = count_rows(dst, schema, table)
                size_mb = nbytes / 1024 / 1024

                print(f"  {table}: {rows:,} rows ({size_mb:.1f} MB) in {elapsed:.1f}s")
                total_tables += 1
                total_bytes += nbytes

            except Exception as e:
                dst.rollback()
                errors.append(f"{schema}.{table}: {e}")
                print(f"  {table}: ERROR — {e}")

        src.close()
        print()

    # Reset sequences to match migrated data
    print("Resetting sequences...")
    with dst.cursor() as cur:
        cur.execute("""
            SELECT s.schemaname, s.sequencename,
                   d.refobjsubid::int AS colnum,
                   c.relname AS tablename,
                   a.attname AS colname
            FROM pg_sequences s
            JOIN pg_class seq ON seq.relname = s.sequencename
                AND seq.relnamespace = (SELECT oid FROM pg_namespace WHERE nspname = s.schemaname)
            JOIN pg_depend d ON d.objid = seq.oid AND d.deptype = 'a'
            JOIN pg_class c ON c.oid = d.refobjid
            JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum = d.refobjsubid
            WHERE s.schemaname IN ('players','games','teams','intelligence','axiom','features')
        """)
        for schema, seq, _, table, col in cur.fetchall():
            try:
                cur.execute(
                    f"SELECT setval('{schema}.{seq}', "
                    f"COALESCE((SELECT MAX({col}) FROM {schema}.{table}), 1))"
                )
            except Exception:
                pass
    dst.commit()

    dst.close()

    print("=" * 60)
    print(f"Done: {total_tables} tables, {total_bytes / 1024 / 1024:.0f} MB total")
    if errors:
        print(f"\n{len(errors)} errors:")
        for e in errors:
            print(f"  - {e}")
    print("=" * 60)


if __name__ == "__main__":
    migrate()
