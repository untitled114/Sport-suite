#!/usr/bin/env python3
"""
Complete nba_reference Database Consolidation
=============================================
Migrates ALL missing data to fix 40% data loss issue.

Missing Data Discovered:
- player_profile: 423 records (40%)
- player_game_logs: 648 records (33%)
- player_rolling_stats: 923 records (16%)

Migration Order (respects foreign key dependencies):
1. player_profile FIRST (referenced by other tables)
2. player_game_logs (requires player_profile)
3. player_rolling_stats (requires player_profile)
"""
import psycopg2
from datetime import datetime
import os

# Source database (complete data)
SOURCE_DB = {
    'host': 'localhost',
    'port': 5536,
    'user': os.getenv('DB_USER', 'nba_user'),
    'password': os.getenv('DB_PASSWORD'),
    'database': 'nba_players'
}

# Target database (incomplete data)
TARGET_DB = {
    'host': 'localhost',
    'port': 5540,
    'user': os.getenv('DB_USER', 'nba_user'),
    'password': os.getenv('DB_PASSWORD'),
    'database': 'nba_reference'
}


def migrate_player_profile():
    """STEP 1: Migrate missing player_profile records (fixes foreign key issue)."""
    print("\n" + "="*80)
    print("STEP 1: Migrating player_profile (Foundation Data)")
    print("="*80)

    source_conn = psycopg2.connect(**SOURCE_DB)
    target_conn = psycopg2.connect(**TARGET_DB)

    try:
        # Get all player_ids from source
        source_cur = source_conn.cursor()
        source_cur.execute("SELECT player_id FROM player_profile ORDER BY player_id")
        source_ids = {row[0] for row in source_cur.fetchall()}
        print(f"  Source (port 5536): {len(source_ids)} players")

        # Get all player_ids from target
        target_cur = target_conn.cursor()
        target_cur.execute("SELECT player_id FROM player_profile")
        target_ids = {row[0] for row in target_cur.fetchall()}
        print(f"  Target (port 5540): {len(target_ids)} players")

        # Find missing player_ids
        missing_ids = source_ids - target_ids
        print(f"  Missing: {len(missing_ids)} players")

        if not missing_ids:
            print("  ✓ No missing players to migrate")
            return 0

        # Fetch full records from source
        print(f"\n  Migrating {len(missing_ids)} players...")
        placeholders = ','.join(['%s'] * len(missing_ids))
        source_cur.execute(f"""
            SELECT * FROM player_profile
            WHERE player_id IN ({placeholders})
            ORDER BY player_id
        """, tuple(missing_ids))
        rows = source_cur.fetchall()

        print(f"  Fetched {len(rows)} records from source")

        # Get column names from source
        source_cur.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'player_profile'
            ORDER BY ordinal_position
        """)
        source_columns = [row[0] for row in source_cur.fetchall()]

        # Get column names from target
        target_cur = target_conn.cursor()
        target_cur.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'player_profile'
            ORDER BY ordinal_position
        """)
        target_columns = [row[0] for row in target_cur.fetchall()]

        # Use only columns that exist in BOTH databases
        common_columns = [c for c in source_columns if c in target_columns]
        print(f"  Common columns: {len(common_columns)} (source: {len(source_columns)}, target: {len(target_columns)})")

        # Map column positions
        col_indices = [source_columns.index(c) for c in common_columns]

        # Extract only common columns from rows
        filtered_rows = [[row[i] for i in col_indices] for row in rows]

        # Insert into target
        cols_str = ', '.join(common_columns)
        vals_str = ', '.join(['%s'] * len(common_columns))

        insert_query = f"""
            INSERT INTO player_profile ({cols_str})
            VALUES ({vals_str})
            ON CONFLICT (player_id) DO NOTHING
        """

        target_cur.executemany(insert_query, filtered_rows)
        target_conn.commit()

        inserted = target_cur.rowcount
        print(f"  ✓ Inserted {inserted} players into target")

        return inserted

    finally:
        source_conn.close()
        target_conn.close()


def migrate_player_game_logs():
    """STEP 2: Migrate missing player_game_logs (November 2025)."""
    print("\n" + "="*80)
    print("STEP 2: Migrating player_game_logs (November 2025)")
    print("="*80)

    source_conn = psycopg2.connect(**SOURCE_DB)
    target_conn = psycopg2.connect(**TARGET_DB)

    try:
        # Get all November 2025 game_log_ids from source
        source_cur = source_conn.cursor()
        source_cur.execute("""
            SELECT game_log_id, game_date
            FROM player_game_logs
            WHERE game_date >= '2025-11-01'
            ORDER BY game_log_id
        """)
        source_keys = {(row[0], row[1]) for row in source_cur.fetchall()}
        print(f"  Source (port 5536): {len(source_keys)} game logs")

        # Get all November 2025 game_log_ids from target
        target_cur = target_conn.cursor()
        target_cur.execute("""
            SELECT game_log_id, game_date
            FROM player_game_logs
            WHERE game_date >= '2025-11-01'
        """)
        target_keys = {(row[0], row[1]) for row in target_cur.fetchall()}
        print(f"  Target (port 5540): {len(target_keys)} game logs")

        # Find missing keys
        missing_keys = source_keys - target_keys
        print(f"  Missing: {len(missing_keys)} game logs")

        if not missing_keys:
            print("  ✓ No missing game logs to migrate")
            return 0

        # Fetch full records from source
        print(f"\n  Migrating {len(missing_keys)} game logs...")
        missing_ids = [k[0] for k in missing_keys]
        placeholders = ','.join(['%s'] * len(missing_ids))
        source_cur.execute(f"""
            SELECT * FROM player_game_logs
            WHERE game_log_id IN ({placeholders})
            ORDER BY game_log_id
        """, tuple(missing_ids))
        rows = source_cur.fetchall()

        print(f"  Fetched {len(rows)} records from source")

        # Get column names
        source_cur.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'player_game_logs'
            ORDER BY ordinal_position
        """)
        columns = [row[0] for row in source_cur.fetchall()]

        # Insert into target (batch processing for TimescaleDB)
        target_cur = target_conn.cursor()
        cols_str = ', '.join(columns)
        vals_str = ', '.join(['%s'] * len(columns))

        insert_query = f"""
            INSERT INTO player_game_logs ({cols_str})
            VALUES ({vals_str})
            ON CONFLICT (game_log_id, game_date) DO NOTHING
        """

        # Batch insert (100 rows at a time for TimescaleDB performance)
        batch_size = 100
        total_inserted = 0

        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            target_cur.executemany(insert_query, batch)
            target_conn.commit()
            total_inserted += target_cur.rowcount
            if (i + batch_size) % 300 == 0:
                print(f"    Progress: {min(i + batch_size, len(rows))}/{len(rows)} rows")

        print(f"  ✓ Inserted {total_inserted} game logs into target")

        return total_inserted

    finally:
        source_conn.close()
        target_conn.close()


def migrate_player_rolling_stats():
    """STEP 3: Migrate missing player_rolling_stats (November 2025)."""
    print("\n" + "="*80)
    print("STEP 3: Migrating player_rolling_stats (November 2025)")
    print("="*80)

    source_conn = psycopg2.connect(**SOURCE_DB)
    target_conn = psycopg2.connect(**TARGET_DB)

    try:
        # Get all November 2025 rolling stats from source
        source_cur = source_conn.cursor()
        source_cur.execute("""
            SELECT player_id, as_of_date, window_size
            FROM player_rolling_stats
            WHERE as_of_date >= '2025-11-01'
        """)
        source_keys = {(row[0], row[1], row[2]) for row in source_cur.fetchall()}
        print(f"  Source (port 5536): {len(source_keys)} rolling stats")

        # Get all November 2025 rolling stats from target
        target_cur = target_conn.cursor()
        target_cur.execute("""
            SELECT player_id, as_of_date, window_size
            FROM player_rolling_stats
            WHERE as_of_date >= '2025-11-01'
        """)
        target_keys = {(row[0], row[1], row[2]) for row in target_cur.fetchall()}
        print(f"  Target (port 5540): {len(target_keys)} rolling stats")

        # Find missing keys
        missing_keys = source_keys - target_keys
        print(f"  Missing: {len(missing_keys)} rolling stats")

        if not missing_keys:
            print("  ✓ No missing rolling stats to migrate")
            return 0

        # Fetch full records from source
        print(f"\n  Migrating {len(missing_keys)} rolling stats...")

        # Build WHERE clause for composite key
        conditions = []
        params = []
        for player_id, as_of_date, window_size in missing_keys:
            conditions.append("(player_id = %s AND as_of_date = %s AND window_size = %s)")
            params.extend([player_id, as_of_date, window_size])

        where_clause = " OR ".join(conditions)

        source_cur.execute(f"""
            SELECT * FROM player_rolling_stats
            WHERE {where_clause}
        """, params)
        rows = source_cur.fetchall()

        print(f"  Fetched {len(rows)} records from source")

        # Get column names from source
        source_cur.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'player_rolling_stats'
            ORDER BY ordinal_position
        """)
        source_columns = [row[0] for row in source_cur.fetchall()]

        # Get column names from target
        target_cur.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'player_rolling_stats'
            ORDER BY ordinal_position
        """)
        target_columns = [row[0] for row in target_cur.fetchall()]

        # Use only columns that exist in BOTH databases
        common_columns = [c for c in source_columns if c in target_columns]
        print(f"  Common columns: {len(common_columns)} (source: {len(source_columns)}, target: {len(target_columns)})")

        # Map column positions
        col_indices = [source_columns.index(c) for c in common_columns]

        # Extract only common columns from rows
        filtered_rows = [[row[i] for i in col_indices] for row in rows]

        # Insert into target
        cols_str = ', '.join(common_columns)
        vals_str = ', '.join(['%s'] * len(common_columns))

        insert_query = f"""
            INSERT INTO player_rolling_stats ({cols_str})
            VALUES ({vals_str})
            ON CONFLICT (player_id, as_of_date, window_size) DO NOTHING
        """

        # Batch insert
        batch_size = 100
        total_inserted = 0

        for i in range(0, len(filtered_rows), batch_size):
            batch = filtered_rows[i:i + batch_size]
            target_cur.executemany(insert_query, batch)
            target_conn.commit()
            total_inserted += target_cur.rowcount
            if (i + batch_size) % 300 == 0:
                print(f"    Progress: {min(i + batch_size, len(filtered_rows))}/{len(filtered_rows)} rows")

        print(f"  ✓ Inserted {total_inserted} rolling stats into target")

        return total_inserted

    finally:
        source_conn.close()
        target_conn.close()


def verify_consolidation():
    """STEP 4: Verify consolidation completed successfully."""
    print("\n" + "="*80)
    print("STEP 4: Verification")
    print("="*80)

    source_conn = psycopg2.connect(**SOURCE_DB)
    target_conn = psycopg2.connect(**TARGET_DB)

    source_cur = source_conn.cursor()
    target_cur = target_conn.cursor()

    success = True

    # Verify player_profile count
    source_cur.execute("SELECT COUNT(*) FROM player_profile")
    source_count = source_cur.fetchone()[0]
    target_cur.execute("SELECT COUNT(*) FROM player_profile")
    target_count = target_cur.fetchone()[0]

    print(f"\n  player_profile:")
    print(f"    Source: {source_count}")
    print(f"    Target: {target_count}")
    if source_count == target_count:
        print(f"    ✓ PASS - Counts match")
    else:
        print(f"    ✗ FAIL - Missing {source_count - target_count} players")
        success = False

    # Verify player_game_logs count (Nov 2025)
    source_cur.execute("SELECT COUNT(*) FROM player_game_logs WHERE game_date >= '2025-11-01'")
    source_count = source_cur.fetchone()[0]
    target_cur.execute("SELECT COUNT(*) FROM player_game_logs WHERE game_date >= '2025-11-01'")
    target_count = target_cur.fetchone()[0]

    print(f"\n  player_game_logs (Nov 2025):")
    print(f"    Source: {source_count}")
    print(f"    Target: {target_count}")
    if source_count == target_count:
        print(f"    ✓ PASS - Counts match (expected 1,921)")
    else:
        print(f"    ✗ FAIL - Missing {source_count - target_count} game logs")
        success = False

    # Verify player_rolling_stats count (Nov 2025)
    source_cur.execute("SELECT COUNT(*) FROM player_rolling_stats WHERE as_of_date >= '2025-11-01'")
    source_count = source_cur.fetchone()[0]
    target_cur.execute("SELECT COUNT(*) FROM player_rolling_stats WHERE as_of_date >= '2025-11-01'")
    target_count = target_cur.fetchone()[0]

    print(f"\n  player_rolling_stats (Nov 2025):")
    print(f"    Source: {source_count}")
    print(f"    Target: {target_count}")
    if source_count == target_count:
        print(f"    ✓ PASS - Counts match (expected 5,731)")
    else:
        print(f"    ✗ FAIL - Missing {source_count - target_count} rolling stats")
        success = False

    # Test specific players (previously returned 0 games)
    test_players = ['Gary Trent Jr', 'Kelly Oubre Jr', 'Michael Porter Jr', 'Jase Richardson']
    print(f"\n  Testing specific players:")
    for player in test_players:
        target_cur.execute("""
            SELECT COUNT(*)
            FROM player_game_logs pgl
            JOIN player_profile pp ON pgl.player_id = pp.player_id
            WHERE pp.full_name = %s AND pgl.game_date >= '2025-11-01'
        """, (player,))
        count = target_cur.fetchone()[0]
        status = "✓" if count > 0 else "✗"
        print(f"    {status} {player}: {count} games")
        if count == 0 and player != 'Jase Richardson':  # Jase might be a rookie with 0 games
            success = False

    source_conn.close()
    target_conn.close()

    return success


def main():
    print("="*80)
    print("  Complete nba_reference Database Consolidation")
    print("  Fixing 40% Data Loss Issue")
    print("="*80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nMigration Order (respects foreign key dependencies):")
    print("  1. player_profile (foundation - referenced by other tables)")
    print("  2. player_game_logs (requires player_profile)")
    print("  3. player_rolling_stats (requires player_profile)")

    try:
        # Step 1: Migrate player_profile FIRST
        profile_inserted = migrate_player_profile()

        # Step 2: Migrate player_game_logs
        game_logs_inserted = migrate_player_game_logs()

        # Step 3: Migrate player_rolling_stats
        rolling_stats_inserted = migrate_player_rolling_stats()

        # Step 4: Verify
        success = verify_consolidation()

        # Summary
        print("\n" + "="*80)
        print("MIGRATION SUMMARY")
        print("="*80)
        print(f"  player_profile migrated: {profile_inserted}")
        print(f"  player_game_logs migrated: {game_logs_inserted}")
        print(f"  player_rolling_stats migrated: {rolling_stats_inserted}")
        print(f"  Verification: {'✓ PASS' if success else '✗ FAIL'}")

        if success:
            print(f"\n✓ CONSOLIDATION SUCCESSFUL!")
            print(f"\nNext Steps:")
            print(f"  1. Feature extraction now has complete data")
            print(f"  2. Run predictions to verify 5 POINTS tier A picks")
            print(f"  3. nba_reference (port 5540) is now production-ready")
        else:
            print(f"\n✗ Consolidation verification failed. Check counts above.")

    except Exception as e:
        print(f"\n✗ Migration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
