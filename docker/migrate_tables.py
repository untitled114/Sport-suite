#!/usr/bin/env python3
"""
Migrate missing tables to consolidated nba_reference database.
"""
import psycopg2
import os

# Source databases
PLAYER_DB = {
    'host': 'localhost',
    'port': 5536,
    'user': os.getenv('DB_USER', 'nba_user'),
    'password': os.getenv('DB_PASSWORD'),
    'database': 'nba_players'
}

GAMES_DB = {
    'host': 'localhost',
    'port': 5537,
    'user': os.getenv('DB_USER', 'nba_user'),
    'password': os.getenv('DB_PASSWORD'),
    'database': 'nba_games'
}

# Target database
TARGET_DB = {
    'host': 'localhost',
    'port': 5540,
    'user': os.getenv('DB_USER', 'nba_user'),
    'password': os.getenv('DB_PASSWORD'),
    'database': 'nba_reference'
}

def migrate_player_season_stats():
    """Migrate player_season_stats from port 5536 to port 5540."""
    print("\n1. Migrating player_season_stats...")

    source_conn = psycopg2.connect(**PLAYER_DB)
    target_conn = psycopg2.connect(**TARGET_DB)

    try:
        # Fetch all records from source
        source_cur = source_conn.cursor()
        source_cur.execute("SELECT * FROM player_season_stats ORDER BY player_id, season")
        rows = source_cur.fetchall()

        print(f"   Fetched {len(rows)} records from source")

        # Insert into target
        target_cur = target_conn.cursor()

        insert_query = """
        INSERT INTO player_season_stats (
            player_id, season, games_played, minutes_per_game,
            ppg, rpg, apg, spg, bpg, tpg,
            fg_pct, three_pt_pct, ft_pct, usage_rate, true_shooting_pct,
            per, ppg_per100, rpg_per100, apg_per100, created_at
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        ON CONFLICT (player_id, season) DO NOTHING
        """

        target_cur.executemany(insert_query, rows)
        target_conn.commit()

        print(f"   ✓ Inserted {target_cur.rowcount} records into target")

    finally:
        source_conn.close()
        target_conn.close()


def migrate_team_game_logs():
    """Migrate team_game_logs from port 5537 to port 5540."""
    print("\n2. Migrating team_game_logs...")

    source_conn = psycopg2.connect(**GAMES_DB)
    target_conn = psycopg2.connect(**TARGET_DB)

    try:
        # Fetch all records from source
        source_cur = source_conn.cursor()
        source_cur.execute("SELECT * FROM team_game_logs ORDER BY game_log_id")
        rows = source_cur.fetchall()

        print(f"   Fetched {len(rows)} records from source")

        # Insert into target
        target_cur = target_conn.cursor()

        insert_query = """
        INSERT INTO team_game_logs (
            game_log_id, team_abbrev, game_id, game_date, season, opponent, is_home,
            points, possessions, pace, offensive_rating, defensive_rating,
            fg_made, fg_attempted, three_pt_made, three_pt_attempted,
            ft_made, ft_attempted, rebounds, assists, turnovers, created_at
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        ON CONFLICT (game_log_id) DO NOTHING
        """

        target_cur.executemany(insert_query, rows)
        target_conn.commit()

        print(f"   ✓ Inserted {target_cur.rowcount} records into target")

    finally:
        source_conn.close()
        target_conn.close()


def verify_migration():
    """Verify tables exist and have data."""
    print("\n3. Verifying migration...")

    conn = psycopg2.connect(**TARGET_DB)
    cur = conn.cursor()

    # Check player_season_stats
    cur.execute("SELECT COUNT(*) FROM player_season_stats")
    count = cur.fetchone()[0]
    print(f"   player_season_stats: {count} records")

    # Check team_game_logs
    cur.execute("SELECT COUNT(*) FROM team_game_logs")
    count = cur.fetchone()[0]
    print(f"   team_game_logs: {count} records")

    conn.close()
    print("\n✓ Migration complete!\n")


if __name__ == '__main__':
    print("="*60)
    print("  Table Migration: nba_reference Database")
    print("="*60)

    try:
        migrate_player_season_stats()
        migrate_team_game_logs()
        verify_migration()

    except Exception as e:
        print(f"\n✗ Migration failed: {e}")
        import traceback
        traceback.print_exc()
