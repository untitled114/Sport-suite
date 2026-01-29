#!/usr/bin/env python3
"""
Deduplicate player_game_logs Table
===================================
Removes duplicate entries for the same player-date pair.

Issue: 31.5% of current season entries are duplicates with corrupted data
- Example: Bennedict Mathurin 2025-10-23 has 2 entries (45 min vs 2707 min)

Strategy:
1. Identify duplicates: GROUP BY player_id, game_date HAVING COUNT(*) > 1
2. Keep best entry: Prefer valid minutes (< 60), then most recent insert
3. Delete duplicates: DELETE FROM WHERE id NOT IN (kept_ids)
"""

import psycopg2
import sys
import logging
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database config
DB_CONFIG = {
    'host': 'localhost',
    'port': 5536,
    'user': os.getenv('DB_USER', 'nba_user'),
    'password': os.getenv('DB_PASSWORD'),
    'database': 'nba_players'
}


def analyze_duplicates(conn):
    """Analyze duplicate entries before cleanup"""
    logger.info("=" * 80)
    logger.info("ANALYZING DUPLICATES")
    logger.info("=" * 80)

    query = """
    SELECT
        COUNT(*) as total_entries,
        COUNT(DISTINCT (player_id, game_date)) as unique_player_games,
        COUNT(*) - COUNT(DISTINCT (player_id, game_date)) as duplicates,
        ROUND(100.0 * (COUNT(*) - COUNT(DISTINCT (player_id, game_date))) / COUNT(*), 2) as dup_pct
    FROM player_game_logs;
    """

    with conn.cursor() as cur:
        cur.execute(query)
        row = cur.fetchone()
        total, unique, dups, pct = row

        logger.info(f"Total entries: {total:,}")
        logger.info(f"Unique player-games: {unique:,}")
        logger.info(f"Duplicates: {dups:,} ({pct}%)")

    # Sample duplicates
    query = """
    SELECT
        pp.full_name,
        pgl.game_date,
        COUNT(*) as dup_count,
        array_agg(pgl.minutes_played ORDER BY pgl.minutes_played) as minutes
    FROM player_game_logs pgl
    JOIN player_profile pp ON pgl.player_id = pp.player_id
    GROUP BY pp.full_name, pgl.game_date
    HAVING COUNT(*) > 1
    ORDER BY dup_count DESC
    LIMIT 10;
    """

    logger.info("\nSample duplicates (top 10):")
    with conn.cursor() as cur:
        cur.execute(query)
        for row in cur.fetchall():
            player, date, count, minutes = row
            logger.info(f"  {player} {date}: {count} entries, minutes: {minutes}")


def deduplicate_gamelogs(conn, dry_run=False):
    """Remove duplicate player_game_logs entries"""
    logger.info("\n" + "=" * 80)
    logger.info("DEDUPLICATING PLAYER_GAME_LOGS")
    logger.info("=" * 80)

    if dry_run:
        logger.info("⚠️  DRY RUN MODE - No changes will be made\n")

    # Strategy: For each duplicate group, keep the entry with:
    # 1. Valid minutes (< 60)
    # 2. If multiple valid or all invalid, keep most complete data
    # 3. As last resort, keep lowest ID (oldest insert)

    query = """
    WITH duplicates AS (
        -- Find all player-game pairs with duplicates
        SELECT player_id, game_date
        FROM player_game_logs
        GROUP BY player_id, game_date
        HAVING COUNT(*) > 1
    ),
    ranked_entries AS (
        -- Rank entries within each duplicate group
        SELECT
            pgl.game_log_id,
            pgl.player_id,
            pgl.game_date,
            pgl.minutes_played,
            pgl.steals,
            pgl.blocks,
            pgl.turnovers,
            -- Prefer entries with valid minutes (< 60)
            CASE
                WHEN pgl.minutes_played < 60 THEN 1
                ELSE 2
            END as minutes_rank,
            -- Prefer entries with complete stats (non-null steals/blocks/turnovers)
            CASE
                WHEN pgl.steals IS NOT NULL AND pgl.blocks IS NOT NULL AND pgl.turnovers IS NOT NULL THEN 1
                ELSE 2
            END as completeness_rank,
            -- Use game_log_id as tiebreaker (keep oldest)
            pgl.game_log_id as id_rank
        FROM player_game_logs pgl
        INNER JOIN duplicates d ON pgl.player_id = d.player_id AND pgl.game_date = d.game_date
    ),
    entries_to_keep AS (
        -- Select the best entry from each duplicate group
        SELECT DISTINCT ON (player_id, game_date)
            game_log_id
        FROM ranked_entries
        ORDER BY player_id, game_date, minutes_rank, completeness_rank, id_rank
    ),
    entries_to_delete AS (
        -- All duplicate entries except the ones we're keeping
        SELECT pgl.game_log_id
        FROM player_game_logs pgl
        INNER JOIN duplicates d ON pgl.player_id = d.player_id AND pgl.game_date = d.game_date
        WHERE pgl.game_log_id NOT IN (SELECT game_log_id FROM entries_to_keep)
    )
    SELECT COUNT(*) FROM entries_to_delete;
    """

    with conn.cursor() as cur:
        cur.execute(query)
        to_delete = cur.fetchone()[0]
        logger.info(f"Entries to delete: {to_delete:,}")

    if to_delete == 0:
        logger.info("✅ No duplicates found")
        return 0

    if dry_run:
        logger.info("\n✅ Dry run complete - no changes made")
        return to_delete

    # Execute deletion
    delete_query = """
    WITH duplicates AS (
        SELECT player_id, game_date
        FROM player_game_logs
        GROUP BY player_id, game_date
        HAVING COUNT(*) > 1
    ),
    ranked_entries AS (
        SELECT
            pgl.game_log_id,
            pgl.player_id,
            pgl.game_date,
            CASE
                WHEN pgl.minutes_played < 60 THEN 1
                ELSE 2
            END as minutes_rank,
            CASE
                WHEN pgl.steals IS NOT NULL AND pgl.blocks IS NOT NULL AND pgl.turnovers IS NOT NULL THEN 1
                ELSE 2
            END as completeness_rank,
            pgl.game_log_id as id_rank
        FROM player_game_logs pgl
        INNER JOIN duplicates d ON pgl.player_id = d.player_id AND pgl.game_date = d.game_date
    ),
    entries_to_keep AS (
        SELECT DISTINCT ON (player_id, game_date)
            game_log_id
        FROM ranked_entries
        ORDER BY player_id, game_date, minutes_rank, completeness_rank, id_rank
    )
    DELETE FROM player_game_logs
    WHERE game_log_id IN (
        SELECT pgl.game_log_id
        FROM player_game_logs pgl
        INNER JOIN duplicates d ON pgl.player_id = d.player_id AND pgl.game_date = d.game_date
        WHERE pgl.game_log_id NOT IN (SELECT game_log_id FROM entries_to_keep)
    );
    """

    logger.info("\nExecuting deletion...")
    with conn.cursor() as cur:
        cur.execute(delete_query)
        deleted = cur.rowcount
        conn.commit()
        logger.info(f"✅ Deleted {deleted:,} duplicate entries")

    return deleted


def verify_cleanup(conn):
    """Verify no duplicates remain"""
    logger.info("\n" + "=" * 80)
    logger.info("VERIFYING CLEANUP")
    logger.info("=" * 80)

    query = """
    SELECT
        COUNT(*) - COUNT(DISTINCT (player_id, game_date)) as remaining_dups
    FROM player_game_logs;
    """

    with conn.cursor() as cur:
        cur.execute(query)
        remaining = cur.fetchone()[0]

        if remaining == 0:
            logger.info("✅ No duplicates remaining")
        else:
            logger.error(f"❌ Still have {remaining:,} duplicates")
            return False

    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Deduplicate player_game_logs table')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be deleted without actually deleting')
    args = parser.parse_args()

    try:
        # Connect to database
        logger.info("Connecting to nba_players database...")
        conn = psycopg2.connect(**DB_CONFIG)
        logger.info("✅ Connected\n")

        # Analyze duplicates
        analyze_duplicates(conn)

        # Deduplicate
        deleted = deduplicate_gamelogs(conn, dry_run=args.dry_run)

        if not args.dry_run and deleted > 0:
            # Verify cleanup
            verify_cleanup(conn)

        logger.info("\n" + "=" * 80)
        logger.info("DEDUPLICATION COMPLETE")
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.error(f"Error: {e}")
        return 1

    finally:
        if 'conn' in locals():
            conn.close()


if __name__ == '__main__':
    sys.exit(main())
