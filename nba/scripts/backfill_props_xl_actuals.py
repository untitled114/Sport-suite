#!/usr/bin/env python3
"""
Backfill actual_value for nba_props_xl from player_game_logs
=============================================================
Populates actual_value in nba_props_xl table by matching props to
actual game results in player_game_logs.

Handles player name normalization to avoid deduplication issues.

Usage:
    python3 backfill_props_xl_actuals.py --start-date 2025-10-30 --end-date 2025-11-10
    python3 backfill_props_xl_actuals.py --date 2025-11-07
    python3 backfill_props_xl_actuals.py --all
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import psycopg2

from nba.config.database import get_intelligence_db_config, get_players_db_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Database configs
PLAYERS_DB = get_players_db_config()
INTEL_DB = get_intelligence_db_config()

# Stat type mapping from nba_props_xl to player_game_logs columns
STAT_MAP = {
    "POINTS": "points",
    "REBOUNDS": "rebounds",
    "ASSISTS": "assists",
    "THREES": "three_pointers_made",
    "STEALS": "steals",
    "BLOCKS": "blocks",
}


def normalize_player_name(name):
    """
    Normalize player name for matching
    Handles common variations
    """
    if not name:
        return None

    # Basic normalization
    normalized = name.strip().lower()

    # Handle common suffixes
    suffixes = [" jr.", " jr", " sr.", " sr", " iii", " ii", " iv"]
    for suffix in suffixes:
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)].strip()

    return normalized


def backfill_results(start_date=None, end_date=None, specific_date=None):
    """
    Backfill actual_value for props in nba_props_xl

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        specific_date: Specific date to backfill (YYYY-MM-DD)
    """
    conn_players = psycopg2.connect(**PLAYERS_DB)
    conn_intel = psycopg2.connect(**INTEL_DB)

    # Build date filter with parameterized queries
    date_filter = ""
    date_params = []
    if specific_date:
        date_filter = "AND game_date = %s"
        date_params.append(specific_date)
    elif start_date and end_date:
        date_filter = "AND game_date BETWEEN %s AND %s"
        date_params.extend([start_date, end_date])
    elif start_date:
        date_filter = "AND game_date >= %s"
        date_params.append(start_date)

    # Get props that need backfilling from nba_props_xl
    query_props = f"""
    SELECT DISTINCT player_name, game_date, stat_type
    FROM nba_props_xl
    WHERE actual_value IS NULL
      AND stat_type IN ('POINTS', 'REBOUNDS', 'ASSISTS', 'THREES', 'STEALS', 'BLOCKS')
      {date_filter}
    ORDER BY game_date, player_name
    """

    cursor_intel = conn_intel.cursor()
    cursor_intel.execute(query_props, date_params or None)
    props_to_fill = cursor_intel.fetchall()
    cursor_intel.close()

    logger.info(f"Found {len(props_to_fill)} player-game-stat combinations to backfill")

    if len(props_to_fill) == 0:
        logger.info("✅ No props to backfill")
        conn_intel.close()
        conn_players.close()
        return

    updates = 0
    not_found = 0
    batch_size = 100

    for idx, (player_name, prop_game_date, stat_type) in enumerate(props_to_fill, 1):
        # Map stat type to column name
        stat_column = STAT_MAP.get(stat_type)
        if not stat_column:
            logger.warning(f"Unknown stat type: {stat_type}")
            continue

        # Query actual result from player_game_logs
        # Join with player_profile to match by name
        query_actual = f"""
        SELECT pgl.{stat_column}
        FROM player_game_logs pgl
        JOIN player_profile pp ON pgl.player_id = pp.player_id
        WHERE LOWER(TRIM(pp.full_name)) = LOWER(TRIM(%s))
          AND pgl.game_date = %s
        LIMIT 1
        """

        cursor_players = conn_players.cursor()
        cursor_players.execute(query_actual, (player_name, prop_game_date))
        result = cursor_players.fetchone()
        cursor_players.close()

        if result and result[0] is not None:
            actual_value = float(result[0])

            # Update nba_props_xl with actual result
            update_query = """
            UPDATE nba_props_xl
            SET actual_value = %s
            WHERE LOWER(TRIM(player_name)) = LOWER(TRIM(%s))
              AND game_date = %s
              AND stat_type = %s
              AND actual_value IS NULL
            """

            cursor_update = conn_intel.cursor()
            cursor_update.execute(
                update_query, (actual_value, player_name, prop_game_date, stat_type)
            )
            rows_updated = cursor_update.rowcount
            cursor_update.close()

            updates += rows_updated

            if updates % batch_size == 0:
                conn_intel.commit()
                logger.info(f"Progress: {idx}/{len(props_to_fill)} processed, {updates} updated")
        else:
            not_found += 1

    # Final commit
    conn_intel.commit()
    conn_intel.close()
    conn_players.close()

    logger.info(f"\n{'='*80}")
    logger.info(f"✅ BACKFILL COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Props updated: {updates}")
    logger.info(f"Not found: {not_found} (game not played or player didn't play)")
    logger.info(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description="Backfill actual_value for nba_props_xl from player_game_logs"
    )
    parser.add_argument("--date", help="Specific date to backfill (YYYY-MM-DD)")
    parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")
    parser.add_argument("--all", action="store_true", help="Backfill all missing actual_value")

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("BACKFILL ACTUAL VALUES FOR NBA_PROPS_XL")
    logger.info("=" * 80)

    if args.all:
        logger.info("Mode: Backfill ALL missing actual_value")
        backfill_results()
    elif args.date:
        logger.info(f"Mode: Specific date - {args.date}")
        backfill_results(specific_date=args.date)
    elif args.start_date:
        logger.info(f"Mode: Date range - {args.start_date} to {args.end_date or 'present'}")
        backfill_results(start_date=args.start_date, end_date=args.end_date)
    else:
        logger.error("Must specify --date, --start-date, or --all")
        parser.print_help()
        return

    logger.info("")


if __name__ == "__main__":
    main()
