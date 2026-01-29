#!/usr/bin/env python3
"""
Populate actual_value in nba_props_xl from player_game_logs

Uses name normalization to match players across different ID systems.
"""
import psycopg2
from datetime import datetime, timedelta
import argparse
import sys
from pathlib import Path
import os

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.name_normalizer import NameNormalizer
from betting_xl.utils.logging_config import setup_logging, get_logger, add_logging_args

# Logger will be configured in main block
logger = get_logger(__name__)

def get_current_season():
    """
    Calculate current NBA season based on date.
    NBA season uses END year (2025-26 season = 2026).
    Season starts in October, so Oct-Dec uses next year's number.
    """
    now = datetime.now()
    return now.year + 1 if now.month >= 10 else now.year

# Database configs
DB_INTELLIGENCE = {
    'host': 'localhost',
    'port': 5539,
    'user': os.getenv('DB_USER', 'nba_user'),
    'password': os.getenv('DB_PASSWORD'),
    'database': 'nba_intelligence'
}

DB_PLAYERS = {
    'host': 'localhost',
    'port': 5536,
    'user': os.getenv('DB_USER', 'nba_user'),
    'password': os.getenv('DB_PASSWORD'),
    'database': 'nba_players'
}

STAT_MAP = {
    'POINTS': 'points',
    'REBOUNDS': 'rebounds',
    'ASSISTS': 'assists',
    'THREES': 'three_pointers_made',
    'BLOCKS': 'blocks',
    'STEALS': 'steals'
}

def populate_actual_values(days_back=14):
    """
    Populate actual_value in nba_props_xl from player_game_logs

    Args:
        days_back: Number of days to backfill
    """
    conn_intel = psycopg2.connect(**DB_INTELLIGENCE)
    conn_players = psycopg2.connect(**DB_PLAYERS)

    start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

    print(f"\n{'='*80}")
    print(f"POPULATING ACTUAL VALUES FROM {start_date} TO TODAY")
    print(f"{'='*80}\n")

    try:
        cursor_intel = conn_intel.cursor()
        cursor_players = conn_players.cursor()

        # Initialize name normalizer
        normalizer = NameNormalizer()
        print("[OK] Name normalizer loaded")

        # For each stat type
        for stat_type, stat_column in STAT_MAP.items():
            print(f"\nProcessing {stat_type}...")

            # Get game logs with player names (join with player_profile)
            current_season = get_current_season()
            cursor_players.execute(f"""
                SELECT p.full_name, g.game_date, g.{stat_column}
                FROM player_game_logs g
                JOIN player_profile p ON g.player_id = p.player_id
                WHERE g.game_date >= %s AND g.season = %s
            """, (start_date, current_season))

            game_logs = cursor_players.fetchall()
            print(f"  Found {len(game_logs):,} game logs")

            # Build normalized game log mapping: (normalized_name, game_date) → stat_value
            game_log_map = {}
            for player_name, game_date, stat_value in game_logs:
                normalized = normalizer.normalize_name(player_name)
                # DEBUG: Print first entry to see date format
                if len(game_log_map) == 0:
                    print(f"  DEBUG: First game log - name='{player_name}' → '{normalized}', date={game_date} (type={type(game_date).__name__})")
                game_log_map[(normalized, game_date)] = stat_value

            print(f"  Built map with {len(game_log_map):,} unique (player, date) pairs")

            # Get all props without actuals for this stat type
            cursor_intel.execute("""
                SELECT id, player_name, game_date
                FROM nba_props_xl
                WHERE game_date >= %s
                  AND stat_type = %s
                  AND actual_value IS NULL
            """, (start_date, stat_type))

            props_to_update = cursor_intel.fetchall()
            print(f"  Found {len(props_to_update):,} props without actuals")

            # Match props to game logs using normalized names
            updated = 0
            not_found = 0
            for i, (prop_id, prop_player_name, prop_game_date) in enumerate(props_to_update):
                # Normalize prop player name
                normalized_prop = normalizer.normalize_name(prop_player_name)

                # DEBUG: Print first prop to see date format
                if i == 0:
                    print(f"  DEBUG: First prop - name='{prop_player_name}' → '{normalized_prop}', date={prop_game_date} (type={type(prop_game_date).__name__})")

                # Look up in game log map
                key = (normalized_prop, prop_game_date)
                if key in game_log_map:
                    stat_value = game_log_map[key]

                    cursor_intel.execute("""
                        UPDATE nba_props_xl
                        SET actual_value = %s
                        WHERE id = %s
                    """, (stat_value, prop_id))

                    updated += 1
                else:
                    not_found += 1

            conn_intel.commit()
            print(f"  [OK] Updated {updated:,} props")
            if not_found > 0:
                print(f"  [WARN]  Could not find actuals for {not_found:,} props")

        # Build set of players who actually played (exclude DNPs)
        print("\n[SEARCH] Building DNP filter (players who actually played)...")
        cursor_players.execute("""
            SELECT p.full_name, g.game_date
            FROM player_game_logs g
            JOIN player_profile p ON g.player_id = p.player_id
            WHERE g.game_date >= %s
        """, (start_date,))

        played_set = set()
        for player_name, game_date in cursor_players.fetchall():
            normalized = normalizer.normalize_name(player_name)
            played_set.add((normalized, game_date))

        print(f"  Found {len(played_set):,} (player, date) pairs who actually played")

        # Summary
        print(f"\n{'='*80}")
        print("SUMMARY (Excluding DNPs)")
        print(f"{'='*80}\n")

        # Get all props to filter manually
        cursor_intel.execute("""
            SELECT player_name, game_date, stat_type, actual_value
            FROM nba_props_xl
            WHERE game_date >= %s
            ORDER BY game_date DESC, stat_type
        """, (start_date,))

        all_props = cursor_intel.fetchall()

        # Group by (game_date, stat_type) and filter out DNPs
        from collections import defaultdict
        stats = defaultdict(lambda: {'total': 0, 'populated': 0})

        for player_name, game_date, stat_type, actual_value in all_props:
            normalized = normalizer.normalize_name(player_name)

            # Only count props where player actually played
            if (normalized, game_date) in played_set:
                key = (game_date, stat_type)
                stats[key]['total'] += 1
                if actual_value is not None:
                    stats[key]['populated'] += 1

        # Print results
        print(f"{'Date':<12} {'Market':<10} {'Total':<10} {'Populated':<12} {'Coverage':<10}")
        print(f"{'-'*60}")

        for (game_date, stat_type), counts in sorted(stats.items(), key=lambda x: (x[0][0], x[0][1]), reverse=True):
            total = counts['total']
            populated = counts['populated']
            pct = round(100.0 * populated / total, 1) if total > 0 else 0.0
            print(f"{game_date} {stat_type:<10} {total:<10} {populated:<12} {pct}%")

        print(f"\n{'='*80}\n")

    finally:
        conn_intel.close()
        conn_players.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Populate actual values in nba_props_xl')
    parser.add_argument('--days', type=int, default=14, help='Days to backfill (default: 14)')
    add_logging_args(parser)  # Adds --debug and --quiet flags
    args = parser.parse_args()

    # Setup unified logging
    setup_logging('populate_actuals', debug=args.debug, quiet=args.quiet)
    logger.info(f"Populating actuals for last {args.days} days")

    populate_actual_values(days_back=args.days)
