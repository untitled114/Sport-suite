#!/usr/bin/env python3
"""
PrizePicks Direct Loader
========================
Loads PrizePicks props from direct API fetch into nba_props_xl table.

Handles PrizePicks-specific fields:
- odds_type: standard/goblin/demon
- trending_count: pick popularity
- board_time: when line was first posted
- pp_updated_at: PrizePicks updated_at timestamp
- adjusted_odds: null/true/false
- projection_id: PrizePicks internal ID

Usage:
    python load_prizepicks_to_db.py --fetch          # Fetch and load in one step
    python load_prizepicks_to_db.py --file data.json # Load from JSON file
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import psycopg2

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[3]))

from nba.betting_xl.fetchers.fetch_prizepicks_direct import PrizePicksDirectFetcher

# Database config
DB_CONFIG = {
    "host": os.getenv("NBA_INT_DB_HOST", "localhost"),
    "port": int(os.getenv("NBA_INT_DB_PORT", 5539)),
    "database": os.getenv("NBA_INT_DB_NAME", "nba_intelligence"),
    "user": os.getenv("NBA_INT_DB_USER", os.getenv("DB_USER", "mlb_user")),
    "password": os.getenv("NBA_INT_DB_PASSWORD", os.getenv("DB_PASSWORD")),
    "connect_timeout": 10,
}

# Valid stat types (must match database constraint)
VALID_STAT_TYPES = {
    "POINTS",
    "REBOUNDS",
    "ASSISTS",
    "THREES",
    "PRA",
    "PR",
    "PA",
    "RA",
    "STEALS",
    "BLOCKS",
    "TURNOVERS",
    "BLKS_STLS",
    "FANTASY_SCORE",
}


class PrizePicksLoader:
    """Loads PrizePicks props into nba_props_xl table"""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.conn = None
        self.cursor = None
        self.stats = defaultdict(int)

    def connect(self):
        """Connect to PostgreSQL"""
        if self.verbose:
            print(
                f"Connecting to PostgreSQL: {DB_CONFIG['database']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}"
            )

        self.conn = psycopg2.connect(**DB_CONFIG)
        self.cursor = self.conn.cursor()

        if self.verbose:
            print("[OK] PostgreSQL connected\n")

    def disconnect(self):
        """Disconnect from PostgreSQL"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()

        if self.verbose:
            print("\n[OK] Disconnected from database")

    def get_player_id(self, player_name: str) -> int:
        """Get player_id from name (hash-based for now)"""
        return abs(hash(player_name)) % (10**9)

    def parse_timestamp(self, ts: str) -> Optional[datetime]:
        """Parse ISO timestamp string to datetime"""
        if not ts:
            return None
        try:
            # Handle timezone offset
            ts = ts.replace("Z", "+00:00")
            return datetime.fromisoformat(ts)
        except (ValueError, AttributeError):
            return None

    def insert_prop(self, prop: Dict[str, Any]) -> bool:
        """
        Insert single PrizePicks prop into database.

        Args:
            prop: Prop dictionary from fetcher

        Returns:
            True if inserted, False if skipped
        """
        # Extract fields
        player_name = prop.get("player_name")
        stat_type = prop.get("stat_type")
        book_name = prop.get("book_name", "prizepicks")
        game_date = prop.get("game_date")
        line = prop.get("line")

        # Validate required fields
        if not all([player_name, stat_type, game_date, line]):
            self.stats["skipped_missing_fields"] += 1
            return False

        # Validate stat type
        if stat_type not in VALID_STAT_TYPES:
            self.stats[f"skipped_stat_{stat_type}"] += 1
            return False

        # Get player_id
        player_id = self.get_player_id(player_name)

        # Parse timestamps
        board_time = self.parse_timestamp(prop.get("board_time"))
        pp_updated_at = self.parse_timestamp(prop.get("updated_at"))
        fetch_timestamp = self.parse_timestamp(prop.get("fetch_timestamp")) or datetime.now()

        # Extract PrizePicks-specific fields
        odds_type = prop.get("odds_type", "standard")
        trending_count = prop.get("trending_count")
        adjusted_odds = prop.get("adjusted_odds")
        projection_id = prop.get("projection_id")
        is_promo = prop.get("is_promo", False)
        flash_sale_line = prop.get("flash_sale_line_score")

        # Insert query with PrizePicks-specific columns
        query = """
            INSERT INTO nba_props_xl (
                player_id, player_name, stat_type, book_name, game_date, game_time,
                over_line, over_odds, under_line, under_odds,
                game_id, player_team, opponent_team, is_home,
                fetch_timestamp, source_url, is_active,
                odds_type, trending_count, board_time, pp_updated_at,
                adjusted_odds, projection_id, is_promo, flash_sale_line
            ) VALUES (
                %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s
            )
            ON CONFLICT (player_id, game_date, stat_type, book_name, fetch_timestamp)
            DO UPDATE SET
                over_line = EXCLUDED.over_line,
                trending_count = EXCLUDED.trending_count,
                pp_updated_at = EXCLUDED.pp_updated_at,
                adjusted_odds = EXCLUDED.adjusted_odds,
                is_promo = EXCLUDED.is_promo,
                flash_sale_line = EXCLUDED.flash_sale_line,
                updated_at = CURRENT_TIMESTAMP
            RETURNING id;
        """

        values = (
            player_id,
            player_name,
            stat_type,
            book_name.lower(),  # Ensure lowercase for constraint
            game_date,
            prop.get("game_time"),
            float(line),
            prop.get("over_odds", -110),
            float(line),
            prop.get("under_odds", -110),
            prop.get("game_id", ""),
            prop.get("player_team"),
            prop.get("opponent_team"),
            prop.get("is_home"),
            fetch_timestamp,
            "prizepicks_direct",
            True,
            odds_type,
            trending_count,
            board_time,
            pp_updated_at,
            adjusted_odds,
            projection_id,
            is_promo,
            flash_sale_line,
        )

        try:
            self.cursor.execute(query, values)
            result = self.cursor.fetchone()
            if result:
                self.stats["inserted"] += 1
                return True
            else:
                self.stats["updated"] += 1
                return True
        except psycopg2.Error as e:
            self.conn.rollback()  # Rollback failed transaction
            self.stats["errors"] += 1
            if self.verbose and self.stats["errors"] <= 5:  # Only show first 5 errors
                print(f"  [ERROR] {player_name} {stat_type}: {str(e)[:100]}")
            return False

    def load_props(self, props: List[Dict[str, Any]]) -> int:
        """
        Load list of props into database.

        Args:
            props: List of prop dictionaries

        Returns:
            Number of props loaded
        """
        if self.verbose:
            print(f"Loading {len(props)} props to database...")
            print()

        for prop in props:
            self.insert_prop(prop)

        # Commit all inserts
        self.conn.commit()

        return self.stats["inserted"] + self.stats["updated"]

    def calculate_consensus_metrics(self):
        """
        Calculate consensus metrics for all props.

        Uses database function to calculate consensus_line from all books.
        This enables deflation/inflation calculations for two-energy strategy.
        """
        if self.verbose:
            print("\nCalculating consensus metrics...")

        try:
            self.cursor.execute("SELECT update_all_consensus_metrics();")
            result = self.cursor.fetchone()
            rows_updated = result[0] if result else 0
            self.conn.commit()

            if self.verbose:
                print(f"[OK] Updated consensus metrics ({rows_updated} rows)")

        except Exception as e:
            self.conn.rollback()
            if self.verbose:
                print(f"[WARN] Consensus metrics error: {e}")

    def print_summary(self):
        """Print loading summary"""
        print("\n" + "=" * 60)
        print("PRIZEPICKS LOAD SUMMARY")
        print("=" * 60)
        print(f"Inserted:  {self.stats['inserted']}")
        print(f"Updated:   {self.stats['updated']}")
        print(f"Errors:    {self.stats['errors']}")

        # Print skipped stats
        skipped = {k: v for k, v in self.stats.items() if k.startswith("skipped")}
        if skipped:
            print("\nSkipped:")
            for reason, count in sorted(skipped.items()):
                print(f"  {reason}: {count}")

        print("=" * 60)


def fetch_and_load(
    state_code: str = "FL",
    include_combos: bool = False,
    all_stats: bool = False,
    verbose: bool = True,
) -> int:
    """
    Fetch from PrizePicks API and load to database.

    Args:
        state_code: US state code
        include_combos: Include combo player props
        all_stats: Include all stat types
        verbose: Verbose output

    Returns:
        Number of props loaded
    """
    # Fetch props
    with PrizePicksDirectFetcher(
        state_code=state_code,
        include_combos=include_combos,
        all_stats=all_stats,
        verbose=verbose,
    ) as fetcher:
        props = fetcher.fetch()

    if not props:
        print("[WARN] No props fetched from PrizePicks")
        return 0

    # Load to database
    loader = PrizePicksLoader(verbose=verbose)
    loader.connect()

    try:
        loaded = loader.load_props(props)
        # Calculate consensus so deflation works for two-energy strategy
        loader.calculate_consensus_metrics()
        loader.print_summary()
        return loaded
    finally:
        loader.disconnect()


def load_from_file(filepath: str, verbose: bool = True) -> int:
    """
    Load props from JSON file.

    Args:
        filepath: Path to JSON file
        verbose: Verbose output

    Returns:
        Number of props loaded
    """
    if verbose:
        print(f"Loading from file: {filepath}")

    with open(filepath, "r") as f:
        data = json.load(f)

    # Handle different JSON formats
    if isinstance(data, list):
        props = data
    elif isinstance(data, dict) and "props" in data:
        props = data["props"]
    else:
        props = [data]

    if verbose:
        print(f"Found {len(props)} props in file")

    # Load to database
    loader = PrizePicksLoader(verbose=verbose)
    loader.connect()

    try:
        loaded = loader.load_props(props)
        # Calculate consensus so deflation works for two-energy strategy
        loader.calculate_consensus_metrics()
        loader.print_summary()
        return loaded
    finally:
        loader.disconnect()


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description="Load PrizePicks props to database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch and load in one step
  python load_prizepicks_to_db.py --fetch

  # Fetch with options
  python load_prizepicks_to_db.py --fetch --state NY --include-combos

  # Load from existing JSON file
  python load_prizepicks_to_db.py --file prizepicks_data.json
        """,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--fetch", action="store_true", help="Fetch from API and load")
    group.add_argument("--file", type=str, help="Load from JSON file")

    parser.add_argument("--state", type=str, default="FL", help="US state code (default: FL)")
    parser.add_argument("--include-combos", action="store_true", help="Include combo player props")
    parser.add_argument("--all-stats", action="store_true", help="Include all stat types")
    parser.add_argument("--quiet", action="store_true", help="Quiet mode")

    args = parser.parse_args()
    verbose = not args.quiet

    if args.fetch:
        loaded = fetch_and_load(
            state_code=args.state,
            include_combos=args.include_combos,
            all_stats=args.all_stats,
            verbose=verbose,
        )
    else:
        loaded = load_from_file(args.file, verbose=verbose)

    if loaded > 0:
        print(f"\n[OK] Loaded {loaded} PrizePicks props to database")
    else:
        print("\n[WARN] No props loaded")


if __name__ == "__main__":
    main()
