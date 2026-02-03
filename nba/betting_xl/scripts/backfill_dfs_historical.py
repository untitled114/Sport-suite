#!/usr/bin/env python3
"""
DFS Historical Backfill Script
==============================
Fetches historical props from BettingPros for DFS platforms:
- Underdog (book_id=36): Oct 2023 → Oct 2025
- PrizePicks (book_id=37): Oct 2023 → Oct 2025

Usage:
    # Backfill both platforms
    python backfill_dfs_historical.py

    # Backfill specific platform
    python backfill_dfs_historical.py --platform underdog
    python backfill_dfs_historical.py --platform prizepicks

    # Backfill specific date range
    python backfill_dfs_historical.py --start 2024-01-01 --end 2024-03-31

    # Resume from last checkpoint
    python backfill_dfs_historical.py --resume

    # Dry run (don't save to database)
    python backfill_dfs_historical.py --dry-run
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import psycopg2

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[3]))

from nba.betting_xl.fetchers.fetch_bettingpros import BettingProsFetcher

# Database config
DB_CONFIG = {
    "host": os.getenv("NBA_DB_HOST", "localhost"),
    "port": int(os.getenv("NBA_INT_DB_PORT", 5539)),
    "database": os.getenv("NBA_INT_DB_NAME", "nba_intelligence"),
    "user": os.getenv("DB_USER", "mlb_user"),
    "password": os.getenv("DB_PASSWORD"),
    "connect_timeout": 10,
}

# DFS platforms to backfill
DFS_PLATFORMS = {
    "underdog": 36,
    "prizepicks": 37,
}

# Markets to fetch
MARKETS = {
    "points": 156,
    "rebounds": 157,
    "assists": 151,
    "threes": 162,
}

# Checkpoint file for resuming
CHECKPOINT_FILE = Path(__file__).parent / "backfill_checkpoint.json"


class DFSBackfiller:
    """Backfills historical DFS props from BettingPros"""

    def __init__(
        self,
        platforms: List[str] = None,
        start_date: str = "2023-10-24",
        end_date: str = "2025-10-26",
        dry_run: bool = False,
        verbose: bool = True,
    ):
        """
        Initialize backfiller.

        Args:
            platforms: List of platforms to backfill ('underdog', 'prizepicks')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            dry_run: Don't save to database
            verbose: Verbose output
        """
        self.platforms = platforms or list(DFS_PLATFORMS.keys())
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
        self.dry_run = dry_run
        self.verbose = verbose

        self.conn = None
        self.cursor = None
        self.stats = {
            "dates_processed": 0,
            "props_fetched": 0,
            "props_inserted": 0,
            "props_updated": 0,
            "errors": 0,
            "skipped_dates": 0,
        }

    def connect(self):
        """Connect to database"""
        if self.dry_run:
            print("[DRY RUN] Skipping database connection")
            return

        print(f"Connecting to {DB_CONFIG['database']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}")
        self.conn = psycopg2.connect(**DB_CONFIG)
        self.cursor = self.conn.cursor()
        print("[OK] Connected\n")

    def disconnect(self):
        """Disconnect from database"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()

    def get_player_id(self, player_name: str) -> int:
        """Generate player_id from name (hash-based)"""
        return abs(hash(player_name)) % (10**9)

    def save_checkpoint(self, current_date: str, platform: str):
        """Save checkpoint for resuming"""
        checkpoint = {
            "last_date": current_date,
            "last_platform": platform,
            "timestamp": datetime.now().isoformat(),
            "stats": self.stats,
        }
        with open(CHECKPOINT_FILE, "w") as f:
            json.dump(checkpoint, f, indent=2)

    def load_checkpoint(self) -> Optional[Dict]:
        """Load checkpoint if exists"""
        if CHECKPOINT_FILE.exists():
            with open(CHECKPOINT_FILE, "r") as f:
                return json.load(f)
        return None

    def insert_prop(self, prop: Dict[str, Any], platform: str, game_date: str) -> bool:
        """Insert single prop into database"""
        if self.dry_run:
            return True

        player_name = prop.get("player_name")
        stat_type = prop.get("stat_type")
        line = prop.get("line")

        if not all([player_name, stat_type, line]):
            return False

        player_id = self.get_player_id(player_name)

        query = """
            INSERT INTO nba_props_xl (
                player_id, player_name, stat_type, book_name, game_date,
                over_line, over_odds, under_line, under_odds,
                game_id, player_team, opponent_team, is_home,
                fetch_timestamp, source_url, is_active
            ) VALUES (
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s
            )
            ON CONFLICT (player_id, game_date, stat_type, book_name, fetch_timestamp)
            DO UPDATE SET
                over_line = EXCLUDED.over_line,
                over_odds = EXCLUDED.over_odds,
                updated_at = CURRENT_TIMESTAMP
            RETURNING id;
        """

        values = (
            player_id,
            player_name,
            stat_type,
            platform,
            game_date,
            float(line),
            prop.get("over_odds", -110),
            float(line),
            prop.get("under_odds", -110),
            prop.get("game_id", ""),
            prop.get("player_team", ""),
            prop.get("opponent_team", ""),
            prop.get("is_home"),
            datetime.strptime(
                game_date, "%Y-%m-%d"
            ),  # Use game_date as fetch_timestamp for historical
            f"bettingpros_backfill_{platform}",
            True,
        )

        try:
            self.cursor.execute(query, values)
            result = self.cursor.fetchone()
            if result:
                self.stats["props_inserted"] += 1
            else:
                self.stats["props_updated"] += 1
            return True
        except psycopg2.Error as e:
            self.conn.rollback()
            self.stats["errors"] += 1
            if self.verbose and self.stats["errors"] <= 5:
                print(f"  [ERROR] {player_name} {stat_type}: {str(e)[:80]}")
            return False

    def fetch_date(self, date_str: str, platform: str, book_id: int) -> List[Dict]:
        """Fetch all props for a single date and platform"""
        all_props = []

        try:
            fetcher = BettingProsFetcher(date=date_str, books="priority", verbose=False)

            for market_name, market_id in MARKETS.items():
                props = fetcher.fetch_market(
                    market_name, market_id, book_id=book_id, book_name=platform
                )
                all_props.extend(props)

                # Rate limiting
                time.sleep(0.5)

        except Exception as e:
            if self.verbose:
                print(f"  [ERROR] {date_str} {platform}: {str(e)[:60]}")
            self.stats["errors"] += 1

        return all_props

    def is_game_day(self, date: datetime.date) -> bool:
        """Check if date likely has NBA games (rough heuristic)"""
        # NBA season: late October to mid-April (regular) + playoffs to June
        month = date.month

        # Off-season: July to mid-October
        if month in (7, 8, 9):
            return False
        if month == 10 and date.day < 20:
            return False
        if month == 6 and date.day > 20:
            return False

        return True

    def run(self, resume: bool = False):
        """Run the backfill"""
        print("=" * 70)
        print("DFS HISTORICAL BACKFILL")
        print("=" * 70)
        print(f"Platforms: {', '.join(self.platforms)}")
        print(f"Date range: {self.start_date} → {self.end_date}")
        print(f"Dry run: {self.dry_run}")
        print("=" * 70 + "\n")

        # Connect to database
        self.connect()

        # Load checkpoint if resuming
        start_date = self.start_date
        if resume:
            checkpoint = self.load_checkpoint()
            if checkpoint:
                start_date = datetime.strptime(checkpoint["last_date"], "%Y-%m-%d").date()
                print(f"[RESUME] Resuming from {start_date}")
                print(f"  Previous stats: {checkpoint['stats']}\n")

        # Iterate through dates
        current_date = start_date
        total_days = (self.end_date - start_date).days + 1

        try:
            while current_date <= self.end_date:
                date_str = current_date.strftime("%Y-%m-%d")
                day_num = (current_date - start_date).days + 1

                # Skip non-game days
                if not self.is_game_day(current_date):
                    if self.verbose:
                        print(f"[{day_num}/{total_days}] {date_str}: Skipping (off-season)")
                    self.stats["skipped_dates"] += 1
                    current_date += timedelta(days=1)
                    continue

                # Fetch for each platform
                day_props = 0
                for platform in self.platforms:
                    book_id = DFS_PLATFORMS[platform]
                    props = self.fetch_date(date_str, platform, book_id)

                    if props:
                        for prop in props:
                            self.insert_prop(prop, platform, date_str)
                        day_props += len(props)
                        self.stats["props_fetched"] += len(props)

                # Commit after each day
                if not self.dry_run and self.conn:
                    self.conn.commit()

                self.stats["dates_processed"] += 1

                if self.verbose:
                    print(f"[{day_num}/{total_days}] {date_str}: {day_props} props")

                # Save checkpoint every 10 days
                if self.stats["dates_processed"] % 10 == 0:
                    self.save_checkpoint(date_str, self.platforms[-1])

                # Rate limiting between days
                time.sleep(1)

                current_date += timedelta(days=1)

        except KeyboardInterrupt:
            print("\n\n[INTERRUPTED] Saving checkpoint...")
            self.save_checkpoint(date_str, self.platforms[-1])

        finally:
            # Final commit
            if not self.dry_run and self.conn:
                self.conn.commit()

            self.disconnect()

        # Print summary
        self.print_summary()

    def print_summary(self):
        """Print backfill summary"""
        print("\n" + "=" * 70)
        print("BACKFILL SUMMARY")
        print("=" * 70)
        print(f"Dates processed: {self.stats['dates_processed']}")
        print(f"Dates skipped:   {self.stats['skipped_dates']}")
        print(f"Props fetched:   {self.stats['props_fetched']}")
        print(f"Props inserted:  {self.stats['props_inserted']}")
        print(f"Props updated:   {self.stats['props_updated']}")
        print(f"Errors:          {self.stats['errors']}")
        print("=" * 70)

        # Clean up checkpoint on successful completion
        if self.stats["errors"] == 0 and CHECKPOINT_FILE.exists():
            CHECKPOINT_FILE.unlink()
            print("[OK] Checkpoint cleared (completed successfully)")


def main():
    parser = argparse.ArgumentParser(
        description="Backfill historical DFS props from BettingPros",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Backfill all DFS platforms (Oct 2023 - Oct 2025)
  python backfill_dfs_historical.py

  # Backfill only PrizePicks
  python backfill_dfs_historical.py --platform prizepicks

  # Backfill specific date range
  python backfill_dfs_historical.py --start 2024-01-01 --end 2024-06-30

  # Resume interrupted backfill
  python backfill_dfs_historical.py --resume

  # Dry run (test without saving)
  python backfill_dfs_historical.py --dry-run --start 2024-01-01 --end 2024-01-07
        """,
    )

    parser.add_argument(
        "--platform",
        choices=["underdog", "prizepicks", "all"],
        default="all",
        help="Platform to backfill (default: all)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2023-10-24",
        help="Start date (YYYY-MM-DD, default: 2023-10-24)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2025-10-26",
        help="End date (YYYY-MM-DD, default: 2025-10-26)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't save to database",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Less verbose output",
    )

    args = parser.parse_args()

    # Determine platforms
    if args.platform == "all":
        platforms = list(DFS_PLATFORMS.keys())
    else:
        platforms = [args.platform]

    # Run backfill
    backfiller = DFSBackfiller(
        platforms=platforms,
        start_date=args.start,
        end_date=args.end,
        dry_run=args.dry_run,
        verbose=not args.quiet,
    )

    backfiller.run(resume=args.resume)


if __name__ == "__main__":
    main()
