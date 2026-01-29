#!/usr/bin/env python3
"""
Database Loader for NBA XL Props
================================
Loads multi-book props from JSON files into nba_props_xl table (PostgreSQL + MongoDB).

This loader:
1. Reads props from fetcher JSON output
2. Inserts into nba_props_xl table (PostgreSQL - created by migration)
3. Inserts into MongoDB nba_betting_xl.nba_props_xl (nested structure)
4. Calculates consensus metrics using database functions
5. Handles duplicates and updates

Usage:
    python load_props_to_db.py --file lines/bettingpros_2025-11-05.json
    python load_props_to_db.py --directory lines/  # Load all JSON files
    python load_props_to_db.py --file lines/file.json --skip-mongodb  # PostgreSQL only
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import psycopg2
from pymongo import MongoClient

from nba.betting_xl.loaders.opponent_mapper import OpponentMapper
from nba.betting_xl.utils.logging_config import add_logging_args, get_logger, setup_logging

# Logger will be configured in main()
logger = get_logger(__name__)

# Valid stat types (matches database constraint)
# Combo props like POINTS_ASSISTS are not supported
VALID_STAT_TYPES = {"POINTS", "REBOUNDS", "ASSISTS", "THREES"}

# Database connections
# IMPORTANT: Use port 5539 (nba_intelligence) - the ORIGINAL/LEGACY database
# Port 5540 (nba_reference) is the consolidated DB that gives mixed/bad predictions
DB_CONFIG = {
    "host": os.getenv("NBA_INT_DB_HOST", "localhost"),
    "port": int(os.getenv("NBA_INT_DB_PORT", 5539)),
    "database": os.getenv("NBA_INT_DB_NAME", "nba_intelligence"),
    "user": os.getenv("NBA_INT_DB_USER", os.getenv("DB_USER", "nba_user")),
    "password": os.getenv("NBA_INT_DB_PASSWORD", os.getenv("DB_PASSWORD")),
    "connect_timeout": int(os.getenv("NBA_DB_CONNECT_TIMEOUT", 10)),
}

MONGO_CONFIG = {
    "uri": os.getenv("NBA_MONGO_URI", "mongodb://${MONGO_USER}:${MONGO_PASSWORD}@localhost:27017/"),
    "database": os.getenv("NBA_MONGO_DB", "nba_betting_xl"),
    "collection": os.getenv("NBA_MONGO_COLLECTION", "nba_props_xl"),
    "timeout_ms": int(os.getenv("NBA_MONGO_TIMEOUT_MS", 8000)),
}


class PropsLoader:
    """Loads multi-book props into nba_props_xl table (PostgreSQL + MongoDB)"""

    def __init__(self, verbose: bool = True, use_mongodb: bool = True):
        """
        Initialize loader.

        Args:
            verbose: Enable verbose logging
            use_mongodb: Enable MongoDB integration (dual writes)
        """
        self.verbose = verbose
        self.use_mongodb = use_mongodb
        self.conn = None
        self.cursor = None
        self.mongo_client = None
        self.mongo_collection = None

        # Track skipped props by reason (for clean summary output)
        self.skipped_counts = defaultdict(int)

        # Track MongoDB aggregation for upserts
        self.mongo_buffer = defaultdict(
            lambda: {
                "player_name": None,
                "game_date": None,
                "stat_type": None,
                "opponent_team": None,
                "is_home": None,
                "game_id": None,
                "game_time": None,
                "actual_value": None,
                "books": [],
            }
        )

    def connect(self):
        """Connect to PostgreSQL and MongoDB"""
        if self.verbose:
            print(
                f"Connecting to PostgreSQL: {DB_CONFIG['database']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}"
            )

        try:
            self.conn = psycopg2.connect(**DB_CONFIG)
        except Exception as exc:
            raise RuntimeError(
                f"❌ Failed to connect to PostgreSQL ({DB_CONFIG['database']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}): {exc}"
            ) from exc

        self.cursor = self.conn.cursor()

        if self.verbose:
            print("✅ PostgreSQL connected")

        # Connect to MongoDB
        if self.use_mongodb:
            try:
                if self.verbose:
                    print(
                        f"Connecting to MongoDB: {MONGO_CONFIG['database']}.{MONGO_CONFIG['collection']}"
                    )

                timeout = MONGO_CONFIG["timeout_ms"]
                self.mongo_client = MongoClient(
                    MONGO_CONFIG["uri"],
                    serverSelectionTimeoutMS=timeout,
                    connectTimeoutMS=timeout,
                    socketTimeoutMS=timeout,
                )
                self.mongo_db = self.mongo_client[MONGO_CONFIG["database"]]
                self.mongo_collection = self.mongo_db[MONGO_CONFIG["collection"]]

                # Test connection
                self.mongo_client.admin.command("ping")

                if self.verbose:
                    print("✅ MongoDB connected\n")
            except Exception as e:
                warning = f"⚠️  MongoDB connection failed: {e}"
                print(warning)
                if self.verbose:
                    print("   Continuing with PostgreSQL only\n")
                self.use_mongodb = False
        else:
            if self.verbose:
                print("⚠️  MongoDB disabled (--skip-mongodb flag)\n")

    def disconnect(self):
        """Disconnect from PostgreSQL and MongoDB"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()

        if self.mongo_client:
            self.mongo_client.close()

        if self.verbose:
            print("\n✅ Disconnected from all databases")

    def load_json_file(self, filepath: str) -> List[Dict[str, Any]]:
        """
        Load props from JSON file.

        Args:
            filepath: Path to JSON file

        Returns:
            List of prop dictionaries
        """
        with open(filepath, "r") as f:
            data = json.load(f)

        # Handle different JSON formats
        if isinstance(data, list):
            props = data
        elif isinstance(data, dict) and "props" in data:
            props = data["props"]
        else:
            props = [data]

        if self.verbose:
            print(f"Loaded {len(props)} props from {filepath}")

        return props

    def get_or_create_player_id(self, player_name: str) -> int:
        """
        Get player_id for player name, or create placeholder if not exists.

        For now, we'll use a simple hash of the name as player_id.
        In production, this should lookup from player_profile table.

        Args:
            player_name: Player's full name

        Returns:
            player_id (integer)
        """
        # For MVP: Use hash of name as player_id
        # In production: SELECT player_id FROM player_profile WHERE full_name = %s
        return abs(hash(player_name)) % (10**9)  # Keep it under 1 billion

    def insert_prop(self, prop: Dict[str, Any]) -> bool:
        """
        Insert single prop into database.

        Args:
            prop: Prop dictionary

        Returns:
            True if inserted, False if skipped (duplicate)
        """
        # Extract fields from prop
        player_name = prop.get("player_name")
        stat_type = prop.get("stat_type")
        book_name = prop.get("book_name")
        game_date = prop.get("game_date")
        over_line = prop.get("over_line") or prop.get("line")
        under_line = prop.get("under_line") or prop.get("line")
        over_odds = prop.get("over_odds", -110)
        under_odds = prop.get("under_odds", -110)
        game_id = prop.get("game_id", "")
        game_time = prop.get("game_time")
        opponent_team = prop.get("opponent_team") or None  # Use NULL instead of empty string
        is_home = prop.get("is_home")
        consensus_line = prop.get("consensus_line")
        actual_value = prop.get("actual_value")  # ⭐ CRITICAL for training
        fetch_timestamp = prop.get("fetch_timestamp", datetime.now().isoformat())
        source = prop.get("source", "bettingpros")

        # Validate required fields
        if not all([player_name, stat_type, book_name, game_date, over_line]):
            if self.verbose:
                print(f"  ⚠️  Skipping invalid prop: {player_name} - missing required fields")
            return False

        # Filter out invalid stat types (combo props like PA, PR, RA, PRA)
        if stat_type not in VALID_STAT_TYPES:
            self.skipped_counts[f"combo_{stat_type}"] += 1
            return False

        # Get player_id
        player_id = self.get_or_create_player_id(player_name)

        # Check for duplicate from same day (prevent loading same props multiple times)
        # Only check if this is a current-day prop (not historical backfill)
        if game_date >= datetime.now().date().isoformat():
            check_query = """
                SELECT id FROM nba_props_xl
                WHERE player_id = %s AND game_date = %s
                  AND stat_type = %s AND book_name = %s
                  AND DATE(fetch_timestamp) = CURRENT_DATE
                LIMIT 1
            """
            self.cursor.execute(check_query, (player_id, game_date, stat_type, book_name))
            if self.cursor.fetchone():
                return False  # Skip duplicate from today

        # Insert query (using ON CONFLICT to handle duplicates)
        query = """
            INSERT INTO nba_props_xl (
                player_id, player_name, stat_type, book_name, game_date, game_time,
                over_line, over_odds, under_line, under_odds,
                game_id, opponent_team, is_home,
                consensus_line, actual_value, fetch_timestamp, source_url, is_active
            ) VALUES (
                %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s, %s, %s
            )
            ON CONFLICT (player_id, game_date, stat_type, book_name, fetch_timestamp)
            DO UPDATE SET
                over_line = EXCLUDED.over_line,
                over_odds = EXCLUDED.over_odds,
                under_line = EXCLUDED.under_line,
                under_odds = EXCLUDED.under_odds,
                actual_value = EXCLUDED.actual_value,
                updated_at = CURRENT_TIMESTAMP
            RETURNING id;
        """

        values = (
            player_id,
            player_name,
            stat_type,
            book_name,
            game_date,
            game_time,
            float(over_line),
            over_odds,
            float(under_line),
            under_odds,
            game_id,
            opponent_team,
            is_home,
            float(consensus_line) if consensus_line else None,
            float(actual_value) if actual_value is not None else None,
            fetch_timestamp,
            source,
            True,
        )

        try:
            self.cursor.execute(query, values)
            result = self.cursor.fetchone()

            inserted = bool(result)

            # Also insert into MongoDB if enabled
            if inserted and self.use_mongodb:
                self._buffer_mongodb_prop(prop)

            return inserted

        except Exception as e:
            # Rollback to clear the aborted transaction state
            self.conn.rollback()
            if self.verbose:
                print(f"  ❌ Error inserting prop {player_name} - {stat_type} - {book_name}: {e}")
            return False

    def _buffer_mongodb_prop(self, prop: Dict[str, Any]):
        """
        Buffer props for MongoDB batch insert.

        Props are grouped by (player_name, game_date, stat_type) to create
        nested documents with books array.

        Args:
            prop: Prop dictionary from JSON
        """
        player_name = prop.get("player_name")
        game_date = prop.get("game_date")
        stat_type = prop.get("stat_type")
        book_name = prop.get("book_name")

        # Create unique key for grouping
        key = (player_name, game_date, stat_type)

        # Initialize document fields (only once per unique prop)
        if self.mongo_buffer[key]["player_name"] is None:
            self.mongo_buffer[key].update(
                {
                    "player_name": player_name,
                    "game_date": (
                        datetime.fromisoformat(game_date)
                        if isinstance(game_date, str)
                        else game_date
                    ),
                    "stat_type": stat_type,
                    "opponent_team": prop.get("opponent_team"),
                    "is_home": prop.get("is_home"),
                    "game_id": prop.get("game_id", ""),
                    "game_time": prop.get("game_time"),
                    "actual_value": (
                        float(prop["actual_value"])
                        if prop.get("actual_value") is not None
                        else None
                    ),
                }
            )

        # Add book to books array
        book_doc = {
            "book_name": book_name,
            "over_line": float(prop.get("over_line") or prop.get("line")),
            "over_odds": prop.get("over_odds", -110),
            "under_line": float(prop.get("under_line") or prop.get("line")),
            "under_odds": prop.get("under_odds", -110),
            "fetch_timestamp": (
                datetime.fromisoformat(prop.get("fetch_timestamp"))
                if isinstance(prop.get("fetch_timestamp"), str)
                else datetime.now()
            ),
        }

        self.mongo_buffer[key]["books"].append(book_doc)

    def _flush_mongodb_buffer(self):
        """
        Flush buffered props to MongoDB using upsert.

        This calculates consensus metrics and inserts/updates documents.
        """
        if not self.use_mongodb or not self.mongo_buffer:
            return

        if self.verbose:
            print(f"\nFlushing {len(self.mongo_buffer)} props to MongoDB...")

        upserted = 0
        errors = 0

        for _key, doc in self.mongo_buffer.items():
            try:
                # Calculate line shopping metrics from books array
                lines = [book["over_line"] for book in doc["books"]]

                if not lines:
                    continue

                consensus_line = sum(lines) / len(lines)
                min_line = min(lines)
                max_line = max(lines)
                line_spread = max_line - min_line
                line_std_dev = (
                    (sum((x - consensus_line) ** 2 for x in lines) / len(lines)) ** 0.5
                    if len(lines) > 1
                    else 0.0
                )
                line_coef_variation = (
                    (line_std_dev / consensus_line * 100) if consensus_line > 0 else 0.0
                )

                # Find softest/hardest books
                softest_book = min(doc["books"], key=lambda b: b["over_line"])["book_name"]
                hardest_book = max(doc["books"], key=lambda b: b["over_line"])["book_name"]

                # Calculate book deviations
                deviations = {}
                for book in doc["books"]:
                    book_key = book["book_name"].lower().replace(" ", "")
                    deviations[book_key] = book["over_line"] - consensus_line

                # Add line_shopping subdocument
                doc["line_shopping"] = {
                    "consensus_line": round(consensus_line, 2),
                    "min_line": min_line,
                    "max_line": max_line,
                    "line_spread": round(line_spread, 2),
                    "line_std_dev": round(line_std_dev, 3),
                    "num_books": len(lines),
                    "line_coef_variation": round(line_coef_variation, 2),
                    "softest_book": softest_book,
                    "hardest_book": hardest_book,
                    "softest_vs_consensus": round(min_line - consensus_line, 2),
                    "hardest_vs_consensus": round(max_line - consensus_line, 2),
                    "books_agree": line_spread < 0.5,
                    "books_disagree": line_spread >= 1.5,
                    "line_spread_percentile": 0.0,  # Placeholder - would need historical data
                    "deviations": deviations,
                }

                # Add metadata
                doc["xl_metadata"] = {"version": "1.0", "last_updated": datetime.now()}

                doc["created_at"] = datetime.now()
                doc["updated_at"] = datetime.now()

                # Upsert to MongoDB
                filter_doc = {
                    "player_name": doc["player_name"],
                    "game_date": doc["game_date"],
                    "stat_type": doc["stat_type"],
                }

                self.mongo_collection.update_one(filter_doc, {"$set": doc}, upsert=True)

                upserted += 1

            except Exception as e:
                errors += 1
                if self.verbose:
                    print(f"  ❌ Error upserting MongoDB doc: {e}")

        if self.verbose:
            print(f"✅ MongoDB flush complete: {upserted} upserted, {errors} errors")

        # Clear buffer
        self.mongo_buffer.clear()

    def load_props(self, props: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Load multiple props into database.

        Args:
            props: List of prop dictionaries

        Returns:
            Dictionary with stats (inserted, updated, errors)
        """
        stats = {"total": len(props), "inserted": 0, "skipped": 0, "errors": 0}

        if self.verbose:
            print(f"\nLoading {len(props)} props into database...")

        for i, prop in enumerate(props, 1):
            if self.verbose and i % 100 == 0:
                print(f"  Progress: {i}/{len(props)} ({i/len(props)*100:.1f}%)")

            try:
                inserted = self.insert_prop(prop)
                if inserted:
                    stats["inserted"] += 1
                else:
                    stats["skipped"] += 1

            except Exception as e:
                stats["errors"] += 1
                if self.verbose:
                    print(f"  ❌ Error on prop {i}: {e}")

        # Commit PostgreSQL transaction
        self.conn.commit()

        # Flush MongoDB buffer
        if self.use_mongodb:
            self._flush_mongodb_buffer()

        if self.verbose:
            print(f"\n✅ Load complete:")
            print(f"   Total:    {stats['total']}")
            print(f"   Inserted: {stats['inserted']}")
            print(f"   Skipped:  {stats['skipped']}")
            print(f"   Errors:   {stats['errors']}")

        return stats

    def calculate_consensus_metrics(self, game_date: str = None):
        """
        Calculate consensus metrics for props.

        Uses database function to calculate:
        - consensus_line (average across all books)
        - min_over_line (softest OVER)
        - max_over_line (hardest OVER)
        - line_spread (max - min)
        - softest_book
        - hardest_book

        Args:
            game_date: Optional date filter (YYYY-MM-DD) - currently not used by DB function
        """
        if self.verbose:
            print(f"\nCalculating consensus metrics...")

        try:
            # Update all active props using database function
            query = "SELECT update_all_consensus_metrics();"
            self.cursor.execute(query)
            result = self.cursor.fetchone()
            rows_updated = result[0] if result else 0

            self.conn.commit()

            if self.verbose:
                print(f"✅ Updated consensus metrics ({rows_updated} rows affected)")

        except psycopg2.errors.UndefinedFunction as e:
            # Database function doesn't exist yet - skip gracefully
            self.conn.rollback()
            if self.verbose:
                print(f"⚠️  Consensus metrics function not available (skipping)")

        except Exception as e:
            # Other errors - log but don't crash
            self.conn.rollback()
            if self.verbose:
                print(f"⚠️  Error calculating consensus metrics: {e}")

    def get_line_shopping_summary(self, game_date: str = None):
        """
        Get summary of line shopping opportunities.

        Args:
            game_date: Optional date filter (YYYY-MM-DD)
        """
        query = """
            SELECT
                stat_type,
                COUNT(DISTINCT (player_name, game_date, stat_type)) as num_props,
                AVG(line_spread) as avg_spread,
                MAX(line_spread) as max_spread,
                COUNT(*) FILTER (WHERE line_spread >= 1.5) as opportunities_1_5,
                COUNT(*) FILTER (WHERE line_spread >= 2.5) as opportunities_2_5,
                COUNT(*) FILTER (WHERE line_spread >= 3.5) as opportunities_3_5
            FROM (
                SELECT DISTINCT ON (player_name, game_date, stat_type)
                    player_name, game_date, stat_type, line_spread
                FROM nba_props_xl
                WHERE is_active = true
        """

        if game_date:
            query += f" AND game_date = '{game_date}'"

        query += """
            ) unique_props
            GROUP BY stat_type
            ORDER BY stat_type;
        """

        self.cursor.execute(query)
        results = self.cursor.fetchall()

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"LINE SHOPPING OPPORTUNITIES SUMMARY")
            print(f"{'='*70}")

            for row in results:
                stat_type, num_props, avg_spread, max_spread, opp_1_5, opp_2_5, opp_3_5 = row
                print(f"\n{stat_type}:")
                print(f"  Total props: {num_props}")
                print(
                    f"  Avg spread:  {avg_spread:.2f} points"
                    if avg_spread
                    else "  Avg spread:  N/A"
                )
                print(
                    f"  Max spread:  {max_spread:.2f} points"
                    if max_spread
                    else "  Max spread:  N/A"
                )
                print(
                    f"  Spread ≥1.5: {opp_1_5} props ({opp_1_5/num_props*100:.1f}%)"
                    if num_props > 0
                    else ""
                )
                print(
                    f"  Spread ≥2.5: {opp_2_5} props ({opp_2_5/num_props*100:.1f}%)"
                    if num_props > 0
                    else ""
                )
                print(
                    f"  Spread ≥3.5: {opp_3_5} props ({opp_3_5/num_props*100:.1f}%)"
                    if num_props > 0
                    else ""
                )

            print(f"{'='*70}\n")


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description="Load NBA XL props into database")
    parser.add_argument("--file", type=str, help="JSON file to load")
    parser.add_argument("--directory", type=str, help="Directory of JSON files to load")
    parser.add_argument("--date", type=str, help="Game date filter (YYYY-MM-DD)")
    parser.add_argument("--skip-consensus", action="store_true", help="Skip consensus calculation")
    parser.add_argument("--skip-mongodb", action="store_true", help="Skip MongoDB writes")
    add_logging_args(parser)  # Adds --debug and --quiet flags

    args = parser.parse_args()

    # Setup unified logging
    setup_logging("load_props", debug=args.debug, quiet=args.quiet)
    logger.info("Starting props database loader")

    # Validate inputs
    if not args.file and not args.directory:
        parser.error("Must specify --file or --directory")

    # Get list of files to load
    files_to_load = []

    if args.file:
        files_to_load.append(args.file)
    elif args.directory:
        directory = Path(args.directory)
        if not directory.is_dir():
            print(f"❌ Directory not found: {args.directory}")
            sys.exit(1)

        files_to_load = list(directory.glob("*.json"))
        files_to_load = [str(f) for f in files_to_load]

    if not files_to_load:
        print("❌ No JSON files found")
        sys.exit(1)

    # Create loader
    loader = PropsLoader(verbose=not args.quiet, use_mongodb=not args.skip_mongodb)

    try:
        # Connect to database
        logger.debug("Connecting to database...")
        loader.connect()
        logger.info("Database connection established")

        # Initialize opponent mapper for enriching props
        opponent_mapper = OpponentMapper(verbose=not args.quiet)

        # Load all files
        total_stats = {"total": 0, "inserted": 0, "skipped": 0, "errors": 0}

        for filepath in files_to_load:
            print(f"\n{'='*70}")
            print(f"Loading file: {filepath}")
            print(f"{'='*70}")

            # Load props from JSON
            props = loader.load_json_file(filepath)

            # CRITICAL FIX (Dec 22, 2025): Enrich opponent_team and is_home from schedule
            # BettingPros API doesn't provide opponent data, so we derive it from ESPN schedule
            props = opponent_mapper.enrich_props(props)

            # Insert into database
            stats = loader.load_props(props)

            # Accumulate stats
            for key in total_stats:
                total_stats[key] += stats[key]

        # Calculate consensus metrics
        if not args.skip_consensus:
            loader.calculate_consensus_metrics(game_date=args.date)

            # Show line shopping summary
            loader.get_line_shopping_summary(game_date=args.date)

        # Overall summary
        print(f"\n{'='*70}")
        print(f"OVERALL SUMMARY")
        print(f"{'='*70}")
        print(f"Files loaded:  {len(files_to_load)}")
        print(f"Total props:   {total_stats['total']}")
        print(f"Inserted:      {total_stats['inserted']}")
        print(f"Skipped:       {total_stats['skipped']}")
        print(f"Errors:        {total_stats['errors']}")

        # Show breakdown of skipped combo props (clean summary instead of per-prop spam)
        combo_skipped = {k: v for k, v in loader.skipped_counts.items() if k.startswith("combo_")}
        if combo_skipped:
            combo_total = sum(combo_skipped.values())
            combo_types = ", ".join(
                f"{k.replace('combo_', '')}:{v}" for k, v in sorted(combo_skipped.items())
            )
            print(f"Combo props:   {combo_total} filtered ({combo_types})")

        print(f"{'='*70}\n")

    finally:
        loader.disconnect()


if __name__ == "__main__":
    main()
