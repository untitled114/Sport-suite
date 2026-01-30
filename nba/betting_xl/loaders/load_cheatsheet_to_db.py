#!/usr/bin/env python3
"""
Load BettingPros Cheat Sheet Data to Database
==============================================
Loads projection, EV, ratings, and hit rate data from cheat sheet JSON files
into the cheatsheet_data table.

Usage:
    python load_cheatsheet_to_db.py --file lines/cheatsheet_underdog_*.json
    python load_cheatsheet_to_db.py --date 2026-01-02
    python load_cheatsheet_to_db.py --latest
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import psycopg2

# Database connection
DB_CONFIG = {
    "host": "localhost",
    "port": 5539,
    "database": "nba_intelligence",
    "user": os.getenv("DB_USER", "nba_user"),
    "password": os.getenv("DB_PASSWORD"),
}


def get_db_connection():
    """Get database connection"""
    return psycopg2.connect(**DB_CONFIG)


def load_json_file(filepath: Path) -> List[Dict[str, Any]]:
    """Load props from JSON file"""
    with open(filepath, "r") as f:
        data = json.load(f)
    return data.get("props", [])


def insert_cheatsheet_data(conn, props: List[Dict[str, Any]], platform: str) -> int:
    """
    Insert cheat sheet data into database.

    Uses UPSERT to handle duplicates (same player/date/stat/platform).

    Returns:
        Number of rows inserted/updated
    """
    cursor = conn.cursor()

    # PrizePicks = reference only (lower payouts), Underdog = use for betting
    use_for_betting = platform == "underdog"

    insert_sql = """
        INSERT INTO cheatsheet_data (
            player_name, game_date, stat_type, platform,
            line, over_odds, under_odds,
            projection, projection_diff, bet_rating, expected_value, ev_pct,
            probability, recommended_side,
            opp_rank, opp_value,
            hit_rate_l5, hit_rate_l15, hit_rate_season,
            l5_over, l5_under, l15_over, l15_under, season_over, season_under,
            fetch_timestamp, use_for_betting
        ) VALUES (
            %(player_name)s, %(game_date)s, %(stat_type)s, %(platform)s,
            %(line)s, %(over_odds)s, %(under_odds)s,
            %(projection)s, %(projection_diff)s, %(bet_rating)s, %(expected_value)s, %(ev_pct)s,
            %(probability)s, %(recommended_side)s,
            %(opp_rank)s, %(opp_value)s,
            %(hit_rate_l5)s, %(hit_rate_l15)s, %(hit_rate_season)s,
            %(l5_over)s, %(l5_under)s, %(l15_over)s, %(l15_under)s, %(season_over)s, %(season_under)s,
            %(fetch_timestamp)s, %(use_for_betting)s
        )
        ON CONFLICT (player_name, game_date, stat_type, platform)
        DO UPDATE SET
            line = EXCLUDED.line,
            over_odds = EXCLUDED.over_odds,
            under_odds = EXCLUDED.under_odds,
            projection = EXCLUDED.projection,
            projection_diff = EXCLUDED.projection_diff,
            bet_rating = EXCLUDED.bet_rating,
            expected_value = EXCLUDED.expected_value,
            ev_pct = EXCLUDED.ev_pct,
            probability = EXCLUDED.probability,
            recommended_side = EXCLUDED.recommended_side,
            opp_rank = EXCLUDED.opp_rank,
            opp_value = EXCLUDED.opp_value,
            hit_rate_l5 = EXCLUDED.hit_rate_l5,
            hit_rate_l15 = EXCLUDED.hit_rate_l15,
            hit_rate_season = EXCLUDED.hit_rate_season,
            l5_over = EXCLUDED.l5_over,
            l5_under = EXCLUDED.l5_under,
            l15_over = EXCLUDED.l15_over,
            l15_under = EXCLUDED.l15_under,
            season_over = EXCLUDED.season_over,
            season_under = EXCLUDED.season_under,
            fetch_timestamp = EXCLUDED.fetch_timestamp,
            use_for_betting = EXCLUDED.use_for_betting
    """

    rows_affected = 0

    for prop in props:
        try:
            # Parse fetch_timestamp
            fetch_ts = prop.get("fetch_timestamp", "")
            if fetch_ts:
                try:
                    fetch_timestamp = datetime.fromisoformat(fetch_ts.replace("Z", "+00:00"))
                except (ValueError, AttributeError):
                    fetch_timestamp = datetime.now()
            else:
                fetch_timestamp = datetime.now()

            # Prepare record
            record = {
                "player_name": prop.get("player_name"),
                "game_date": prop.get("game_date"),
                "stat_type": prop.get("stat_type"),
                "platform": platform,
                "line": prop.get("line"),
                "over_odds": prop.get("over_odds"),
                "under_odds": prop.get("under_odds"),
                "projection": prop.get("projection"),
                "projection_diff": prop.get("projection_diff"),
                "bet_rating": prop.get("bet_rating"),
                "expected_value": prop.get("expected_value"),
                "ev_pct": prop.get("ev_pct"),
                "probability": prop.get("probability"),
                "recommended_side": prop.get("recommended_side"),
                "opp_rank": prop.get("opp_rank"),
                "opp_value": prop.get("opp_value"),
                "hit_rate_l5": prop.get("hit_rate_l5"),
                "hit_rate_l15": prop.get("hit_rate_l15"),
                "hit_rate_season": prop.get("hit_rate_season"),
                "l5_over": prop.get("l5_over"),
                "l5_under": prop.get("l5_under"),
                "l15_over": prop.get("l15_over"),
                "l15_under": prop.get("l15_under"),
                "season_over": prop.get("season_over"),
                "season_under": prop.get("season_under"),
                "fetch_timestamp": fetch_timestamp,
                "use_for_betting": use_for_betting,
            }

            cursor.execute(insert_sql, record)
            rows_affected += 1

        except (psycopg2.Error, KeyError, TypeError, ValueError) as e:
            print(
                f"  [WARN] Failed to insert {prop.get('player_name')} {prop.get('stat_type')}: {e}"
            )
            continue

    conn.commit()
    cursor.close()

    return rows_affected


def find_cheatsheet_files(date: str = None, latest: bool = False) -> List[Path]:
    """Find cheat sheet JSON files"""
    lines_dir = Path(__file__).parent.parent / "lines"

    # Find all cheatsheet files
    files = list(lines_dir.glob("cheatsheet_*.json"))

    if not files:
        return []

    # Sort by modification time (newest first)
    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

    if latest:
        # Return most recent underdog and prizepicks files
        latest_files = []
        found_underdog = False
        found_prizepicks = False

        for f in files:
            if "underdog" in f.name and not found_underdog:
                latest_files.append(f)
                found_underdog = True
            elif "prizepicks" in f.name and not found_prizepicks:
                latest_files.append(f)
                found_prizepicks = True

            if found_underdog and found_prizepicks:
                break

        return latest_files

    if date:
        # Filter by date in filename or content
        date_files = []
        for f in files:
            # Check if date is in filename
            if date.replace("-", "") in f.name or date in f.name:
                date_files.append(f)
                continue

            # Check content
            try:
                with open(f, "r") as fp:
                    data = json.load(fp)
                    props = data.get("props", [])
                    if props and props[0].get("game_date") == date:
                        date_files.append(f)
            except (json.JSONDecodeError, IOError, KeyError, IndexError):
                pass

        return date_files

    return files


def main():
    parser = argparse.ArgumentParser(description="Load cheat sheet data to database")
    parser.add_argument("--file", type=str, help="Specific JSON file to load")
    parser.add_argument("--date", type=str, help="Load files for specific date (YYYY-MM-DD)")
    parser.add_argument("--latest", action="store_true", help="Load most recent files")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be loaded without inserting"
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("LOADING CHEATSHEET DATA TO DATABASE")
    print("=" * 60 + "\n")

    # Find files to load
    if args.file:
        files = [Path(args.file)]
    elif args.date:
        files = find_cheatsheet_files(date=args.date)
    elif args.latest:
        files = find_cheatsheet_files(latest=True)
    else:
        print("Error: Must specify --file, --date, or --latest")
        sys.exit(1)

    if not files:
        print("[WARN] No cheat sheet files found")
        sys.exit(1)

    print(f"Found {len(files)} file(s) to load:")
    for f in files:
        print(f"  - {f.name}")
    print()

    if args.dry_run:
        print("[DRY RUN] Would load the above files")
        return

    # Connect to database
    conn = get_db_connection()

    total_loaded = 0

    for filepath in files:
        print(f"Loading {filepath.name}...")

        # Determine platform from filename
        if "underdog" in filepath.name:
            platform = "underdog"
        elif "prizepicks" in filepath.name:
            platform = "prizepicks"
        else:
            platform = "unknown"

        # Load props from file
        props = load_json_file(filepath)
        print(f"  Found {len(props)} props")

        # Filter to only OVER recommendations (since model only predicts overs)
        over_props = [p for p in props if p.get("recommended_side") == "over"]
        print(f"  OVER recommendations: {len(over_props)}")

        # Insert into database
        rows = insert_cheatsheet_data(conn, over_props, platform)
        print(f"  Loaded: {rows} rows\n")

        total_loaded += rows

    conn.close()

    print("=" * 60)
    print(f"TOTAL LOADED: {total_loaded} cheat sheet records")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
