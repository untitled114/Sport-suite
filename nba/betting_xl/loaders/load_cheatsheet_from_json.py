#!/usr/bin/env python3
"""
Load BettingPros historical cheatsheet data from JSON to cheatsheet_data table.

Usage:
    python load_cheatsheet_from_json.py --file /path/to/checkpoint.json
    python load_cheatsheet_from_json.py --dir /path/to/lines/  # Load all checkpoint files
"""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_values

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    "host": "localhost",
    "port": 5539,
    "database": "nba_intelligence",
    "user": os.getenv("DB_USER", "nba_user"),
    "password": os.getenv("DB_PASSWORD"),
}

# Valid stat types for cheatsheet
VALID_STAT_TYPES = {"POINTS", "REBOUNDS", "ASSISTS", "THREES", "STEALS", "BLOCKS"}


def load_json_file(filepath: str) -> list:
    """Load props from JSON checkpoint file."""
    logger.info(f"Loading {filepath}...")

    with open(filepath, "r") as f:
        data = json.load(f)

    props = data.get("props", [])
    logger.info(f"  Found {len(props):,} props in file")

    return props


def transform_prop(prop: dict) -> dict:
    """Transform JSON prop to cheatsheet_data format."""
    # Map source to platform
    source = prop.get("source", "bettingpros")
    if "bettingpros" in source.lower():
        platform = "bettingpros"
    else:
        platform = source

    # Parse game_date
    game_date = prop.get("game_date")
    if isinstance(game_date, str):
        game_date = game_date[:10]  # Take YYYY-MM-DD part

    # Parse fetch_timestamp
    fetch_ts = prop.get("fetch_timestamp")
    if isinstance(fetch_ts, str):
        try:
            fetch_ts = datetime.fromisoformat(fetch_ts.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            fetch_ts = datetime.now()

    return {
        "player_name": prop.get("player_name"),
        "game_date": game_date,
        "stat_type": prop.get("stat_type", "").upper(),
        "platform": platform,
        "line": prop.get("line") or prop.get("over_line"),
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
        "fetch_timestamp": fetch_ts,
        "use_for_betting": True,
    }


def load_to_database(props: list, batch_size: int = 1000):
    """Load props to cheatsheet_data table with upsert."""
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    # Filter valid stat types
    valid_props = [p for p in props if p.get("stat_type", "").upper() in VALID_STAT_TYPES]
    logger.info(f"  Filtered to {len(valid_props):,} props with valid stat types")

    # Transform props
    transformed = [transform_prop(p) for p in valid_props]

    # Filter out props with missing required fields
    transformed = [p for p in transformed if p["player_name"] and p["game_date"] and p["stat_type"]]
    logger.info(f"  {len(transformed):,} props have required fields")

    # Deduplicate by (player_name, game_date, stat_type, platform) - keep last occurrence
    seen = {}
    for p in transformed:
        key = (p["player_name"], p["game_date"], p["stat_type"], p["platform"])
        seen[key] = p  # Later entries overwrite earlier ones
    transformed = list(seen.values())
    logger.info(f"  {len(transformed):,} unique props after deduplication")

    if not transformed:
        logger.warning("  No valid props to load")
        conn.close()
        return 0

    # Upsert query
    insert_query = """
    INSERT INTO cheatsheet_data (
        player_name, game_date, stat_type, platform, line,
        over_odds, under_odds, projection, projection_diff, bet_rating,
        expected_value, ev_pct, probability, recommended_side,
        opp_rank, opp_value, hit_rate_l5, hit_rate_l15, hit_rate_season,
        l5_over, l5_under, l15_over, l15_under, season_over, season_under,
        fetch_timestamp, use_for_betting
    ) VALUES %s
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
        fetch_timestamp = EXCLUDED.fetch_timestamp
    """

    # Convert to tuples for execute_values
    values = [
        (
            p["player_name"],
            p["game_date"],
            p["stat_type"],
            p["platform"],
            p["line"],
            p["over_odds"],
            p["under_odds"],
            p["projection"],
            p["projection_diff"],
            p["bet_rating"],
            p["expected_value"],
            p["ev_pct"],
            p["probability"],
            p["recommended_side"],
            p["opp_rank"],
            p["opp_value"],
            p["hit_rate_l5"],
            p["hit_rate_l15"],
            p["hit_rate_season"],
            p["l5_over"],
            p["l5_under"],
            p["l15_over"],
            p["l15_under"],
            p["season_over"],
            p["season_under"],
            p["fetch_timestamp"],
            p["use_for_betting"],
        )
        for p in transformed
    ]

    # Batch insert
    total_loaded = 0
    for i in range(0, len(values), batch_size):
        batch = values[i : i + batch_size]
        try:
            execute_values(cursor, insert_query, batch)
            conn.commit()
            total_loaded += len(batch)
            if (i // batch_size) % 10 == 0:
                logger.info(f"    Loaded {total_loaded:,}/{len(values):,} props...")
        except (psycopg2.Error, KeyError, TypeError, ValueError) as e:
            logger.error(f"  Error loading batch at {i}: {e}")
            conn.rollback()

    cursor.close()
    conn.close()

    logger.info(f"  ✅ Loaded {total_loaded:,} props to cheatsheet_data")
    return total_loaded


def find_checkpoint_files(directory: str) -> list:
    """Find all historical checkpoint JSON files in directory."""
    dir_path = Path(directory)
    files = list(dir_path.glob("*historical*checkpoint*.json"))
    files.sort(key=lambda f: f.stat().st_mtime)  # Sort by modification time
    return files


def main():
    parser = argparse.ArgumentParser(description="Load cheatsheet data from JSON to database")
    parser.add_argument("--file", type=str, help="Single JSON file to load")
    parser.add_argument("--dir", type=str, help="Directory containing checkpoint files")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for inserts")
    args = parser.parse_args()

    if not args.file and not args.dir:
        # Default to lines directory
        args.dir = str(Path(__file__).resolve().parent.parent.parent / "nba/betting_xl/lines/")

    files_to_load = []

    if args.file:
        files_to_load = [Path(args.file)]
    elif args.dir:
        files_to_load = find_checkpoint_files(args.dir)

    if not files_to_load:
        logger.error("No checkpoint files found")
        return

    logger.info(f"Found {len(files_to_load)} checkpoint files to load")

    total_loaded = 0
    for filepath in files_to_load:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {filepath.name}")
        logger.info(f"{'='*60}")

        props = load_json_file(str(filepath))
        loaded = load_to_database(props, args.batch_size)
        total_loaded += loaded

    logger.info(f"\n{'='*60}")
    logger.info(f"COMPLETE: Loaded {total_loaded:,} total props to cheatsheet_data")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
