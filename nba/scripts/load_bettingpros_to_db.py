#!/usr/bin/env python3
"""
NBA BettingPros Props Loader
Loads historical props from BettingPros JSON files into nba_intelligence database

Usage:
    python load_bettingpros_to_db.py --input-dir nba/data/bettingpros_props/
    python load_bettingpros_to_db.py --input-dir nba/data/bettingpros_props/ --dry-run
"""

import argparse
import glob
import json
import os
from datetime import datetime

import psycopg2

# Database connection
DB_CONFIG = {
    "host": "localhost",
    "port": 5539,
    "user": os.getenv("DB_USER", "nba_user"),
    "password": os.getenv("DB_PASSWORD"),
    "database": "nba_intelligence",
}

# Player database for lookups
PLAYERS_DB_CONFIG = {
    "host": "localhost",
    "port": 5536,
    "user": os.getenv("DB_USER", "nba_user"),
    "password": os.getenv("DB_PASSWORD"),
    "database": "nba_players",
}

# Stat type mapping
STAT_TYPE_MAP = {
    "points": "POINTS",
    "rebounds": "REBOUNDS",
    "assists": "ASSISTS",
    "threes": "THREES",
    "steals": "STEALS",
    "blocks": "BLOCKS",
}


def get_player_id(player_slug, conn):
    """
    Map BettingPros player slug to database player_id

    Args:
        player_slug: e.g., 'lebron-james'
        conn: Database connection to nba_players

    Returns:
        player_id or None
    """
    # Try exact slug match first
    cursor = conn.cursor()

    # BettingPros uses hyphens, our DB might use different format
    # Try multiple approaches

    # Approach 1: Direct match on full_name (convert slug to name)
    name_parts = player_slug.split("-")
    first_name = name_parts[0].capitalize()
    last_name = " ".join(p.capitalize() for p in name_parts[1:])
    full_name = f"{first_name} {last_name}"

    cursor.execute(
        """
        SELECT player_id FROM player_profile
        WHERE LOWER(full_name) = LOWER(%s)
        LIMIT 1
    """,
        (full_name,),
    )

    result = cursor.fetchone()
    if result:
        return result[0]

    # Approach 2: Try last name match (common for unique players)
    cursor.execute(
        """
        SELECT player_id FROM player_profile
        WHERE LOWER(full_name) LIKE LOWER(%s)
        LIMIT 1
    """,
        (f"%{last_name}%",),
    )

    result = cursor.fetchone()
    if result:
        return result[0]

    # Not found
    return None


def parse_bettingpros_file(filepath):
    """
    Parse a BettingPros JSON file

    Returns:
        List of prop dicts with: player_slug, stat_type, game_date, line, actual, projection, etc.
    """
    with open(filepath, "r") as f:
        data = json.load(f)

    # Extract player slug and stat type from filename
    # Format: {player-slug}_{stat-type}_{season}.json
    basename = os.path.basename(filepath)
    parts = basename.replace(".json", "").split("_")

    if len(parts) < 3:
        print(f"⚠️  Skipping malformed filename: {basename}")
        return []

    # Player slug is everything before the last 2 parts
    player_slug = "_".join(parts[:-2])
    stat_type = parts[-2]
    season = parts[-1]

    # Canonical stat type
    stat_canonical = STAT_TYPE_MAP.get(stat_type, stat_type.upper())

    analyses = data.get("analyses", [])

    props = []
    for analysis in analyses:
        event = analysis.get("event", {})
        prop_offer = analysis.get("propOffer", {})

        # Skip if missing critical data
        line = prop_offer.get("line")
        actual = prop_offer.get("score")

        if line is None:
            continue  # No line offered

        # Extract fields
        prop = {
            "player_slug": player_slug,
            "stat_type": stat_canonical,
            "game_date": prop_offer.get("date"),
            "line": line,
            "actual": actual,
            "over_odds": prop_offer.get("cost_inverse"),
            "under_odds": prop_offer.get("cost"),
            "bettingpros_projection": prop_offer.get("projection"),
            "bettingpros_recommendation": prop_offer.get("recommendation"),
            "season": int(season) if season.isdigit() else None,
            "event_id": event.get("id"),
            "raw_json": json.dumps(analysis),
        }

        props.append(prop)

    return props


def load_props_to_database(props, players_conn, intel_conn, dry_run=False):
    """
    Load props into nba_intelligence database

    Args:
        props: List of prop dicts
        players_conn: Connection to nba_players (for lookups)
        intel_conn: Connection to nba_intelligence
        dry_run: If True, don't commit

    Returns:
        (inserted, skipped, errors)
    """
    cursor = intel_conn.cursor()

    inserted = 0
    skipped = 0
    errors = 0

    for prop in props:
        try:
            # Get player_id
            player_id = get_player_id(prop["player_slug"], players_conn)

            if player_id is None:
                print(f"⚠️  Player not found: {prop['player_slug']}")
                skipped += 1
                continue

            # Check if already exists
            cursor.execute(
                """
                SELECT COUNT(*) FROM nba_prop_lines
                WHERE player_id = %s
                  AND game_date = %s
                  AND stat_type = %s
                  AND source = 'bettingpros'
            """,
                (player_id, prop["game_date"], prop["stat_type"]),
            )

            exists = cursor.fetchone()[0] > 0

            if exists:
                skipped += 1
                continue

            # Insert
            cursor.execute(
                """
                INSERT INTO nba_prop_lines (
                    source,
                    player_id,
                    stat_type,
                    line,
                    actual_result,
                    odds_over,
                    odds_under,
                    game_date,
                    bettingpros_projection,
                    bettingpros_recommendation,
                    season,
                    source_data
                ) VALUES (
                    'bettingpros',
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
            """,
                (
                    player_id,
                    prop["stat_type"],
                    prop["line"],
                    prop["actual"],
                    prop["over_odds"],
                    prop["under_odds"],
                    prop["game_date"],
                    prop["bettingpros_projection"],
                    prop["bettingpros_recommendation"],
                    prop["season"],
                    prop["raw_json"],
                ),
            )

            inserted += 1

        except (psycopg2.Error, KeyError, TypeError, ValueError) as e:
            print(f"❌ Error loading prop: {str(e)}")
            print(f"   Prop: {prop['player_slug']} {prop['stat_type']} {prop['game_date']}")
            errors += 1

    if not dry_run:
        intel_conn.commit()

    return inserted, skipped, errors


def main():
    parser = argparse.ArgumentParser(description="Load NBA BettingPros props to database")
    parser.add_argument("--input-dir", required=True, help="Directory with BettingPros JSON files")
    parser.add_argument("--dry-run", action="store_true", help="Parse but don't insert")

    args = parser.parse_args()

    print("=" * 80)
    print("NBA BETTINGPROS PROPS LOADER")
    print("=" * 80)

    # Find all JSON files
    pattern = os.path.join(args.input_dir, "*.json")
    files = glob.glob(pattern)

    print(f"\nFound {len(files)} JSON files in {args.input_dir}")

    if args.dry_run:
        print("\n🔍 DRY RUN MODE - No data will be inserted")

    # Connect to databases
    print("\nConnecting to databases...")
    try:
        players_conn = psycopg2.connect(**PLAYERS_DB_CONFIG)
        intel_conn = psycopg2.connect(**DB_CONFIG)
        print("✅ Connected to nba_players (port 5536)")
        print("✅ Connected to nba_intelligence (port 5539)")
    except (psycopg2.Error, KeyError, TypeError, ValueError) as e:
        print(f"❌ Database connection failed: {str(e)}")
        return

    # Parse all files
    print(f"\nParsing {len(files)} files...")
    all_props = []

    for filepath in files:
        props = parse_bettingpros_file(filepath)
        all_props.extend(props)

    print(f"✅ Parsed {len(all_props)} total props")

    # Load to database
    print(f"\nLoading props to database...")
    inserted, skipped, errors = load_props_to_database(
        all_props, players_conn, intel_conn, dry_run=args.dry_run
    )

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Inserted: {inserted}")
    print(f"Skipped (duplicates): {skipped}")
    print(f"Errors: {errors}")
    print(f"Total: {len(all_props)}")

    if not args.dry_run:
        print(f"\n✅ Data committed to database")
    else:
        print(f"\n🔍 Dry run complete - no data inserted")

    # Close connections
    players_conn.close()
    intel_conn.close()


if __name__ == "__main__":
    main()
