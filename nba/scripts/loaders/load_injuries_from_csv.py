#!/usr/bin/env python3
"""
Load NBA Injuries from CSV - PRAGMATIC SOLUTION
================================================
Manual CSV upload system for tracking injuries.
Much more reliable than scraping dynamic ESPN pages.

CSV Format:
-----------
player_name,status,injury_description
Jayson Tatum,Out,Left ankle sprain
LeBron James,Questionable,Ankle soreness
"""

import csv
import os
import sys
from datetime import datetime

import psycopg2


def load_injuries_from_csv(csv_path):
    """Load injuries from CSV file"""

    conn = psycopg2.connect(
        host="localhost",
        port=5536,
        user=os.getenv("DB_USER", "nba_user"),
        password=os.getenv("DB_PASSWORD"),
        database="nba_players",
    )

    cursor = conn.cursor()

    # Clear existing injuries
    cursor.execute("DELETE FROM player_injury_status")
    print("Cleared existing injury records")

    saved = 0
    not_found = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            player_name = row["player_name"].strip()
            status = row["status"].strip()
            description = row.get("injury_description", "").strip()

            # Try to find player
            name_variations = [
                player_name,
                player_name.replace(" Jr.", ""),
                player_name.replace(" III", ""),
                player_name.replace(" II", ""),
            ]

            player_id = None
            for name in name_variations:
                cursor.execute(
                    """
                    SELECT player_id FROM player_profile
                    WHERE full_name = %s
                    LIMIT 1
                """,
                    (name,),
                )

                result = cursor.fetchone()
                if result:
                    player_id = result[0]
                    break

            if not player_id:
                not_found.append(player_name)
                continue

            # Insert injury
            cursor.execute(
                """
                INSERT INTO player_injury_status
                (player_id, injury_status, injury_description, data_source)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (player_id)
                DO UPDATE SET
                    injury_status = EXCLUDED.injury_status,
                    injury_description = EXCLUDED.injury_description,
                    last_updated = CURRENT_TIMESTAMP
            """,
                (player_id, status, description, "MANUAL_CSV"),
            )

            saved += 1
            print(f"  ✅ {player_name} - {status}")

    conn.commit()
    cursor.close()
    conn.close()

    print(f"\n✅ Saved {saved} injury records")

    if not_found:
        print(f"\n⚠️  Players not found ({len(not_found)}):")
        for name in not_found:
            print(f"   - {name}")

    return saved


def create_template_csv(output_path):
    """Create template CSV for manual entry"""

    with open(output_path, "w") as f:
        f.write("player_name,status,injury_description\n")
        f.write("# Valid statuses: Out, Questionable, Day-To-Day, Doubtful\n")
        f.write("# Example entries:\n")
        f.write("# Jayson Tatum,Out,Left ankle sprain\n")
        f.write("# LeBron James,Questionable,Ankle soreness\n")

    print(f"✅ Template created: {output_path}")
    print(f"\nEdit this file and run:")
    print(f"  python3 nba/scripts/loaders/load_injuries_from_csv.py {output_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("NBA INJURY LOADER - MANUAL CSV")
    print("=" * 60)

    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  Create template:")
        print("    python3 nba/scripts/loaders/load_injuries_from_csv.py --create-template [path]")
        print("\n  Load injuries:")
        print("    python3 nba/scripts/loaders/load_injuries_from_csv.py injuries.csv")
        sys.exit(1)

    if sys.argv[1] == "--create-template":
        output = sys.argv[2] if len(sys.argv) > 2 else "nba/data/injuries_template.csv"
        create_template_csv(output)
    else:
        csv_path = sys.argv[1]
        load_injuries_from_csv(csv_path)
