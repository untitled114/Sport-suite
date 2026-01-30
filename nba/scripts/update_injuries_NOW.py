#!/usr/bin/env python3
"""
QUICK INJURY UPDATE - Use correct ESPN API
===========================================
Run this before tonight's predictions!

Usage:
    python3 nba/scripts/update_injuries_NOW.py
"""

import os
from datetime import datetime

import psycopg2
import requests

DB_CONFIG = {
    "host": "localhost",
    "port": 5539,  # nba_intelligence database (has injuries table)
    "user": os.getenv("DB_USER", "nba_user"),
    "password": os.getenv("DB_PASSWORD"),
    "database": "nba_intelligence",
}


def fetch_injuries():
    """Fetch from correct ESPN injuries endpoint"""
    print("\n" + "=" * 80)
    print("NBA INJURY UPDATE - ESPN API")
    print("=" * 80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    url = "https://site.web.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"

    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()

        injuries = []
        for team in data.get("injuries", []):
            team_name = team.get("displayName")
            for inj in team.get("injuries", []):
                athlete = inj.get("athlete", {})
                injuries.append(
                    {
                        "name": athlete.get("displayName"),
                        "espn_id": (
                            athlete.get("links", [{}])[0].get("href", "").split("/")[-1]
                            if athlete.get("links")
                            else None
                        ),
                        "status": inj.get("status"),
                        "description": inj.get("shortComment", ""),
                    }
                )

        print(f"✅ Fetched {len(injuries)} injuries from ESPN\n")
        return injuries

    except (psycopg2.Error, KeyError, TypeError, ValueError) as e:
        print(f"❌ Failed to fetch: {e}")
        return []


def load_to_database(injuries):
    """Load injuries into database"""
    if not injuries:
        print("No injuries to load")
        return

    try:
        # Connect to nba_intelligence (port 5539) for injury_report table
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # Clear old injuries from today (keep historical)
        cursor.execute("DELETE FROM injury_report WHERE report_date = CURRENT_DATE")
        print(f"🗑️  Cleared today's injury data\n")

        # Connect to nba_players (port 5536) to get player_profile
        player_db_config = {
            "host": "localhost",
            "port": 5536,
            "user": os.getenv("DB_USER", "nba_user"),
            "password": os.getenv("DB_PASSWORD"),
            "database": "nba_players",
        }
        player_conn = psycopg2.connect(**player_db_config)
        player_cursor = player_conn.cursor()

        # Get all players for name matching (use DISTINCT ON to handle duplicates)
        player_cursor.execute(
            "SELECT DISTINCT ON (full_name) player_id, full_name FROM player_profile ORDER BY full_name, player_id DESC"
        )
        player_map = {
            full_name.lower(): player_id for player_id, full_name in player_cursor.fetchall()
        }

        player_cursor.close()
        player_conn.close()

        # Insert new injuries
        inserted = 0
        skipped = 0

        print("📝 Loading injuries:\n")
        for inj in injuries:
            try:
                inj_name = inj["name"]
                if not inj_name:
                    skipped += 1
                    continue

                # Try exact match
                player_id = player_map.get(inj_name.lower())

                if not player_id:
                    skipped += 1
                    print(f"   ⚠️  No match: {inj_name}")
                    continue

                status = (
                    inj["status"].upper().replace(" ", "_").replace("-", "_")[:50]
                )  # Truncate status too
                injury_desc = (inj["description"] or "Unknown")[:100]  # Truncate to 100 chars

                # Insert with correct schema (player_id, report_date, status, injury_type)
                cursor.execute(
                    """
                    INSERT INTO injury_report
                    (player_id, report_date, status, injury_type, updated_at)
                    VALUES (%s, CURRENT_DATE, %s, %s, NOW())
                """,
                    (player_id, status, injury_desc),
                )

                print(f"   ✅ {inj_name}: {status}")
                inserted += 1

            except (psycopg2.Error, KeyError, TypeError, ValueError) as e:
                print(f"   ❌ Error inserting {inj.get('name', 'Unknown')}: {e}")
                skipped += 1
                continue

        conn.commit()
        cursor.close()
        conn.close()

        print(f"\n✅ Loaded {inserted} injuries")
        if skipped > 0:
            print(f"   ⚠️  Skipped {skipped} (no player match)")

    except (psycopg2.Error, KeyError, TypeError, ValueError) as e:
        print(f"❌ Database error: {e}")


def main():
    injuries = fetch_injuries()
    if injuries:
        load_to_database(injuries)

        print("\n" + "=" * 80)
        print("✅ INJURY UPDATE COMPLETE")
        print("=" * 80)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("\n💡 Next: Run daily predictions")
        print("   python3 nba/betting/daily_predictions_PRODUCTION.py")
    else:
        print("\n❌ No injuries fetched - check network/API")


if __name__ == "__main__":
    main()
