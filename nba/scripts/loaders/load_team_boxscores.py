#!/usr/bin/env python3
"""
Load team game log box score stats from NBA API
Populates: fg_made, fg_attempted, three_pt_made, three_pt_attempted,
           ft_made, ft_attempted, rebounds, assists, turnovers
"""

import sys
from datetime import datetime
from pathlib import Path

import psycopg2

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from nba.config.database import get_games_db_config

sys.path.append(str(Path(__file__).parent.parent.parent / "nba/scripts/utilities"))
from nba_api_wrapper import NBAApiWrapper

DB_CONFIG = get_games_db_config()


def main():
    print("\n" + "=" * 80)
    print("LOADING TEAM GAME LOG BOX SCORES FROM NBA API")
    print("=" * 80 + "\n")

    api = NBAApiWrapper(requests_per_minute=15)
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # Get all team game logs that need box score data
    cur.execute(
        """
        SELECT game_log_id, team_abbrev, game_id, game_date, opponent, is_home
        FROM team_game_logs
        WHERE fg_made IS NULL
        ORDER BY game_date, game_id
        LIMIT 100;
    """
    )

    logs = cur.fetchall()
    print(f"Found {len(logs)} team game logs needing box score data")
    print(f"Processing first 100 to test...\n")

    if len(logs) == 0:
        print("✅ All team game logs already have box score data!")
        cur.close()
        conn.close()
        return 0

    fixed_count = 0
    for i, (log_id, team, game_id, game_date, opp, _is_home) in enumerate(logs, 1):
        try:
            print(f"[{i}/{len(logs)}] {game_date} {team} vs {opp}...", end=" ")

            # Extract season from game_id (format: 002XXYYYYYY)
            if game_id.startswith("002"):
                season_year = int(game_id[3:5])
                season = 2000 + season_year
            else:
                season = int(game_date.split("-")[0])

            # Fetch team game log from NBA API
            team_logs = api.get_team_game_logs(team, season)

            if not team_logs:
                print("⚠️  No data from API")
                continue

            # Find matching game in team logs
            matching_game = None
            for log in team_logs:
                if log.get("GAME_ID") == game_id:
                    matching_game = log
                    break

            if not matching_game:
                print("⚠️  Game not found in logs")
                continue

            # Extract box score stats
            fg_made = matching_game.get("FGM")
            fg_attempted = matching_game.get("FGA")
            three_pt_made = matching_game.get("FG3M")
            three_pt_attempted = matching_game.get("FG3A")
            ft_made = matching_game.get("FTM")
            ft_attempted = matching_game.get("FTA")
            rebounds = matching_game.get("REB")
            assists = matching_game.get("AST")
            turnovers = matching_game.get("TOV")

            # Update database
            cur.execute(
                """
                UPDATE team_game_logs
                SET fg_made = %s,
                    fg_attempted = %s,
                    three_pt_made = %s,
                    three_pt_attempted = %s,
                    ft_made = %s,
                    ft_attempted = %s,
                    rebounds = %s,
                    assists = %s,
                    turnovers = %s
                WHERE game_log_id = %s;
            """,
                (
                    fg_made,
                    fg_attempted,
                    three_pt_made,
                    three_pt_attempted,
                    ft_made,
                    ft_attempted,
                    rebounds,
                    assists,
                    turnovers,
                    log_id,
                ),
            )

            conn.commit()
            fixed_count += 1
            print(f"✅ {fg_made}/{fg_attempted} FG, {rebounds} REB, {assists} AST")

        except (psycopg2.Error, KeyError, TypeError, ValueError) as e:
            print(f"❌ Error: {e}")
            conn.rollback()
            continue

    print(f"\n✅ Updated {fixed_count}/{len(logs)} team game logs")

    # Show remaining work
    cur.execute("SELECT COUNT(*) FROM team_game_logs WHERE fg_made IS NULL;")
    remaining = cur.fetchone()[0]
    print(f"⚠️  {remaining} team game logs still need box score data")
    print(f"\nRun this script multiple times to load all data (rate limited to 15 req/min)")

    cur.close()
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
