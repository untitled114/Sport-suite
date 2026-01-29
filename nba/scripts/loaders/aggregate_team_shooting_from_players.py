#!/usr/bin/env python3
"""
Aggregate Team Game Log Shooting Stats from Player Game Logs
============================================================
Populates team_game_logs shooting stats by summing player stats:
- fg_made, fg_attempted
- three_pt_made, three_pt_attempted
- ft_made, ft_attempted
- rebounds, assists, turnovers

This avoids NBA API issues and uses data we already have.

Usage:
    python aggregate_team_shooting_from_players.py
"""

import logging
import psycopg2
from psycopg2.extras import execute_batch
import os

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database connections
PLAYERS_DB = {
    'host': 'localhost',
    'port': 5536,
    'database': 'nba_players',
    'user': os.getenv('DB_USER', 'nba_user'),
    'password': os.getenv('DB_PASSWORD')
}

GAMES_DB = {
    'host': 'localhost',
    'port': 5537,
    'database': 'nba_games',
    'user': os.getenv('DB_USER', 'nba_user'),
    'password': os.getenv('DB_PASSWORD')
}


def aggregate_team_shooting_stats():
    """Aggregate player game logs into team game logs shooting stats"""

    players_conn = psycopg2.connect(**PLAYERS_DB)
    games_conn = psycopg2.connect(**GAMES_DB)

    try:
        players_cur = players_conn.cursor()
        games_cur = games_conn.cursor()

        # Get count of team game logs needing shooting stats
        games_cur.execute("""
            SELECT COUNT(*)
            FROM team_game_logs
            WHERE fg_made IS NULL
        """)
        total_to_fix = games_cur.fetchone()[0]

        logger.info(f"Found {total_to_fix} team game logs needing shooting stats")

        if total_to_fix == 0:
            logger.info("✅ All team game logs already have shooting stats!")
            return

        # Get team game logs that need shooting stats
        games_cur.execute("""
            SELECT game_log_id, team_abbrev, game_id, game_date
            FROM team_game_logs
            WHERE fg_made IS NULL
            ORDER BY game_date, game_id
        """)

        team_logs = games_cur.fetchall()
        logger.info(f"Processing {len(team_logs)} team game logs...")

        updates = []
        fixed_count = 0
        failed_count = 0

        for i, (log_id, team, game_id, game_date) in enumerate(team_logs, 1):
            if i % 500 == 0:
                logger.info(f"  Progress: {i}/{len(team_logs)} ({100*i//len(team_logs)}%)")

            # Aggregate player stats for this team in this game
            # Use team_abbrev from player_game_logs (not profile) to handle trades
            players_cur.execute("""
                SELECT
                    SUM(fg_made) as fg_made,
                    SUM(fg_attempted) as fg_attempted,
                    SUM(three_pointers_made) as three_pt_made,
                    SUM(three_pt_attempted) as three_pt_attempted,
                    SUM(ft_made) as ft_made,
                    SUM(ft_attempted) as ft_attempted,
                    SUM(rebounds) as rebounds,
                    SUM(assists) as assists,
                    SUM(turnovers) as turnovers
                FROM player_game_logs
                WHERE team_abbrev = %s
                  AND game_id = %s
                  AND game_date = %s
            """, (team, game_id, game_date))

            result = players_cur.fetchone()

            if result and result[0] is not None:  # Check if we got data
                updates.append((
                    result[0],  # fg_made
                    result[1],  # fg_attempted
                    result[2],  # three_pt_made
                    result[3],  # three_pt_attempted
                    result[4],  # ft_made
                    result[5],  # ft_attempted
                    result[6],  # rebounds
                    result[7],  # assists
                    result[8],  # turnovers
                    log_id
                ))
                fixed_count += 1
            else:
                failed_count += 1
                if failed_count <= 10:  # Only log first 10 failures
                    logger.warning(f"  No player data for {team} in game {game_id} on {game_date}")

        # Batch update team game logs
        if updates:
            logger.info(f"Updating {len(updates)} team game logs...")

            update_query = """
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
                WHERE game_log_id = %s
            """

            execute_batch(games_cur, update_query, updates, page_size=1000)
            games_conn.commit()

            logger.info(f"✅ Updated {len(updates)} team game logs")

        if failed_count > 0:
            logger.warning(f"⚠️  Failed to find player data for {failed_count} team game logs")
            logger.info("This is expected if players changed teams mid-season or for old seasons")

        # Final check
        games_cur.execute("SELECT COUNT(*) FROM team_game_logs WHERE fg_made IS NULL")
        remaining = games_cur.fetchone()[0]

        if remaining > 0:
            logger.warning(f"⚠️  {remaining} team game logs still missing shooting stats")
        else:
            logger.info("🎉 All team game logs now have shooting stats!")

    except Exception as e:
        logger.error(f"Error: {e}")
        games_conn.rollback()
        raise
    finally:
        players_conn.close()
        games_conn.close()


def main():
    logger.info("="*80)
    logger.info("AGGREGATING TEAM SHOOTING STATS FROM PLAYER GAME LOGS")
    logger.info("="*80)

    aggregate_team_shooting_stats()

    logger.info("\n✅ Done!")


if __name__ == "__main__":
    main()
