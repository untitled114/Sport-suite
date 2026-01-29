#!/usr/bin/env python3
"""
Calculate Team Season Stats from Games Data
============================================
Calculates team_season_stats including:
- Pace (average from team_game_logs)
- Offensive Rating (Points per 100 possessions)
- Defensive Rating (Points allowed per 100 possessions)

Uses games table to get opponent points for defensive rating.

Usage:
    python calculate_team_stats.py --season 2020 2021 2022 2023 2024
"""

import argparse
import logging
import psycopg2
from psycopg2.extras import execute_values
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database connections
GAMES_DB = {
    'host': 'localhost',
    'port': 5537,
    'database': 'nba_games',
    'user': os.getenv('DB_USER', 'nba_user'),
    'password': os.getenv('DB_PASSWORD')
}

TEAM_DB = {
    'host': 'localhost',
    'port': 5538,
    'database': 'nba_team',
    'user': os.getenv('DB_USER', 'nba_user'),
    'password': os.getenv('DB_PASSWORD')
}


def calculate_team_season_stats(seasons: list):
    """Calculate and insert team season stats from games data"""

    games_conn = psycopg2.connect(**GAMES_DB)
    team_conn = psycopg2.connect(**TEAM_DB)

    try:
        for season in seasons:
            logger.info(f"Calculating team stats for season {season}...")

            games_cur = games_conn.cursor()

            # Get all teams for this season
            # Use 'season' column, not EXTRACT(YEAR FROM game_date)
            # NBA 2025-26 season (season=2026) has games from Oct 2025 to Apr 2026
            games_cur.execute("""
                SELECT DISTINCT team_abbrev
                FROM team_game_logs
                WHERE season = %s
                ORDER BY team_abbrev
            """, (season,))
            teams = [row[0] for row in games_cur.fetchall()]

            if not teams:
                logger.warning(f"No teams found for season {season}")
                continue

            insert_data = []

            for team in teams:
                # Calculate team stats from games table
                # Get home games (team scored = home_score, opponent = away_score)
                # Note: pace may be NULL for current season - use COALESCE to default to 100
                games_cur.execute("""
                    SELECT
                        COUNT(*) as games,
                        AVG(g.home_score) as avg_points_scored,
                        AVG(g.away_score) as avg_points_allowed,
                        AVG(COALESCE(tgl.pace, 100.0)) as avg_pace
                    FROM games g
                    JOIN team_game_logs tgl ON g.game_id = tgl.game_id AND tgl.team_abbrev = %s
                    WHERE g.home_team = %s
                      AND g.season = %s
                      AND g.home_score IS NOT NULL
                """, (team, team, season))
                home_data = games_cur.fetchone()

                # Get away games (team scored = away_score, opponent = home_score)
                games_cur.execute("""
                    SELECT
                        COUNT(*) as games,
                        AVG(g.away_score) as avg_points_scored,
                        AVG(g.home_score) as avg_points_allowed,
                        AVG(COALESCE(tgl.pace, 100.0)) as avg_pace
                    FROM games g
                    JOIN team_game_logs tgl ON g.game_id = tgl.game_id AND tgl.team_abbrev = %s
                    WHERE g.away_team = %s
                      AND g.season = %s
                      AND g.away_score IS NOT NULL
                """, (team, team, season))
                away_data = games_cur.fetchone()

                # Combine home and away stats
                total_games = (home_data[0] or 0) + (away_data[0] or 0)

                if total_games == 0:
                    logger.warning(f"No games found for {team} in {season}")
                    continue

                # Weighted average
                home_games = home_data[0] or 0
                away_games = away_data[0] or 0

                avg_points_scored = (
                    (home_data[1] or 0) * home_games +
                    (away_data[1] or 0) * away_games
                ) / total_games if total_games > 0 else 0

                avg_points_allowed = (
                    (home_data[2] or 0) * home_games +
                    (away_data[2] or 0) * away_games
                ) / total_games if total_games > 0 else 0

                avg_pace = (
                    (home_data[3] or 0) * home_games +
                    (away_data[3] or 0) * away_games
                ) / total_games if total_games > 0 else 100.0

                # Calculate offensive and defensive rating
                # Rating = Points per 100 possessions
                # Possessions ≈ Pace (which is possessions per 48 minutes)
                # Offensive Rating = (Points Scored / Pace) * 100
                # Defensive Rating = (Points Allowed / Pace) * 100

                offensive_rating = (avg_points_scored / avg_pace) * 100 if avg_pace > 0 else None
                defensive_rating = (avg_points_allowed / avg_pace) * 100 if avg_pace > 0 else None

                insert_data.append((
                    team,
                    season,
                    round(avg_pace, 2),
                    round(offensive_rating, 2) if offensive_rating else None,
                    round(defensive_rating, 2) if defensive_rating else None,
                    None,  # def_rating_vs_pg (requires positional data)
                    None,  # def_rating_vs_sg
                    None,  # def_rating_vs_sf
                    None,  # def_rating_vs_pf
                    None,  # def_rating_vs_c
                    None   # pace_neutral_off_rating
                ))

                logger.debug(f"  {team}: {total_games} games, Pace={avg_pace:.1f}, OffRtg={offensive_rating:.1f}, DefRtg={defensive_rating:.1f}")

            # Insert into team database
            if insert_data:
                team_cur = team_conn.cursor()
                insert_query = """
                    INSERT INTO team_season_stats
                    (team_abbrev, season, pace, offensive_rating, defensive_rating,
                     def_rating_vs_pg, def_rating_vs_sg, def_rating_vs_sf,
                     def_rating_vs_pf, def_rating_vs_c, pace_neutral_off_rating)
                    VALUES %s
                    ON CONFLICT (team_abbrev, season)
                    DO UPDATE SET
                        pace = EXCLUDED.pace,
                        offensive_rating = EXCLUDED.offensive_rating,
                        defensive_rating = EXCLUDED.defensive_rating
                """

                execute_values(team_cur, insert_query, insert_data)
                team_conn.commit()

                logger.info(f"✅ Calculated and inserted {len(insert_data)} team season stats for {season}")
            else:
                logger.warning(f"No team stats calculated for {season}")

    except Exception as e:
        logger.error(f"Error calculating team stats: {e}")
        raise
    finally:
        games_conn.close()
        team_conn.close()


def main():
    parser = argparse.ArgumentParser(description='Calculate team season stats from game logs')
    parser.add_argument('--season', type=int, nargs='+', default=[2020, 2021, 2022, 2023, 2024],
                       help='Seasons to calculate (years, e.g., 2020 2021 2022)')

    args = parser.parse_args()

    logger.info("Starting team season stats calculation...")
    logger.info(f"Seasons: {args.season}")
    calculate_team_season_stats(args.season)
    logger.info("🎉 Team season stats calculation complete!")


if __name__ == "__main__":
    main()
