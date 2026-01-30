"""
NBA Team Data Loader
Loads team data and season stats into nba_team_db (port 5538)

Usage:
    # Load all team data for 2023-24
    python load_nba_teams.py --season 2023-24

    # Load multiple seasons
    python load_nba_teams.py --season 2021-22 2022-23 2023-24
"""

import argparse
import logging
import os
import sys

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

# Add utilities to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utilities"))
from nba_api_wrapper import NBAApiWrapper

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Database connection params
DB_CONFIG = {
    "host": "localhost",
    "port": 5538,
    "database": "nba_team",
    "user": os.getenv("DB_USER", "nba_user"),
    "password": os.getenv("DB_PASSWORD"),
}


class NBATeamLoader:
    """Loads NBA team data into database"""

    def __init__(self):
        self.api = NBAApiWrapper(requests_per_minute=15)
        self.conn = None

    def connect(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(**DB_CONFIG)
            logger.info("✅ Connected to nba_team database")
        except (psycopg2.Error, KeyError, TypeError, ValueError) as e:
            logger.error(f"❌ Database connection failed: {e}")
            raise

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

    def load_teams(self) -> int:
        """
        Load all NBA teams

        Returns:
            Number of teams loaded
        """
        logger.info("Loading NBA teams...")

        # Fetch teams from API
        teams_df = self.api.get_all_teams()

        if teams_df.empty:
            logger.warning("No teams found")
            return 0

        # Prepare data for insertion
        insert_data = []
        for _, row in teams_df.iterrows():
            # Parse conference/division from team data
            # Note: Static API doesn't include this, using defaults
            insert_data.append(
                (
                    row["id"],
                    row["abbreviation"],
                    row["full_name"],
                    (
                        "East"
                        if row["id"]
                        in [
                            1610612737,
                            1610612738,
                            1610612739,
                            1610612751,
                            1610612765,
                            1610612752,
                            1610612756,
                            1610612748,
                            1610612753,
                            1610612755,
                            1610612749,
                            1610612764,
                            1610612741,
                            1610612754,
                            1610612766,
                        ]
                        else "West"
                    ),
                    None,  # division - can be filled later
                )
            )

        # Insert into database
        cursor = self.conn.cursor()
        insert_query = """
            INSERT INTO teams (team_id, team_abbrev, team_name, conference, division)
            VALUES %s
            ON CONFLICT (team_id)
            DO UPDATE SET
                team_abbrev = EXCLUDED.team_abbrev,
                team_name = EXCLUDED.team_name,
                conference = EXCLUDED.conference
        """

        execute_values(cursor, insert_query, insert_data)
        self.conn.commit()

        logger.info(f"✅ Loaded {len(insert_data)} teams")
        return len(insert_data)

    def load_team_season_stats(self, season: str = "2023-24") -> int:
        """
        Load team season stats (aggregated from game logs)

        Args:
            season: NBA season (e.g., "2023-24")

        Returns:
            Number of team stat records loaded
        """
        logger.info(f"Loading team season stats for {season}...")

        # Get all teams
        teams_df = self.api.get_all_teams()

        if teams_df.empty:
            logger.warning("No teams found")
            return 0

        insert_data = []
        season_year = int(season.split("-")[0])

        # Fetch game logs for each team and aggregate
        for _, team in teams_df.iterrows():
            try:
                logger.info(f"Fetching stats for {team['abbreviation']}...")

                # Get team game logs
                game_logs = self.api.get_team_game_logs(team["id"], season=season)

                if game_logs.empty:
                    logger.warning(f"No game logs for {team['abbreviation']}")
                    continue

                # Calculate aggregate stats
                # Note: Column names vary by endpoint, adjust as needed
                avg_pace = (
                    game_logs.get("PACE", pd.Series([0])).mean()
                    if "PACE" in game_logs.columns
                    else None
                )

                # Offensive/Defensive rating approximations
                # These would ideally come from advanced stats endpoints
                avg_pts = game_logs.get("PTS", pd.Series([0])).mean()
                avg_opp_pts = (
                    game_logs.get("OPP_PTS", pd.Series([0])).mean()
                    if "OPP_PTS" in game_logs.columns
                    else None
                )

                insert_data.append(
                    (
                        team["abbreviation"],
                        season_year,
                        avg_pace,
                        None,  # offensive_rating (requires possessions data)
                        None,  # defensive_rating
                        None,  # def_rating_vs_pg
                        None,  # def_rating_vs_sg
                        None,  # def_rating_vs_sf
                        None,  # def_rating_vs_pf
                        None,  # def_rating_vs_c
                        None,  # pace_neutral_off_rating
                    )
                )

            except (psycopg2.Error, KeyError, TypeError, ValueError) as e:
                logger.warning(f"Failed to load stats for {team['abbreviation']}: {e}")
                continue

        if not insert_data:
            logger.warning("No valid team stats to insert")
            return 0

        # Insert into database
        cursor = self.conn.cursor()
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

        execute_values(cursor, insert_query, insert_data)
        self.conn.commit()

        logger.info(f"✅ Loaded {len(insert_data)} team season stat records")
        return len(insert_data)


def main():
    parser = argparse.ArgumentParser(description="Load NBA team data into database")
    parser.add_argument(
        "--season",
        type=str,
        nargs="+",
        default=["2023-24"],
        help="NBA seasons to load (e.g., 2023-24)",
    )

    args = parser.parse_args()

    loader = NBATeamLoader()

    try:
        loader.connect()

        # Load teams
        loader.load_teams()

        # Load season stats for each season
        for season in args.season:
            loader.load_team_season_stats(season=season)

        logger.info("🎉 Team data loading complete!")

    except (psycopg2.Error, KeyError, TypeError, ValueError) as e:
        logger.error(f"❌ Loading failed: {e}")
        raise
    finally:
        loader.close()


if __name__ == "__main__":
    main()
