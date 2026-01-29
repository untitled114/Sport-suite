"""
NBA Player Data Loader
Loads player profiles, season stats, and game logs into nba_players_db (port 5536)

Usage:
    # Load all data for 2023-24 season
    python load_nba_players.py --season 2023-24

    # Load multiple seasons
    python load_nba_players.py --season 2021-22 2022-23 2023-24

    # Load specific player
    python load_nba_players.py --player-id 2544 --season 2023-24
"""

import sys
import os
import argparse
import logging
from typing import List, Optional
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

# Add utilities to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utilities'))
from nba_api_wrapper import NBAApiWrapper

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database connection params
DB_CONFIG = {
    'host': 'localhost',
    'port': 5536,
    'database': 'nba_players',
    'user': os.getenv('DB_USER', 'nba_user'),
    'password': os.getenv('DB_PASSWORD')
}


class NBAPlayerLoader:
    """Loads NBA player data into database"""

    def __init__(self):
        self.api = NBAApiWrapper(requests_per_minute=20)
        self.conn = None

    def connect(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(**DB_CONFIG)
            logger.info("✅ Connected to nba_players database")
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            raise

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

    def load_player_profiles(self, active_only: bool = True) -> int:
        """
        Load player profile data

        Args:
            active_only: If True, only load active players

        Returns:
            Number of players loaded
        """
        logger.info("Loading player profiles...")

        # Fetch players from API
        players_df = self.api.get_all_players(active_only=active_only)

        if players_df.empty:
            logger.warning("No players found")
            return 0

        # Prepare data for insertion
        insert_data = []
        for _, row in players_df.iterrows():
            insert_data.append((
                row['id'],
                row['full_name'],
                None,  # position (not in static API, will update later)
                None,  # height_inches
                None,  # weight_lbs
                None,  # draft_year
                None   # team_abbrev
            ))

        # Insert into database using UPSERT
        cursor = self.conn.cursor()
        insert_query = """
            INSERT INTO player_profile (player_id, full_name, position, height_inches, weight_lbs, draft_year, team_abbrev)
            VALUES %s
            ON CONFLICT (player_id)
            DO UPDATE SET
                full_name = EXCLUDED.full_name
        """

        execute_values(cursor, insert_query, insert_data)
        self.conn.commit()

        logger.info(f"✅ Loaded {len(insert_data)} player profiles")
        return len(insert_data)

    def load_season_stats(self, season: str = "2023-24") -> int:
        """
        Load player season stats

        Args:
            season: NBA season (e.g., "2023-24")

        Returns:
            Number of stat records loaded
        """
        logger.info(f"Loading season stats for {season}...")

        # Fetch season stats from API
        stats_df = self.api.get_player_season_stats(season=season)

        if stats_df.empty:
            logger.warning(f"No season stats found for {season}")
            return 0

        # Get existing player IDs to avoid foreign key violations
        cursor = self.conn.cursor()
        cursor.execute("SELECT player_id FROM player_profile")
        valid_player_ids = set(row[0] for row in cursor.fetchall())
        logger.info(f"Found {len(valid_player_ids)} existing player profiles")

        # Map API columns to database columns
        # Note: Column names vary by API endpoint, adjust as needed
        column_mapping = {
            'PLAYER_ID': 'player_id',
            'GP': 'games_played',
            'PTS': 'ppg',
            'REB': 'rpg',
            'AST': 'apg',
            'MIN': 'mpg',
            'FG_PCT': 'fg_pct',
            'FG3_PCT': 'three_pt_pct',
            'FT_PCT': 'ft_pct',
            'USG_PCT': 'usage_rate',
            'TS_PCT': 'true_shooting_pct',
            # 'PER': 'per'  # PER not in this endpoint
        }

        # Prepare data for insertion
        insert_data = []
        season_year = int(season.split('-')[0])  # "2023-24" -> 2023

        for _, row in stats_df.iterrows():
            try:
                player_id = int(row.get('PLAYER_ID', 0))

                # Skip players not in player_profile table
                if player_id not in valid_player_ids:
                    continue
                # Calculate per-100 possessions (approximate: stat / min * 48 / possession_factor)
                # Simplified: stat / min * (100 / 48.0)
                ppg_per100 = None
                rpg_per100 = None
                apg_per100 = None
                if 'MIN' in row and row['MIN'] > 0:
                    factor = 100.0 / row['MIN']
                    if 'PTS' in row:
                        ppg_per100 = float(row['PTS']) * factor
                    if 'REB' in row:
                        rpg_per100 = float(row['REB']) * factor
                    if 'AST' in row:
                        apg_per100 = float(row['AST']) * factor

                # Calculate True Shooting % if not provided: TS% = PTS / (2 * (FGA + 0.44 * FTA))
                ts_pct = row.get('TS_PCT') if pd.notna(row.get('TS_PCT')) else None
                if ts_pct is None and 'FGA' in row and 'FTA' in row:
                    fga = float(row.get('FGA', 0))
                    fta = float(row.get('FTA', 0))
                    pts = float(row.get('PTS', 0))
                    denominator = 2 * (fga + 0.44 * fta)
                    if denominator > 0:
                        ts_pct = pts / denominator

                # Usage % - try from API first, calculate if not available
                usage_pct = row.get('USG_PCT') if pd.notna(row.get('USG_PCT')) else None
                # If not available, leave as None (requires team stats to calculate properly)

                # PER - simplified approximation (full PER requires team stats)
                per = None
                if 'MIN' in row and row['MIN'] > 0:
                    # Simplified PER = (PTS + REB + AST + STL + BLK - TOV) / MIN * 15
                    per = (float(row.get('PTS', 0)) + float(row.get('REB', 0)) +
                           float(row.get('AST', 0)) + float(row.get('STL', 0)) +
                           float(row.get('BLK', 0)) - float(row.get('TOV', 0))) / float(row['MIN']) * 15

                insert_data.append((
                    player_id,
                    season_year,
                    int(row.get('GP', 0)),
                    float(row.get('MIN', 0.0)),  # minutes_per_game
                    float(row.get('PTS', 0.0)),
                    float(row.get('REB', 0.0)),
                    float(row.get('AST', 0.0)),
                    float(row.get('STL', 0.0)) if 'STL' in row else 0.0,
                    float(row.get('BLK', 0.0)) if 'BLK' in row else 0.0,
                    float(row.get('TOV', 0.0)) if 'TOV' in row else 0.0,
                    float(row.get('FG_PCT', 0.0)) if pd.notna(row.get('FG_PCT')) else None,
                    float(row.get('FG3_PCT', 0.0)) if pd.notna(row.get('FG3_PCT')) else None,
                    float(row.get('FT_PCT', 0.0)) if pd.notna(row.get('FT_PCT')) else None,
                    usage_pct,
                    ts_pct,
                    per,
                    ppg_per100,
                    rpg_per100,
                    apg_per100
                ))
            except Exception as e:
                logger.warning(f"Skipping player {row.get('PLAYER_ID', 'unknown')}: {e}")
                continue

        if not insert_data:
            logger.warning("No valid season stats to insert")
            return 0

        # Insert into database
        cursor = self.conn.cursor()
        insert_query = """
            INSERT INTO player_season_stats
            (player_id, season, games_played, minutes_per_game, ppg, rpg, apg, spg, bpg, tpg,
             fg_pct, three_pt_pct, ft_pct, usage_rate, true_shooting_pct, per,
             ppg_per100, rpg_per100, apg_per100)
            VALUES %s
            ON CONFLICT (player_id, season)
            DO UPDATE SET
                games_played = EXCLUDED.games_played,
                minutes_per_game = EXCLUDED.minutes_per_game,
                ppg = EXCLUDED.ppg,
                rpg = EXCLUDED.rpg,
                apg = EXCLUDED.apg,
                spg = EXCLUDED.spg,
                bpg = EXCLUDED.bpg,
                tpg = EXCLUDED.tpg,
                fg_pct = EXCLUDED.fg_pct,
                three_pt_pct = EXCLUDED.three_pt_pct,
                ft_pct = EXCLUDED.ft_pct,
                usage_rate = EXCLUDED.usage_rate,
                true_shooting_pct = EXCLUDED.true_shooting_pct,
                ppg_per100 = EXCLUDED.ppg_per100,
                rpg_per100 = EXCLUDED.rpg_per100,
                apg_per100 = EXCLUDED.apg_per100
        """

        execute_values(cursor, insert_query, insert_data)
        self.conn.commit()

        logger.info(f"✅ Loaded {len(insert_data)} season stat records")
        return len(insert_data)

    def load_player_game_logs(self, player_id: int, season: str = "2023-24") -> int:
        """
        Load game logs for a specific player

        Args:
            player_id: NBA player ID
            season: NBA season (e.g., "2023-24")

        Returns:
            Number of game logs loaded
        """
        logger.info(f"Loading game logs for player {player_id} ({season})...")

        # Fetch game logs from API
        logs_df = self.api.get_player_game_logs(player_id=player_id, season=season)

        if logs_df.empty:
            logger.warning(f"No game logs found for player {player_id}")
            return 0

        # Prepare data for insertion
        insert_data = []
        season_year = int(season.split('-')[0])

        for _, row in logs_df.iterrows():
            try:
                # Parse matchup (e.g., "LAL vs. BOS" or "LAL @ BOS")
                matchup = row.get('MATCHUP', '')
                is_home = 'vs.' in matchup

                # Get opponent abbreviation
                opponent_abbrev = None
                if '@' in matchup:
                    opponent_abbrev = matchup.split('@')[-1].strip()
                elif 'vs.' in matchup:
                    opponent_abbrev = matchup.split('vs.')[-1].strip()

                insert_data.append((
                    player_id,
                    row.get('Game_ID', ''),
                    pd.to_datetime(row.get('GAME_DATE')),
                    season_year,
                    row.get('TEAM_ABBREVIATION', ''),
                    opponent_abbrev,
                    is_home,
                    int(row.get('MIN', 0)) if pd.notna(row.get('MIN')) and row.get('MIN', '') != '' else 0,
                    int(row.get('PTS', 0)),
                    int(row.get('REB', 0)),
                    int(row.get('AST', 0)),
                    int(row.get('STL', 0)),
                    int(row.get('BLK', 0)),
                    int(row.get('TOV', 0)),
                    int(row.get('FG3M', 0)),
                    int(row.get('FGM', 0)),
                    int(row.get('FGA', 0)),
                    int(row.get('PLUS_MINUS', 0)) if pd.notna(row.get('PLUS_MINUS')) else 0
                ))
            except Exception as e:
                logger.warning(f"Skipping game {row.get('Game_ID', 'unknown')}: {e}")
                continue

        if not insert_data:
            logger.warning("No valid game logs to insert")
            return 0

        # Insert into database
        cursor = self.conn.cursor()
        insert_query = """
            INSERT INTO player_game_logs
            (player_id, game_id, game_date, season, team_abbrev, opponent_abbrev, is_home,
             minutes_played, points, rebounds, assists, steals, blocks, turnovers,
             three_pointers_made, fg_made, fg_attempted, plus_minus)
            VALUES %s
            ON CONFLICT (player_id, game_id)
            DO UPDATE SET
                season = EXCLUDED.season,
                minutes_played = EXCLUDED.minutes_played,
                points = EXCLUDED.points,
                rebounds = EXCLUDED.rebounds,
                assists = EXCLUDED.assists,
                steals = EXCLUDED.steals,
                blocks = EXCLUDED.blocks,
                turnovers = EXCLUDED.turnovers,
                three_pointers_made = EXCLUDED.three_pointers_made,
                fg_made = EXCLUDED.fg_made,
                fg_attempted = EXCLUDED.fg_attempted,
                plus_minus = EXCLUDED.plus_minus
        """

        execute_values(cursor, insert_query, insert_data)
        self.conn.commit()

        logger.info(f"✅ Loaded {len(insert_data)} game logs for player {player_id}")
        return len(insert_data)

    def load_sample_game_logs(self, season: str = "2023-24", sample_size: int = 10) -> int:
        """
        Load game logs for a sample of top players (for testing)

        Args:
            season: NBA season
            sample_size: Number of players to sample

        Returns:
            Total game logs loaded
        """
        logger.info(f"Loading sample game logs ({sample_size} players) for {season}...")

        # Get season stats to find top players
        stats_df = self.api.get_player_season_stats(season=season)

        if stats_df.empty:
            logger.warning("No season stats available to sample from")
            return 0

        # Get top N players by minutes played
        top_players = stats_df.nlargest(sample_size, 'MIN')['PLAYER_ID'].tolist()

        total_logs = 0
        for player_id in top_players:
            try:
                logs_count = self.load_player_game_logs(player_id, season)
                total_logs += logs_count
            except Exception as e:
                logger.warning(f"Failed to load logs for player {player_id}: {e}")
                continue

        logger.info(f"✅ Loaded {total_logs} total game logs for {sample_size} players")
        return total_logs


def main():
    parser = argparse.ArgumentParser(description='Load NBA player data into database')
    parser.add_argument('--season', type=str, nargs='+', default=['2023-24'],
                       help='NBA seasons to load (e.g., 2023-24)')
    parser.add_argument('--player-id', type=int, help='Load specific player by ID')
    parser.add_argument('--sample-size', type=int, default=10,
                       help='Number of players to sample for game logs (default: 10)')
    parser.add_argument('--full', action='store_true',
                       help='Load full game logs for all players (slow)')

    args = parser.parse_args()

    loader = NBAPlayerLoader()

    try:
        loader.connect()

        # Load player profiles
        loader.load_player_profiles(active_only=True)

        # Load season stats for each season
        for season in args.season:
            loader.load_season_stats(season=season)

            # Load game logs
            if args.player_id:
                # Load specific player
                loader.load_player_game_logs(args.player_id, season)
            elif args.full:
                # Load all players (very slow)
                logger.warning("Full load not yet implemented - use --sample-size instead")
            else:
                # Load sample
                loader.load_sample_game_logs(season, args.sample_size)

        logger.info("🎉 Player data loading complete!")

    except Exception as e:
        logger.error(f"❌ Loading failed: {e}")
        raise
    finally:
        loader.close()


if __name__ == "__main__":
    main()
