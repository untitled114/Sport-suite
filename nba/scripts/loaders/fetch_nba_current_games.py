"""
Fetch Current Season Games from NBA API
Uses LeagueGameFinder + BoxScore endpoints to get 2025-26 season game logs.

This fixes Gap #6: 2025-26 Season Game Logs (using correct NBA API player IDs)

Usage:
    # Fetch all 2025-26 games
    python fetch_nba_current_games.py

    # Fetch and insert into database
    python fetch_nba_current_games.py --insert
"""

import sys
import os
import argparse
import logging
from typing import Dict, List, Set
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import time
from datetime import datetime

# Add utilities to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utilities'))
from nba_api_wrapper import NBAApiWrapper

# Direct imports for specific endpoints
from nba_api.stats.endpoints import LeagueGameFinder, BoxScoreTraditionalV2

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


class NBACurrentGameFetcher:
    """Fetches current season games from NBA API"""

    def __init__(self):
        self.api = NBAApiWrapper(requests_per_minute=15)
        self.conn = None
        self.player_logs = []
        self.games_processed = set()

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

    def fetch_current_season_games(self, season: str = "2025-26") -> pd.DataFrame:
        """
        Fetch all games for current season

        Args:
            season: Season string (e.g., "2025-26")

        Returns:
            DataFrame with game information
        """
        logger.info(f"\n🔍 Fetching games for {season} season...")

        try:
            time.sleep(2)
            games_finder = LeagueGameFinder(
                season_nullable=season,
                season_type_nullable="Regular Season",
                league_id_nullable="00"
            )

            df = games_finder.get_data_frames()[0]

            logger.info(f"✅ Found {len(df)} game records")

            # Get unique games (each game appears twice, once per team)
            unique_games = df['GAME_ID'].nunique()
            logger.info(f"📊 Unique games: {unique_games}")

            return df

        except Exception as e:
            logger.error(f"❌ Failed to fetch games: {e}")
            return pd.DataFrame()

    def fetch_box_score(self, game_id: str) -> Dict:
        """
        Fetch box score for a specific game

        Args:
            game_id: NBA game ID

        Returns:
            Dictionary with player and team stats
        """
        logger.info(f"  Fetching box score for game {game_id}...")

        try:
            time.sleep(2)  # Rate limit
            box_score = BoxScoreTraditionalV2(game_id=game_id)

            dfs = box_score.get_data_frames()

            return {
                'player_stats': dfs[0] if len(dfs) > 0 else pd.DataFrame(),
                'team_stats': dfs[1] if len(dfs) > 1 else pd.DataFrame()
            }

        except Exception as e:
            logger.warning(f"  ⚠️  Failed to fetch box score: {e}")
            return {
                'player_stats': pd.DataFrame(),
                'team_stats': pd.DataFrame()
            }

    def parse_box_score(self, box_score: Dict, game_id: str, game_date: str, season: int = 2025) -> List[Dict]:
        """
        Parse box score into player game logs

        Args:
            box_score: Box score data
            game_id: Game ID
            game_date: Game date (YYYY-MM-DD)
            season: Season year

        Returns:
            List of player game log dictionaries
        """
        player_stats = box_score['player_stats']

        if player_stats.empty:
            logger.warning(f"    ⚠️  No player stats in box score")
            return []

        player_logs = []

        for _, row in player_stats.iterrows():
            try:
                # Skip DNP players
                if pd.isna(row['MIN']) or row['MIN'] is None or row['MIN'] == 0:
                    continue

                # Convert minutes from decimal to integer
                minutes_played = int(float(row['MIN']))

                if minutes_played == 0:
                    continue

                # Parse matchup to determine home/away
                matchup = row.get('MATCHUP', '')
                is_home = 'vs.' in matchup

                # Get opponent
                team_abbrev = row['TEAM_ABBREVIATION']
                if '@' in matchup:
                    opponent_abbrev = matchup.split('@')[-1].strip()
                elif 'vs.' in matchup:
                    opponent_abbrev = matchup.split('vs.')[-1].strip()
                else:
                    opponent_abbrev = None

                player_log = {
                    'player_id': int(row['PLAYER_ID']),
                    'game_id': game_id,
                    'game_date': game_date,
                    'season': season,
                    'team_abbrev': team_abbrev,
                    'opponent_abbrev': opponent_abbrev,
                    'is_home': is_home,
                    'minutes_played': minutes_played,
                    'points': int(row['PTS']) if pd.notna(row['PTS']) else 0,
                    'rebounds': int(row['REB']) if pd.notna(row['REB']) else 0,
                    'assists': int(row['AST']) if pd.notna(row['AST']) else 0,
                    'steals': int(row['STL']) if pd.notna(row['STL']) else 0,
                    'blocks': int(row['BLK']) if pd.notna(row['BLK']) else 0,
                    'turnovers': int(row['TO']) if pd.notna(row['TO']) else 0,
                    'three_pointers_made': int(row['FG3M']) if pd.notna(row['FG3M']) else 0,
                    'fg_made': int(row['FGM']) if pd.notna(row['FGM']) else 0,
                    'fg_attempted': int(row['FGA']) if pd.notna(row['FGA']) else 0,
                    'plus_minus': int(row['PLUS_MINUS']) if pd.notna(row['PLUS_MINUS']) else 0
                }

                player_logs.append(player_log)

            except Exception as e:
                logger.warning(f"      ⚠️  Failed to parse player: {e}")
                continue

        logger.info(f"    ✅ Parsed {len(player_logs)} player logs")
        return player_logs

    def insert_game_logs(self, game_logs: List[Dict]) -> int:
        """
        Insert game logs into database

        Args:
            game_logs: List of game log dictionaries

        Returns:
            Number of records inserted
        """
        if not game_logs:
            logger.warning("No game logs to insert")
            return 0

        cursor = self.conn.cursor()

        # Get valid player IDs
        cursor.execute("SELECT player_id FROM player_profile")
        valid_player_ids = set(row[0] for row in cursor.fetchall())
        logger.info(f"Found {len(valid_player_ids)} valid player IDs in database")

        # Prepare data
        insert_data = []
        skipped_players = set()

        for log in game_logs:
            # Skip if player not in database
            if log['player_id'] not in valid_player_ids:
                skipped_players.add(log['player_id'])
                continue

            insert_data.append((
                log['player_id'],
                log['game_id'],
                log['game_date'],
                log['season'],
                log['team_abbrev'],
                log['opponent_abbrev'],
                log['is_home'],
                log['minutes_played'],
                log['points'],
                log['rebounds'],
                log['assists'],
                log['steals'],
                log['blocks'],
                log['turnovers'],
                log['three_pointers_made'],
                log['fg_made'],
                log['fg_attempted'],
                log['plus_minus']
            ))

        if not insert_data:
            logger.warning("No valid game logs to insert (all players skipped)")
            if skipped_players:
                logger.warning(f"⚠️  Skipped {len(skipped_players)} players not in database")
            return 0

        # Insert with UPSERT
        insert_query = """
            INSERT INTO player_game_logs
            (player_id, game_id, game_date, season, team_abbrev, opponent_abbrev, is_home,
             minutes_played, points, rebounds, assists, steals, blocks, turnovers,
             three_pointers_made, fg_made, fg_attempted, plus_minus)
            VALUES %s
            ON CONFLICT (player_id, game_id)
            DO UPDATE SET
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

        logger.info(f"✅ Inserted {len(insert_data)} game logs into database")

        if skipped_players:
            logger.warning(f"⚠️  Skipped {len(skipped_players)} players not in database")

        return len(insert_data)

    def run(self, season: str = "2025-26", insert: bool = False):
        """
        Main execution flow

        Args:
            season: Season to fetch
            insert: Whether to insert into database
        """
        try:
            # Fetch games
            games_df = self.fetch_current_season_games(season)

            if games_df.empty:
                logger.warning("No games found")
                return

            # Get unique game IDs
            unique_game_ids = games_df['GAME_ID'].unique()
            logger.info(f"\n🎮 Processing {len(unique_game_ids)} unique games...")

            # Process each game
            for i, game_id in enumerate(unique_game_ids, 1):
                # Get game info
                game_row = games_df[games_df['GAME_ID'] == game_id].iloc[0]
                game_date = pd.to_datetime(game_row['GAME_DATE']).strftime('%Y-%m-%d')

                logger.info(f"\n[{i}/{len(unique_game_ids)}] Game {game_id} ({game_date})")

                # Fetch box score
                box_score = self.fetch_box_score(game_id)

                # Parse player logs
                player_logs = self.parse_box_score(box_score, game_id, game_date, season=2025)

                if player_logs:
                    self.player_logs.extend(player_logs)
                    self.games_processed.add(game_id)

            # Summary
            logger.info(f"\n📊 FETCH SUMMARY:")
            logger.info(f"   Games processed: {len(self.games_processed)}")
            logger.info(f"   Player logs extracted: {len(self.player_logs)}")

            # Insert into database
            if insert and self.player_logs:
                logger.info(f"\n💾 Inserting into database...")
                self.insert_game_logs(self.player_logs)

                logger.info(f"\n✅ Successfully fetched and stored current season games!")
            elif not insert:
                logger.info(f"\n✅ Fetch complete (use --insert to save to database)")

        except Exception as e:
            logger.error(f"❌ Fetch failed: {e}")
            if self.conn:
                self.conn.rollback()
            raise


def main():
    parser = argparse.ArgumentParser(description='Fetch current season games from NBA API')
    parser.add_argument('--season', type=str, default='2025-26',
                       help='Season to fetch (default: 2025-26)')
    parser.add_argument('--insert', action='store_true',
                       help='Insert into database (default: False)')

    args = parser.parse_args()

    fetcher = NBACurrentGameFetcher()

    try:
        fetcher.connect()
        fetcher.run(season=args.season, insert=args.insert)

    except Exception as e:
        logger.error(f"❌ Failed: {e}")
        raise
    finally:
        fetcher.close()


if __name__ == "__main__":
    main()
