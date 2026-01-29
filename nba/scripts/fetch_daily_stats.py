#!/usr/bin/env python3
"""
Daily NBA Stats Updater
========================
Fetches the latest NBA player stats and updates the database.
Fetches player game logs and updates rolling stats.

Usage:
    # Fetch yesterday's games (default)
    python3 nba/scripts/fetch_daily_stats.py

    # Fetch specific date
    python3 nba/scripts/fetch_daily_stats.py --date 2025-10-29

    # Fetch last N days
    python3 nba/scripts/fetch_daily_stats.py --days 3
"""

import argparse
import requests
import psycopg2
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
import time
import json
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.team_utils import normalize_team_abbrev

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Database configs
DB_PLAYERS = {
    'host': 'localhost',
    'port': 5536,
    'database': 'nba_players',
    'user': os.getenv('DB_USER', 'nba_user'),
    'password': os.getenv('DB_PASSWORD')
}

DB_GAMES = {
    'host': 'localhost',
    'port': 5537,
    'database': 'nba_games',
    'user': os.getenv('DB_USER', 'nba_user'),
    'password': os.getenv('DB_PASSWORD')
}

# ESPN NBA API
ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
ESPN_BOXSCORE_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"


class NBAStatsUpdater:
    """Fetch and update daily NBA stats"""

    def __init__(self):
        self.conn_players = None
        self.conn_games = None
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
        })

    def connect(self):
        """Connect to all databases"""
        self.conn_players = psycopg2.connect(**DB_PLAYERS)
        self.conn_games = psycopg2.connect(**DB_GAMES)
        logger.info("✅ Connected to all NBA databases")

    def close(self):
        """Close all connections"""
        if self.conn_players:
            self.conn_players.close()
        if self.conn_games:
            self.conn_games.close()

    def get_games_for_date(self, date_str):
        """Fetch games for a specific date from ESPN API"""
        logger.info(f"\n{'='*100}")
        logger.info(f"FETCHING NBA GAMES FOR {date_str}")
        logger.info(f"{'='*100}")

        # Convert to ESPN date format (YYYYMMDD)
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        espn_date = date_obj.strftime('%Y%m%d')

        params = {
            'dates': espn_date,
            'limit': 50
        }

        try:
            response = self.session.get(ESPN_SCOREBOARD_URL, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()

            events = data.get('events', [])
            logger.info(f"✅ Found {len(events)} games")

            return events

        except Exception as e:
            logger.error(f"❌ Error fetching games: {e}")
            return []

    def get_boxscore(self, game_id):
        """Fetch detailed boxscore for a game"""
        params = {'event': game_id}

        try:
            response = self.session.get(ESPN_BOXSCORE_URL, params=params, timeout=15)
            response.raise_for_status()
            return response.json()

        except Exception as e:
            logger.error(f"❌ Error fetching boxscore for game {game_id}: {e}")
            return None

    def map_player_to_db_id(self, espn_id, player_name):
        """Map ESPN player ID to database player_id"""
        cursor = self.conn_players.cursor()

        # Try name matching
        cursor.execute("""
            SELECT player_id FROM player_profile
            WHERE LOWER(full_name) = LOWER(%s)
            LIMIT 1
        """, (player_name,))

        result = cursor.fetchone()
        cursor.close()

        if result:
            return result[0]

        logger.warning(f"⚠️  No DB mapping for {player_name} (ESPN ID: {espn_id})")
        return None

    def load_player_game(self, player_stats, game_info, labels):
        """Load player game into player_game_logs table"""
        espn_id = player_stats.get('athlete', {}).get('id')
        player_name = player_stats.get('athlete', {}).get('displayName', '')

        player_id = self.map_player_to_db_id(espn_id, player_name)
        if not player_id:
            return False

        cursor = self.conn_players.cursor()

        # Extract stats - ESPN uses array format matching labels
        stats_array = player_stats.get('stats', [])
        stats_dict = dict(zip(labels, stats_array)) if labels and stats_array else {}

        # Parse stats (skip season totals or invalid data)
        try:
            min_str = str(stats_dict.get('MIN', '0'))

            # Skip incomplete stats
            if '--' in min_str or min_str == '' or min_str == 'None':
                return False

            minutes = float(min_str.split(':')[0] if ':' in min_str else min_str)

            # Skip DNPs and season totals
            if minutes == 0 or minutes > 60:
                return False

            points = int(stats_dict.get('PTS', 0) or 0)
            rebounds = int(stats_dict.get('REB', 0) or 0)
            assists = int(stats_dict.get('AST', 0) or 0)
            threes = int(stats_dict.get('3PM', 0) or 0)
            fgm = int(stats_dict.get('FGM', 0) or 0)
            fga = int(stats_dict.get('FGA', 0) or 0)

        except (ValueError, TypeError):
            return False

        # Calculate season: NBA season runs Oct-Jun, so Oct-Dec = next year's season
        game_year = int(game_info['game_date'][:4])
        game_month = int(game_info['game_date'][5:7])
        season = game_year + 1 if game_month >= 10 else game_year

        game_data = {
            'player_id': player_id,
            'game_id': f"{game_info['game_date']}_{game_info['team']}",  # Simple game ID
            'game_date': game_info['game_date'],
            'season': season,  # 2025-26 season = 2026
            'team_abbrev': game_info['team'],
            'opponent_abbrev': game_info['opponent'],
            'is_home': game_info['is_home'],
            'minutes_played': int(minutes),
            'points': points,
            'rebounds': rebounds,
            'assists': assists,
            'three_pointers_made': threes,
            'fg_made': fgm,
            'fg_attempted': fga
        }

        # Check if exists (use composite key: player_id + game_date)
        cursor.execute("""
            SELECT player_id FROM player_game_logs
            WHERE player_id = %s AND game_date = %s
        """, (player_id, game_data['game_date']))

        existing = cursor.fetchone()

        if existing:
            # Update existing
            cursor.execute("""
                UPDATE player_game_logs
                SET game_id = %s, season = %s, team_abbrev = %s, opponent_abbrev = %s, is_home = %s,
                    minutes_played = %s, points = %s, rebounds = %s, assists = %s,
                    three_pointers_made = %s, fg_made = %s, fg_attempted = %s
                WHERE player_id = %s AND game_date = %s
            """, (
                game_data['game_id'], game_data['season'],
                game_data['team_abbrev'], game_data['opponent_abbrev'], game_data['is_home'],
                game_data['minutes_played'], game_data['points'], game_data['rebounds'],
                game_data['assists'], game_data['three_pointers_made'], game_data['fg_made'],
                game_data['fg_attempted'], player_id, game_data['game_date']
            ))
            action = "updated"
        else:
            # Insert new
            cursor.execute("""
                INSERT INTO player_game_logs (
                    player_id, game_id, game_date, season, team_abbrev, opponent_abbrev, is_home,
                    minutes_played, points, rebounds, assists, three_pointers_made, fg_made, fg_attempted
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, tuple(game_data.values()))
            action = "inserted"

        self.conn_players.commit()
        cursor.close()

        logger.info(f"  ✅ {player_name}: {action} ({points}p {rebounds}r {assists}a)")
        return True

    def process_game(self, event, fetch_date):
        """Process a single game from ESPN data"""
        game_id = event.get('id')

        # Get team info
        competitions = event.get('competitions', [])
        if not competitions:
            return

        comp = competitions[0]

        # Only process completed games (STATUS_FINAL)
        status = comp.get('status', {}).get('type', {}).get('name', '')
        if status != 'STATUS_FINAL':
            return

        competitors = comp.get('competitors', [])

        if len(competitors) != 2:
            return

        home_team = next((c for c in competitors if c.get('homeAway') == 'home'), None)
        away_team = next((c for c in competitors if c.get('homeAway') == 'away'), None)

        if not home_team or not away_team:
            return

        # Use fetch_date instead of ESPN's unreliable date field
        game_date = fetch_date

        # Normalize team abbreviations
        home_abbrev = normalize_team_abbrev(home_team['team']['abbreviation'])
        away_abbrev = normalize_team_abbrev(away_team['team']['abbreviation'])

        logger.info(f"\n📊 Processing: {away_abbrev} @ {home_abbrev} ({game_date})")

        # Fetch boxscore
        boxscore = self.get_boxscore(game_id)
        if not boxscore:
            return

        time.sleep(0.5)  # Rate limiting

        # Process box score data
        boxscore_data = boxscore.get('boxscore', {})
        players = boxscore_data.get('players', [])

        player_count = 0

        for team_data in players:
            team_abbrev_raw = team_data.get('team', {}).get('abbreviation', '')
            team_abbrev = normalize_team_abbrev(team_abbrev_raw)

            is_home = team_abbrev == home_abbrev
            opponent = away_abbrev if is_home else home_abbrev

            game_info = {
                'game_date': game_date,
                'team': team_abbrev,
                'opponent': opponent,
                'is_home': is_home
            }

            # Process players
            for stat_group in team_data.get('statistics', []):
                labels = stat_group.get('labels', [])

                for player in stat_group.get('athletes', []):
                    if self.load_player_game(player, game_info, labels):
                        player_count += 1

        logger.info(f"  📈 Loaded: {player_count} players")

    def run(self, start_date, days=1):
        """Main execution"""
        logger.info(f"\n{'='*100}")
        logger.info("NBA DAILY STATS UPDATER")
        logger.info(f"{'='*100}")
        logger.info(f"Fetching stats from {start_date} ({days} day(s))")

        self.connect()

        total_games = 0

        for i in range(days):
            date_obj = datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=i)
            date_str = date_obj.strftime('%Y-%m-%d')

            events = self.get_games_for_date(date_str)

            for event in events:
                self.process_game(event, date_str)
                total_games += 1

        logger.info(f"\n{'='*100}")
        logger.info(f"✅ COMPLETE - Processed {total_games} games")
        logger.info(f"{'='*100}")

        self.close()


def main():
    parser = argparse.ArgumentParser(description='Fetch latest NBA player stats')
    parser.add_argument('--date', type=str, help='Specific date (YYYY-MM-DD), defaults to yesterday')
    parser.add_argument('--days', type=int, default=1, help='Number of days to fetch (default: 1)')

    args = parser.parse_args()

    # Default to yesterday (games completed)
    if args.date:
        start_date = args.date
    else:
        yesterday = datetime.now() - timedelta(days=1)
        start_date = yesterday.strftime('%Y-%m-%d')

    updater = NBAStatsUpdater()
    updater.run(start_date, args.days)


if __name__ == '__main__':
    main()
