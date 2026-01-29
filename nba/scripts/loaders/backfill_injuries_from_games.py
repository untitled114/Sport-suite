#!/usr/bin/env python3
"""
Backfill Historical Injury Data
================================
Infers player injuries from game participation data.

Logic:
- For each game date, check which players were "active" (played recently)
- If an active player's team played but the player didn't, mark as OUT
- "Active" = played at least 1 game in the prior 14 days

This provides approximate injury data for training features.

Usage:
    python3 backfill_injuries_from_games.py --start 2023-10-24 --end 2025-10-23
    python3 backfill_injuries_from_games.py --start 2023-10-24 --end 2025-10-23 --dry-run
"""

import argparse
import logging
import os
from collections import defaultdict
from datetime import datetime, timedelta

import psycopg2

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Database configs
DB_PLAYERS = {
    "host": "localhost",
    "port": 5536,
    "user": os.getenv("DB_USER", "nba_user"),
    "password": os.getenv("DB_PASSWORD"),
    "database": "nba_players",
}

DB_GAMES = {
    "host": "localhost",
    "port": 5537,
    "user": os.getenv("DB_USER", "nba_user"),
    "password": os.getenv("DB_PASSWORD"),
    "database": "nba_games",
}

DB_INTELLIGENCE = {
    "host": "localhost",
    "port": 5539,
    "user": os.getenv("DB_USER", "nba_user"),
    "password": os.getenv("DB_PASSWORD"),
    "database": "nba_intelligence",
}

# Players must have played within this window to be considered "active"
ACTIVE_WINDOW_DAYS = 14
# Minimum games in window to be considered active roster player
MIN_GAMES_TO_BE_ACTIVE = 2


class InjuryBackfiller:
    def __init__(self):
        self.conn_players = None
        self.conn_games = None
        self.conn_intel = None

    def connect(self):
        self.conn_players = psycopg2.connect(**DB_PLAYERS)
        self.conn_games = psycopg2.connect(**DB_GAMES)
        self.conn_intel = psycopg2.connect(**DB_INTELLIGENCE)
        logger.info("Connected to all databases")

    def close(self):
        if self.conn_players:
            self.conn_players.close()
        if self.conn_games:
            self.conn_games.close()
        if self.conn_intel:
            self.conn_intel.close()

    def get_game_dates(self, start_date: str, end_date: str) -> list:
        """Get all unique game dates in the range"""
        cur = self.conn_games.cursor()
        cur.execute(
            """
            SELECT DISTINCT game_date
            FROM games
            WHERE game_date BETWEEN %s AND %s
            ORDER BY game_date
        """,
            (start_date, end_date),
        )
        dates = [row[0] for row in cur.fetchall()]
        cur.close()
        return dates

    def get_teams_playing_on_date(self, game_date) -> set:
        """Get all teams that played on a given date"""
        cur = self.conn_games.cursor()
        cur.execute(
            """
            SELECT home_team, away_team
            FROM games
            WHERE game_date = %s
        """,
            (game_date,),
        )
        teams = set()
        for home, away in cur.fetchall():
            teams.add(home)
            teams.add(away)
        cur.close()
        return teams

    def get_active_players_for_team(
        self, team_abbrev: str, as_of_date, window_days: int = ACTIVE_WINDOW_DAYS
    ) -> dict:
        """
        Get players who were active (played recently) for a team.
        Returns dict of {player_id: player_name}
        """
        start_window = as_of_date - timedelta(days=window_days)

        cur = self.conn_players.cursor()
        cur.execute(
            """
            SELECT pgl.player_id, pp.full_name, COUNT(*) as games_played
            FROM player_game_logs pgl
            JOIN player_profile pp ON pgl.player_id = pp.player_id
            WHERE pgl.team_abbrev = %s
              AND pgl.game_date >= %s
              AND pgl.game_date < %s
              AND pgl.minutes_played > 0
            GROUP BY pgl.player_id, pp.full_name
            HAVING COUNT(*) >= %s
        """,
            (team_abbrev, start_window, as_of_date, MIN_GAMES_TO_BE_ACTIVE),
        )

        active_players = {row[0]: row[1] for row in cur.fetchall()}
        cur.close()
        return active_players

    def get_players_who_played(self, team_abbrev: str, game_date) -> set:
        """Get player_ids who actually played in a game"""
        cur = self.conn_players.cursor()
        cur.execute(
            """
            SELECT DISTINCT player_id
            FROM player_game_logs
            WHERE team_abbrev = %s
              AND game_date = %s
              AND minutes_played > 0
        """,
            (team_abbrev, game_date),
        )
        players = {row[0] for row in cur.fetchall()}
        cur.close()
        return players

    def infer_injuries_for_date(self, game_date) -> list:
        """
        Infer which players were OUT on a given date.
        Returns list of (player_id, player_name, team_abbrev)
        """
        injuries = []
        teams_playing = self.get_teams_playing_on_date(game_date)

        for team in teams_playing:
            # Get active roster (players who played recently)
            active_players = self.get_active_players_for_team(team, game_date)

            if not active_players:
                continue

            # Get who actually played today
            played_today = self.get_players_who_played(team, game_date)

            # Anyone active but didn't play = likely injured/OUT
            for player_id, player_name in active_players.items():
                if player_id not in played_today:
                    injuries.append((player_id, player_name, team))

        return injuries

    def insert_injuries(self, game_date, injuries: list, dry_run: bool = False):
        """Insert inferred injuries into injury_report table"""
        if not injuries:
            return 0

        if dry_run:
            logger.info(f"  [DRY RUN] Would insert {len(injuries)} injuries for {game_date}")
            for _pid, name, team in injuries[:5]:
                logger.info(f"    - {name} ({team})")
            if len(injuries) > 5:
                logger.info(f"    ... and {len(injuries) - 5} more")
            return len(injuries)

        cur = self.conn_intel.cursor()

        # Insert injuries (ON CONFLICT DO NOTHING to avoid duplicates)
        insert_count = 0
        for player_id, player_name, _team in injuries:
            try:
                cur.execute(
                    """
                    INSERT INTO injury_report (player_id, report_date, status, injury_type, confidence)
                    VALUES (%s, %s, 'OUT', 'Inferred from DNP', 0.70)
                    ON CONFLICT DO NOTHING
                """,
                    (player_id, game_date),
                )
                insert_count += cur.rowcount
            except Exception as e:
                logger.debug(f"Error inserting injury for {player_name}: {e}")

        self.conn_intel.commit()
        cur.close()
        return insert_count

    def backfill(self, start_date: str, end_date: str, dry_run: bool = False):
        """Main backfill process"""
        logger.info("=" * 80)
        logger.info("INJURY DATA BACKFILL")
        logger.info("=" * 80)
        logger.info(f"Date range: {start_date} to {end_date}")
        logger.info(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")
        logger.info(f"Active window: {ACTIVE_WINDOW_DAYS} days")
        logger.info(f"Min games to be active: {MIN_GAMES_TO_BE_ACTIVE}")
        logger.info("=" * 80)

        self.connect()

        try:
            game_dates = self.get_game_dates(start_date, end_date)
            logger.info(f"Found {len(game_dates)} game dates to process")

            total_injuries = 0
            dates_processed = 0

            for game_date in game_dates:
                injuries = self.infer_injuries_for_date(game_date)
                inserted = self.insert_injuries(game_date, injuries, dry_run)
                total_injuries += len(injuries)
                dates_processed += 1

                if dates_processed % 50 == 0:
                    logger.info(
                        f"  Processed {dates_processed}/{len(game_dates)} dates, {total_injuries} injuries found"
                    )

            logger.info("=" * 80)
            logger.info(f"COMPLETE: Processed {dates_processed} dates")
            logger.info(f"Total injuries inferred: {total_injuries}")
            logger.info(f"Average per game day: {total_injuries / max(dates_processed, 1):.1f}")
            logger.info("=" * 80)

        finally:
            self.close()


def main():
    parser = argparse.ArgumentParser(description="Backfill historical injury data")
    parser.add_argument("--start", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be inserted without making changes"
    )

    args = parser.parse_args()

    backfiller = InjuryBackfiller()
    backfiller.backfill(args.start, args.end, args.dry_run)


if __name__ == "__main__":
    main()
