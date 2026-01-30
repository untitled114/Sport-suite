#!/usr/bin/env python3
"""
NBA Pick Results Tracker
=========================
Populates actual results for completed picks.

This script:
1. Finds picks with result='PENDING' for completed games
2. Queries player_game_logs for actual stat values
3. Updates pick_results with actual_result, result, profit
4. Calculates win/loss based on side (OVER/UNDER) and line

Usage:
    python3 track_results.py
    python3 track_results.py --date 2025-11-07
    python3 track_results.py --backfill-days 7
"""

import argparse
import logging
import os
from datetime import datetime, timedelta

import pandas as pd
import psycopg2

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

try:
    from nba.config import STAT_COLUMN_MAP, get_intelligence_db_config, get_players_db_config

    DB_INTELLIGENCE = get_intelligence_db_config()
    DB_PLAYERS = get_players_db_config()
except ImportError:
    # Fallback for standalone execution
    DB_INTELLIGENCE = {
        "host": "localhost",
        "port": 5539,
        "user": os.getenv("DB_USER", "nba_user"),
        "password": os.getenv("DB_PASSWORD"),
        "database": "nba_intelligence",
    }
    DB_PLAYERS = {
        "host": "localhost",
        "port": 5536,
        "user": os.getenv("DB_USER", "nba_user"),
        "password": os.getenv("DB_PASSWORD"),
        "database": "nba_players",
    }
    # Note: Use correct column name 'three_pointers_made' (not 'threes_made')
    STAT_COLUMN_MAP = {
        "POINTS": "points",
        "REBOUNDS": "rebounds",
        "ASSISTS": "assists",
        "THREES": "three_pointers_made",
    }


class ResultsTracker:
    """
    Tracks results for NBA betting picks.
    Updates pick_results table with actual outcomes.
    """

    def __init__(self):
        self.conn_intelligence = None
        self.conn_players = None

    def connect(self):
        """Connect to databases"""
        self.conn_intelligence = psycopg2.connect(**DB_INTELLIGENCE)
        self.conn_players = psycopg2.connect(**DB_PLAYERS)
        logger.info("[OK] Connected to databases")

    def get_pending_picks(self, game_date=None):
        """
        Get picks with PENDING results for completed games.

        Args:
            game_date: Optional specific date to check (YYYY-MM-DD)

        Returns:
            DataFrame with pending picks
        """
        if game_date:
            date_filter = "AND game_date = %s"
            params = (game_date,)
        else:
            # Get picks from last 7 days with completed games
            date_filter = """
                AND game_date >= CURRENT_DATE - INTERVAL '7 days'
                AND game_date < CURRENT_DATE
            """
            params = ()

        query = f"""
            SELECT
                pick_id,
                player_name,
                game_date,
                stat_type,
                side,
                best_line,
                prediction,
                edge,
                confidence
            FROM pick_results
            WHERE (result IS NULL OR result = 'PENDING')
              {date_filter}
            ORDER BY game_date DESC;
        """

        df = pd.read_sql_query(query, self.conn_intelligence, params=params)
        return df

    def get_actual_stat(self, player_name, game_date, stat_type):
        """
        Query actual stat value from player_game_logs.

        Args:
            player_name: Player's full name
            game_date: Game date (YYYY-MM-DD)
            stat_type: 'POINTS', 'REBOUNDS', 'ASSISTS', 'THREES'

        Returns:
            Actual stat value (float) or None if not found
        """
        stat_column = STAT_COLUMN_MAP.get(stat_type)

        if not stat_column:
            logger.error(f"Unknown stat type: {stat_type}")
            return None

        query = f"""
            SELECT {stat_column}
            FROM player_game_logs
            WHERE player_name = %s
              AND game_date = %s
            LIMIT 1;
        """

        try:
            result = pd.read_sql_query(query, self.conn_players, params=(player_name, game_date))

            if len(result) > 0:
                return float(result.iloc[0][stat_column])
            else:
                return None

        except (psycopg2.Error, KeyError, TypeError, ValueError) as e:
            logger.error(f"Error querying stat: {e}")
            return None

    def calculate_result(self, actual_value, line, side):
        """
        Calculate bet result (WIN/LOSS/PUSH) and profit.

        Args:
            actual_value: Actual stat value
            line: Betting line
            side: 'OVER' or 'UNDER'

        Returns:
            (result_str, profit_float)
        """
        if actual_value is None:
            return ("PENDING", 0.0)

        # Check result
        if side == "OVER":
            if actual_value > line:
                return ("WIN", 1.0)
            elif actual_value < line:
                return ("LOSS", -1.1)  # Standard -110 odds
            else:
                return ("PUSH", 0.0)
        elif side == "UNDER":
            if actual_value < line:
                return ("WIN", 1.0)
            elif actual_value > line:
                return ("LOSS", -1.1)
            else:
                return ("PUSH", 0.0)
        else:
            logger.error(f"Unknown side: {side}")
            return ("PENDING", 0.0)

    def update_pick_result(self, pick_id, actual_result, result, profit):
        """
        Update pick_results table with actual outcome.

        Args:
            pick_id: Pick ID to update
            actual_result: Actual stat value
            result: 'WIN', 'LOSS', 'PUSH', or 'PENDING'
            profit: Profit/loss amount
        """
        cursor = self.conn_intelligence.cursor()

        update_query = """
            UPDATE pick_results
            SET actual_result = %s,
                result = %s,
                profit = %s,
                result_updated_at = CURRENT_TIMESTAMP
            WHERE pick_id = %s;
        """

        cursor.execute(update_query, (actual_result, result, profit, pick_id))
        self.conn_intelligence.commit()

    def process_pending_picks(self, game_date=None):
        """
        Main processing logic - update all pending picks.

        Args:
            game_date: Optional specific date to process
        """
        logger.info("=" * 80)
        logger.info("PROCESSING PENDING PICKS")
        logger.info("=" * 80)

        # Get pending picks
        pending_picks = self.get_pending_picks(game_date=game_date)

        if len(pending_picks) == 0:
            logger.info("[OK] No pending picks to process")
            return

        logger.info(f"Found {len(pending_picks)} pending picks to process\n")

        # Process each pick
        wins = 0
        losses = 0
        pushes = 0
        still_pending = 0

        for _idx, pick in pending_picks.iterrows():
            player_name = pick["player_name"]
            game_date = pick["game_date"]
            stat_type = pick["stat_type"]
            side = pick["side"]
            line = pick["best_line"]

            # Get actual stat
            actual_value = self.get_actual_stat(player_name, game_date, stat_type)

            if actual_value is None:
                logger.debug(f"⏳ No result yet: {player_name} {stat_type} ({game_date})")
                still_pending += 1
                continue

            # Calculate result
            result, profit = self.calculate_result(actual_value, line, side)

            # Update database
            self.update_pick_result(pick["pick_id"], actual_value, result, profit)

            # Log result
            result_symbol = "[OK]" if result == "WIN" else "[ERROR]" if result == "LOSS" else "[-]"
            logger.info(
                f"{result_symbol} {result}: {player_name} {stat_type} {side} {line} "
                f"(actual: {actual_value}, edge: {pick['edge']:.1f}, "
                f"conf: {pick['confidence']})"
            )

            # Count results
            if result == "WIN":
                wins += 1
            elif result == "LOSS":
                losses += 1
            elif result == "PUSH":
                pushes += 1

        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("RESULTS SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total Processed: {wins + losses + pushes}")
        logger.info(f"  Wins: {wins}")
        logger.info(f"  Losses: {losses}")
        logger.info(f"  Pushes: {pushes}")
        logger.info(f"Still Pending: {still_pending}")

        if wins + losses > 0:
            win_rate = wins / (wins + losses) * 100
            total_profit = (wins * 1.0) + (losses * -1.1)
            roi = total_profit / (wins + losses) * 100

            logger.info(f"\nWin Rate: {win_rate:.1f}%")
            logger.info(f"ROI: {roi:+.2f}%")
        logger.info("=" * 80)

    def run(self, game_date=None):
        """Main execution"""
        try:
            self.connect()
            self.process_pending_picks(game_date=game_date)
        finally:
            if self.conn_intelligence:
                self.conn_intelligence.close()
            if self.conn_players:
                self.conn_players.close()


def main():
    parser = argparse.ArgumentParser(description="Track results for NBA betting picks")
    parser.add_argument("--date", help="Specific game date to process (YYYY-MM-DD)", default=None)
    parser.add_argument(
        "--backfill-days", type=int, help="Process last N days of picks", default=None
    )

    args = parser.parse_args()

    # If backfill requested, process each day
    if args.backfill_days:
        logger.info(f"Backfilling last {args.backfill_days} days...")
        tracker = ResultsTracker()
        tracker.connect()

        for days_ago in range(args.backfill_days):
            date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
            logger.info(f"\nProcessing {date}...")
            tracker.process_pending_picks(game_date=date)

        tracker.conn_intelligence.close()
        tracker.conn_players.close()
    else:
        # Process single date or default (last 7 days)
        tracker = ResultsTracker()
        tracker.run(game_date=args.date)


if __name__ == "__main__":
    main()
