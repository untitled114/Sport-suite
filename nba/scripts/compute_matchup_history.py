#!/usr/bin/env python3
"""
Matchup History Updater - Computes H2H stats from player_game_logs
===================================================================

Populates the matchup_history table with player vs opponent performance stats.
Used by V3 model's H2H features during live prediction.

Usage:
    # Full rebuild (all players)
    python3 nba/scripts/compute_matchup_history.py

    # Single player
    python3 nba/scripts/compute_matchup_history.py --player "LeBron James"

    # Incremental (only players who played in last N days)
    python3 nba/scripts/compute_matchup_history.py --incremental --days 7

Database: consolidated TimescaleDB (port 5500) - players + intelligence schemas
"""

import argparse
import logging
import math
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import psycopg2
from psycopg2.extras import execute_values

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from nba.config.database import get_intelligence_db_config, get_players_db_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MatchupHistoryUpdater:
    """Compute and update matchup_history from player_game_logs."""

    STAT_TYPES = ["POINTS", "REBOUNDS", "ASSISTS", "THREES"]
    STAT_COL_MAP = {
        "POINTS": "points",
        "REBOUNDS": "rebounds",
        "ASSISTS": "assists",
        "THREES": "three_pointers_made",
    }

    def __init__(self):
        self.conn_players = None
        self.conn_intel = None

    def connect(self):
        self.conn_players = psycopg2.connect(**get_players_db_config())
        self.conn_intel = psycopg2.connect(**get_intelligence_db_config())
        logger.info("Connected to players + intelligence databases")

    def close(self):
        if self.conn_players:
            self.conn_players.close()
        if self.conn_intel:
            self.conn_intel.close()

    def ensure_table(self):
        """Ensure matchup_history table exists with computed_as_of_date column."""
        with self.conn_intel.cursor() as cur:
            # Create table if not exists
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS matchup_history (
                    player_name VARCHAR(255) NOT NULL,
                    opponent_team VARCHAR(10) NOT NULL,
                    stat_type VARCHAR(20) NOT NULL,
                    games_played INT NOT NULL DEFAULT 0,
                    avg_stat DECIMAL(5,2),
                    std_stat DECIMAL(5,2),
                    L3_stat DECIMAL(5,2), L5_stat DECIMAL(5,2),
                    L10_stat DECIMAL(5,2), L20_stat DECIMAL(5,2),
                    home_avg_stat DECIMAL(5,2), away_avg_stat DECIMAL(5,2),
                    last_matchup_date DATE,
                    days_since_last INT,
                    sample_quality DECIMAL(3,2),
                    recency_weight DECIMAL(3,2),
                    computed_as_of_date DATE,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (player_name, opponent_team, stat_type)
                )
            """
            )
            # Add computed_as_of_date if missing (for existing tables)
            cur.execute(
                """
                DO $$
                BEGIN
                    ALTER TABLE matchup_history ADD COLUMN computed_as_of_date DATE;
                EXCEPTION WHEN duplicate_column THEN NULL;
                END $$;
            """
            )
            self.conn_intel.commit()
        logger.info("matchup_history table ready")

    def get_player_list(self, incremental_days=None, player_name=None):
        """Get list of players to process."""
        with self.conn_players.cursor() as cur:
            if player_name:
                cur.execute(
                    "SELECT DISTINCT pp.full_name FROM player_profile pp "
                    "WHERE LOWER(pp.full_name) = LOWER(%s)",
                    (player_name,),
                )
            elif incremental_days:
                cutoff = (date.today() - timedelta(days=incremental_days)).isoformat()
                cur.execute(
                    """SELECT DISTINCT pp.full_name
                    FROM player_game_logs pgl
                    JOIN player_profile pp ON pgl.player_id = pp.player_id
                    WHERE pgl.game_date >= %s AND pgl.minutes_played > 0""",
                    (cutoff,),
                )
            else:
                cur.execute(
                    """SELECT DISTINCT pp.full_name
                    FROM player_game_logs pgl
                    JOIN player_profile pp ON pgl.player_id = pp.player_id
                    WHERE pgl.minutes_played > 0"""
                )
            return [row[0] for row in cur.fetchall()]

    def get_player_games_vs_opponent(self, player_name, opponent_team):
        """Get all games for a player against a specific opponent, chronologically."""
        with self.conn_players.cursor() as cur:
            cur.execute(
                """
                SELECT pgl.game_date, pgl.is_home, pgl.points, pgl.rebounds,
                       pgl.assists, pgl.three_pointers_made, pgl.minutes_played
                FROM player_game_logs pgl
                JOIN player_profile pp ON pgl.player_id = pp.player_id
                WHERE LOWER(pp.full_name) = LOWER(%s)
                  AND pgl.opponent_abbrev = %s
                  AND pgl.minutes_played > 0
                ORDER BY pgl.game_date ASC
                """,
                (player_name, opponent_team),
            )
            cols = ["game_date", "is_home", "points", "rebounds", "assists", "threes", "minutes"]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def get_opponents_for_player(self, player_name):
        """Get all opponents a player has faced."""
        with self.conn_players.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT pgl.opponent_abbrev
                FROM player_game_logs pgl
                JOIN player_profile pp ON pgl.player_id = pp.player_id
                WHERE LOWER(pp.full_name) = LOWER(%s)
                  AND pgl.minutes_played > 0
                """,
                (player_name,),
            )
            return [row[0] for row in cur.fetchall()]

    def compute_matchup_row(self, games, player_name, opponent_team, stat_type):
        """Compute a single matchup_history row from game data.

        Only computes stats for the primary stat type — cross-stat H2H
        features had zero model importance and were removed.
        """
        if not games:
            return None

        stat_col = self.STAT_COL_MAP[stat_type].replace("three_pointers_made", "threes")
        n = len(games)
        today = date.today()

        row = {"player_name": player_name, "opponent_team": opponent_team, "stat_type": stat_type}
        row["games_played"] = n

        # Compute only the primary stat
        all_vals = [float(g[stat_col] or 0) for g in games]
        row["avg_stat"] = float(round(np.mean(all_vals), 2)) if all_vals else 0.0
        row["std_stat"] = float(round(np.std(all_vals), 2)) if len(all_vals) >= 2 else 0.0

        # Rolling windows (most recent N games)
        recent = list(reversed(all_vals))  # Most recent first
        for window, label in [(3, "L3"), (5, "L5"), (10, "L10"), (20, "L20")]:
            subset = recent[:window]
            row[f"{label}_stat"] = float(round(np.mean(subset), 2)) if subset else 0.0

        # Home/away averages
        home_vals = [float(g[stat_col] or 0) for g in games if g["is_home"]]
        away_vals = [float(g[stat_col] or 0) for g in games if not g["is_home"]]
        row["home_avg_stat"] = float(round(np.mean(home_vals), 2)) if home_vals else 0.0
        row["away_avg_stat"] = float(round(np.mean(away_vals), 2)) if away_vals else 0.0

        # Quality metrics
        last_game = games[-1]  # Most recent (sorted ASC)
        last_date = last_game["game_date"]
        if isinstance(last_date, str):
            last_date = datetime.strptime(last_date, "%Y-%m-%d").date()
        row["last_matchup_date"] = last_date
        row["days_since_last"] = (today - last_date).days

        # Sample quality: Wilson score lower bound (confidence in the estimate)
        row["sample_quality"] = round(min(1.0, n / 20.0), 2)

        # Recency weight: exponential decay (tau=90 days)
        row["recency_weight"] = round(math.exp(-row["days_since_last"] / 90.0), 2)

        row["computed_as_of_date"] = today

        return row

    def upsert_rows(self, rows):
        """Batch upsert rows into matchup_history."""
        if not rows:
            return 0

        columns = [
            "player_name",
            "opponent_team",
            "stat_type",
            "games_played",
            "avg_stat",
            "std_stat",
            "L3_stat",
            "L5_stat",
            "L10_stat",
            "L20_stat",
            "home_avg_stat",
            "away_avg_stat",
            "last_matchup_date",
            "days_since_last",
            "sample_quality",
            "recency_weight",
            "computed_as_of_date",
        ]

        values = [[row.get(c) for c in columns] for row in rows]

        update_cols = [c for c in columns if c not in ("player_name", "opponent_team", "stat_type")]
        update_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in update_cols)
        update_clause += ", updated_at = CURRENT_TIMESTAMP"

        sql = f"""
            INSERT INTO matchup_history ({', '.join(columns)})
            VALUES %s
            ON CONFLICT (player_name, opponent_team, stat_type)
            DO UPDATE SET {update_clause}
        """

        with self.conn_intel.cursor() as cur:
            execute_values(cur, sql, values, page_size=500)
        self.conn_intel.commit()
        return len(values)

    def run(self, incremental_days=None, player_name=None):
        """Main execution."""
        self.connect()
        self.ensure_table()

        players = self.get_player_list(incremental_days, player_name)
        logger.info(f"Processing {len(players)} players")

        total_rows = 0
        batch = []
        batch_size = 1000

        for i, pname in enumerate(players, 1):
            opponents = self.get_opponents_for_player(pname)

            for opp in opponents:
                games = self.get_player_games_vs_opponent(pname, opp)
                if not games:
                    continue

                # Create one row per stat_type
                for stat_type in self.STAT_TYPES:
                    row = self.compute_matchup_row(games, pname, opp, stat_type)
                    if row:
                        batch.append(row)

            # Flush batch periodically
            if len(batch) >= batch_size:
                total_rows += self.upsert_rows(batch)
                batch = []

            if i % 50 == 0:
                logger.info(f"  [{i}/{len(players)}] {total_rows} rows upserted so far")

        # Final flush
        if batch:
            total_rows += self.upsert_rows(batch)

        logger.info(f"Done: {total_rows} matchup_history rows upserted for {len(players)} players")
        self.close()


def main():
    parser = argparse.ArgumentParser(description="Compute matchup_history from player_game_logs")
    parser.add_argument("--player", type=str, help="Single player name")
    parser.add_argument("--incremental", action="store_true", help="Only process recent players")
    parser.add_argument(
        "--days", type=int, default=7, help="Lookback days for incremental (default: 7)"
    )
    args = parser.parse_args()

    updater = MatchupHistoryUpdater()
    updater.run(
        incremental_days=args.days if args.incremental else None,
        player_name=args.player,
    )


if __name__ == "__main__":
    main()
