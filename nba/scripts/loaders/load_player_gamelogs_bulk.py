#!/usr/bin/env python3
"""
Bulk Player Game Logs Loader (Optimized)
==========================================
Loads ALL player game logs efficiently with proper error handling.

Fetches top ~500 players per season (by minutes) and loads their game logs.
Uses transaction rollback to handle foreign key errors gracefully.

Usage:
    python load_player_gamelogs_bulk.py --season 2021-22 2022-23 2023-24 2024-25
"""

import argparse
import logging
import os
import sys
import time

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

# Add utilities to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utilities"))
from nba_api_wrapper import NBAApiWrapper

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Database connection - use centralized config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from nba.config.database import get_players_db_config

DB_CONFIG = get_players_db_config()


def load_game_logs_for_season(season: str, min_minutes: int = 100, limit: int = 600):
    """
    Load game logs for all meaningful players in a season

    Args:
        season: NBA season (e.g., "2021-22")
        min_minutes: Minimum total minutes to include player
        limit: Max players to load per season
    """
    api = NBAApiWrapper(requests_per_minute=20)
    conn = psycopg2.connect(**DB_CONFIG)

    try:
        logger.info(f"=" * 60)
        logger.info(f"Loading player game logs for {season}")
        logger.info(f"=" * 60)

        # Get season stats to find active players
        logger.info("Fetching player season stats...")
        stats_df = api.get_player_season_stats(season=season)

        if stats_df.empty:
            logger.warning(f"No season stats found for {season}")
            return 0

        # Filter players by minimum minutes per game and get top N
        # MIN column is per-game minutes, not total
        if "MIN" in stats_df.columns:
            stats_df["MPG"] = pd.to_numeric(stats_df["MIN"], errors="coerce").fillna(0)
            min_mpg = 1.0  # At least 1 minute per game average
            filtered = stats_df[stats_df["MPG"] >= min_mpg]
            top_players = filtered.nlargest(limit, "MPG")
        else:
            top_players = stats_df.head(limit)

        logger.info(f"Found {len(top_players)} players to load (top {limit} by MPG)")

        # Get existing player IDs from profile table
        cur = conn.cursor()
        cur.execute("SELECT player_id FROM player_profile")
        existing_ids = {row[0] for row in cur.fetchall()}
        logger.info(f"Found {len(existing_ids)} players in profile table")

        # Load game logs for each player
        total_loaded = 0
        skipped = 0
        failed = 0

        for idx, (_, player) in enumerate(top_players.iterrows(), 1):
            player_id = int(player["PLAYER_ID"])
            player_name = player.get("PLAYER_NAME", f"Player {player_id}")

            # Skip if not in profile table
            if player_id not in existing_ids:
                logger.warning(
                    f"[{idx}/{len(top_players)}] Skipping {player_name} (ID={player_id}) - not in profile table"
                )
                skipped += 1
                continue

            try:
                logger.info(f"[{idx}/{len(top_players)}] Loading {player_name} (ID={player_id})...")

                # Fetch game logs from API
                logs_df = api.get_player_game_logs(player_id=player_id, season=season)

                if logs_df.empty:
                    logger.warning(f"  No game logs found")
                    continue

                # Prepare data for insertion
                insert_data = []
                season_year = int(season.split("-")[0])

                for _, row in logs_df.iterrows():
                    try:
                        # Parse matchup (e.g., "LAL vs. BOS" or "LAL @ BOS")
                        matchup = row.get("MATCHUP", "")
                        is_home = "vs." in matchup

                        # Extract team_abbrev and opponent from matchup
                        team_abbrev = None
                        opponent_abbrev = None

                        if "@" in matchup:
                            # Away game: "LAL @ BOS"
                            parts = matchup.split("@")
                            team_abbrev = parts[0].strip()
                            opponent_abbrev = parts[-1].strip()
                        elif "vs." in matchup:
                            # Home game: "LAL vs. BOS"
                            parts = matchup.split("vs.")
                            team_abbrev = parts[0].strip()
                            opponent_abbrev = parts[-1].strip()

                        insert_data.append(
                            (
                                player_id,
                                row.get("Game_ID", ""),
                                pd.to_datetime(row.get("GAME_DATE")),
                                season_year,
                                team_abbrev,  # Extracted from MATCHUP
                                opponent_abbrev,
                                is_home,
                                (
                                    int(row.get("MIN", 0))
                                    if pd.notna(row.get("MIN")) and str(row.get("MIN", "")).strip()
                                    else 0
                                ),
                                int(row.get("PTS", 0)),
                                int(row.get("REB", 0)),
                                int(row.get("AST", 0)),
                                int(row.get("STL", 0)),
                                int(row.get("BLK", 0)),
                                int(row.get("TOV", 0)),
                                int(row.get("FG3M", 0)),
                                int(row.get("FGM", 0)),
                                int(row.get("FGA", 0)),
                                int(row.get("FG3A", 0)),  # ADDED: 3-point attempts
                                int(row.get("FTM", 0)),  # ADDED: Free throws made
                                int(row.get("FTA", 0)),  # ADDED: Free throw attempts
                                (
                                    int(row.get("PLUS_MINUS", 0))
                                    if pd.notna(row.get("PLUS_MINUS"))
                                    else 0
                                ),
                            )
                        )
                    except (psycopg2.Error, KeyError, TypeError, ValueError) as e:
                        logger.warning(f"  Skipping game log row: {e}")
                        continue

                if not insert_data:
                    logger.warning(f"  No valid game logs to insert")
                    continue

                # Insert with transaction (allows rollback on error)
                cur = conn.cursor()

                insert_query = """
                    INSERT INTO player_game_logs
                    (player_id, game_id, game_date, season, team_abbrev, opponent_abbrev,
                     is_home, minutes_played, points, rebounds, assists, steals, blocks,
                     turnovers, three_pointers_made, fg_made, fg_attempted,
                     three_pt_attempted, ft_made, ft_attempted, plus_minus)
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
                        three_pt_attempted = EXCLUDED.three_pt_attempted,
                        ft_made = EXCLUDED.ft_made,
                        ft_attempted = EXCLUDED.ft_attempted,
                        plus_minus = EXCLUDED.plus_minus
                """

                try:
                    execute_values(cur, insert_query, insert_data)
                    conn.commit()
                    logger.info(f"  ✅ Loaded {len(insert_data)} game logs")
                    total_loaded += len(insert_data)
                except (psycopg2.Error, KeyError, TypeError, ValueError) as e:
                    conn.rollback()
                    logger.error(f"  ❌ Insert failed: {e}")
                    failed += 1
                    continue

            except (psycopg2.Error, KeyError, TypeError, ValueError) as e:
                logger.error(f"  ❌ Failed to load player {player_id}: {e}")
                failed += 1
                continue

        logger.info(f"")
        logger.info(f"{'='*60}")
        logger.info(f"Season {season} Summary:")
        logger.info(f"  Total game logs loaded: {total_loaded}")
        logger.info(f"  Players skipped (not in profile): {skipped}")
        logger.info(f"  Players failed: {failed}")
        logger.info(f"{'='*60}")

        return total_loaded

    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Load NBA player game logs (bulk)")
    parser.add_argument(
        "--season",
        type=str,
        nargs="+",
        default=["2023-24"],
        help="NBA seasons to load (e.g., 2021-22 2022-23)",
    )

    args = parser.parse_args()

    start_time = time.time()

    try:
        for season in args.season:
            load_game_logs_for_season(season, min_minutes=100, limit=600)

        elapsed = time.time() - start_time
        logger.info(f"")
        logger.info(f"🎉 All seasons complete! Total time: {elapsed/60:.1f} minutes")

    except (psycopg2.Error, KeyError, TypeError, ValueError) as e:
        logger.error(f"❌ Loading failed: {e}")
        raise


if __name__ == "__main__":
    main()
