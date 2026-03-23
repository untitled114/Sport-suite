#!/usr/bin/env python3
"""
Compute Team Rolling Stats (Point-in-Time)
==========================================
Populates teams.team_rolling_stats with 10-game rolling averages
computed strictly from games BEFORE each game date (no target leakage).

Data sources:
- games.team_game_logs: points scored per team per game
- Self-join on game_id + opponent: points allowed

Pace/ratings note:
- team_game_logs.pace and .offensive_rating contain incorrect values
- We approximate pace from total points in a game:
    estimated_pace = (team_pts + opp_pts) / 2 * (48/48) ≈ avg game score
    NBA average: ~112 pts per team → pace ~100 possessions
    We use: pace ≈ (team_pts + opp_pts) / 2.24 (league avg ~112 pts / 100 poss)
- Offensive rating: team_pts / estimated_possessions * 100
- Defensive rating: opp_pts / estimated_possessions * 100

Usage:
    # Full backfill (all data from 2023-10-01 onward)
    python compute_team_rolling_stats.py

    # Incremental (last N days for daily pipeline)
    python compute_team_rolling_stats.py --incremental --days 7

    # Custom start date
    python compute_team_rolling_stats.py --start-date 2025-10-01
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import psycopg2
from psycopg2.extras import execute_values

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from nba.config.database import get_games_db_config, get_team_db_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

EST = ZoneInfo("America/New_York")

# League-average points per possession (approx 1.12 pts/poss in modern NBA)
# Used to estimate possessions: estimated_poss = (team_pts + opp_pts) / 2 / LEAGUE_PTS_PER_POSS
LEAGUE_PTS_PER_POSS = 1.12

WINDOW_SIZE = 10


def get_team_game_history(games_conn):
    """
    Fetch all completed team games with points scored and points allowed.

    Returns list of dicts sorted by (team_abbrev, game_date).
    """
    query = """
    SELECT
        a.team_abbrev,
        a.game_date,
        a.game_id,
        a.points AS team_pts,
        b.points AS opp_pts,
        a.is_home
    FROM team_game_logs a
    JOIN team_game_logs b
        ON a.game_id = b.game_id
        AND a.team_abbrev = b.opponent
    WHERE a.points IS NOT NULL
      AND b.points IS NOT NULL
      AND a.team_abbrev != a.opponent
    ORDER BY a.team_abbrev, a.game_date, a.game_id
    """
    with games_conn.cursor() as cur:
        cur.execute(query)
        columns = [d[0] for d in cur.description]
        rows = cur.fetchall()

    return [dict(zip(columns, row)) for row in rows]


def compute_rolling_stats(game_history, start_date=None):
    """
    Compute point-in-time rolling stats for each team-game.

    For each game a team plays, we look at the PREVIOUS `WINDOW_SIZE` games
    (strictly before that game_date) and compute averages.

    Args:
        game_history: List of dicts from get_team_game_history, sorted by team+date.
        start_date: Only emit rows for games on or after this date (None = all).

    Returns:
        List of tuples ready for INSERT into team_rolling_stats.
    """
    # Group games by team
    team_games = {}
    for g in game_history:
        team = g["team_abbrev"]
        if team not in team_games:
            team_games[team] = []
        team_games[team].append(g)

    results = []
    total_teams = len(team_games)

    for team_idx, (team, games) in enumerate(sorted(team_games.items())):
        # Games are already sorted by date
        for i, game in enumerate(games):
            game_date = game["game_date"]

            # Skip if before start_date
            if start_date and game_date < start_date:
                continue

            # Get the previous WINDOW_SIZE games (strictly before this game)
            # Since games are sorted by (team, date, game_id), the previous
            # entries in the list are the prior games
            window_games = games[max(0, i - WINDOW_SIZE) : i]

            if len(window_games) < 3:
                # Need at least 3 games for meaningful stats
                continue

            # Compute rolling averages
            n = len(window_games)
            total_pts = sum(g["team_pts"] for g in window_games)
            total_pts_allowed = sum(g["opp_pts"] for g in window_games)
            wins = sum(1 for g in window_games if g["team_pts"] > g["opp_pts"])
            losses = n - wins

            avg_points = total_pts / n
            avg_points_allowed = total_pts_allowed / n

            # Estimate pace per game: total points / 2 / league_pts_per_poss
            # This gives estimated possessions, which is a proxy for pace
            estimated_paces = []
            off_ratings = []
            def_ratings = []
            for g in window_games:
                total_game_pts = g["team_pts"] + g["opp_pts"]
                est_poss = total_game_pts / 2.0 / LEAGUE_PTS_PER_POSS
                if est_poss > 0:
                    estimated_paces.append(est_poss)
                    off_ratings.append(g["team_pts"] / est_poss * 100.0)
                    def_ratings.append(g["opp_pts"] / est_poss * 100.0)

            avg_pace = sum(estimated_paces) / len(estimated_paces) if estimated_paces else None
            avg_off_rating = sum(off_ratings) / len(off_ratings) if off_ratings else None
            avg_def_rating = sum(def_ratings) / len(def_ratings) if def_ratings else None
            net_rating = (
                (avg_off_rating - avg_def_rating)
                if avg_off_rating is not None and avg_def_rating is not None
                else None
            )
            win_pct = wins / n if n > 0 else None

            results.append(
                (
                    team,  # team_abbrev
                    game_date,  # as_of_date
                    WINDOW_SIZE,  # window_size
                    n,  # games_in_window
                    round(avg_points, 2),  # avg_points
                    round(avg_points_allowed, 2),  # avg_points_allowed
                    round(avg_pace, 2) if avg_pace is not None else None,  # avg_pace
                    round(avg_off_rating, 2) if avg_off_rating is not None else None,
                    round(avg_def_rating, 2) if avg_def_rating is not None else None,
                    round(net_rating, 2) if net_rating is not None else None,
                    wins,  # wins_in_window
                    losses,  # losses_in_window
                    round(win_pct, 4) if win_pct is not None else None,  # win_pct
                )
            )

        if (team_idx + 1) % 10 == 0:
            logger.info(f"  Processed {team_idx + 1}/{total_teams} teams...")

    # Deduplicate: if a team has two games on the same date (rare),
    # keep the later entry (which includes the earlier same-day game in its window)
    deduped = {}
    for row in results:
        key = (row[0], row[1], row[2])  # (team_abbrev, as_of_date, window_size)
        deduped[key] = row  # last one wins (later game in the day)

    if len(deduped) < len(results):
        logger.info(
            f"  Deduplicated {len(results)} -> {len(deduped)} rows "
            f"({len(results) - len(deduped)} same-date duplicates)"
        )

    return list(deduped.values())


def upsert_rolling_stats(team_conn, rows):
    """
    Insert rolling stats with upsert on (team_abbrev, as_of_date, window_size).
    """
    if not rows:
        logger.warning("No rows to insert.")
        return 0

    insert_sql = """
    INSERT INTO team_rolling_stats (
        team_abbrev, as_of_date, window_size, games_in_window,
        avg_points, avg_points_allowed, avg_pace,
        avg_offensive_rating, avg_defensive_rating, net_rating,
        wins_in_window, losses_in_window, win_pct
    ) VALUES %s
    ON CONFLICT (team_abbrev, as_of_date, window_size) DO UPDATE SET
        games_in_window = EXCLUDED.games_in_window,
        avg_points = EXCLUDED.avg_points,
        avg_points_allowed = EXCLUDED.avg_points_allowed,
        avg_pace = EXCLUDED.avg_pace,
        avg_offensive_rating = EXCLUDED.avg_offensive_rating,
        avg_defensive_rating = EXCLUDED.avg_defensive_rating,
        net_rating = EXCLUDED.net_rating,
        wins_in_window = EXCLUDED.wins_in_window,
        losses_in_window = EXCLUDED.losses_in_window,
        win_pct = EXCLUDED.win_pct
    """

    # Need autocommit off for batch insert
    old_autocommit = team_conn.autocommit
    team_conn.autocommit = False

    try:
        with team_conn.cursor() as cur:
            # Insert in batches of 1000
            batch_size = 1000
            total_inserted = 0
            for i in range(0, len(rows), batch_size):
                batch = rows[i : i + batch_size]
                execute_values(cur, insert_sql, batch)
                total_inserted += len(batch)
                if total_inserted % 5000 == 0 or total_inserted == len(rows):
                    logger.info(f"  Upserted {total_inserted}/{len(rows)} rows...")

        team_conn.commit()
        return total_inserted
    except Exception:
        team_conn.rollback()
        raise
    finally:
        team_conn.autocommit = old_autocommit


def main():
    parser = argparse.ArgumentParser(description="Compute team rolling stats")
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Only compute for recent days (use with --days)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=14,
        help="Number of days to look back in incremental mode (default: 14)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date for backfill (YYYY-MM-DD). Default: 2023-10-01",
    )
    args = parser.parse_args()

    now_est = datetime.now(EST)

    if args.incremental:
        start_date = (now_est - timedelta(days=args.days)).date()
        logger.info(f"Incremental mode: computing last {args.days} days (from {start_date})")
    elif args.start_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
        logger.info(f"Custom start date: {start_date}")
    else:
        from datetime import date

        start_date = date(2023, 10, 1)
        logger.info(f"Full backfill mode from {start_date}")

    logger.info("Connecting to databases...")
    games_conn = psycopg2.connect(**get_games_db_config())
    games_conn.autocommit = True
    team_conn = psycopg2.connect(**get_team_db_config())

    try:
        logger.info("Fetching team game history from team_game_logs...")
        game_history = get_team_game_history(games_conn)
        logger.info(f"  Loaded {len(game_history)} team-game records")

        logger.info(f"Computing {WINDOW_SIZE}-game rolling stats (start_date={start_date})...")
        rows = compute_rolling_stats(game_history, start_date=start_date)
        logger.info(f"  Computed {len(rows)} rolling stat rows")

        if rows:
            logger.info("Upserting into teams.team_rolling_stats...")
            count = upsert_rolling_stats(team_conn, rows)
            logger.info(f"Done! Upserted {count} rows into team_rolling_stats.")
        else:
            logger.info("No rows computed — nothing to insert.")

        # Print summary
        with team_conn.cursor() as cur:
            team_conn.autocommit = True
            cur.execute("SELECT COUNT(*) FROM team_rolling_stats")
            total = cur.fetchone()[0]
            cur.execute("SELECT MIN(as_of_date), MAX(as_of_date) FROM team_rolling_stats")
            date_range = cur.fetchone()
            cur.execute("SELECT COUNT(DISTINCT team_abbrev) FROM team_rolling_stats")
            n_teams = cur.fetchone()[0]
            logger.info(
                f"Table summary: {total} rows, {n_teams} teams, "
                f"dates {date_range[0]} to {date_range[1]}"
            )

    finally:
        games_conn.close()
        team_conn.close()


if __name__ == "__main__":
    main()
