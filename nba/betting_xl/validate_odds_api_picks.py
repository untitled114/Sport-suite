#!/usr/bin/env python3
"""
Validate Odds API Picks
========================
Measures win rates of Odds API picks by:
- Filter name
- Multiplier bucket
- Stat type
- Hit rate buckets
- Opponent defense rank

Reads picks from predictions/odds_api_picks_*.json and joins
with actual results from player_game_logs.

Usage:
    python3 validate_odds_api_picks.py --days 7
    python3 validate_odds_api_picks.py --start 2026-01-15 --end 2026-01-24
    python3 validate_odds_api_picks.py --file predictions/odds_api_picks_20260125.json
"""

import argparse
import glob
import json
import os
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import psycopg2

# Database configs
INTELLIGENCE_DB = {
    "host": "localhost",
    "port": 5539,
    "database": "nba_intelligence",
    "user": os.getenv("DB_USER", "nba_user"),
    "password": os.getenv("DB_PASSWORD"),
}

PLAYERS_DB = {
    "host": "localhost",
    "port": 5536,
    "database": "nba_players",
    "user": os.getenv("DB_USER", "nba_user"),
    "password": os.getenv("DB_PASSWORD"),
}

# Stat type -> game log column mapping
STAT_COLUMN_MAP = {
    "POINTS": "points",
    "REBOUNDS": "rebounds",
    "ASSISTS": "assists",
    "THREES": "three_pointers_made",
}

COMBO_STAT_MAP = {
    "PA": ("points", "assists"),
    "PR": ("points", "rebounds"),
    "RA": ("rebounds", "assists"),
    "PRA": ("points", "rebounds", "assists"),
}


def load_picks_from_files(start_date: str, end_date: str) -> list:
    """
    Load Odds API picks from prediction JSON files in the date range.

    Returns list of pick dicts with actual results joined.
    """
    predictions_dir = Path(__file__).parent / "predictions"
    all_picks = []

    # Parse date range
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()

    # Scan for files matching pattern
    for filepath in sorted(predictions_dir.glob("odds_api_picks_*.json")):
        # Extract date from filename (odds_api_picks_YYYYMMDD.json or odds_api_picks_YYYY-MM-DD.json)
        stem = filepath.stem.replace("odds_api_picks_", "")
        try:
            if "-" in stem:
                file_date = datetime.strptime(stem, "%Y-%m-%d").date()
            else:
                file_date = datetime.strptime(stem, "%Y%m%d").date()
        except ValueError:
            continue

        if not (start <= file_date <= end):
            continue

        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            picks = data.get("picks", [])
            game_date = data.get("date", str(file_date))

            for pick in picks:
                pick["game_date"] = game_date
                all_picks.append(pick)

        except (json.JSONDecodeError, KeyError) as e:
            print(f"  Warning: Could not read {filepath}: {e}")

    return all_picks


def load_picks_from_file(filepath: str) -> list:
    """Load picks from a single JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)

    picks = data.get("picks", [])
    game_date = data.get("date", "")

    for pick in picks:
        pick["game_date"] = game_date

    return picks


def get_actual_result(cursor, player_name: str, game_date: str, stat_type: str) -> dict:
    """
    Get actual game result for a pick.

    Returns dict with actual_value and result (WIN/LOSS/PUSH), or None.
    """
    stat_col = STAT_COLUMN_MAP.get(stat_type)
    combo_cols = COMBO_STAT_MAP.get(stat_type)

    if stat_col:
        cursor.execute(
            f"""
            SELECT g.{stat_col}
            FROM player_game_logs g
            JOIN player_profile p ON g.player_id = p.player_id
            WHERE LOWER(p.full_name) = LOWER(%s) AND g.game_date = %s
            LIMIT 1
        """,
            (player_name, game_date),
        )
    elif combo_cols:
        sum_expr = " + ".join([f"g.{col}" for col in combo_cols])
        cursor.execute(
            f"""
            SELECT {sum_expr} as combo
            FROM player_game_logs g
            JOIN player_profile p ON g.player_id = p.player_id
            WHERE LOWER(p.full_name) = LOWER(%s) AND g.game_date = %s
            LIMIT 1
        """,
            (player_name, game_date),
        )
    else:
        return None

    row = cursor.fetchone()
    if row is None or row[0] is None:
        return None

    actual = float(row[0])
    line = float(STAT_COLUMN_MAP.get("line", 0))
    return {"actual_value": actual}


def grade_picks(picks: list) -> list:
    """
    Grade picks by joining with actual results from player_game_logs.

    Returns list of picks with actual_value and result fields added.
    """
    conn = psycopg2.connect(**PLAYERS_DB)
    cursor = conn.cursor()

    graded = []

    for pick in picks:
        player_name = pick["player_name"]
        game_date = pick.get("game_date", "")
        stat_type = pick["stat_type"]
        line = float(pick.get("line", 0))

        stat_col = STAT_COLUMN_MAP.get(stat_type)
        combo_cols = COMBO_STAT_MAP.get(stat_type)

        if stat_col:
            cursor.execute(
                f"""
                SELECT g.{stat_col}
                FROM player_game_logs g
                JOIN player_profile p ON g.player_id = p.player_id
                WHERE LOWER(p.full_name) = LOWER(%s) AND g.game_date = %s
                LIMIT 1
            """,
                (player_name, game_date),
            )
        elif combo_cols:
            sum_expr = " + ".join([f"g.{col}" for col in combo_cols])
            cursor.execute(
                f"""
                SELECT {sum_expr} as combo
                FROM player_game_logs g
                JOIN player_profile p ON g.player_id = p.player_id
                WHERE LOWER(p.full_name) = LOWER(%s) AND g.game_date = %s
                LIMIT 1
            """,
                (player_name, game_date),
            )
        else:
            continue

        row = cursor.fetchone()

        if row is None or row[0] is None:
            continue

        actual = float(row[0])
        pick["actual_value"] = actual

        if actual > line:
            pick["result"] = "WIN"
        elif actual < line:
            pick["result"] = "LOSS"
        else:
            pick["result"] = "PUSH"

        graded.append(pick)

    cursor.close()
    conn.close()

    return graded


def analyze_by_bucket(results: list, label: str, key_fn, buckets: list):
    """Analyze win rate by bucketed values."""
    print(f"\n{'='*60}")
    print(f"WIN RATE BY {label.upper()}")
    print(f"{'='*60}")

    bucket_stats = defaultdict(lambda: {"wins": 0, "losses": 0, "pushes": 0})

    for r in results:
        val = key_fn(r)
        if val is None:
            continue

        for b_name, b_min, b_max in buckets:
            if b_min <= val < b_max:
                if r["result"] == "WIN":
                    bucket_stats[b_name]["wins"] += 1
                elif r["result"] == "LOSS":
                    bucket_stats[b_name]["losses"] += 1
                else:
                    bucket_stats[b_name]["pushes"] += 1
                break

    print(f"{'Bucket':<25} {'W':>4} {'L':>4} {'P':>3} {'Total':>6} {'WR':>8}")
    print("-" * 55)

    for b_name, _b_min, _b_max in buckets:
        stats = bucket_stats[b_name]
        total = stats["wins"] + stats["losses"]
        if total > 0:
            wr = stats["wins"] / total * 100
            print(
                f"{b_name:<25} {stats['wins']:>4} {stats['losses']:>4} {stats['pushes']:>3} {total:>6} {wr:>7.1f}%"
            )
        else:
            print(f"{b_name:<25} {'-':>4} {'-':>4} {'-':>3} {0:>6} {'-':>8}")


def main():
    parser = argparse.ArgumentParser(description="Validate Odds API picks against actual results")
    parser.add_argument("--days", type=int, default=7, help="Number of days to analyze")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--file", type=str, help="Single picks file to validate")

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("ODDS API PICKS VALIDATION")
    print("=" * 60)

    # Load picks
    if args.file:
        print(f"File: {args.file}")
        picks = load_picks_from_file(args.file)
    else:
        if args.start and args.end:
            start_date = args.start
            end_date = args.end
        else:
            end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=args.days)).strftime("%Y-%m-%d")

        print(f"Date Range: {start_date} to {end_date}")
        picks = load_picks_from_files(start_date, end_date)

    print(f"Picks loaded: {len(picks)}")
    print("=" * 60)

    if not picks:
        print("\nNo picks found. Make sure:")
        print("  1. Odds API picks have been generated for these dates")
        print("  2. Files exist in predictions/odds_api_picks_*.json")
        return

    # Grade picks against actuals
    graded = grade_picks(picks)
    print(f"Picks graded: {len(graded)} (out of {len(picks)} loaded)")

    if not graded:
        print("\nNo graded results. Make sure player game logs exist for these dates.")
        return

    # Overall stats
    wins = sum(1 for r in graded if r["result"] == "WIN")
    losses = sum(1 for r in graded if r["result"] == "LOSS")
    pushes = sum(1 for r in graded if r["result"] == "PUSH")
    total = wins + losses

    if total > 0:
        wr = wins / total * 100
        roi = (wins * 0.91 - losses) / total * 100  # Assuming -110 odds
        print(f"\nOVERALL: {wins}W / {losses}L / {pushes}P = {wr:.1f}% WR")
        print(f"ROI (at -110): {roi:+.1f}%")
    else:
        print("\nNo graded results (no wins or losses)")
        return

    # =================================================================
    # BY FILTER NAME
    # =================================================================
    print(f"\n{'='*60}")
    print("WIN RATE BY FILTER")
    print(f"{'='*60}")

    filter_stats = defaultdict(lambda: {"wins": 0, "losses": 0})
    for r in graded:
        fn = r.get("filter_name", "unknown")
        if r["result"] == "WIN":
            filter_stats[fn]["wins"] += 1
        elif r["result"] == "LOSS":
            filter_stats[fn]["losses"] += 1

    print(f"{'Filter':<30} {'W':>4} {'L':>4} {'Total':>6} {'WR':>8}")
    print("-" * 55)
    for fn in sorted(filter_stats.keys()):
        s = filter_stats[fn]
        t = s["wins"] + s["losses"]
        wr = s["wins"] / t * 100 if t > 0 else 0
        print(f"{fn:<30} {s['wins']:>4} {s['losses']:>4} {t:>6} {wr:>7.1f}%")

    # =================================================================
    # BY STAT TYPE
    # =================================================================
    print(f"\n{'='*60}")
    print("WIN RATE BY STAT TYPE")
    print(f"{'='*60}")

    stat_stats = defaultdict(lambda: {"wins": 0, "losses": 0})
    for r in graded:
        st = r["stat_type"]
        if r["result"] == "WIN":
            stat_stats[st]["wins"] += 1
        elif r["result"] == "LOSS":
            stat_stats[st]["losses"] += 1

    print(f"{'Stat Type':<20} {'W':>4} {'L':>4} {'Total':>6} {'WR':>8}")
    print("-" * 45)
    for st in sorted(stat_stats.keys()):
        s = stat_stats[st]
        t = s["wins"] + s["losses"]
        wr = s["wins"] / t * 100 if t > 0 else 0
        print(f"{st:<20} {s['wins']:>4} {s['losses']:>4} {t:>6} {wr:>7.1f}%")

    # =================================================================
    # BY MULTIPLIER BUCKET
    # =================================================================
    analyze_by_bucket(
        graded,
        "Pick6 Multiplier",
        lambda r: r.get("pick6_multiplier"),
        [
            ("mult < 0.8", 0.0, 0.8),
            ("mult 0.8-1.0", 0.8, 1.0),
            ("mult 1.0-1.2", 1.0, 1.2),
            ("mult 1.2-1.5", 1.2, 1.5),
            ("mult 1.5-2.0", 1.5, 2.0),
            ("mult 2.0-5.0", 2.0, 5.0),
            ("mult 5.0+ (TRAP)", 5.0, 100.0),
        ],
    )

    # =================================================================
    # BY L5 HIT RATE
    # =================================================================
    analyze_by_bucket(
        graded,
        "L5 Hit Rate",
        lambda r: (r.get("hit_rate_l5") or 0) * 100,
        [
            ("L5 < 60%", 0, 60),
            ("L5 60-80%", 60, 80),
            ("L5 80-99%", 80, 99.9),
            ("L5 100%", 99.9, 101),
        ],
    )

    # =================================================================
    # BY OPPONENT RANK
    # =================================================================
    analyze_by_bucket(
        graded,
        "Opponent Defense Rank",
        lambda r: r.get("opp_rank"),
        [
            ("Top 10 (tough)", 1, 11),
            ("11-20", 11, 21),
            ("21-30 (weak)", 21, 31),
        ],
    )

    # =================================================================
    # BY BET RATING
    # =================================================================
    analyze_by_bucket(
        graded,
        "Bet Rating",
        lambda r: r.get("bet_rating"),
        [
            ("1-2 stars", 1, 3),
            ("3 stars", 3, 4),
            ("4 stars", 4, 5),
            ("5 stars", 5, 6),
        ],
    )

    # =================================================================
    # DAILY BREAKDOWN
    # =================================================================
    print(f"\n{'='*60}")
    print("DAILY BREAKDOWN")
    print(f"{'='*60}")

    daily_stats = defaultdict(lambda: {"wins": 0, "losses": 0, "pushes": 0})
    for r in graded:
        d = r.get("game_date", "unknown")
        if r["result"] == "WIN":
            daily_stats[d]["wins"] += 1
        elif r["result"] == "LOSS":
            daily_stats[d]["losses"] += 1
        else:
            daily_stats[d]["pushes"] += 1

    print(f"{'Date':<15} {'W':>4} {'L':>4} {'P':>3} {'Total':>6} {'WR':>8}")
    print("-" * 45)
    for d in sorted(daily_stats.keys()):
        s = daily_stats[d]
        t = s["wins"] + s["losses"]
        wr = s["wins"] / t * 100 if t > 0 else 0
        print(f"{d:<15} {s['wins']:>4} {s['losses']:>4} {s['pushes']:>3} {t:>6} {wr:>7.1f}%")

    print("\n" + "=" * 60)
    print("Note: Need 50+ samples per bucket for statistical significance")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
