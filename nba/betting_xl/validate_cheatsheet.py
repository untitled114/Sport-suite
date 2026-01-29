#!/usr/bin/env python3
"""
Validate BettingPros Cheat Sheet Recommendations
=================================================
Measures win rates of cheat sheet picks by:
- Bet rating (1-5 stars)
- EV threshold
- Hit rate filters
- Projection diff

This helps find the "sweet spot" for using cheat sheet data.

Usage:
    python validate_cheatsheet.py --days 7
    python validate_cheatsheet.py --start 2026-01-01 --end 2026-01-07
"""

import argparse
import os
from collections import defaultdict
from datetime import datetime, timedelta

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


def get_cheatsheet_with_actuals(
    start_date: str, end_date: str, platform: str = "underdog", stat_types: list = None
):
    """
    Get cheat sheet recommendations joined with actual results.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        platform: Platform filter ('underdog', 'all', etc.)
        stat_types: Optional list of stat types to filter (e.g., ['PA', 'PR', 'RA'])

    Returns list of dicts with cheat sheet data + actual_value + result (WIN/LOSS/PUSH)
    """
    conn_intel = psycopg2.connect(**INTELLIGENCE_DB)
    conn_players = psycopg2.connect(**PLAYERS_DB)

    # Build query with optional stat type filter
    query = """
        SELECT player_name, game_date, stat_type, platform,
               line, projection, projection_diff, bet_rating, ev_pct,
               probability, hit_rate_l5, hit_rate_l15, hit_rate_season,
               opp_rank
        FROM cheatsheet_data
        WHERE game_date BETWEEN %s AND %s
          AND recommended_side = 'over'
          AND platform = %s
    """
    params = [start_date, end_date, platform]

    if stat_types:
        placeholders = ", ".join(["%s"] * len(stat_types))
        query += f" AND stat_type IN ({placeholders})"
        params.extend(stat_types)

    query += " ORDER BY game_date, player_name"

    # Get cheat sheet data (OVER recommendations only)
    cursor = conn_intel.cursor()
    cursor.execute(query, params)

    cheatsheet_rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    cursor.close()
    conn_intel.close()

    if not cheatsheet_rows:
        print(f"No cheat sheet data found for {start_date} to {end_date}")
        return []

    # Get actual results from player_game_logs
    cursor_players = conn_players.cursor()

    results = []
    for row in cheatsheet_rows:
        data = dict(zip(columns, row))
        player_name = data["player_name"]
        game_date = data["game_date"]
        stat_type = data["stat_type"]
        line = float(data["line"])

        # Map stat_type to column in player_game_logs
        stat_column_map = {
            "POINTS": "points",
            "REBOUNDS": "rebounds",
            "ASSISTS": "assists",
            "THREES": "three_pointers_made",
        }

        # Combo stat mappings (PA, PR, RA, PRA)
        combo_stat_map = {
            "PA": ("points", "assists"),  # Points + Assists
            "PR": ("points", "rebounds"),  # Points + Rebounds
            "RA": ("rebounds", "assists"),  # Rebounds + Assists
            "PRA": ("points", "rebounds", "assists"),  # Points + Rebounds + Assists
            # Legacy names (for backwards compatibility)
            "POINTS_ASSISTS": ("points", "assists"),
            "POINTS_REBOUNDS": ("points", "rebounds"),
            "REBOUNDS_ASSISTS": ("rebounds", "assists"),
        }

        stat_col = stat_column_map.get(stat_type)
        combo_cols = combo_stat_map.get(stat_type)

        if stat_col:
            # Single stat - join with player_profile to match by name
            cursor_players.execute(
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
            # Combo stat - sum the columns
            sum_expr = " + ".join([f"g.{col}" for col in combo_cols])
            cursor_players.execute(
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

        result_row = cursor_players.fetchone()

        if result_row and result_row[0] is not None:
            actual = float(result_row[0])
            data["actual_value"] = actual

            # Determine result (OVER bet)
            if actual > line:
                data["result"] = "WIN"
            elif actual < line:
                data["result"] = "LOSS"
            else:
                data["result"] = "PUSH"

            results.append(data)

    cursor_players.close()
    conn_players.close()

    return results


def analyze_by_filter(results: list, filter_name: str, filter_fn, buckets: list):
    """Analyze win rate by a specific filter with buckets"""
    print(f"\n{'='*60}")
    print(f"WIN RATE BY {filter_name.upper()}")
    print(f"{'='*60}")

    bucket_stats = defaultdict(lambda: {"wins": 0, "losses": 0, "pushes": 0})

    for r in results:
        bucket = filter_fn(r)
        if bucket is None:
            continue

        # Find which bucket this falls into
        for b_name, b_min, b_max in buckets:
            if b_min <= bucket < b_max:
                if r["result"] == "WIN":
                    bucket_stats[b_name]["wins"] += 1
                elif r["result"] == "LOSS":
                    bucket_stats[b_name]["losses"] += 1
                else:
                    bucket_stats[b_name]["pushes"] += 1
                break

    print(f"{'Bucket':<20} {'W':>5} {'L':>5} {'P':>3} {'Total':>6} {'WR':>8} {'vs 52.4%':>10}")
    print("-" * 60)

    for b_name, _b_min, _b_max in buckets:
        stats = bucket_stats[b_name]
        total = stats["wins"] + stats["losses"]
        if total > 0:
            wr = stats["wins"] / total * 100
            edge = wr - 52.4  # breakeven for -110
            edge_str = f"+{edge:.1f}%" if edge > 0 else f"{edge:.1f}%"
            print(
                f"{b_name:<20} {stats['wins']:>5} {stats['losses']:>5} {stats['pushes']:>3} {total:>6} {wr:>7.1f}% {edge_str:>10}"
            )
        else:
            print(f"{b_name:<20} {'-':>5} {'-':>5} {'-':>3} {0:>6} {'-':>8} {'-':>10}")


def main():
    parser = argparse.ArgumentParser(description="Validate cheat sheet recommendations")
    parser.add_argument("--days", type=int, default=7, help="Number of days to analyze")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--platform", type=str, default="underdog", help="Platform to analyze")
    parser.add_argument(
        "--stat-types", type=str, help="Comma-separated stat types (e.g., PA,PR,RA)"
    )

    args = parser.parse_args()

    # Parse stat types if provided
    stat_types = None
    if args.stat_types:
        stat_types = [s.strip().upper() for s in args.stat_types.split(",")]

    # Determine date range
    if args.start and args.end:
        start_date = args.start
        end_date = args.end
    else:
        end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=args.days)).strftime("%Y-%m-%d")

    print("\n" + "=" * 60)
    print("BETTINGPROS CHEAT SHEET VALIDATION")
    print("=" * 60)
    print(f"Date Range: {start_date} to {end_date}")
    print(f"Platform: {args.platform}")
    if stat_types:
        print(f"Stat Types: {', '.join(stat_types)}")
    print("Bet Type: OVER only")
    print("=" * 60)

    # Get data with actuals
    results = get_cheatsheet_with_actuals(start_date, end_date, args.platform, stat_types)

    if not results:
        print("\nNo results with actual values found.")
        print("Make sure:")
        print("  1. Cheat sheet data is loaded for these dates")
        print("  2. Player game logs exist for these dates")
        return

    # Overall stats
    wins = sum(1 for r in results if r["result"] == "WIN")
    losses = sum(1 for r in results if r["result"] == "LOSS")
    pushes = sum(1 for r in results if r["result"] == "PUSH")
    total = wins + losses

    print(
        f"\nOVERALL: {wins}W / {losses}L / {pushes}P = {wins/total*100:.1f}% WR"
        if total > 0
        else "\nNo graded results"
    )
    print(f"Total picks analyzed: {len(results)}")

    if total == 0:
        return

    # Analyze by BET RATING (1-5 stars)
    analyze_by_filter(
        results,
        "BET RATING (Stars)",
        lambda r: r.get("bet_rating"),
        [
            ("1 star", 1, 2),
            ("2 stars", 2, 3),
            ("3 stars", 3, 4),
            ("4 stars", 4, 5),
            ("5 stars", 5, 6),
        ],
    )

    # Analyze by EV%
    analyze_by_filter(
        results,
        "EXPECTED VALUE (EV%)",
        lambda r: r.get("ev_pct"),
        [
            ("EV < 10%", 0, 10),
            ("EV 10-20%", 10, 20),
            ("EV 20-30%", 20, 30),
            ("EV 30-40%", 30, 40),
            ("EV 40%+", 40, 100),
        ],
    )

    # Analyze by PROJECTION DIFF
    analyze_by_filter(
        results,
        "PROJECTION DIFF (Proj - Line)",
        lambda r: float(r.get("projection_diff") or 0),
        [
            ("Diff < 1", -10, 1),
            ("Diff 1-2", 1, 2),
            ("Diff 2-3", 2, 3),
            ("Diff 3-5", 3, 5),
            ("Diff 5+", 5, 50),
        ],
    )

    # Analyze by HIT RATE L5
    analyze_by_filter(
        results,
        "HIT RATE LAST 5 GAMES",
        lambda r: float(r.get("hit_rate_l5") or 0) * 100,
        [
            ("L5 < 40%", 0, 40),
            ("L5 40-60%", 40, 60),
            ("L5 60-80%", 60, 80),
            ("L5 80%+", 80, 101),
        ],
    )

    # Analyze by OPP RANK
    analyze_by_filter(
        results,
        "OPPONENT DEFENSE RANK",
        lambda r: r.get("opp_rank"),
        [
            ("Top 10 (tough)", 1, 11),
            ("11-20", 11, 21),
            ("21-30 (weak)", 21, 31),
        ],
    )

    # Analyze by STAT TYPE
    print(f"\n{'='*60}")
    print("WIN RATE BY STAT TYPE")
    print(f"{'='*60}")

    stat_stats = defaultdict(lambda: {"wins": 0, "losses": 0})
    for r in results:
        st = r["stat_type"]
        if r["result"] == "WIN":
            stat_stats[st]["wins"] += 1
        elif r["result"] == "LOSS":
            stat_stats[st]["losses"] += 1

    print(f"{'Stat Type':<20} {'W':>5} {'L':>5} {'Total':>6} {'WR':>8}")
    print("-" * 45)
    for st in sorted(stat_stats.keys()):
        s = stat_stats[st]
        total = s["wins"] + s["losses"]
        wr = s["wins"] / total * 100 if total > 0 else 0
        print(f"{st:<20} {s['wins']:>5} {s['losses']:>5} {total:>6} {wr:>7.1f}%")

    # Best combinations
    print(f"\n{'='*60}")
    print("BEST FILTER COMBINATIONS")
    print(f"{'='*60}")

    # High rating + high EV
    high_conf = [r for r in results if r.get("bet_rating", 0) >= 4 and (r.get("ev_pct") or 0) >= 20]
    if high_conf:
        wins = sum(1 for r in high_conf if r["result"] == "WIN")
        losses = sum(1 for r in high_conf if r["result"] == "LOSS")
        if wins + losses > 0:
            print(
                f"Rating ≥4 AND EV ≥20%: {wins}W/{losses}L = {wins/(wins+losses)*100:.1f}% ({wins+losses} picks)"
            )

    # High rating + high proj diff
    high_diff = [
        r
        for r in results
        if r.get("bet_rating", 0) >= 4 and float(r.get("projection_diff") or 0) >= 2
    ]
    if high_diff:
        wins = sum(1 for r in high_diff if r["result"] == "WIN")
        losses = sum(1 for r in high_diff if r["result"] == "LOSS")
        if wins + losses > 0:
            print(
                f"Rating ≥4 AND Diff ≥2: {wins}W/{losses}L = {wins/(wins+losses)*100:.1f}% ({wins+losses} picks)"
            )

    # 5-star only
    five_star = [r for r in results if r.get("bet_rating", 0) == 5]
    if five_star:
        wins = sum(1 for r in five_star if r["result"] == "WIN")
        losses = sum(1 for r in five_star if r["result"] == "LOSS")
        if wins + losses > 0:
            print(
                f"5-Star Only: {wins}W/{losses}L = {wins/(wins+losses)*100:.1f}% ({wins+losses} picks)"
            )

    # Rebounds Meta Filter (Jan 2026)
    # Criteria: projection_diff >= 1.0, hit_rate_season >= 0.6, hit_rate_l15 >= 0.6
    rebounds_meta = [
        r
        for r in results
        if r.get("stat_type") == "REBOUNDS"
        and float(r.get("projection_diff") or 0) >= 1.0
        and float(r.get("hit_rate_season") or 0) >= 0.6
        and float(r.get("hit_rate_l15") or 0) >= 0.6
    ]
    if rebounds_meta:
        wins = sum(1 for r in rebounds_meta if r["result"] == "WIN")
        losses = sum(1 for r in rebounds_meta if r["result"] == "LOSS")
        if wins + losses > 0:
            print(
                f"REBOUNDS Meta (Diff≥1, Season≥60%, L15≥60%): {wins}W/{losses}L = {wins/(wins+losses)*100:.1f}% ({wins+losses} picks)"
            )

    # Rebounds baseline (no filter) for comparison
    rebounds_all = [r for r in results if r.get("stat_type") == "REBOUNDS"]
    if rebounds_all:
        wins = sum(1 for r in rebounds_all if r["result"] == "WIN")
        losses = sum(1 for r in rebounds_all if r["result"] == "LOSS")
        if wins + losses > 0:
            print(
                f"REBOUNDS Baseline (all): {wins}W/{losses}L = {wins/(wins+losses)*100:.1f}% ({wins+losses} picks)"
            )

    print("\n" + "=" * 60)
    print("Note: Need 50+ samples per bucket for statistical significance")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
