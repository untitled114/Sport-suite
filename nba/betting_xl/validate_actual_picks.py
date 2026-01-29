#!/usr/bin/env python3
"""
Validate Actual Published Picks
================================
Validates the 4-10 picks per day that were actually generated and published,
not all props that met the filters.

Reads from predictions/xl_picks_YYYY-MM-DD.json files and compares against
actual_value in nba_props_xl.

Usage:
    python3 validate_actual_picks.py --start-date 2025-11-07 --end-date 2025-11-10
    python3 validate_actual_picks.py --date 2025-11-07
"""

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import psycopg2

DB_CONFIG = {
    "host": "localhost",
    "port": 5539,
    "user": os.getenv("DB_USER", "nba_user"),
    "password": os.getenv("DB_PASSWORD"),
    "database": "nba_intelligence",
}


def load_picks_for_date(date_str: str) -> dict:
    """Load picks JSON file for a specific date"""
    # Try both naming formats
    patterns = [
        f"predictions/xl_picks_{date_str}.json",
        f"predictions/xl_picks_{date_str.replace('-', '')}.json",
    ]

    for pattern in patterns:
        filepath = Path(pattern)
        if filepath.exists():
            with open(filepath, "r") as f:
                return json.load(f)

    return None


def get_actual_value(conn, player_name: str, game_date: str, stat_type: str) -> float:
    """Get actual_value from nba_props_xl"""
    cursor = conn.cursor()

    query = """
    SELECT actual_value
    FROM nba_props_xl
    WHERE LOWER(TRIM(player_name)) = LOWER(TRIM(%s))
      AND game_date = %s
      AND stat_type = %s
      AND actual_value IS NOT NULL
    LIMIT 1
    """

    cursor.execute(query, (player_name, game_date, stat_type))
    result = cursor.fetchone()
    cursor.close()

    return result[0] if result else None


def validate_pick(pick: dict, actual_value: float) -> dict:
    """Validate a single pick"""
    side = pick["side"]

    # Get the best line (softest)
    if pick.get("top_3_lines"):
        best_line = pick["top_3_lines"][0]["line"]
    else:
        best_line = pick.get("consensus_line")

    if actual_value is None:
        return {"status": "NO_DATA", "result": None}

    # Determine win/loss/push
    if side == "OVER":
        if actual_value > best_line:
            result = "WIN"
        elif actual_value == best_line:
            result = "PUSH"
        else:
            result = "LOSS"
    else:  # UNDER
        if actual_value < best_line:
            result = "WIN"
        elif actual_value == best_line:
            result = "PUSH"
        else:
            result = "LOSS"

    return {
        "status": "COMPLETE",
        "result": result,
        "line": best_line,
        "actual": actual_value,
        "edge": (
            pick.get("top_3_lines", [{}])[0].get("edge_pct", 0) if pick.get("top_3_lines") else 0
        ),
    }


def validate_date(date_str: str) -> dict:
    """Validate all picks for a specific date"""
    picks_data = load_picks_for_date(date_str)

    if not picks_data:
        return {
            "date": date_str,
            "error": "No picks file found",
            "total": 0,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
            "no_data": 0,
        }

    picks = picks_data.get("picks", [])
    conn = psycopg2.connect(**DB_CONFIG)

    results = {
        "date": date_str,
        "total": len(picks),
        "wins": 0,
        "losses": 0,
        "pushes": 0,
        "no_data": 0,
        "by_market": defaultdict(lambda: {"wins": 0, "losses": 0, "pushes": 0, "no_data": 0}),
        "by_tier": defaultdict(lambda: {"wins": 0, "losses": 0, "pushes": 0, "no_data": 0}),
        "picks_detail": [],
    }

    for pick in picks:
        player_name = pick["player_name"]
        stat_type = pick["stat_type"]
        tier = pick.get("filter_tier", "unknown")

        actual_value = get_actual_value(conn, player_name, date_str, stat_type)
        validation = validate_pick(pick, actual_value)

        if validation["status"] == "NO_DATA":
            results["no_data"] += 1
            results["by_market"][stat_type]["no_data"] += 1
            results["by_tier"][tier]["no_data"] += 1
        else:
            result = validation["result"]
            if result == "WIN":
                results["wins"] += 1
                results["by_market"][stat_type]["wins"] += 1
                results["by_tier"][tier]["wins"] += 1
            elif result == "LOSS":
                results["losses"] += 1
                results["by_market"][stat_type]["losses"] += 1
                results["by_tier"][tier]["losses"] += 1
            elif result == "PUSH":
                results["pushes"] += 1
                results["by_market"][stat_type]["pushes"] += 1
                results["by_tier"][tier]["pushes"] += 1

        results["picks_detail"].append(
            {
                "player": player_name,
                "market": stat_type,
                "tier": tier,
                "side": pick["side"],
                "line": validation.get("line"),
                "actual": validation.get("actual"),
                "result": validation.get("result", "NO_DATA"),
                "edge": validation.get("edge", 0),
            }
        )

    conn.close()
    return results


def print_results(all_results: list):
    """Print validation results"""
    print("\n" + "=" * 80)
    print("ACTUAL PICKS VALIDATION")
    print("=" * 80)

    total_wins = sum(r["wins"] for r in all_results)
    total_losses = sum(r["losses"] for r in all_results)
    total_pushes = sum(r["pushes"] for r in all_results)
    total_no_data = sum(r["no_data"] for r in all_results)
    total_bets = total_wins + total_losses + total_pushes

    print(f"\nDate Range: {all_results[0]['date']} to {all_results[-1]['date']}")
    print(f"Total Picks Published: {sum(r['total'] for r in all_results)}")
    print(f"Completed Games: {total_bets} | No Data: {total_no_data}")

    print("\n" + "-" * 80)
    print("OVERALL RESULTS")
    print("-" * 80)
    print(f"Wins: {total_wins}")
    print(f"Losses: {total_losses}")
    print(f"Pushes: {total_pushes}")

    if total_bets > 0:
        win_rate = total_wins / total_bets * 100
        roi = ((total_wins * 1.91) - total_bets) / total_bets * 100  # Assuming -110 juice
        print(f"\nWin Rate: {win_rate:.1f}%")
        print(f"ROI: {roi:+.2f}%")
        print(f"Units: {((total_wins * 1.91) - total_bets):.2f}u")

    # By date
    print("\n" + "-" * 80)
    print("BY DATE")
    print("-" * 80)
    print(f"{'Date':<12} {'Picks':<6} {'W':<4} {'L':<4} {'P':<4} {'WR%':<7} {'ROI%':<8}")
    print("-" * 80)

    for result in all_results:
        picks_with_data = result["wins"] + result["losses"] + result["pushes"]
        if picks_with_data > 0:
            wr = result["wins"] / picks_with_data * 100
            roi_day = ((result["wins"] * 1.91) - picks_with_data) / picks_with_data * 100
            print(
                f"{result['date']:<12} {result['total']:<6} {result['wins']:<4} {result['losses']:<4} {result['pushes']:<4} {wr:<7.1f} {roi_day:+<8.2f}"
            )

    # By market
    print("\n" + "-" * 80)
    print("BY MARKET")
    print("-" * 80)

    by_market = defaultdict(lambda: {"wins": 0, "losses": 0, "pushes": 0})
    for result in all_results:
        for market, stats in result["by_market"].items():
            by_market[market]["wins"] += stats["wins"]
            by_market[market]["losses"] += stats["losses"]
            by_market[market]["pushes"] += stats["pushes"]

    print(f"{'Market':<12} {'W':<4} {'L':<4} {'P':<4} {'WR%':<7} {'ROI%':<8}")
    print("-" * 80)
    for market, stats in sorted(by_market.items()):
        total_market = stats["wins"] + stats["losses"] + stats["pushes"]
        if total_market > 0:
            wr = stats["wins"] / total_market * 100
            roi_market = ((stats["wins"] * 1.91) - total_market) / total_market * 100
            print(
                f"{market:<12} {stats['wins']:<4} {stats['losses']:<4} {stats['pushes']:<4} {wr:<7.1f} {roi_market:+<8.2f}"
            )

    # By tier
    print("\n" + "-" * 80)
    print("BY TIER")
    print("-" * 80)

    by_tier = defaultdict(lambda: {"wins": 0, "losses": 0, "pushes": 0})
    for result in all_results:
        for tier, stats in result["by_tier"].items():
            by_tier[tier]["wins"] += stats["wins"]
            by_tier[tier]["losses"] += stats["losses"]
            by_tier[tier]["pushes"] += stats["pushes"]

    print(f"{'Tier':<12} {'W':<4} {'L':<4} {'P':<4} {'WR%':<7} {'ROI%':<8}")
    print("-" * 80)
    for tier, stats in sorted(by_tier.items()):
        total_tier = stats["wins"] + stats["losses"] + stats["pushes"]
        if total_tier > 0:
            wr = stats["wins"] / total_tier * 100
            roi_tier = ((stats["wins"] * 1.91) - total_tier) / total_tier * 100
            print(
                f"{tier:<12} {stats['wins']:<4} {stats['losses']:<4} {stats['pushes']:<4} {wr:<7.1f} {roi_tier:+<8.2f}"
            )

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Validate actual published picks")
    parser.add_argument("--date", help="Specific date (YYYY-MM-DD)")
    parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")

    args = parser.parse_args()

    if args.date:
        dates = [args.date]
    elif args.start_date and args.end_date:
        start = datetime.strptime(args.start_date, "%Y-%m-%d")
        end = datetime.strptime(args.end_date, "%Y-%m-%d")
        dates = []
        current = start
        while current <= end:
            dates.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)
    else:
        print("Error: Must specify --date or --start-date and --end-date")
        return

    all_results = []
    for date_str in dates:
        result = validate_date(date_str)
        if result["total"] > 0:
            all_results.append(result)

    if not all_results:
        print("No picks found for specified dates")
        return

    print_results(all_results)


if __name__ == "__main__":
    main()
