#!/usr/bin/env python3
"""
Validate Predictions Against Actual Results
=============================================
Compare predicted picks against actual game results.

Validates all pick sources:
- XL picks (xl_picks_*.json) - ML model predictions
- PRO picks (pro_picks_*.json) - BettingPros cheatsheet filters
- ODDS_API picks (odds_api_picks_*.json) - Pick6 multiplier strategy

Usage:
    # Single date
    python3 validate_predictions.py --date 2025-11-10

    # Date range
    python3 validate_predictions.py --start-date 2025-11-01 --end-date 2025-11-30

    # Specific system only
    python3 validate_predictions.py --start-date 2025-11-01 --end-date 2025-11-30 --system xl

    # Verbose output with all picks
    python3 validate_predictions.py --start-date 2025-11-01 --end-date 2025-11-30 --verbose
"""

import argparse
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import psycopg2

from nba.config.database import get_players_db_config

# Stat type -> game log column mapping
STAT_COLUMN_MAP = {
    "POINTS": "points",
    "REBOUNDS": "rebounds",
    "ASSISTS": "assists",
    "THREES": "three_pointers_made",
}

# Combo stat mappings
COMBO_STAT_MAP = {
    "PA": ("points", "assists"),
    "PR": ("points", "rebounds"),
    "RA": ("rebounds", "assists"),
    "PRA": ("points", "rebounds", "assists"),
}


def load_all_predictions(date: str, predictions_dir: str = "predictions") -> Dict[str, List]:
    """
    Load predictions from all sources (XL, PRO, ODDS_API).

    Returns dict with keys: 'xl', 'pro', 'odds_api', each containing list of picks.
    """
    date_compact = date.replace("-", "")
    result = {"xl": [], "pro": [], "odds_api": [], "two_energy": []}

    # XL picks
    xl_files = [
        Path(predictions_dir) / f"xl_picks_{date}.json",
        Path(predictions_dir) / f"xl_picks_{date_compact}.json",
    ]
    for filepath in xl_files:
        if filepath.exists():
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                for pick in data.get("picks", []):
                    # Capture model_version (xl or v3)
                    model_version = pick.get("model_version", "xl")
                    pick["_source"] = f"xl_{model_version}"  # e.g., "xl_xl" or "xl_v3"
                    pick["_model_version"] = model_version
                    pick["_game_date"] = date
                    pick["_filter"] = pick.get("filter_tier", "unknown")
                    # Normalize field names
                    if "best_line" not in pick and "line" in pick:
                        pick["best_line"] = pick["line"]
                    result["xl"].append(pick)
                break
            except (json.JSONDecodeError, KeyError):
                continue

    # PRO picks
    pro_files = [
        Path(predictions_dir) / f"pro_picks_{date}.json",
        Path(predictions_dir) / f"pro_picks_{date_compact}.json",
    ]
    for filepath in pro_files:
        if filepath.exists():
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                for pick in data.get("picks", []):
                    pick["_source"] = "pro"
                    pick["_game_date"] = date
                    pick["_filter"] = pick.get("filter_name", pick.get("filter_tier", "pro"))
                    # Normalize field names
                    if "best_line" not in pick:
                        pick["best_line"] = pick.get("line")
                    if "prediction" not in pick:
                        pick["prediction"] = pick.get("projection", pick.get("line", 0))
                    if "side" not in pick:
                        pick["side"] = "OVER"
                    result["pro"].append(pick)
                break
            except (json.JSONDecodeError, KeyError):
                continue

    # ODDS_API picks
    odds_files = [
        Path(predictions_dir) / f"odds_api_picks_{date}.json",
        Path(predictions_dir) / f"odds_api_picks_{date_compact}.json",
    ]
    for filepath in odds_files:
        if filepath.exists():
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                for pick in data.get("picks", []):
                    pick["_source"] = "odds_api"
                    pick["_game_date"] = date
                    pick["_filter"] = pick.get("filter_name", pick.get("source", "odds_api"))
                    # Normalize field names
                    if "best_line" not in pick:
                        pick["best_line"] = pick.get("line")
                    if "prediction" not in pick:
                        pick["prediction"] = pick.get("projection", pick.get("line", 0))
                    if "side" not in pick:
                        pick["side"] = "OVER"
                    result["odds_api"].append(pick)
                break
            except (json.JSONDecodeError, KeyError):
                continue

    # TWO_ENERGY picks
    two_energy_files = [
        Path(predictions_dir) / f"two_energy_picks_{date}.json",
        Path(predictions_dir) / f"two_energy_picks_{date_compact}.json",
    ]
    for filepath in two_energy_files:
        if filepath.exists():
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                for pick in data.get("picks", []):
                    pick["_source"] = "two_energy"
                    pick["_game_date"] = date
                    pick["_filter"] = pick.get("filter_name", pick.get("filter", "two_energy"))
                    # Normalize field names
                    if "best_line" not in pick:
                        pick["best_line"] = pick.get("line")
                    if "prediction" not in pick:
                        pick["prediction"] = pick.get("line", 0)
                    if "side" not in pick:
                        pick["side"] = pick.get("side", "OVER")
                    result["two_energy"].append(pick)
                break
            except (json.JSONDecodeError, KeyError):
                continue

    return result


def normalize_name(name: str) -> str:
    """Normalize player name for matching (remove Jr, III, etc.)"""
    suffixes = [" Jr", " Jr.", " III", " II", " IV", " Sr", " Sr."]
    normalized = name.strip()
    for suffix in suffixes:
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)].strip()
    return normalized.lower()


def get_actual_results(date: str) -> Dict[str, Dict]:
    """Get actual game results from database, keyed by normalized player name."""
    conn = psycopg2.connect(**get_players_db_config())
    cursor = conn.cursor()

    query = """
        SELECT
            p.full_name,
            l.points,
            l.rebounds,
            l.assists,
            l.three_pointers_made,
            l.opponent_abbrev
        FROM player_game_logs l
        JOIN player_profile p ON l.player_id = p.player_id
        WHERE l.game_date = %s
    """

    cursor.execute(query, (date,))
    results = {}

    for row in cursor.fetchall():
        name, points, rebounds, assists, threes, opponent = row
        normalized = normalize_name(name)

        stats = {
            "POINTS": points or 0,
            "REBOUNDS": rebounds or 0,
            "ASSISTS": assists or 0,
            "THREES": threes or 0,
            # Combo stats
            "PA": (points or 0) + (assists or 0),
            "PR": (points or 0) + (rebounds or 0),
            "RA": (rebounds or 0) + (assists or 0),
            "PRA": (points or 0) + (rebounds or 0) + (assists or 0),
        }

        results[normalized] = stats
        results[name] = stats  # Also store original name

    cursor.close()
    conn.close()

    return results


def validate_pick(pick: Dict, actuals: Dict) -> Dict:
    """Validate a single pick against actual results."""
    player = pick.get("player_name", "Unknown")
    stat_type = pick.get("stat_type", "POINTS")
    line = pick.get("best_line") or pick.get("line") or 0
    side = pick.get("side", "OVER")
    prediction = pick.get("prediction", line)

    # Try to find actual result
    actual = None
    normalized_player = normalize_name(player)

    if normalized_player in actuals:
        actual = actuals[normalized_player].get(stat_type)
    elif player in actuals:
        actual = actuals[player].get(stat_type)

    if actual is None:
        return {"status": "NO_DATA", "reason": f"No game log for {player}"}

    # Determine outcome
    if side == "OVER":
        won = actual > line
        push = actual == line
    else:
        won = actual < line
        push = actual == line

    if push:
        profit = 0.0
        outcome = "PUSH"
    elif won:
        profit = 0.909  # Win at -110
        outcome = "WIN"
    else:
        profit = -1.0
        outcome = "LOSS"

    return {
        "status": "VALIDATED",
        "outcome": outcome,
        "actual": actual,
        "line": line,
        "prediction": prediction,
        "diff": actual - line,
        "profit": profit,
    }


def validate_date_range(
    start_date: str,
    end_date: str,
    predictions_dir: str = "predictions",
    system_filter: Optional[str] = None,
    verbose: bool = False,
) -> Dict:
    """
    Validate predictions across a date range with detailed breakdowns.
    """
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()

    # Aggregate stats by system
    by_system: Dict[str, Dict] = defaultdict(
        lambda: {"wins": 0, "losses": 0, "pushes": 0, "profit": 0.0, "picks": []}
    )

    # Aggregate stats by system + market
    by_system_market: Dict[str, Dict[str, Dict]] = defaultdict(
        lambda: defaultdict(lambda: {"wins": 0, "losses": 0, "pushes": 0, "profit": 0.0})
    )

    # Aggregate stats by filter
    by_filter: Dict[str, Dict] = defaultdict(
        lambda: {"wins": 0, "losses": 0, "pushes": 0, "profit": 0.0, "system": None}
    )

    # Daily results by system
    daily_by_system: Dict[str, List] = defaultdict(list)

    # Aggregate stats by model version (xl vs v3 for XL picks)
    by_model_version: Dict[str, Dict] = defaultdict(
        lambda: {"wins": 0, "losses": 0, "pushes": 0, "profit": 0.0}
    )

    all_picks = []

    current = start
    while current <= end:
        date_str = current.strftime("%Y-%m-%d")

        predictions = load_all_predictions(date_str, predictions_dir)
        actuals = get_actual_results(date_str)

        if not actuals:
            current += timedelta(days=1)
            continue

        for system, picks in predictions.items():
            if system_filter and system != system_filter:
                continue

            day_wins = 0
            day_losses = 0
            day_profit = 0.0

            for pick in picks:
                validation = validate_pick(pick, actuals)

                if validation["status"] != "VALIDATED":
                    continue

                market = pick.get("stat_type", "UNKNOWN")
                filter_name = pick.get("_filter", "unknown")
                outcome = validation["outcome"]
                profit = validation["profit"]

                # Store pick detail
                pick_detail = {
                    "date": date_str,
                    "player": pick.get("player_name"),
                    "market": market,
                    "side": pick.get("side", "OVER"),
                    "line": validation["line"],
                    "actual": validation["actual"],
                    "prediction": validation["prediction"],
                    "outcome": outcome,
                    "profit": profit,
                    "system": system,
                    "filter": filter_name,
                }
                all_picks.append(pick_detail)
                by_system[system]["picks"].append(pick_detail)

                # Update aggregates
                if outcome == "WIN":
                    by_system[system]["wins"] += 1
                    by_system_market[system][market]["wins"] += 1
                    by_filter[filter_name]["wins"] += 1
                    day_wins += 1
                elif outcome == "LOSS":
                    by_system[system]["losses"] += 1
                    by_system_market[system][market]["losses"] += 1
                    by_filter[filter_name]["losses"] += 1
                    day_losses += 1
                else:  # PUSH
                    by_system[system]["pushes"] += 1
                    by_system_market[system][market]["pushes"] += 1
                    by_filter[filter_name]["pushes"] += 1

                by_system[system]["profit"] += profit
                by_system_market[system][market]["profit"] += profit
                by_filter[filter_name]["profit"] += profit
                by_filter[filter_name]["system"] = system
                day_profit += profit

                # Track by model version (for XL picks)
                model_version = pick.get("_model_version", "xl")
                if system == "xl":
                    if outcome == "WIN":
                        by_model_version[model_version]["wins"] += 1
                    elif outcome == "LOSS":
                        by_model_version[model_version]["losses"] += 1
                    else:
                        by_model_version[model_version]["pushes"] += 1
                    by_model_version[model_version]["profit"] += profit

            if day_wins + day_losses > 0:
                daily_by_system[system].append(
                    {
                        "date": date_str,
                        "wins": day_wins,
                        "losses": day_losses,
                        "profit": day_profit,
                    }
                )

        current += timedelta(days=1)

    # Print results
    # Determine which systems have data
    active_systems = [
        s
        for s in ["xl", "pro", "odds_api", "two_energy"]
        if by_system[s]["wins"] + by_system[s]["losses"] + by_system[s]["pushes"] > 0
    ]
    system_labels = [s.upper() for s in active_systems]

    print()
    print("  Running Validation")
    print(f"  Checking {', '.join(system_labels)} picks" if system_labels else "  No picks found")
    print()
    print("=" * 90)
    print(f"PICK VALIDATION REPORT: {start_date} to {end_date}")
    print("=" * 90)

    # Overall summary by system
    print()
    print("-" * 90)
    print("RESULTS BY SYSTEM")
    print("-" * 90)
    print(
        f"{'System':<12} {'Graded':<8} {'W':<6} {'L':<6} {'P':<4} {'Win Rate':<10} {'ROI':<10} {'Profit':<10}"
    )
    print("-" * 90)

    total_wins = 0
    total_losses = 0
    total_pushes = 0
    total_profit = 0.0

    for system in active_systems:
        stats = by_system[system]
        wins, losses, pushes = stats["wins"], stats["losses"], stats["pushes"]
        graded = wins + losses + pushes
        profit = stats["profit"]

        if graded > 0:
            wr = wins / graded * 100
            roi = profit / graded * 100
            print(
                f"{system.upper():<12} {graded:<8} {wins:<6} {losses:<6} {pushes:<4} {wr:>6.1f}%    {roi:>+6.1f}%    {profit:>+.2f}u"
            )

            total_wins += wins
            total_losses += losses
            total_pushes += pushes
            total_profit += profit

    # Total
    total_graded = total_wins + total_losses + total_pushes
    if total_graded > 0:
        total_wr = total_wins / total_graded * 100
        total_roi = total_profit / total_graded * 100
        print("-" * 90)
        print(
            f"{'TOTAL':<12} {total_graded:<8} {total_wins:<6} {total_losses:<6} {total_pushes:<4} {total_wr:>6.1f}%    {total_roi:>+6.1f}%    {total_profit:>+.2f}u"
        )

    # Results by model version (XL vs V3)
    if by_model_version:
        print("\n" + "-" * 90)
        print("RESULTS BY MODEL VERSION (XL picks only)")
        print("-" * 90)
        print(
            f"{'Model':<12} {'Graded':<8} {'W':<6} {'L':<6} {'P':<4} {'Win Rate':<10} {'ROI':<10} {'Profit':<10}"
        )
        print("-" * 90)

        for version in ["xl", "v3"]:
            stats = by_model_version.get(
                version, {"wins": 0, "losses": 0, "pushes": 0, "profit": 0.0}
            )
            wins, losses, pushes = stats["wins"], stats["losses"], stats["pushes"]
            graded = wins + losses + pushes
            profit = stats["profit"]

            if graded > 0:
                wr = wins / graded * 100
                roi = profit / graded * 100
                print(
                    f"{version.upper():<12} {graded:<8} {wins:<6} {losses:<6} {pushes:<4} {wr:>6.1f}%    {roi:>+6.1f}%    {profit:>+.2f}u"
                )

    # Results by system + market
    print("\n" + "-" * 90)
    print("RESULTS BY SYSTEM + MARKET")
    print("-" * 90)
    print(f"{'System':<10} {'Market':<12} {'W':<5} {'L':<5} {'P':<4} {'Win Rate':<10} {'ROI':<10}")
    print("-" * 90)

    for system in active_systems:
        markets = by_system_market[system]
        if not markets:
            continue
        for market in sorted(markets.keys()):
            stats = markets[market]
            wins, losses, pushes = stats["wins"], stats["losses"], stats["pushes"]
            graded = wins + losses + pushes
            if graded > 0:
                wr = wins / graded * 100
                roi = stats["profit"] / graded * 100
                print(
                    f"{system.upper():<10} {market:<12} {wins:<5} {losses:<5} {pushes:<4} {wr:>6.1f}%    {roi:>+6.1f}%"
                )

    # Results by filter
    print("\n" + "-" * 90)
    print("RESULTS BY FILTER/TIER")
    print("-" * 90)
    print(f"{'Filter':<30} {'System':<8} {'W':<5} {'L':<5} {'Win Rate':<10} {'ROI':<10}")
    print("-" * 90)

    # Sort filters by win rate (descending)
    sorted_filters = sorted(
        by_filter.items(),
        key=lambda x: (
            x[1]["wins"] / (x[1]["wins"] + x[1]["losses"])
            if (x[1]["wins"] + x[1]["losses"]) > 0
            else 0
        ),
        reverse=True,
    )

    for filter_name, stats in sorted_filters:
        wins, losses = stats["wins"], stats["losses"]
        graded = wins + losses + stats["pushes"]
        if graded >= 2:  # Only show filters with 2+ picks
            wr = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
            roi = stats["profit"] / graded * 100
            system = stats["system"] or "?"
            # Truncate long filter names
            display_name = filter_name[:28] + ".." if len(filter_name) > 30 else filter_name
            print(
                f"{display_name:<30} {system.upper():<8} {wins:<5} {losses:<5} {wr:>6.1f}%    {roi:>+6.1f}%"
            )

    # Daily breakdown by system
    print("\n" + "-" * 90)
    print("DAILY BREAKDOWN BY SYSTEM")
    print("-" * 90)

    for system in active_systems:
        daily = daily_by_system[system]
        if not daily:
            continue
        print(f"\n{system.upper()}:")
        for day in daily:
            total = day["wins"] + day["losses"]
            wr = day["wins"] / total * 100 if total > 0 else 0
            print(
                f"  {day['date']}: {day['wins']}W-{day['losses']}L ({wr:.0f}%) | {day['profit']:+.2f}u"
            )

    # Verbose: show all picks
    if verbose:
        print("\n" + "-" * 90)
        print("ALL VALIDATED PICKS")
        print("-" * 90)

        for pick in sorted(all_picks, key=lambda x: (x["date"], x["system"])):
            icon = "✓" if pick["outcome"] == "WIN" else ("✗" if pick["outcome"] == "LOSS" else "—")
            side_char = "O" if pick["side"] == "OVER" else "U"
            print(
                f"{icon} {pick['date']} [{pick['system'].upper():<7}] {pick['player'][:22]:<22} "
                f"{pick['market']:<10} {side_char}{pick['line']:<5} → {pick['actual']:<4} "
                f"[{pick['filter'][:20]}]"
            )

    print("\n" + "=" * 90)

    return {
        "by_system": dict(by_system),
        "by_system_market": dict(by_system_market),
        "by_filter": dict(by_filter),
        "all_picks": all_picks,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate predictions against actual results")
    parser.add_argument("--date", help="Single date to validate (YYYY-MM-DD)")
    parser.add_argument("--start-date", help="Start date for range validation (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date for range validation (YYYY-MM-DD)")
    parser.add_argument(
        "--predictions-dir",
        default="predictions",
        help="Directory containing prediction files (default: predictions/)",
    )
    parser.add_argument(
        "--system",
        choices=["xl", "pro", "odds_api", "two_energy"],
        help="Filter to specific system only",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show all individual picks",
    )
    args = parser.parse_args()

    if args.start_date and args.end_date:
        start = args.start_date
        end = args.end_date
    elif args.date:
        start = args.date
        end = args.date
    else:
        # Default to last 7 days
        today = datetime.now().date()
        end = today.strftime("%Y-%m-%d")
        start = (today - timedelta(days=7)).strftime("%Y-%m-%d")

    validate_date_range(
        start,
        end,
        args.predictions_dir,
        system_filter=args.system,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
