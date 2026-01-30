#!/usr/bin/env python3
"""
Show Today's Picks - Clean, actionable output
Usage: python3 show_picks.py [--date YYYY-MM-DD]
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PREDICTIONS_DIR = Path(__file__).parent / "predictions"


def load_json(filepath):
    """Load JSON file if it exists."""
    if filepath.exists():
        with open(filepath) as f:
            return json.load(f)
    return None


def print_header(text):
    """Print section header."""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print("=" * 60)


def show_xl_picks(date_str):
    """Display XL model picks."""
    filepath = PREDICTIONS_DIR / f"xl_picks_{date_str}.json"
    data = load_json(filepath)

    if not data or not data.get("picks"):
        print("\n  No XL picks for today.")
        return

    print_header(f"XL MODEL PICKS ({len(data['picks'])})")
    print(f"  Strategy: {data.get('strategy', 'N/A')}")
    print(f"  Markets: {', '.join(data.get('markets_enabled', []))}")
    print()

    for pick in data["picks"]:
        player = pick["player_name"]
        stat = pick["stat_type"]
        side = pick["side"]
        line = pick["best_line"]
        book = pick["best_book"].upper()
        edge = pick["edge_pct"]
        pred = pick["prediction"]
        opp = pick.get("opponent_team", "?")
        home = "vs" if pick.get("is_home") else "@"

        print(f"  {player}")
        print(f"    {stat} {side} {line} ({book})")
        print(f"    Prediction: {pred:.1f} | Edge: {edge:+.1f}% | {home} {opp}")
        print()


def show_pro_picks(date_str):
    """Display pro/cheatsheet picks."""
    filepath = PREDICTIONS_DIR / f"pro_picks_{date_str}.json"
    data = load_json(filepath)

    if not data or not data.get("picks"):
        print("\n  No pro picks for today.")
        return

    print_header(f"PRO PICKS ({len(data['picks'])})")

    for pick in data["picks"]:
        player = pick.get("player_name", "Unknown")
        stat = pick.get("stat_type", "?")
        side = pick.get("side", "?")
        line = pick.get("line", "?")
        reason = pick.get("reason", pick.get("filter", ""))

        print(f"  {player}: {stat} {side} {line}")
        if reason:
            print(f"    Reason: {reason}")
    print()


def show_summary(date_str):
    """Display daily summary."""
    filepath = PREDICTIONS_DIR / f"daily_summary_{date_str}.json"
    data = load_json(filepath)

    if not data:
        return

    print_header("SUMMARY")

    xl = data.get("xl_predictions", {})
    pro = data.get("pro_picks", {})
    odds = data.get("odds_api_picks", {})

    print(f"  XL Model:    {xl.get('total', 0)} picks")
    print(f"  Pro Picks:   {pro.get('total', 0)} picks")
    print(f"  Odds API:    {odds.get('total', 0)} picks")
    print(f"  {'─'*30}")
    print(f"  TOTAL:       {data.get('total_all_sources', 0)} picks")
    print()


def main():
    parser = argparse.ArgumentParser(description="Show today's betting picks")
    parser.add_argument("--date", "-d", help="Date (YYYY-MM-DD)", default=None)
    parser.add_argument("--xl-only", action="store_true", help="Show only XL picks")
    parser.add_argument("--pro-only", action="store_true", help="Show only pro picks")
    args = parser.parse_args()

    date_str = args.date or datetime.now().strftime("%Y-%m-%d")

    print(f"\n  NBA PICKS FOR {date_str}")
    print(f"  Generated: {datetime.now().strftime('%I:%M %p')}")

    if args.xl_only:
        show_xl_picks(date_str)
    elif args.pro_only:
        show_pro_picks(date_str)
    else:
        show_xl_picks(date_str)
        show_pro_picks(date_str)
        show_summary(date_str)

    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
