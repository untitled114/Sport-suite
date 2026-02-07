#!/usr/bin/env python3
"""
Historical DFS Props Backfill via The Odds API
==============================================
Fetches historical DFS multipliers from Pick6 AND Underdog for backtesting.

Uses the historical events + historical event odds endpoints.
Cost: 10 credits per market per region per event.
Adding multiple bookmakers (pick6,underdog) in same region = no extra cost!

Usage:
    python3 -m nba.betting_xl.fetchers.fetch_pick6_historical \
        --start 2026-01-02 --end 2026-01-14 \
        --output the-odds-api-data/historical_dfs_jan1_14.json
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import requests

logger = logging.getLogger(__name__)

API_KEY = os.getenv("ODDS_API_KEY")
BASE_URL = "https://api.the-odds-api.com"
SPORT = "basketball_nba"

# DFS bookmakers to fetch (same region = no extra cost)
BOOKMAKERS = ["pick6", "underdog"]

# Only POINTS + REBOUNDS + alternates to save credits (4 markets = 40 credits/event)
MARKETS_TO_FETCH = [
    "player_points",
    "player_points_alternate",
    "player_rebounds",
    "player_rebounds_alternate",
]

STAT_TYPE_MAP = {
    "player_points": "POINTS",
    "player_points_alternate": "POINTS",
    "player_rebounds": "REBOUNDS",
    "player_rebounds_alternate": "REBOUNDS",
}


def get_historical_events(date_str: str) -> List[Dict]:
    """
    Get NBA events for a historical date.
    Cost: 1 credit per call.
    """
    # Use 6pm ET (11pm UTC) to get full day's games
    date_iso = f"{date_str}T23:00:00Z"

    url = f"{BASE_URL}/v4/historical/sports/{SPORT}/events"
    params = {
        "apiKey": API_KEY,
        "date": date_iso,
    }

    resp = requests.get(url, params=params, timeout=30)
    remaining = resp.headers.get("x-requests-remaining", "?")
    cost = resp.headers.get("x-requests-last", "?")

    if resp.status_code != 200:
        logger.error(f"Events API error {resp.status_code}: {resp.text[:200]}")
        return []

    data = resp.json()
    events = data.get("data", [])
    timestamp = data.get("timestamp", "")

    logger.info(
        f"  [{date_str}] {len(events)} events (snapshot: {timestamp}, cost: {cost}, remaining: {remaining})"
    )
    return events


def get_historical_event_odds(event_id: str, date_str: str) -> Dict:
    """
    Get historical odds for a single event with DFS multipliers.
    Fetches from both Pick6 AND Underdog (same cost since same region).
    Cost: 10 × num_markets × num_regions credits.
    """
    date_iso = f"{date_str}T23:00:00Z"
    markets_str = ",".join(MARKETS_TO_FETCH)
    bookmakers_str = ",".join(BOOKMAKERS)

    url = f"{BASE_URL}/v4/historical/sports/{SPORT}/events/{event_id}/odds"
    params = {
        "apiKey": API_KEY,
        "date": date_iso,
        "regions": "us_dfs",
        "bookmakers": bookmakers_str,
        "markets": markets_str,
        "oddsFormat": "american",
        "includeMultipliers": "true",
    }

    resp = requests.get(url, params=params, timeout=30)
    remaining = resp.headers.get("x-requests-remaining", "?")
    cost = resp.headers.get("x-requests-last", "?")

    if resp.status_code != 200:
        logger.error(f"Event odds API error {resp.status_code} for {event_id}: {resp.text[:200]}")
        return {}

    data = resp.json()
    logger.debug(f"    Event {event_id}: cost={cost}, remaining={remaining}")
    return data.get("data", data)


def extract_alternate_multipliers(bookmaker_data: Dict) -> Dict:
    """Extract multipliers from alternate market data."""
    alt_multipliers = {}

    for market in bookmaker_data.get("markets", []):
        market_key = market.get("key", "")
        if "_alternate" not in market_key:
            continue

        stat_type = STAT_TYPE_MAP.get(market_key)
        if not stat_type:
            continue

        for outcome in market.get("outcomes", []):
            player_name = outcome.get("description", "")
            if not player_name:
                continue

            multiplier = outcome.get("multiplier")
            alt_line = outcome.get("point")

            if multiplier is not None:
                key = (player_name.lower(), stat_type)
                # Keep lowest multiplier (most favorable)
                if key not in alt_multipliers or multiplier < alt_multipliers[key]["multiplier"]:
                    alt_multipliers[key] = {
                        "multiplier": float(multiplier),
                        "alt_line": float(alt_line) if alt_line is not None else None,
                    }

    return alt_multipliers


def parse_event_props(event_data: Dict, game_date: str) -> List[Dict]:
    """Parse props from historical event odds response for ALL DFS bookmakers."""
    props = []

    home_team = event_data.get("home_team", "")
    away_team = event_data.get("away_team", "")
    event_id = event_data.get("id", "")

    for bookmaker in event_data.get("bookmakers", []):
        book_key = bookmaker.get("key", "")
        if book_key not in BOOKMAKERS:
            continue

        # First pass: extract alternate multipliers for this bookmaker
        alt_multipliers = extract_alternate_multipliers(bookmaker)

        # Track which players we've added from main markets for this book
        book_main_keys = set()

        # Second pass: build props from main markets
        for market in bookmaker.get("markets", []):
            market_key = market.get("key", "")
            if "_alternate" in market_key:
                continue  # Skip alternates in main pass

            stat_type = STAT_TYPE_MAP.get(market_key)
            if not stat_type:
                continue

            for outcome in market.get("outcomes", []):
                player_name = outcome.get("description", "")
                line = outcome.get("point")
                side = outcome.get("name", "")

                if not player_name or line is None:
                    continue
                if side.upper() != "OVER":
                    continue

                # Look up real multiplier from alternates
                alt_key = (player_name.lower(), stat_type)
                alt_info = alt_multipliers.get(alt_key)

                if alt_info:
                    multiplier = alt_info["multiplier"]
                else:
                    multiplier = float(outcome.get("multiplier", 1.0))

                props.append(
                    {
                        "player": player_name,
                        "stat_type": stat_type,
                        "line": float(line),
                        "multiplier": multiplier,
                        "book": book_key,
                        "game_date": game_date,
                        "home_team": home_team,
                        "away_team": away_team,
                        "event_id": event_id,
                    }
                )
                book_main_keys.add((player_name.lower(), stat_type))

        # Also add players only in alternates for this bookmaker
        for (name_lower, stat_type), alt_info in alt_multipliers.items():
            if (name_lower, stat_type) not in book_main_keys and alt_info.get(
                "alt_line"
            ) is not None:
                props.append(
                    {
                        "player": name_lower.title(),
                        "stat_type": stat_type,
                        "line": alt_info["alt_line"],
                        "multiplier": alt_info["multiplier"],
                        "book": book_key,
                        "game_date": game_date,
                        "home_team": home_team,
                        "away_team": away_team,
                        "event_id": event_id,
                        "alternate_only": True,
                    }
                )

    return props


def fetch_historical_range(start_date: str, end_date: str) -> List[Dict]:
    """
    Fetch Pick6 data for a date range using historical API.
    """
    all_props = []
    total_credits_used = 0

    current = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        logger.info(f"Fetching {date_str}...")

        # Get events for this date
        events = get_historical_events(date_str)
        total_credits_used += 1
        time.sleep(1)

        if not events:
            logger.info(f"  No events for {date_str}")
            current += timedelta(days=1)
            continue

        # Filter to NBA games on this date
        target_events = []
        for event in events:
            commence_time = event.get("commence_time", "")
            if commence_time:
                try:
                    utc_time = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
                    est_time = utc_time - timedelta(hours=5)
                    if est_time.strftime("%Y-%m-%d") == date_str:
                        target_events.append(event)
                except ValueError as e:
                    logger.debug(f"Skipping event with unparseable date '{commence_time}': {e}")

        logger.info(f"  {len(target_events)} games on {date_str}")

        # Fetch odds for each event
        day_props = []
        for event in target_events:
            event_id = event.get("id", "")
            if not event_id:
                continue

            event_odds = get_historical_event_odds(event_id, date_str)
            est_cost = len(MARKETS_TO_FETCH) * 10  # 10 per market per region
            total_credits_used += est_cost
            time.sleep(1)  # Rate limit

            if event_odds:
                props = parse_event_props(event_odds, date_str)
                day_props.extend(props)

        # Dedup by (player, stat_type, book) keeping lowest multiplier per book
        deduped = {}
        for p in day_props:
            key = (p["player"].lower(), p["stat_type"], p["game_date"], p.get("book", "pick6"))
            if key not in deduped or p["multiplier"] < deduped[key]["multiplier"]:
                deduped[key] = p
        day_props = list(deduped.values())

        logger.info(f"  {len(day_props)} props (deduped)")
        all_props.extend(day_props)
        current += timedelta(days=1)

    logger.info(f"\nTotal: {len(all_props)} props, ~{total_credits_used} credits used")
    return all_props


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fetch historical Pick6 data")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--output",
        default="the-odds-api-data/historical_dfs_backfill.json",
        help="Output JSON file",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Only count events, don't fetch odds"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if args.dry_run:
        logger.info("DRY RUN - counting events only (1 credit per day)")
        total_events = 0
        current = datetime.strptime(args.start, "%Y-%m-%d")
        end = datetime.strptime(args.end, "%Y-%m-%d")
        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            events = get_historical_events(date_str)
            target = []
            for event in events:
                ct = event.get("commence_time", "")
                if ct:
                    try:
                        utc = datetime.fromisoformat(ct.replace("Z", "+00:00"))
                        est = utc - timedelta(hours=5)
                        if est.strftime("%Y-%m-%d") == date_str:
                            target.append(event)
                    except (ValueError, AttributeError) as e:
                        logger.debug(f"Skipping event with unparseable date '{ct}': {e}")
            total_events += len(target)
            time.sleep(1)
            current += timedelta(days=1)

        est_cost = (
            total_events * len(MARKETS_TO_FETCH) * 10
            + (end - datetime.strptime(args.start, "%Y-%m-%d")).days
            + 1
        )
        logger.info(f"\nDry run results:")
        logger.info(f"  Total events: {total_events}")
        logger.info(f"  Markets per event: {len(MARKETS_TO_FETCH)}")
        logger.info(f"  Estimated credit cost: {est_cost}")
        return

    props = fetch_historical_range(args.start, args.end)

    if props:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(props, f, indent=2)

        # Summary by bookmaker
        book_counts = {}
        for p in props:
            book = p.get("book", "pick6")
            book_counts[book] = book_counts.get(book, 0) + 1

        logger.info(f"\nSaved {len(props)} props to {output_path}")
        logger.info("By bookmaker:")
        for book, count in sorted(book_counts.items()):
            logger.info(f"  {book}: {count} props")
    else:
        logger.warning("No props fetched")


if __name__ == "__main__":
    main()
