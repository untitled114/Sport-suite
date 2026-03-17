#!/usr/bin/env python3
"""
BettingPros Historical Analytics Backfill
==========================================
Queries /v3/props for every game date in the training window and stores
projections, hit rates, EV, bet rating, and opposition rank in the
bp_historical_analytics table for V4 training.

Also backfills DVP (defense vs position) data for seasons 2023, 2024, 2025
from the server-rendered BettingPros page.

Usage:
    # Full backfill (Oct 2023 → present)
    python backfill_bp_analytics.py

    # Specific date range
    python backfill_bp_analytics.py --start 2024-11-01 --end 2025-03-15

    # DVP only
    python backfill_bp_analytics.py --dvp-only

    # Props only (skip DVP)
    python backfill_bp_analytics.py --skip-dvp

    # Resume from last backfilled date
    python backfill_bp_analytics.py --resume

    # Dry run (count dates, don't fetch)
    python backfill_bp_analytics.py --dry-run
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import psycopg2
import psycopg2.extras
import requests

EST = ZoneInfo("America/New_York")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────

API_URL = "https://api.bettingpros.com/v3/props"
DVP_URL = "https://www.bettingpros.com/nba/defense-vs-position/"

MARKETS = {
    "POINTS": 156,
    "REBOUNDS": 157,
}

API_HEADERS = {
    "x-api-key": os.getenv("BETTINGPROS_API_KEY", ""),
    "x-level": "cHJlbWl1bQ==",
    "accept": "application/json",
    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 Chrome/125.0.0.0 Safari/537.36",
    "accept-encoding": "gzip, deflate",
}

WEB_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 Chrome/125.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
}

# BP uses non-standard abbreviations
TEAM_NORMALIZE = {
    "NOR": "NOP",
    "PHO": "PHX",
    "UTH": "UTA",
    "GS": "GSW",
    "SA": "SAS",
    "NY": "NYK",
    "NO": "NOP",
    "WSH": "WAS",
}

# Default training window
DEFAULT_START = "2023-10-24"  # Start of 2023-24 NBA season


def _connect():
    """Connect to nba_intelligence database."""
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("INTELLIGENCE_DB_PORT", "5539")),
        dbname="nba_intelligence",
        user=os.getenv("DB_USER", "mlb_user"),
        password=os.getenv("DB_PASSWORD", ""),
    )


# ──────────────────────────────────────────────────────────────────────
# PROPS API BACKFILL
# ──────────────────────────────────────────────────────────────────────


def _hit_rate(splits: Dict[str, int]) -> Optional[float]:
    """Compute over hit rate from BP performance splits."""
    over = splits.get("over", 0) or 0
    under = splits.get("under", 0) or 0
    push = splits.get("push", 0) or 0
    total = over + under + push
    if total == 0:
        return None
    return round(over / total, 3)


def _parse_prop(raw: Dict[str, Any], game_date: str, stat_type: str) -> Optional[Dict[str, Any]]:
    """Parse a single prop from the API response into a DB row."""
    participant = raw.get("participant", {})
    player_name = participant.get("name")
    if not player_name:
        return None

    performance = raw.get("performance") or {}
    projection = raw.get("projection") or {}
    opp_rank = (raw.get("extra") or {}).get("opposition_rank") or {}
    scoring = raw.get("scoring") or {}
    over = raw.get("over") or {}
    under = raw.get("under") or {}

    # Build hit rates
    hr_l1 = _hit_rate(performance.get("last_1", {}))
    hr_l5 = _hit_rate(performance.get("last_5", {}))
    hr_l10 = _hit_rate(performance.get("last_10", {}))
    hr_l15 = _hit_rate(performance.get("last_15", {}))
    hr_l20 = _hit_rate(performance.get("last_20", {}))
    hr_season = _hit_rate(performance.get("season", {}))

    return {
        "player_name": player_name,
        "stat_type": stat_type,
        "game_date": game_date,
        "bp_projection": projection.get("value"),
        "bp_projection_diff": projection.get("diff"),
        "bp_probability": projection.get("probability"),
        "bp_expected_value": projection.get("expected_value"),
        "bp_bet_rating": projection.get("bet_rating"),
        "bp_recommended_side": projection.get("recommended_side"),
        "bp_opposition_rank": opp_rank.get("rank"),
        "bp_opposition_value": opp_rank.get("value"),
        "bp_hit_rate_L1": hr_l1,
        "bp_hit_rate_L5": hr_l5,
        "bp_hit_rate_L10": hr_l10,
        "bp_hit_rate_L15": hr_l15,
        "bp_hit_rate_L20": hr_l20,
        "bp_hit_rate_season": hr_season,
        "bp_over_line": over.get("line"),
        "bp_consensus_line": over.get("consensus_line"),
        "bp_over_odds": over.get("odds"),
        "bp_consensus_odds": over.get("consensus_odds"),
        "bp_actual_value": scoring.get("actual"),
        "bp_is_scored": scoring.get("is_scored", False),
    }


def fetch_date_market(date_str: str, market_name: str, market_id: int) -> List[Dict[str, Any]]:
    """Fetch all props for a given date and market from BP API."""
    records = []
    page = 1

    while True:
        params = {
            "sport": "NBA",
            "date": date_str,
            "market_id": market_id,
            "limit": 500,
            "page": page,
            "include_markets": "true",
            "include_counts": "true",
        }

        try:
            r = requests.get(API_URL, params=params, headers=API_HEADERS, timeout=30)
            r.raise_for_status()
            payload = r.json()
        except (requests.RequestException, ValueError) as e:
            logger.warning(f"  API error for {date_str} {market_name} page {page}: {e}")
            break

        props = payload.get("props", [])
        if not props:
            break

        for raw in props:
            parsed = _parse_prop(raw, date_str, market_name)
            if parsed:
                records.append(parsed)

        pagination = payload.get("_pagination", {})
        total_pages = pagination.get("total_pages", 1)
        if page >= total_pages:
            break
        page += 1
        time.sleep(0.3)  # Sub-page rate limit

    return records


def insert_records(conn, records: List[Dict[str, Any]]) -> int:
    """Insert records into bp_historical_analytics with ON CONFLICT skip."""
    if not records:
        return 0

    sql = """
        INSERT INTO bp_historical_analytics (
            player_name, stat_type, game_date,
            bp_projection, bp_projection_diff, bp_probability,
            bp_expected_value, bp_bet_rating, bp_recommended_side,
            bp_opposition_rank, bp_opposition_value,
            bp_hit_rate_L1, bp_hit_rate_L5, bp_hit_rate_L10,
            bp_hit_rate_L15, bp_hit_rate_L20, bp_hit_rate_season,
            bp_over_line, bp_consensus_line, bp_over_odds, bp_consensus_odds,
            bp_actual_value, bp_is_scored
        ) VALUES (
            %(player_name)s, %(stat_type)s, %(game_date)s,
            %(bp_projection)s, %(bp_projection_diff)s, %(bp_probability)s,
            %(bp_expected_value)s, %(bp_bet_rating)s, %(bp_recommended_side)s,
            %(bp_opposition_rank)s, %(bp_opposition_value)s,
            %(bp_hit_rate_L1)s, %(bp_hit_rate_L5)s, %(bp_hit_rate_L10)s,
            %(bp_hit_rate_L15)s, %(bp_hit_rate_L20)s, %(bp_hit_rate_season)s,
            %(bp_over_line)s, %(bp_consensus_line)s, %(bp_over_odds)s, %(bp_consensus_odds)s,
            %(bp_actual_value)s, %(bp_is_scored)s
        )
        ON CONFLICT (player_name, game_date, stat_type) DO NOTHING
    """

    inserted = 0
    with conn.cursor() as cur:
        for rec in records:
            cur.execute(sql, rec)
            inserted += cur.rowcount
    conn.commit()
    return inserted


def get_last_backfilled_date(conn) -> Optional[str]:
    """Get the most recent game_date in bp_historical_analytics."""
    with conn.cursor() as cur:
        cur.execute("SELECT MAX(game_date) FROM bp_historical_analytics")
        row = cur.fetchone()
        if row and row[0]:
            return str(row[0])
    return None


def get_game_dates(conn, start: str, end: str) -> List[str]:
    """Get distinct game dates from nba_props_xl for the backfill window."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT game_date
            FROM nba_props_xl
            WHERE game_date >= %s AND game_date <= %s
              AND stat_type IN ('POINTS', 'REBOUNDS')
              AND actual_value IS NOT NULL
            ORDER BY game_date
            """,
            (start, end),
        )
        return [str(row[0]) for row in cur.fetchall()]


def backfill_props(
    start: str,
    end: str,
    resume: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run the full props API backfill."""
    conn = _connect()

    if resume:
        last = get_last_backfilled_date(conn)
        if last:
            # Start from the day after the last backfilled date
            resume_date = (datetime.strptime(last, "%Y-%m-%d") + timedelta(days=1)).strftime(
                "%Y-%m-%d"
            )
            if resume_date > start:
                start = resume_date
                logger.info(f"Resuming from {start} (last backfilled: {last})")

    game_dates = get_game_dates(conn, start, end)
    logger.info(f"Found {len(game_dates)} game dates from {start} to {end}")

    if dry_run:
        est_calls = len(game_dates) * len(MARKETS)
        est_minutes = est_calls / 60  # ~1 call/sec
        logger.info(f"Dry run: {est_calls} API calls, ~{est_minutes:.0f} minutes")
        conn.close()
        return {"dates": len(game_dates), "est_calls": est_calls}

    total_inserted = 0
    total_fetched = 0
    errors = 0

    for i, date_str in enumerate(game_dates):
        date_records = []

        for market_name, market_id in MARKETS.items():
            try:
                records = fetch_date_market(date_str, market_name, market_id)
                date_records.extend(records)
                total_fetched += len(records)
            except Exception as e:
                logger.error(f"  Error fetching {date_str} {market_name}: {e}")
                errors += 1

            time.sleep(1.0)  # Rate limit between API calls

        if date_records:
            inserted = insert_records(conn, date_records)
            total_inserted += inserted

        if (i + 1) % 10 == 0 or i == len(game_dates) - 1:
            logger.info(
                f"  [{i+1}/{len(game_dates)}] {date_str}: "
                f"{len(date_records)} fetched, {total_inserted} total inserted"
            )

    conn.close()

    logger.info(
        f"\nBackfill complete: {total_fetched} props fetched, "
        f"{total_inserted} inserted, {errors} errors"
    )
    return {
        "dates": len(game_dates),
        "fetched": total_fetched,
        "inserted": total_inserted,
        "errors": errors,
    }


# ──────────────────────────────────────────────────────────────────────
# DVP BACKFILL
# ──────────────────────────────────────────────────────────────────────


def _extract_json_at(text: str, start: int) -> Any:
    """Extract JSON value starting at position using JSONDecoder."""
    decoder = json.JSONDecoder()
    obj, _ = decoder.raw_decode(text, start)
    return obj


def fetch_dvp_season(season: int) -> Dict[str, Any]:
    """Fetch DVP data for a specific season from BP website."""
    url = f"{DVP_URL}?season={season}"
    r = requests.get(url, headers=WEB_HEADERS, timeout=20)
    r.raise_for_status()

    idx = r.text.find("teamStats:")
    if idx < 0:
        logger.error(f"DVP: teamStats not found for season {season}")
        return {}

    start = r.text.index("{", idx)
    raw_teams = _extract_json_at(r.text, start)

    # Extract avg games
    avg_games = None
    ag_idx = r.text.find("avgGamesPlayed:")
    if ag_idx > 0:
        try:
            colon = r.text.index(":", ag_idx)
            comma = r.text.index(",", colon)
            avg_games = int(r.text[colon + 1 : comma].strip())
        except (ValueError, IndexError):
            pass

    # Normalize team abbreviations
    teams = {}
    for abbrev, positions in raw_teams.items():
        teams[TEAM_NORMALIZE.get(abbrev, abbrev)] = positions

    logger.info(f"DVP season {season}: {len(teams)} teams, avg {avg_games} games")
    return teams


def insert_dvp(conn, season: int, teams: Dict[str, Any]) -> int:
    """Insert DVP data into bp_dvp_historical."""
    sql = """
        INSERT INTO bp_dvp_historical (season, team, position, stat_name, value)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (season, team, position, stat_name)
        DO UPDATE SET value = EXCLUDED.value, fetched_at = CURRENT_TIMESTAMP
    """

    inserted = 0
    with conn.cursor() as cur:
        for team, positions in teams.items():
            for pos_name, stats in positions.items():
                if not isinstance(stats, dict):
                    continue
                for stat_name, value in stats.items():
                    if value is None:
                        continue
                    try:
                        cur.execute(sql, (season, team, pos_name, stat_name, float(value)))
                        inserted += 1
                    except (ValueError, TypeError):
                        continue
    conn.commit()
    return inserted


def backfill_dvp(seasons: Optional[List[int]] = None) -> Dict[str, Any]:
    """Backfill DVP data for specified seasons."""
    if seasons is None:
        seasons = [2023, 2024, 2025]

    conn = _connect()
    total = 0

    for season in seasons:
        try:
            teams = fetch_dvp_season(season)
            if teams:
                count = insert_dvp(conn, season, teams)
                total += count
                logger.info(f"  Season {season}: {count} DVP values inserted")
            time.sleep(2.0)  # Be polite between season fetches
        except Exception as e:
            logger.error(f"  DVP error for season {season}: {e}")

    conn.close()
    logger.info(f"\nDVP backfill complete: {total} total values")
    return {"seasons": len(seasons), "values": total}


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Backfill BettingPros historical analytics for V4")
    parser.add_argument("--start", default=DEFAULT_START, help="Start date (YYYY-MM-DD)")
    parser.add_argument(
        "--end",
        default=datetime.now(EST).strftime("%Y-%m-%d"),
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from last backfilled date")
    parser.add_argument("--dry-run", action="store_true", help="Count dates without fetching")
    parser.add_argument("--dvp-only", action="store_true", help="Only backfill DVP data")
    parser.add_argument("--skip-dvp", action="store_true", help="Skip DVP, only do props API")
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=[2023, 2024, 2025],
        help="DVP seasons to backfill",
    )
    args = parser.parse_args()

    if not API_HEADERS["x-api-key"]:
        logger.error("BETTINGPROS_API_KEY not set in environment")
        sys.exit(1)

    print("=" * 70)
    print("BettingPros Historical Analytics Backfill")
    print("=" * 70)

    results = {}

    if not args.dvp_only:
        print(f"\n--- Props API Backfill ({args.start} → {args.end}) ---")
        results["props"] = backfill_props(
            start=args.start,
            end=args.end,
            resume=args.resume,
            dry_run=args.dry_run,
        )

    if not args.skip_dvp and not args.dry_run:
        print(f"\n--- DVP Backfill (seasons: {args.seasons}) ---")
        results["dvp"] = backfill_dvp(seasons=args.seasons)

    print("\n" + "=" * 70)
    print("Results:")
    for key, val in results.items():
        print(f"  {key}: {val}")
    print("=" * 70)


if __name__ == "__main__":
    main()
