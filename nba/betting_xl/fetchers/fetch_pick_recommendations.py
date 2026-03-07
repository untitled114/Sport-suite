#!/usr/bin/env python3
"""
BettingPros Pick Recommendations Fetcher

Session-gated endpoint that returns BP's curated shelf picks for the day.
Each pick includes BP's projection, probability, EV, star rating, and
per-book market_ev — a second independent signal alongside our XL models.

Requires (refreshed by refresh_bp_session.py):
    BETTINGPROS_SESSION_ID   - sessionid cookie
    BETTINGPROS_WEB_API_KEY  - X-Api-Key header (session-generated)
    BETTINGPROS_API_KEY      - personal API key (URL param, stable)

Output: nba/betting_xl/lines/bp_pick_recommendations_{date}.json

Usage:
    python3 nba/betting_xl/fetchers/fetch_pick_recommendations.py
    python3 nba/betting_xl/fetchers/fetch_pick_recommendations.py --date 2026-03-07
    python3 nba/betting_xl/fetchers/fetch_pick_recommendations.py --quiet
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo

log = logging.getLogger(__name__)

_EST = ZoneInfo("America/New_York")

_API_BASE = "https://api.bettingpros.com"
_ENDPOINT = "/v3/pick-recommendations"

# Market ID -> stat type mapping (same as rest of codebase)
_MARKET_STAT = {
    156: "POINTS",
    157: "REBOUNDS",
    151: "ASSISTS",
    162: "THREES",
    160: "STEALS",
    163: "STEALS",
    152: "BLOCKS",
    164: "BLOCKS",
    # Combo markets
    335: "PTS+AST",
    337: "REB+AST",
    129: "GAME_TOTAL",  # team total points
    1261: "PTS+REB+AST",
    1260: "PTS+REB",
    1259: "PTS+AST",
    1258: "REB+AST",
}

_BOOK_NAME = {
    12: "draftkings",
    10: "fanduel",
    19: "betmgm",
    13: "caesars",
    14: "fanatics",
    18: "betrivers",
    33: "espnbet",
    36: "underdog",
    37: "prizepicks",
}


def _get_credentials() -> tuple[str, str, str, str]:
    """Return (session_id, web_api_key, personal_api_key, user_id). Raises if missing."""
    session_id = os.environ.get("BETTINGPROS_SESSION_ID", "")
    web_api_key = os.environ.get("BETTINGPROS_WEB_API_KEY", "")
    personal_key = os.environ.get("BETTINGPROS_API_KEY", "")
    user_id = os.environ.get("BETTINGPROS_USER_ID", "4151919")
    return session_id, web_api_key, personal_key, user_id


def fetch_pick_recommendations(date: str) -> list[dict[str, Any]]:
    """
    Fetch BP pick recommendations for a given date.

    Returns a list of flat pick records. Returns empty list on auth failure
    or any error (non-critical — caller should log and continue).
    """
    session_id, web_api_key, personal_key, user_id = _get_credentials()

    if not session_id or not web_api_key:
        log.warning(
            "BETTINGPROS_SESSION_ID or BETTINGPROS_WEB_API_KEY not set — "
            "run refresh_bp_session.py to capture credentials"
        )
        return []

    params = urllib.parse.urlencode(
        {
            "v": "4",
            "fence": "web",
            "user": user_id,
            "key": personal_key,
            "sport": "NBA",
            "mode": "feed",
            "date": date,
        }
    )
    url = f"{_API_BASE}{_ENDPOINT}?{params}"

    req = urllib.request.Request(url)
    req.add_header("X-Api-Key", web_api_key)
    req.add_header("X-Level", '"cHJlbWl1bQ=="')
    req.add_header("Cookie", f"sessionid={session_id}")
    req.add_header("Accept", "application/json")
    req.add_header(
        "User-Agent",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:  # nosec B310
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:200]
        if e.code == 401 or e.code == 403:
            log.warning(
                f"BP pick-recommendations auth failed (HTTP {e.code}) — "
                f"session may be expired. Re-run refresh_bp_session.py. Body: {body}"
            )
        else:
            log.error(f"BP pick-recommendations HTTP {e.code}: {body}")
        return []
    except Exception as exc:
        log.error(f"BP pick-recommendations request failed: {exc}")
        return []

    picks = []
    for shelf in data.get("shelves", []):
        shelf_label = shelf.get("name", "") or shelf.get("label", "")
        for pick in shelf.get("picks", []):
            record = _parse_pick(pick, shelf_label, date)
            if record:
                picks.append(record)

    return picks


def _parse_pick(pick: dict, shelf_label: str, date: str) -> Optional[dict[str, Any]]:
    """Parse a single pick recommendation into a flat record.

    Response structure (v4 feed):
        pick.offer.participants[0].name  - player name
        pick.offer.market_id             - stat market
        pick.offer.selections[]          - per-side (over/under) data with per-book lines
        pick.expected_value              - BP's EV for the recommended bet
        pick.star_rating                 - 1-5 stars
        pick.justification[]             - model projection, hit rates with key/value/over_under
        pick.other_factors[]             - additional context
    """
    offer = pick.get("offer") or {}

    # Player name
    participants = offer.get("participants") or []
    if not participants:
        return None
    participant = participants[0]
    player_name = participant.get("name", "").strip()
    if not player_name:
        return None

    market_id = offer.get("market_id")
    stat_type = _MARKET_STAT.get(market_id, f"MARKET_{market_id}")

    # Recommended side from justification (most reliable signal)
    justification = pick.get("justification") or []
    rec_side = None
    bp_projection_value = None
    for j in justification:
        if j.get("key") == "model_projection":
            over_under = j.get("over_under", "")
            rec_side = over_under.lower() if over_under else None
            bp_projection_value = j.get("value")
            break

    # Find the matching selection (over or under) to get line + per-book data
    selections = offer.get("selections") or []
    matched_selection = None
    for sel in selections:
        if sel.get("selection") == rec_side:
            matched_selection = sel
            break
    if matched_selection is None and selections:
        matched_selection = selections[0]  # fallback: first selection

    # Consensus/opening line
    opening = (matched_selection or {}).get("opening_line") or {}
    line = opening.get("line")

    # Per-book market_ev from selection.books[*].lines[0].metrics
    book_ev_map: dict[str, float] = {}
    best_ev: Optional[float] = None
    best_book_id: Optional[int] = None
    best_book_line: Optional[float] = None

    for book_entry in (matched_selection or {}).get("books") or []:
        bid = book_entry.get("id")
        if bid is None or bid == 0:  # id=0 is consensus
            continue
        for line_entry in book_entry.get("lines") or []:
            metrics = line_entry.get("metrics") or {}
            ev = metrics.get("market_ev")
            book_line = line_entry.get("line")
            if ev is not None:
                book_name = _BOOK_NAME.get(bid, str(bid))
                book_ev_map[book_name] = ev
                if best_ev is None or ev > best_ev:
                    best_ev = ev
                    best_book_id = bid
                    best_book_line = book_line
            break  # only main line per book

    # Hit rate from other_factors (key == "hit_rate")
    hit_rate_raw: Optional[dict] = None
    for factor in pick.get("other_factors") or []:
        if factor.get("key") == "hit_rate":
            hit_rate_raw = factor.get("value") or {}
            break

    return {
        "player_name": player_name,
        "player_id": participant.get("id"),
        "player_slug": (participant.get("player") or {}).get("slug"),
        "stat_type": stat_type,
        "market_id": market_id,
        "event_id": offer.get("event_id"),
        "game_date": date,
        "shelf_label": shelf_label,
        "line": float(line) if line is not None else None,
        # BP model signals
        "bp_expected_value": (
            float(pick["expected_value"]) if pick.get("expected_value") is not None else None
        ),
        "bp_bet_rating": int(pick["star_rating"]) if pick.get("star_rating") is not None else None,
        "bp_recommended_side": rec_side,
        "bp_projection": float(bp_projection_value) if bp_projection_value is not None else None,
        # Best book
        "best_ev": float(best_ev) if best_ev is not None else None,
        "best_book": _BOOK_NAME.get(best_book_id, str(best_book_id)) if best_book_id else None,
        "best_book_line": float(best_book_line) if best_book_line is not None else None,
        "book_ev_map": book_ev_map,
        # Hit rates from other_factors
        "hit_rate_L15": (
            round(hit_rate_raw["wins"] / hit_rate_raw["games"], 3)
            if hit_rate_raw and hit_rate_raw.get("games")
            else None
        ),
        "hit_rate_wins": hit_rate_raw.get("wins") if hit_rate_raw else None,
        "hit_rate_losses": hit_rate_raw.get("losses") if hit_rate_raw else None,
        "hit_rate_games": hit_rate_raw.get("games") if hit_rate_raw else None,
    }


def save_recommendations(date: str, picks: list[dict]) -> Path:
    """Save picks to lines/ directory. Returns output path."""
    output_dir = Path(__file__).parent.parent / "lines"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"bp_pick_recommendations_{date}.json"

    payload = {
        "date": date,
        "generated_at": datetime.now(_EST).isoformat(),
        "total_picks": len(picks),
        "picks": picks,
    }

    with open(output_file, "w") as f:
        json.dump(payload, f, indent=2)

    return output_file


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Fetch BettingPros pick recommendations")
    parser.add_argument("--date", help="Date YYYY-MM-DD (default: today EST)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    date = args.date or datetime.now(_EST).strftime("%Y-%m-%d")

    picks = fetch_pick_recommendations(date)

    if not picks:
        print(f"No pick recommendations returned for {date}")
        return

    output_file = save_recommendations(date, picks)
    print(f"Saved {len(picks)} pick recommendations to {output_file}")

    if not args.quiet:
        # Print summary by stat type
        from collections import Counter

        stat_counts = Counter(p["stat_type"] for p in picks)
        for stat, count in sorted(stat_counts.items()):
            print(f"  {stat}: {count}")


if __name__ == "__main__":
    main()
