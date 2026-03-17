#!/usr/bin/env python3
"""
BettingPros Analytics Scraper
==============================
Scrapes server-rendered analytics data from BettingPros that is NOT available
via their API. Three data sources:

1. Defense vs Position (DVP) — per-team defensive stats allowed per position
2. League Trends — ATS/ML/Total records by situation (Home Dogs, Road Faves, etc.)
3. Matchups — game-level odds, handle %, expert/system picks, opening lines

All data is embedded in JavaScript variables on public pages.
No API key or auth required.

Usage:
    python fetch_bettingpros_analytics.py                    # Fetch all
    python fetch_bettingpros_analytics.py --page dvp         # DVP only
    python fetch_bettingpros_analytics.py --page trends      # League trends only
    python fetch_bettingpros_analytics.py --page matchups    # Matchups only
    python fetch_bettingpros_analytics.py --save             # Save to JSON
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import requests

EST = ZoneInfo("America/New_York")

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 Chrome/125.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
}

MARKET_NAMES = {127: "moneyline", 128: "total", 129: "spread"}

# Team abbreviation normalization (BP uses NOR, PHO, UTH)
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

OUTPUT_DIR = Path(__file__).parent.parent / "lines"


def _normalize_team(abbrev: str) -> str:
    """Normalize BP team abbreviation to canonical NBA format."""
    return TEAM_NORMALIZE.get(abbrev, abbrev)


def _extract_json_at(text: str, start: int) -> Any:
    """Extract a JSON value starting at the given position using JSONDecoder."""
    decoder = json.JSONDecoder()
    obj, _ = decoder.raw_decode(text, start)
    return obj


# ──────────────────────────────────────────────────────────────────────
# 1. DEFENSE VS POSITION
# ──────────────────────────────────────────────────────────────────────


def fetch_dvp() -> Dict[str, Any]:
    """Fetch Defense vs Position data.

    Returns per-team, per-position defensive stats allowed:
        30 teams x 7 positions (ALL, PG, SG, SF, PF, C, TM) x 9 stats

    Stats: points, rebounds, assists, three_points_made, steals, blocks,
           turnovers, free_throw_perc, field_goals_perc
    """
    url = "https://www.bettingpros.com/nba/defense-vs-position/"
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()

    # Data is in: const bpDefenseVsPositionStats = { ..., teamStats: {...} }
    idx = r.text.find("teamStats:")
    if idx < 0:
        logger.error("DVP: teamStats not found in page")
        return {"teams": {}, "error": "teamStats not found"}

    start = r.text.index("{", idx)
    raw_teams = _extract_json_at(r.text, start)

    # Normalize team abbreviations
    teams = {}
    for abbrev, positions in raw_teams.items():
        teams[_normalize_team(abbrev)] = positions

    # Extract metadata
    season = None
    avg_games = None
    sp_idx = r.text.find("seasonParam:")
    if sp_idx > 0:
        try:
            colon = r.text.index(":", sp_idx)
            comma = r.text.index(",", colon)
            season = int(r.text[colon + 1 : comma].strip())
        except (ValueError, IndexError):
            pass
    ag_idx = r.text.find("avgGamesPlayed:")
    if ag_idx > 0:
        try:
            colon = r.text.index(":", ag_idx)
            comma = r.text.index(",", colon)
            avg_games = int(r.text[colon + 1 : comma].strip())
        except (ValueError, IndexError):
            pass

    result = {
        "source": "bettingpros_dvp",
        "fetch_timestamp": datetime.now(EST).isoformat(),
        "date": datetime.now(EST).strftime("%Y-%m-%d"),
        "season": season,
        "avg_games_played": avg_games,
        "total_teams": len(teams),
        "positions": ["ALL", "PG", "SG", "SF", "PF", "C", "TM"],
        "stats": [
            "points",
            "rebounds",
            "assists",
            "three_points_made",
            "steals",
            "blocks",
            "turnovers",
            "free_throw_perc",
            "field_goals_perc",
        ],
        "teams": teams,
    }

    return result


# ──────────────────────────────────────────────────────────────────────
# 2. LEAGUE TRENDS
# ──────────────────────────────────────────────────────────────────────


def fetch_league_trends() -> Dict[str, Any]:
    """Fetch League Trends data.

    Returns 17 situational betting categories with:
        - ATS/Spread records + net units
        - Moneyline records + net units
        - Total (O/U) records + net units
        - Per-team breakdowns within each category
        - Upcoming game counts
    """
    url = "https://www.bettingpros.com/nba/league-trends/"
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()

    idx = r.text.find("trends: [", r.text.find("bpLeagueTrendsData"))
    if idx < 0:
        logger.error("League Trends: trends array not found")
        return {"trends": [], "error": "trends not found"}

    start = r.text.index("[", idx)
    trends = _extract_json_at(r.text, start)

    # Normalize team abbreviations in per-team breakdowns
    for trend in trends:
        if "teams" in trend and isinstance(trend["teams"], dict):
            normalized = {}
            for abbrev, data in trend["teams"].items():
                normalized[_normalize_team(abbrev)] = data
            trend["teams"] = normalized

    # Extract date context
    date_str = None
    di = r.text.find("weekOrDate:")
    if di > 0:
        try:
            q1 = r.text.index("'", di)
            q2 = r.text.index("'", q1 + 1)
            date_str = r.text[q1 + 1 : q2]
        except (ValueError, IndexError):
            pass

    return {
        "source": "bettingpros_league_trends",
        "fetch_timestamp": datetime.now(EST).isoformat(),
        "date": date_str or datetime.now(EST).strftime("%Y-%m-%d"),
        "total_categories": len(trends),
        "trends": trends,
    }


# ──────────────────────────────────────────────────────────────────────
# 3. MATCHUPS (game odds, handle %, expert picks)
# ──────────────────────────────────────────────────────────────────────


def fetch_matchups() -> Dict[str, Any]:
    """Fetch Matchups data.

    Returns per-game:
        - Spread, moneyline, total lines per book
        - Opening vs current lines (line movement)
        - Handle % (% of money on each side)
        - Expert picks, system picks, combined picks counts
        - BP projections, cover probability, EV, bet rating
    """
    url = "https://www.bettingpros.com/nba/matchups/"
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()

    # Extract events
    bp_idx = r.text.find("bp_matchup")
    if bp_idx < 0:
        bp_idx = 0

    ev_idx = r.text.find("events: [", bp_idx)
    if ev_idx < 0:
        logger.error("Matchups: events array not found")
        return {"games": [], "error": "events not found"}

    ev_start = r.text.index("[", ev_idx)
    events = _extract_json_at(r.text, ev_start)

    # Extract offers (odds + handle + picks)
    of_idx = r.text.find("offers: [", bp_idx)
    offers = []
    if of_idx > 0:
        of_start = r.text.index("[", of_idx)
        offers = _extract_json_at(r.text, of_start)

    # Group offers by event_id
    offers_by_event: Dict[int, List[Dict]] = {}
    for offer in offers:
        eid = offer.get("event_id")
        if eid:
            offers_by_event.setdefault(eid, []).append(offer)

    # Build game-level output
    games = []
    for event in events:
        eid = event["id"]
        home = _normalize_team(event.get("home", ""))
        away = _normalize_team(event.get("visitor", ""))
        scheduled = event.get("scheduled", "")

        home_record = None
        away_record = None
        for p in event.get("participants", []):
            team = p.get("team", {})
            rec = team.get("record", {})
            record_str = f"{rec.get('W', 0)}-{rec.get('L', 0)}"
            if team.get("abbreviation") == event.get("home"):
                home_record = record_str
            else:
                away_record = record_str

        game = {
            "event_id": eid,
            "home": home,
            "away": away,
            "home_record": home_record,
            "away_record": away_record,
            "scheduled": scheduled,
            "status": event.get("status"),
            "venue": event.get("venue", {}).get("name"),
            "markets": {},
        }

        # Process offers for this event
        for offer in offers_by_event.get(eid, []):
            market_id = offer.get("market_id")
            market_name = MARKET_NAMES.get(market_id, str(market_id))

            market_data = {"sides": []}
            for sel in offer.get("selections", []):
                label = sel.get("label", "")
                picks = sel.get("picks", {})
                opening = sel.get("opening_line", {})

                # Get consensus line (book_id=0)
                consensus = {}
                best = {}
                book_lines = {}
                for book in sel.get("books", []):
                    bid = book.get("id")
                    lines = book.get("lines", [])
                    if not lines:
                        continue
                    line_data = lines[0]
                    entry = {
                        "line": line_data.get("line"),
                        "cost": line_data.get("cost"),
                        "updated": line_data.get("updated"),
                        "metrics": line_data.get("metrics", {}),
                    }
                    if bid == 0:
                        consensus = entry
                    elif line_data.get("best"):
                        best = entry
                        best["book_id"] = bid
                    book_lines[bid] = entry

                side = {
                    "label": label,
                    "opening_line": opening.get("line"),
                    "opening_cost": opening.get("cost"),
                    "consensus_line": consensus.get("line"),
                    "consensus_cost": consensus.get("cost"),
                    "best_line": best.get("line"),
                    "best_cost": best.get("cost"),
                    "best_book_id": best.get("book_id"),
                    "handle_pct": picks.get("handle_percentage"),
                    "expert_picks": picks.get("expert", 0),
                    "system_picks": picks.get("system", 0),
                    "combined_picks": picks.get("combined", 0),
                    "projection": (consensus.get("metrics") or {}).get("projection"),
                    "cover_probability": (consensus.get("metrics") or {}).get("cover_probability"),
                    "expected_value": (consensus.get("metrics") or {}).get("expected_value"),
                    "bet_rating": (consensus.get("metrics") or {}).get("bet_rating"),
                    "num_books": len(book_lines) - 1,  # exclude consensus
                }
                market_data["sides"].append(side)

            game["markets"][market_name] = market_data

        games.append(game)

    return {
        "source": "bettingpros_matchups",
        "fetch_timestamp": datetime.now(EST).isoformat(),
        "date": datetime.now(EST).strftime("%Y-%m-%d"),
        "total_games": len(games),
        "games": games,
    }


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────


def _print_dvp(data: Dict) -> None:
    print(f"\n{'='*70}")
    print(f"DEFENSE VS POSITION — {data['total_teams']} teams, season {data.get('season')}")
    print(f"{'='*70}")
    teams = data.get("teams", {})
    print(
        f"{'TEAM':6s} {'PTS':>6s} {'REB':>6s} {'AST':>6s} {'3PM':>6s} {'STL':>6s} {'BLK':>6s} {'TO':>6s}"
    )
    for abbrev in sorted(teams.keys()):
        all_stats = teams[abbrev].get("ALL", {})
        print(
            f"{abbrev:6s} {all_stats.get('points',0):6.1f} {all_stats.get('rebounds',0):6.1f} "
            f"{all_stats.get('assists',0):6.1f} {all_stats.get('three_points_made',0):6.1f} "
            f"{all_stats.get('steals',0):6.1f} {all_stats.get('blocks',0):6.1f} "
            f"{all_stats.get('turnovers',0):6.1f}"
        )


def _print_trends(data: Dict) -> None:
    print(f"\n{'='*70}")
    print(f"LEAGUE TRENDS — {data['total_categories']} categories ({data['date']})")
    print(f"{'='*70}")
    for t in data.get("trends", []):
        name = t.get("display_name", "?")
        upcoming = t.get("upcoming", 0)
        ats = t.get("splits", {}).get("spread", {})
        ml = t.get("splits", {}).get("moneyline", t.get("overall", {}))
        total = t.get("splits", {}).get("total", {})
        ats_units = ats.get("net_units") or 0
        ml_units = ml.get("net_units") or 0
        print(
            f"  {name:30s} | ATS: {ats.get('win',0)}-{ats.get('loss',0)} ({ats_units:+.1f}u) "
            f"| ML: {ml.get('win',0)}-{ml.get('loss',0)} ({ml_units:+.1f}u) "
            f"| {upcoming} upcoming"
        )


def _print_matchups(data: Dict) -> None:
    print(f"\n{'='*70}")
    print(f"MATCHUPS — {data['total_games']} games ({data['date']})")
    print(f"{'='*70}")
    for g in data.get("games", []):
        home = g["home"]
        away = g["away"]
        print(f"\n  {away} @ {home} ({g.get('home_record', '?')})")

        for mkt_name, mkt in g.get("markets", {}).items():
            print(f"    {mkt_name.upper()}:")
            for side in mkt.get("sides", []):
                label = side["label"]
                line = side.get("consensus_line", "?")
                cost = side.get("consensus_cost", "?")
                handle = side.get("handle_pct")
                handle_str = f"{handle:.1f}%" if handle else "  --"
                expert = side.get("expert_picks", 0)
                system = side.get("system_picks", 0)
                opening = side.get("opening_line", "?")
                print(
                    f"      {label:12s} {line:>7} ({cost:>+4}) | "
                    f"open: {opening:>7} | handle: {handle_str:>6} | "
                    f"picks: E={expert} S={system}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape BettingPros analytics (DVP, Trends, Matchups)"
    )
    parser.add_argument("--page", choices=["dvp", "trends", "matchups", "all"], default="all")
    parser.add_argument("--save", action="store_true", help="Save to JSON files")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now(EST).strftime("%Y-%m-%d")

    if args.page in ("dvp", "all"):
        print("Fetching Defense vs Position...")
        dvp = fetch_dvp()
        _print_dvp(dvp)
        if args.save:
            path = OUTPUT_DIR / f"bp_dvp_{today}.json"
            with open(path, "w") as f:
                json.dump(dvp, f, indent=2)
            print(f"\n  Saved: {path}")

    if args.page in ("trends", "all"):
        print("\nFetching League Trends...")
        trends = fetch_league_trends()
        _print_trends(trends)
        if args.save:
            path = OUTPUT_DIR / f"bp_league_trends_{today}.json"
            with open(path, "w") as f:
                json.dump(trends, f, indent=2)
            print(f"\n  Saved: {path}")

    if args.page in ("matchups", "all"):
        print("\nFetching Matchups...")
        matchups = fetch_matchups()
        _print_matchups(matchups)
        if args.save:
            path = OUTPUT_DIR / f"bp_matchups_{today}.json"
            with open(path, "w") as f:
                json.dump(matchups, f, indent=2)
            print(f"\n  Saved: {path}")


if __name__ == "__main__":
    main()
