#!/usr/bin/env python3
"""
BettingPros Defense vs Position Scraper
========================================
Scrapes the server-rendered DVP (Defense vs Position) page from BettingPros.

This data is NOT available via API — it's baked into the HTML.
Provides per-team defensive stats allowed per prop category.

Data returned per team:
    PTS, REB, AST, 3PM, STL, BLK, TO
    + Easy/Tough indicators (green/gray circles on the website)

Usage:
    python fetch_bettingpros_dvp.py                 # Fetch DVP data
    python fetch_bettingpros_dvp.py --save          # Save to JSON
    python fetch_bettingpros_dvp.py --position G    # Filter by position (G/F/C)
"""

import argparse
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup

EST = ZoneInfo("America/New_York")

logger = logging.getLogger(__name__)

DVP_URL = "https://www.bettingpros.com/nba/defense-vs-position/"

# Stat columns in order as they appear on the page
STAT_COLUMNS = ["PTS", "REB", "AST", "3PM", "STL", "BLK", "TO"]


def fetch_dvp(position: Optional[str] = None, season: str = "2025-26") -> Dict[str, Any]:
    """Fetch Defense vs Position data from BettingPros.

    Args:
        position: Filter by position — "G", "F", "C", or None for all.
        season: Season string (e.g., "2025-26").

    Returns:
        Dict with teams list and metadata.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 Chrome/125.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate",
    }

    url = DVP_URL
    if position:
        url += f"?position={position.upper()}"

    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")

    # Find the data table
    table = soup.find("table") or soup.find("div", class_=re.compile(r"dvp|defense|table", re.I))

    teams = []

    # BettingPros uses a React-rendered table — look for rows with team data
    # The structure typically has rows with team name + stat values
    rows = soup.find_all("tr")
    if not rows:
        # Try div-based layout (React component)
        rows = soup.find_all("div", class_=re.compile(r"row|team-row", re.I))

    for row in rows:
        cells = row.find_all(["td", "div"])
        if len(cells) < 7:
            continue

        # Extract team name and record
        text_content = [c.get_text(strip=True) for c in cells]

        # Look for team pattern: "Atlanta (36-31)" or just team names
        team_name = None
        record = None
        stat_values = []

        for text in text_content:
            # Match "TeamName (W-L)" pattern
            match = re.match(r"^([A-Za-z\s.]+)\s*\((\d+-\d+)\)$", text)
            if match and not team_name:
                team_name = match.group(1).strip()
                record = match.group(2)
                continue

            # Try to parse as numeric stat value
            try:
                val = float(text)
                stat_values.append(val)
            except (ValueError, TypeError):
                pass

        if team_name and len(stat_values) >= 6:
            team_data = {
                "team_name": team_name,
                "record": record,
            }
            # Map stat values to columns
            for i, stat in enumerate(STAT_COLUMNS):
                if i < len(stat_values):
                    team_data[stat.lower()] = stat_values[i]

            teams.append(team_data)

    # If table parsing failed, try JSON embedded in script tags
    if not teams:
        scripts = soup.find_all("script")
        for script in scripts:
            text = script.string or ""
            if "dvp" in text.lower() or "defense" in text.lower():
                # Try to extract JSON data from __NEXT_DATA__ or similar
                json_match = re.search(
                    r"__NEXT_DATA__\s*=\s*({.+?})\s*;?\s*</script>", text, re.DOTALL
                )
                if json_match:
                    try:
                        next_data = json.loads(json_match.group(1))
                        logger.info("Found __NEXT_DATA__ with DVP data")
                        # Navigate the Next.js data structure to find DVP
                        props = next_data.get("props", {}).get("pageProps", {})
                        if "teams" in props or "dvp" in props:
                            teams = props.get("teams", props.get("dvp", []))
                    except json.JSONDecodeError:
                        pass

    result = {
        "date": datetime.now(EST).strftime("%Y-%m-%d"),
        "season": season,
        "position_filter": position or "all",
        "source": "bettingpros_dvp",
        "fetch_timestamp": datetime.now(EST).isoformat(),
        "teams": teams,
        "total_teams": len(teams),
    }

    return result


def fetch_matchups() -> Dict[str, Any]:
    """Fetch NBA Matchups page — expert picks, % of bets, % of money.

    Returns:
        Dict with games list including sharp money indicators.
    """
    url = "https://www.bettingpros.com/nba/matchups/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 Chrome/125.0.0.0 Safari/537.36",
        "Accept": "text/html",
        "Accept-Encoding": "gzip, deflate",
    }

    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    games = []

    # Look for matchup cards/rows
    rows = soup.find_all("div", class_=re.compile(r"matchup|game-row|event", re.I))
    if not rows:
        rows = soup.find_all("tr")

    for row in rows:
        text = row.get_text(" ", strip=True)
        # Try to extract: team, spread, expert rating, % of bets, % of money
        teams_match = re.findall(r"([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s*\(\d+-\d+\)", text)
        pcts = re.findall(r"(\d+\.?\d*)%", text)
        spreads = re.findall(r"([+-]\d+\.?\d*)\s*\([+-]\d+\)", text)
        expert = re.findall(r"([+-]\d+\.?\d+)", text)

        if len(teams_match) >= 2:
            game = {
                "home": teams_match[1] if len(teams_match) > 1 else None,
                "away": teams_match[0],
                "pct_bets": pcts[:2] if len(pcts) >= 2 else [],
                "pct_money": pcts[2:4] if len(pcts) >= 4 else [],
                "spreads": spreads,
            }
            games.append(game)

    return {
        "date": datetime.now(EST).strftime("%Y-%m-%d"),
        "source": "bettingpros_matchups",
        "fetch_timestamp": datetime.now(EST).isoformat(),
        "games": games,
        "total_games": len(games),
    }


def fetch_league_trends() -> Dict[str, Any]:
    """Fetch NBA League Trends — ATS records, net units by situation.

    Returns:
        Dict with trend categories and their records.
    """
    url = "https://www.bettingpros.com/nba/league-trends/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 Chrome/125.0.0.0 Safari/537.36",
        "Accept": "text/html",
        "Accept-Encoding": "gzip, deflate",
    }

    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    trends = []

    rows = soup.find_all("tr")
    for row in rows:
        cells = row.find_all("td")
        if len(cells) < 3:
            continue

        text_cells = [c.get_text(strip=True) for c in cells]
        category = text_cells[0] if text_cells else ""

        if not category or category in ("CATEGORY",):
            continue

        trend = {"category": category}

        # Try to extract record, net units, upcoming from remaining cells
        for cell_text in text_cells[1:]:
            # Record pattern: "26-16 (61.9%)"
            rec_match = re.match(r"(\d+-\d+(?:-\d+)?)\s*\((\d+\.?\d*)%\)", cell_text)
            if rec_match:
                trend.setdefault("records", []).append(
                    {
                        "record": rec_match.group(1),
                        "pct": float(rec_match.group(2)),
                    }
                )
                continue

            # Net units: "+7.6u" or "-3.8u"
            units_match = re.match(r"([+-]\d+\.?\d*)u", cell_text)
            if units_match:
                trend.setdefault("net_units", []).append(float(units_match.group(1)))
                continue

            # Upcoming count
            try:
                val = int(cell_text)
                trend.setdefault("upcoming", []).append(val)
            except ValueError:
                pass

        if len(trend) > 1:
            trends.append(trend)

    return {
        "date": datetime.now(EST).strftime("%Y-%m-%d"),
        "source": "bettingpros_league_trends",
        "fetch_timestamp": datetime.now(EST).isoformat(),
        "trends": trends,
        "total_categories": len(trends),
    }


def main():
    parser = argparse.ArgumentParser(description="Scrape BettingPros DVP, Matchups, League Trends")
    parser.add_argument("--save", action="store_true", help="Save to JSON")
    parser.add_argument("--position", choices=["G", "F", "C"], help="DVP position filter")
    parser.add_argument(
        "--page",
        choices=["dvp", "matchups", "trends", "all"],
        default="all",
        help="Which page to scrape",
    )
    args = parser.parse_args()

    output_dir = Path(__file__).parent.parent / "lines"
    output_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now(EST).strftime("%Y-%m-%d")

    results = {}

    if args.page in ("dvp", "all"):
        print("Fetching Defense vs Position...")
        dvp = fetch_dvp(position=args.position)
        results["dvp"] = dvp
        print(f"  {dvp['total_teams']} teams")
        for t in dvp["teams"][:5]:
            print(
                f"    {t.get('team_name', '?'):20s} PTS={t.get('pts','?')} REB={t.get('reb','?')} AST={t.get('ast','?')}"
            )

    if args.page in ("matchups", "all"):
        print("\nFetching Matchups...")
        matchups = fetch_matchups()
        results["matchups"] = matchups
        print(f"  {matchups['total_games']} games")

    if args.page in ("trends", "all"):
        print("\nFetching League Trends...")
        trends = fetch_league_trends()
        results["trends"] = trends
        print(f"  {trends['total_categories']} categories")
        for t in trends["trends"][:5]:
            print(f"    {t.get('category', '?'):30s} {t.get('records', [])}")

    if args.save:
        for key, data in results.items():
            outfile = output_dir / f"bp_{key}_{today}.json"
            with open(outfile, "w") as f:
                json.dump(data, f, indent=2, default=str)
            print(f"\nSaved to: {outfile}")


if __name__ == "__main__":
    main()
