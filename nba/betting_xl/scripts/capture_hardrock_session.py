#!/usr/bin/env python3
"""
Hard Rock Bet Player Props Extractor via Playwright
=====================================================
Opens browser, user logs in + navigates to NBA player props,
script captures the GraphQL response data containing all lines/odds.

Saves raw player props to JSON for building the direct fetcher.

Usage:
    python3 nba/betting_xl/scripts/capture_hardrock_session.py
    python3 nba/betting_xl/scripts/capture_hardrock_session.py --deploy

Requirements:
    pip install playwright && playwright install chromium
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

EST = ZoneInfo("America/New_York")

_SERVER = os.getenv("DEPLOY_HOST", "sportsuite@16.58.146.197")
_SERVER_ENV = "/home/sportsuite/sport-suite/.env"
_OUTPUT_DIR = Path(__file__).parent.parent / "lines"

_HR_BASE = "https://app.hardrock.bet"
_HR_GQL = "https://api.hardrocksportsbook.com/java-graphql/graphql"


class PropsCapture:
    """Captures GraphQL responses containing player props data."""

    def __init__(self):
        self.graphql_responses: list[dict] = []
        self.all_cookies: dict = {}
        self.auth_headers: dict = {}
        self.player_props: list[dict] = []

    def on_response(self, response) -> None:
        url = response.url
        if "graphql" not in url:
            return
        if response.status != 200:
            return

        try:
            body = response.json()
            size = len(response.body())

            # Only care about substantial responses
            if size < 1000:
                return

            self.graphql_responses.append(
                {
                    "url": url,
                    "size": size,
                    "data": body,
                }
            )

            # Look for events with markets
            events = []
            try:
                events = body.get("data", {}).get("betSync", {}).get("events", {}).get("data", [])
            except (AttributeError, TypeError):
                pass

            if not events:
                return

            props_found = 0
            for event in events:
                markets = event.get("markets", [])
                for market in markets:
                    sels = market.get("selections", [])
                    if len(sels) >= 2:  # Over/Under pair
                        props_found += 1

            if props_found > 0:
                print(f"\n  [PROPS] {url[:80]}")
                print(f"    {len(events)} events, {props_found} player prop markets")
                for ev in events[:3]:
                    name = ev.get("name", "?")
                    markets = ev.get("markets", [])
                    print(f"    {name}: {len(markets)} markets")
                    for m in markets[:3]:
                        mname = m.get("name", "?")
                        sels = m.get("selections", [])
                        if sels:
                            sel = sels[0]
                            price = sel.get("price", {})
                            print(
                                f"      {mname} | {sel.get('name','?')} h={sel.get('handicap','?')} odds={price.get('american','?')}"
                            )

        except Exception as e:
            pass

    def on_request(self, request) -> None:
        url = request.url
        headers = dict(request.headers)

        # Capture auth headers
        for key in ("authorization", "x-api-key", "x-auth-token", "x-session-id"):
            val = headers.get(key)
            if val and key not in self.auth_headers:
                self.auth_headers[key] = val
                print(f"  [AUTH] {key}: {val[:60]}...")

        # Capture full cookie header for GraphQL calls
        if "graphql" in url and "cookie" in headers:
            cookie_str = headers["cookie"]
            for part in cookie_str.split(";"):
                part = part.strip()
                if "=" in part:
                    k, v = part.split("=", 1)
                    self.all_cookies[k.strip()] = v.strip()

    def extract_props(self) -> list[dict]:
        """Extract player props from all captured GraphQL responses."""
        props = []
        today = datetime.now(EST).strftime("%Y-%m-%d")

        for resp in self.graphql_responses:
            body = resp["data"]
            try:
                events = body.get("data", {}).get("betSync", {}).get("events", {}).get("data", [])
            except (AttributeError, TypeError):
                continue

            for event in events:
                event_name = event.get("name", "")
                event_id = event.get("id", "")
                event_time = event.get("eventTime", "")
                path = event.get("path", [])

                # Parse teams from name "Team A @ Team B" or path
                home_team = None
                away_team = None
                if " @ " in event_name:
                    parts = event_name.split(" @ ")
                    away_team = parts[0].strip()
                    home_team = parts[1].strip()
                elif " v " in event_name:
                    parts = event_name.split(" v ")
                    home_team = parts[0].strip()
                    away_team = parts[1].strip()

                markets = event.get("markets", [])
                for market in markets:
                    mtype = market.get("type", "")
                    mname = market.get("name", "")
                    group = market.get("group", "")
                    line_val = market.get("line")
                    sels = market.get("selections", [])

                    if len(sels) < 2:
                        continue

                    # Parse over/under from selections
                    over_sel = None
                    under_sel = None
                    for sel in sels:
                        sname = (sel.get("name", "") or "").lower()
                        if "over" in sname:
                            over_sel = sel
                        elif "under" in sname:
                            under_sel = sel

                    if not over_sel and not under_sel:
                        # Try by position (first=over, second=under)
                        if len(sels) == 2:
                            over_sel = sels[0]
                            under_sel = sels[1]

                    if not over_sel:
                        continue

                    # Extract line from handicap or market line
                    handicap = over_sel.get("handicap") or over_sel.get("points") or line_val
                    if handicap is None:
                        continue

                    try:
                        line = float(handicap)
                    except (ValueError, TypeError):
                        continue

                    # Extract odds
                    over_price = over_sel.get("price", {})
                    under_price = (under_sel or {}).get("price", {})

                    over_odds = over_price.get("american", -110)
                    under_odds = under_price.get("american", -110)

                    try:
                        over_odds = int(over_odds)
                    except (ValueError, TypeError):
                        over_odds = -110
                    try:
                        under_odds = int(under_odds)
                    except (ValueError, TypeError):
                        under_odds = -110

                    # Extract player name from market name
                    # Format: "Player Name - Points" or "Points - Player Name"
                    player_name = mname
                    stat_type = mtype

                    prop = {
                        "player_name": player_name,
                        "stat_type": stat_type,
                        "market_name": mname,
                        "market_type": mtype,
                        "market_group": group,
                        "line": line,
                        "over_odds": over_odds,
                        "under_odds": under_odds,
                        "book_name": "hardrock_direct",
                        "game_date": today,
                        "game_id": str(event_id),
                        "event_name": event_name,
                        "home_team": home_team,
                        "away_team": away_team,
                        "source": "hardrock_direct",
                        "fetch_source": "direct",
                        "fetch_timestamp": datetime.now(EST).isoformat(),
                    }
                    props.append(prop)

        return props


def run_capture() -> dict:
    """Launch browser, capture player props from GraphQL."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("ERROR: pip install playwright && playwright install chromium")
        sys.exit(1)

    capture = PropsCapture()

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False,
            args=["--disable-blink-features=AutomationControlled", "--no-sandbox"],
        )
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1440, "height": 900},
            locale="en-US",
            timezone_id="America/New_York",
            geolocation={"latitude": 25.7617, "longitude": -80.1918},
            permissions=["geolocation"],
        )

        page = context.new_page()
        page.on("response", capture.on_response)
        page.on("request", capture.on_request)

        print("=" * 60)
        print("Hard Rock Bet Player Props Capture")
        print("=" * 60)
        print()
        print("INSTRUCTIONS:")
        print("  1. Log in to Hard Rock Bet")
        print("  2. Go to NBA > click a game > Player Props tab")
        print("  3. Click through ALL games' Player Props tabs")
        print("  4. CLOSE the browser when done")
        print()
        print("Props data captured automatically from GraphQL.")
        print("=" * 60)
        print()

        try:
            page.goto(_HR_BASE, wait_until="domcontentloaded", timeout=30_000)
        except Exception as e:
            print(f"  [WARN] {e}")

        print("Capturing... close browser when done.\n")

        try:
            page.wait_for_event("close", timeout=600_000)
        except Exception:
            pass

        # Get cookies before closing
        try:
            for cookie in context.cookies():
                domain = cookie.get("domain", "")
                if "hardrock" in domain:
                    capture.all_cookies[cookie["name"]] = cookie["value"]
        except Exception:
            pass

        try:
            browser.close()
        except Exception:
            pass

    # Extract props from captured GraphQL responses
    props = capture.extract_props()

    return {
        "capture_time": datetime.now(EST).isoformat(),
        "graphql_responses": len(capture.graphql_responses),
        "auth_headers": capture.auth_headers,
        "cookies": {k: v[:80] for k, v in capture.all_cookies.items()},
        "full_cookies": capture.all_cookies,  # Full values for replay
        "player_props": props,
        "raw_responses": [
            {"url": r["url"], "size": r["size"], "data": r["data"]}
            for r in capture.graphql_responses
        ],
    }


def save_results(data: dict) -> Path:
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(EST).strftime("%Y-%m-%d_%H-%M-%S")
    output_file = _OUTPUT_DIR / f"hardrock_props_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2, default=str)
    return output_file


def deploy_cookies(data: dict) -> None:
    """Deploy captured cookies to server .env for the fetcher."""
    cookies = data.get("full_cookies", {})
    updates = {}

    hrd = cookies.get("_hrd", "")
    if hrd:
        updates["HARDROCK_HRD_COOKIE"] = hrd

    ats = cookies.get("ats_token", "")
    if ats:
        updates["HARDROCK_ATS_TOKEN"] = ats

    if not updates:
        print("[deploy] No auth cookies captured.")
        return

    print(f"\nDeploying {len(updates)} cookie(s) to {_SERVER}:{_SERVER_ENV}")
    for key, value in updates.items():
        display = value[:40] + "..." if len(value) > 40 else value
        cmd = (
            f"grep -q '^{key}=' {_SERVER_ENV} "
            f"&& sed -i 's|^{key}=.*|{key}={value}|' {_SERVER_ENV} "
            f"|| echo '{key}={value}' >> {_SERVER_ENV}"
        )
        result = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=accept-new", _SERVER, cmd],
            capture_output=True,
            text=True,
        )
        status = "[+]" if result.returncode == 0 else "[!]"
        print(f"  {status} {key} = {display}")


def print_summary(data: dict) -> None:
    print("\n" + "=" * 60)
    print("CAPTURE SUMMARY")
    print("=" * 60)
    print(f"GraphQL responses: {data['graphql_responses']}")
    print(f"Player props extracted: {len(data['player_props'])}")

    if data["player_props"]:
        # Group by stat type
        types = {}
        for p in data["player_props"]:
            t = p.get("market_type", "unknown")
            types[t] = types.get(t, 0) + 1

        print("\nProps by market type:")
        for t, c in sorted(types.items(), key=lambda x: -x[1]):
            print(f"  {t}: {c}")

        # Show sample
        print("\nSample props:")
        for p in data["player_props"][:10]:
            print(
                f"  {p['player_name'][:25]:25s} | {p['market_type']:20s} | {p['line']:5.1f} | O {p['over_odds']:+d} U {p['under_odds']:+d}"
            )

    auth = data.get("auth_headers", {})
    if auth:
        print(f"\nAuth headers: {list(auth.keys())}")

    cookies = data.get("cookies", {})
    key_cookies = [k for k in cookies if k in ("_hrd", "ats_token", "ats_user")]
    if key_cookies:
        print(f"Key cookies: {key_cookies}")

    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture Hard Rock Bet player props")
    parser.add_argument("--deploy", action="store_true", help="Deploy cookies to server")
    args = parser.parse_args()

    data = run_capture()
    output_file = save_results(data)
    print(f"\nSaved to: {output_file}")
    print_summary(data)

    if args.deploy:
        deploy_cookies(data)


if __name__ == "__main__":
    main()
