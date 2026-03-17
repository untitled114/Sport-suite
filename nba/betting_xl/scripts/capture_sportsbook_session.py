#!/usr/bin/env python3
"""
Universal Sportsbook API Discovery via Playwright
===================================================
Opens a browser through a state proxy (Colorado or Florida), lets you create
an account or log in, then captures ALL API traffic for building direct fetchers.

Supports: Hard Rock, Caesars, BetRivers, or any sportsbook.

Usage:
    # Hard Rock (Florida proxy)
    python3 nba/betting_xl/scripts/capture_sportsbook_session.py --book hardrock

    # Caesars (Colorado proxy)
    python3 nba/betting_xl/scripts/capture_sportsbook_session.py --book caesars

    # BetRivers (Colorado proxy)
    python3 nba/betting_xl/scripts/capture_sportsbook_session.py --book betrivers

    # Custom URL with proxy
    python3 nba/betting_xl/scripts/capture_sportsbook_session.py --url https://example.com --proxy colorado

    # Dump ALL API traffic (verbose)
    python3 nba/betting_xl/scripts/capture_sportsbook_session.py --book caesars --dump-all

    # Deploy captured auth to server
    python3 nba/betting_xl/scripts/capture_sportsbook_session.py --book caesars --deploy

Requirements:
    pip install playwright
    playwright install chromium

Proxy env vars (from .env):
    SPORTSBOOK_PROXY_URL  = http://user:pass_country-us_state-colorado@geo.iproyal.com:12321
    PRIZEPICKS_PROXY_URL  = http://user:pass_country-us_state-florida@geo.iproyal.com:12321
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

EST = ZoneInfo("America/New_York")

_SERVER = os.getenv("DEPLOY_HOST", "sportsuite@16.58.146.197")
_SERVER_ENV = "/home/sportsuite/sport-suite/.env"
_OUTPUT_DIR = Path(__file__).parent.parent / "lines"

# Book configurations
BOOKS = {
    "hardrock": {
        "name": "Hard Rock Bet",
        "url": "https://app.hardrock.bet",
        "proxy": "florida",
        "geo": {"latitude": 25.7617, "longitude": -80.1918},  # Miami
        "intercept_keywords": ["hardrock", "hrdigital", "hardrocksportsbook"],
        "env_prefix": "HARDROCK",
    },
    "caesars": {
        "name": "Caesars Sportsbook",
        "url": "https://sportsbook.caesars.com/us/co/bet/sports/basketball/nba",
        "proxy": "colorado",
        "geo": {"latitude": 39.7392, "longitude": -104.9903},  # Denver
        "intercept_keywords": ["caesars", "americanwagering", "williamhill", "czr"],
        "env_prefix": "CAESARS",
    },
    "betrivers": {
        "name": "BetRivers",
        "url": "https://co.betrivers.com/?page=sportsbook&group=1000093204",
        "proxy": "colorado",
        "geo": {"latitude": 39.7392, "longitude": -104.9903},
        "intercept_keywords": ["betrivers", "kambi", "kambicdn", "rushstreet", "rsi"],
        "env_prefix": "BETRIVERS",
    },
    "espnbet": {
        "name": "theScore Bet (ex-ESPNBet)",
        "url": "https://sportsbook.thescore.bet",
        "proxy": "colorado",
        "geo": {"latitude": 39.7392, "longitude": -104.9903},
        "intercept_keywords": ["thescore", "scorebet", "penn", "espnbet"],
        "env_prefix": "THESCORE",
    },
}

# Common API path patterns to intercept
_API_PATTERNS = [
    "/api/",
    "/v1/",
    "/v2/",
    "/v3/",
    "/v4/",
    "/graphql",
    "/query",
    "/offering",
    "/odds",
    "/events",
    "/markets",
    "/props",
    "/player",
    "/betoffer",
    "/fixture",
    "/sport",
    "/schedule",
    "/lines",
    "/selections",
    "/competitions",
]


class APICapture:
    """Captures and categorizes intercepted API requests."""

    def __init__(self, intercept_keywords: list[str], dump_all: bool = False):
        self.intercept_keywords = intercept_keywords
        self.dump_all = dump_all
        self.requests: list[dict] = []
        self.auth_headers: dict = {}
        self.cookies: dict = {}
        self.api_endpoints: set = set()
        self.data_responses: list[dict] = []

    def _is_relevant(self, url: str) -> bool:
        url_lower = url.lower()
        return any(kw in url_lower for kw in self.intercept_keywords)

    def _is_api(self, url: str) -> bool:
        url_lower = url.lower()
        return any(p in url_lower for p in _API_PATTERNS)

    def on_request(self, request) -> None:
        url = request.url
        headers = dict(request.headers)

        if not self._is_relevant(url) and not self._is_api(url):
            if "json" not in headers.get("accept", ""):
                return

        # Capture interesting headers
        interesting = {}
        for k, v in headers.items():
            if k.lower() in (
                "authorization",
                "x-api-key",
                "x-auth-token",
                "x-session-id",
                "x-platform",
                "x-app-version",
                "cookie",
                "content-type",
                "accept",
                "x-device-id",
                "x-correlation-id",
                "x-requested-with",
                "origin",
                "referer",
                "x-hr-token",
                "x-access-token",
                "x-bwin-accessid",
                "x-level",
                "x-user-id",
            ):
                interesting[k] = v

        entry = {
            "timestamp": datetime.now(EST).isoformat(),
            "method": request.method,
            "url": url,
            "headers": interesting,
        }

        if request.method == "POST":
            try:
                entry["post_data"] = request.post_data[:2000] if request.post_data else None
            except Exception:
                entry["post_data"] = None

        self.requests.append(entry)

        # Extract auth headers
        for key in (
            "authorization",
            "x-api-key",
            "x-auth-token",
            "x-session-id",
            "x-hr-token",
            "x-access-token",
            "x-bwin-accessid",
        ):
            val = headers.get(key)
            if val and key not in self.auth_headers:
                self.auth_headers[key] = val
                print(f"  [AUTH] {key}: {val[:60]}...")

        # Track endpoints
        base_url = url.split("?")[0]
        if self._is_api(url) or self._is_relevant(url):
            self.api_endpoints.add(base_url)

        if self.dump_all and (self._is_relevant(url) or self._is_api(url)):
            print(f"  [{request.method:4s}] {url[:140]}")

    def on_response(self, response) -> None:
        url = response.url
        status = response.status

        if not self._is_relevant(url) and not self._is_api(url):
            return

        if status != 200:
            if self.dump_all:
                print(f"  [RESP {status}] {url[:100]}")
            return

        try:
            content_type = response.headers.get("content-type", "")
            if "json" not in content_type:
                return

            body = response.json()
            size = len(response.body())

            if size < 100:
                return

            entry = {
                "url": url,
                "status": status,
                "size": size,
                "top_keys": (
                    list(body.keys())[:15] if isinstance(body, dict) else f"array[{len(body)}]"
                ),
            }

            if isinstance(body, dict):
                for key in (
                    "events",
                    "markets",
                    "props",
                    "data",
                    "items",
                    "betOffers",
                    "results",
                    "selections",
                    "lines",
                    "over_under_lines",
                    "fixtures",
                    "players",
                    "offerings",
                    "odds",
                    "schedule",
                    "games",
                    "competitions",
                    "matches",
                    "optionMarkets",
                    "shelves",
                    "picks",
                ):
                    if key in body:
                        items = body[key]
                        if isinstance(items, list):
                            entry["data_key"] = key
                            entry["data_count"] = len(items)
                            if items and isinstance(items[0], dict):
                                entry["first_item_keys"] = list(items[0].keys())[:20]
                            break

            self.data_responses.append(entry)

            print(f"\n  [DATA] {url[:120]}")
            print(f"    Size: {size:,} bytes")
            if "data_key" in entry:
                print(f"    {entry['data_key']}[]: {entry['data_count']} items")
                if "first_item_keys" in entry:
                    print(f"    Item keys: {entry['first_item_keys']}")
            else:
                print(f"    Keys: {entry['top_keys']}")
            print()

        except Exception as e:
            if self.dump_all:
                print(f"  [RESP ERR] {url[:80]}: {e}")

    def summary(self) -> dict:
        return {
            "capture_time": datetime.now(EST).isoformat(),
            "total_requests": len(self.requests),
            "unique_api_endpoints": sorted(self.api_endpoints),
            "auth_headers": self.auth_headers,
            "cookies": self.cookies,
            "data_responses": self.data_responses,
            "all_requests": self.requests,
        }


def get_proxy_config(profile: str) -> dict | None:
    """Get Playwright proxy config from env vars."""
    env_map = {
        "colorado": "SPORTSBOOK_PROXY_URL",
        "florida": "PRIZEPICKS_PROXY_URL",
    }
    env_var = env_map.get(profile)
    if not env_var:
        return None

    proxy_url = os.getenv(env_var)
    if not proxy_url:
        print(f"  [WARN] {env_var} not set — running without proxy")
        return None

    parsed = urlparse(proxy_url)
    config = {"server": f"{parsed.scheme}://{parsed.hostname}:{parsed.port}"}
    if parsed.username:
        config["username"] = parsed.username
    if parsed.password:
        config["password"] = parsed.password

    return config


def run_capture(book_config: dict, dump_all: bool = False) -> dict:
    """Launch browser with proxy, capture API traffic."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("ERROR: pip install playwright && playwright install chromium")
        sys.exit(1)

    capture = APICapture(
        intercept_keywords=book_config["intercept_keywords"],
        dump_all=dump_all,
    )

    proxy_config = get_proxy_config(book_config["proxy"])
    geo = book_config["geo"]

    with sync_playwright() as p:
        launch_args = {
            "headless": False,
            "args": [
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
            ],
        }
        if proxy_config:
            launch_args["proxy"] = proxy_config

        browser = p.chromium.launch(**launch_args)

        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1440, "height": 900},
            locale="en-US",
            timezone_id="America/New_York",
            geolocation=geo,
            permissions=["geolocation"],
        )

        page = context.new_page()
        page.on("request", capture.on_request)
        page.on("response", capture.on_response)

        name = book_config["name"]
        url = book_config["url"]
        proxy_state = book_config["proxy"].upper()

        print("=" * 60)
        print(f"Sportsbook API Discovery: {name}")
        print("=" * 60)
        print(f"  URL:   {url}")
        print(f"  Proxy: {proxy_state} ({'active' if proxy_config else 'NOT SET'})")
        print(f"  Geo:   {geo['latitude']}, {geo['longitude']}")
        print()
        print("INSTRUCTIONS:")
        print(f"  1. Create account or log in to {name}")
        print("  2. Navigate to: Sports > Basketball > NBA")
        print("  3. Click on a game > Player Props tab")
        print("  4. Browse several games' player props")
        print("  5. CLOSE THE BROWSER WINDOW when done")
        print()
        print("All API traffic is captured automatically.")
        print("=" * 60)
        print()

        try:
            page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        except Exception as e:
            print(f"  [WARN] Initial load: {e}")

        print("Capturing API traffic... close browser when done.\n")

        try:
            page.wait_for_event("close", timeout=600_000)
        except Exception:
            pass

        try:
            for cookie in context.cookies():
                domain = cookie.get("domain", "")
                if any(kw in domain for kw in book_config["intercept_keywords"]):
                    capture.cookies[cookie["name"]] = {
                        "value": cookie["value"][:80],
                        "domain": domain,
                        "path": cookie.get("path", "/"),
                        "httpOnly": cookie.get("httpOnly", False),
                        "secure": cookie.get("secure", False),
                    }
        except Exception:
            pass

        try:
            browser.close()
        except Exception:
            pass

    return capture.summary()


def save_capture(data: dict, book_name: str) -> Path:
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(EST).strftime("%Y-%m-%d_%H-%M-%S")
    output_file = _OUTPUT_DIR / f"{book_name}_api_capture_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2, default=str)
    return output_file


def deploy_auth(data: dict, env_prefix: str) -> None:
    auth = data.get("auth_headers", {})
    cookies = data.get("cookies", {})
    updates = {}

    if auth.get("authorization"):
        updates[f"{env_prefix}_AUTH_TOKEN"] = auth["authorization"]
    if auth.get("x-api-key"):
        updates[f"{env_prefix}_API_KEY"] = auth["x-api-key"]
    if auth.get("x-auth-token"):
        updates[f"{env_prefix}_AUTH_TOKEN"] = auth["x-auth-token"]
    if auth.get("x-session-id"):
        updates[f"{env_prefix}_SESSION_ID"] = auth["x-session-id"]
    if auth.get("x-bwin-accessid"):
        updates[f"{env_prefix}_ACCESS_ID"] = auth["x-bwin-accessid"]

    for name, cookie in cookies.items():
        if name.lower() in ("sessionid", "session", "token", "access_token", "auth"):
            updates[f"{env_prefix}_COOKIE_{name.upper()}"] = cookie["value"]

    if not updates:
        print("[deploy] No auth credentials captured.")
        return

    print(f"\nDeploying {len(updates)} credential(s) to {_SERVER}:{_SERVER_ENV}")
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
    print(f"\nRequests intercepted: {data['total_requests']}")

    auth = data.get("auth_headers", {})
    if auth:
        print(f"\nAuth headers ({len(auth)}):")
        for k, v in auth.items():
            print(f"  {k}: {v[:60]}...")
    else:
        print("\nNo auth headers captured.")

    cookies = data.get("cookies", {})
    if cookies:
        print(f"\nCookies ({len(cookies)}):")
        for name, info in cookies.items():
            print(f"  {name} ({info['domain']}): {info['value'][:40]}...")

    endpoints = data.get("unique_api_endpoints", [])
    if endpoints:
        print(f"\nAPI endpoints ({len(endpoints)}):")
        for ep in sorted(endpoints):
            print(f"  {ep}")

    responses = data.get("data_responses", [])
    if responses:
        print(f"\nData responses ({len(responses)}):")
        for r in responses:
            print(f"  {r['url'][:100]}")
            if "data_key" in r:
                print(f"    {r['data_key']}[]: {r['data_count']} items")

    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture sportsbook API via Playwright + proxy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available books: {', '.join(BOOKS.keys())}",
    )
    parser.add_argument("--book", choices=list(BOOKS.keys()), help="Sportsbook to capture")
    parser.add_argument("--url", help="Custom URL (use with --proxy)")
    parser.add_argument("--proxy", choices=["colorado", "florida"], help="Proxy state (for --url)")
    parser.add_argument("--deploy", action="store_true", help="Deploy auth to server .env")
    parser.add_argument("--dump-all", action="store_true", help="Print ALL intercepted requests")
    args = parser.parse_args()

    if args.url:
        book_config = {
            "name": "Custom",
            "url": args.url,
            "proxy": args.proxy or "colorado",
            "geo": {"latitude": 39.7392, "longitude": -104.9903},
            "intercept_keywords": [urlparse(args.url).hostname.split(".")[0]],
            "env_prefix": "CUSTOM",
        }
        book_name = "custom"
    elif args.book:
        book_config = BOOKS[args.book]
        book_name = args.book
    else:
        parser.error("Specify --book or --url")
        return

    data = run_capture(book_config, dump_all=args.dump_all)
    output_file = save_capture(data, book_name)
    print(f"\nCapture saved to: {output_file}")
    print_summary(data)

    if args.deploy:
        deploy_auth(data, book_config["env_prefix"])
    elif data.get("auth_headers"):
        print("\nRun with --deploy to push credentials to server.")


if __name__ == "__main__":
    main()
