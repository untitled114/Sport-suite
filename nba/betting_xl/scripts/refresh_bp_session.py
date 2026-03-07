#!/usr/bin/env python3
"""
BettingPros Apple SSO Session Refresher

Opens a real browser window -> user completes Apple SSO manually once ->
captures sessionid cookie + web API key -> optionally deploys to server .env.

The web API key (X-Api-Key header) is intercepted from live API calls made by
the BettingPros frontend after login. It is session-generated and needed to
access session-gated endpoints like pick-recommendations and market-ev.

Usage:
    python3 nba/betting_xl/scripts/refresh_bp_session.py
    python3 nba/betting_xl/scripts/refresh_bp_session.py --deploy
    python3 nba/betting_xl/scripts/refresh_bp_session.py --verify
    python3 nba/betting_xl/scripts/refresh_bp_session.py --deploy --verify

Requirements:
    pip install playwright
    playwright install chromium

Env var written to server:
    BETTINGPROS_SESSION_ID   - sessionid cookie value
    BETTINGPROS_WEB_API_KEY  - X-Api-Key value for session-gated endpoints
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

_SERVER = os.getenv("DEPLOY_HOST", "sportsuite@16.58.146.197")
_SERVER_ENV = "/home/sportsuite/sport-suite/.env"

_BP_LOGIN_URL = "https://secure.bettingpros.com/sign-in"
_BP_PROPS_URL = "https://www.bettingpros.com/nba/picks/player-props/"
_BP_API_HOST = "api.bettingpros.com"
_BP_USER_URL = "https://api.bettingpros.com/v1/users/current"

# Session-gated endpoints to verify after capture
_VERIFY_URL = (
    "https://api.bettingpros.com/v3/pick-recommendations" "?v=4&fence=web&sport=NBA&mode=feed"
)


def capture_session() -> dict:
    """Launch browser, capture sessionid + web API key after Apple SSO."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("ERROR: playwright not installed.")
        print("  pip install playwright && playwright install chromium")
        sys.exit(1)

    captured: dict = {"sessionid": None, "web_api_key": None}

    def _on_request(request) -> None:
        if _BP_API_HOST not in request.url:
            return
        key = request.headers.get("x-api-key")
        if key and not captured["web_api_key"]:
            captured["web_api_key"] = key
            print(f"  [+] Web API key intercepted: {key[:24]}...")

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False,
            args=["--disable-blink-features=AutomationControlled"],
        )
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            )
        )
        page = context.new_page()
        page.on("request", _on_request)

        print("Opening BettingPros sign-in...")
        print("  -> Click 'Sign in with Apple' and complete the flow.")
        print("  -> Waiting up to 5 minutes...")
        print()

        page.goto(_BP_LOGIN_URL, wait_until="domcontentloaded")

        # Wait until the URL leaves /sign-in and /accounts/apple (callback)
        page.wait_for_function(
            """() => {
                const u = window.location.href;
                return !u.includes('/sign-in') && !u.includes('/accounts/apple');
            }""",
            timeout=300_000,
        )

        # Give the frontend a moment to fire API calls
        page.wait_for_timeout(3_000)

        # Capture sessionid
        for cookie in context.cookies():
            domain = cookie.get("domain", "")
            if cookie["name"] == "sessionid" and "bettingpros" in domain:
                captured["sessionid"] = cookie["value"]
                print(f"  [+] sessionid captured: {captured['sessionid'][:24]}...")
                break

        if not captured["sessionid"]:
            print("ERROR: sessionid cookie not found after login.")
            browser.close()
            sys.exit(1)

        # If no API key yet, navigate to props page to trigger calls
        if not captured["web_api_key"]:
            print("  -> Navigating to props page to trigger API calls...")
            page.goto(_BP_PROPS_URL, wait_until="networkidle", timeout=30_000)
            page.wait_for_timeout(3_000)

        if not captured["web_api_key"]:
            # Last resort: try intercepting from a direct API call
            print("  -> Fetching user info to capture key...")
            try:
                resp = page.request.get(
                    _BP_USER_URL,
                    headers={
                        "Cookie": f"sessionid={captured['sessionid']}",
                        "Accept": "application/json",
                    },
                )
                if resp.ok:
                    data = resp.json()
                    # apikey field is the personal key - different from web session key
                    # but capture it as fallback
                    api_key = data.get("apikey")
                    if api_key and not captured["web_api_key"]:
                        print(f"  [!] Only personal API key available: {api_key[:24]}...")
                        print(
                            "      Web session key not captured - props page did not fire API calls."
                        )
            except Exception as exc:
                print(f"  [!] User info fetch failed: {exc}")

        browser.close()

    return captured


def verify_session(captured: dict) -> bool:
    """Test the captured credentials against pick-recommendations endpoint."""
    import urllib.error
    import urllib.request

    sessionid = captured.get("sessionid", "")
    web_key = captured.get("web_api_key", "")
    personal_key = os.getenv("BETTINGPROS_API_KEY", "")

    if not sessionid or not web_key:
        print("[verify] Missing credentials - skipping verify.")
        return False

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    url = _VERIFY_URL + f"&key={personal_key}&date={today}"

    req = urllib.request.Request(url)
    req.add_header("X-Api-Key", web_key)
    req.add_header("X-Level", '"cHJlbWl1bQ=="')
    req.add_header("Cookie", f"sessionid={sessionid}")
    req.add_header("User-Agent", "Mozilla/5.0")
    req.add_header("Accept", "application/json")

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:  # nosec B310
            data = json.loads(resp.read())
            shelves = data.get("shelves", [])
            picks_total = sum(len(s.get("picks", [])) for s in shelves)
            print(f"[verify] OK - {len(shelves)} shelves, {picks_total} picks")
            return True
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:200]
        print(f"[verify] HTTP {e.code}: {body}")
        return False
    except Exception as exc:
        print(f"[verify] Failed: {exc}")
        return False


def deploy_to_server(captured: dict) -> None:
    """SSH credentials to server .env."""
    updates = {}
    if captured.get("sessionid"):
        updates["BETTINGPROS_SESSION_ID"] = captured["sessionid"]
    if captured.get("web_api_key"):
        updates["BETTINGPROS_WEB_API_KEY"] = captured["web_api_key"]

    if not updates:
        print("[deploy] Nothing to deploy.")
        return

    print(f"\nDeploying {len(updates)} credential(s) to {_SERVER}:{_SERVER_ENV}")

    for key, value in updates.items():
        # Upsert: replace if exists, append if not
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
        if result.returncode == 0:
            print(f"  [+] {key} updated on server")
        else:
            print(f"  [!] {key} failed: {result.stderr.strip()}")

    print()
    print("If Axiom needs the new session key, restart it:")
    print(f"  ssh {_SERVER} 'sudo systemctl restart cephalon-axiom'")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh BettingPros session credentials via Apple SSO"
    )
    parser.add_argument(
        "--deploy", action="store_true", help="SSH captured credentials to server .env"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Test captured credentials against pick-recommendations endpoint",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("BettingPros Session Refresher")
    print("=" * 60)

    captured = capture_session()

    print()
    print("Captured:")
    print(f"  BETTINGPROS_SESSION_ID  = {captured.get('sessionid', 'NOT FOUND')}")
    if captured.get("web_api_key"):
        print(f"  BETTINGPROS_WEB_API_KEY = {captured['web_api_key']}")
    else:
        print("  BETTINGPROS_WEB_API_KEY = NOT CAPTURED")

    if args.verify:
        print()
        verify_session(captured)

    if args.deploy:
        deploy_to_server(captured)
    else:
        print()
        print("Run with --deploy to push to server, --verify to test endpoints.")


if __name__ == "__main__":
    main()
