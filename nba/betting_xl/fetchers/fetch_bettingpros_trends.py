#!/usr/bin/env python3
"""
BettingPros Trends, Streaks & Injuries Fetcher
================================================
Exploratory fetcher for BettingPros analytics endpoints:

1. /v3/props/trends   — Line movement trends per prop
2. /v3/props/streaks  — Player streak information (hit/miss runs)
3. /v3/injuries       — BettingPros injury reports

These endpoints are exploratory — they may return 404 or unexpected formats.
The fetcher logs warnings and continues gracefully on any endpoint failure.

Output is raw BP response data saved to JSON (analytics data for enrichment/V4
features, NOT standardized props).

Output files (in nba/betting_xl/lines/):
    bp_trends_{date}.json
    bp_streaks_{date}.json
    bp_injuries_{date}.json

Usage:
    python fetch_bettingpros_trends.py                        # All endpoints
    python fetch_bettingpros_trends.py --endpoint trends      # Trends only
    python fetch_bettingpros_trends.py --endpoint streaks     # Streaks only
    python fetch_bettingpros_trends.py --endpoint injuries    # Injuries only
    python fetch_bettingpros_trends.py --save --quiet         # Save + quiet
    python fetch_bettingpros_trends.py --date 2026-03-15      # Specific date
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from nba.betting_xl.fetchers.base_fetcher import BaseFetcher

logger = logging.getLogger(__name__)

_EST = ZoneInfo("America/New_York")

# BettingPros API market IDs (same as rest of codebase)
MARKET_IDS: Dict[str, int] = {
    "POINTS": 156,
    "REBOUNDS": 157,
    "ASSISTS": 151,
    "THREES": 162,
}

# Reverse lookup for parsing responses
MARKET_ID_TO_NAME: Dict[int, str] = {v: k for k, v in MARKET_IDS.items()}


class BettingProsTrendsFetcher(BaseFetcher):
    """Fetches analytics data from BettingPros trends, streaks, and injuries endpoints.

    These are exploratory endpoints — the fetcher tolerates 404s and unexpected
    response formats, logging warnings instead of raising exceptions.
    """

    API_BASE = "https://api.bettingpros.com"

    # Endpoint paths (appended to API_BASE)
    ENDPOINTS = {
        "trends": "/v3/props/trends",
        "streaks": "/v3/props/streaks",
        "injuries": "/v3/injuries",
    }

    def __init__(
        self,
        date: Optional[str] = None,
        endpoint: str = "all",
        verbose: bool = True,
    ):
        """
        Initialize BettingPros trends/streaks/injuries fetcher.

        Args:
            date: Date to fetch (YYYY-MM-DD). Defaults to today EST.
            endpoint: Which endpoint(s) to hit — 'trends', 'streaks', 'injuries', or 'all'.
            verbose: Enable verbose logging.
        """
        super().__init__(
            source_name="bettingpros_trends",
            rate_limit=2.5,
            max_retries=3,
            timeout=30,
            verbose=verbose,
        )

        self.date = date or datetime.now(_EST).strftime("%Y-%m-%d")
        self.endpoint_filter = endpoint.lower()

        # Validate endpoint filter
        valid_choices = {"all", "trends", "streaks", "injuries"}
        if self.endpoint_filter not in valid_choices:
            raise ValueError(
                f"Invalid endpoint '{endpoint}'. Must be one of: {', '.join(sorted(valid_choices))}"
            )

    def _get_bp_headers(self) -> Dict[str, str]:
        """Build premium BettingPros headers."""
        api_key = os.getenv("BETTINGPROS_API_KEY", "")
        if not api_key:
            logger.warning("BETTINGPROS_API_KEY not set — requests may fail")
        return {
            "x-api-key": api_key,
            "x-level": "cHJlbWl1bQ==",  # base64 "premium"
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate",
            "Referer": "https://www.bettingpros.com/",
            "Origin": "https://www.bettingpros.com",
        }

    # ------------------------------------------------------------------
    # Trends endpoint
    # ------------------------------------------------------------------

    def fetch_trends(self) -> Dict[str, Any]:
        """Fetch line movement trends from /v3/props/trends.

        Queries each active market and collects trend data. If the endpoint
        returns 404 or an unexpected format, logs a warning and returns
        partial results.

        Returns:
            Dict with 'markets' key mapping market name -> list of trend records.
        """
        logger.info(f"[{self.source_name}] Fetching trends for {self.date}")
        results: Dict[str, List[Dict[str, Any]]] = {}
        url = f"{self.API_BASE}{self.ENDPOINTS['trends']}"

        for market_name, market_id in MARKET_IDS.items():
            params: Dict[str, Any] = {
                "sport": "NBA",
                "market_id": market_id,
                "date": self.date,
            }

            response = self._make_request(
                url=url,
                method="GET",
                params=params,
                headers=self._get_bp_headers(),
            )

            records = self._safe_parse_response(response, f"trends/{market_name}")
            if records is not None:
                results[market_name] = records
                if self.verbose:
                    count = len(records) if isinstance(records, list) else 1
                    logger.info(f"  {market_name}: {count} trend records")

        total = sum(len(v) for v in results.values() if isinstance(v, list))
        logger.info(f"[{self.source_name}] Trends: {total} records across {len(results)} markets")

        return {
            "endpoint": "trends",
            "date": self.date,
            "fetched_at": datetime.now(_EST).isoformat(),
            "markets": results,
            "total_records": total,
        }

    # ------------------------------------------------------------------
    # Streaks endpoint
    # ------------------------------------------------------------------

    def fetch_streaks(self) -> Dict[str, Any]:
        """Fetch player streak data from /v3/props/streaks.

        Queries each active market for streak information (consecutive
        over/under hits per player).

        Returns:
            Dict with 'markets' key mapping market name -> list of streak records.
        """
        logger.info(f"[{self.source_name}] Fetching streaks for {self.date}")
        results: Dict[str, List[Dict[str, Any]]] = {}
        url = f"{self.API_BASE}{self.ENDPOINTS['streaks']}"

        for market_name, market_id in MARKET_IDS.items():
            params: Dict[str, Any] = {
                "sport": "NBA",
                "market_id": market_id,
                "date": self.date,
            }

            response = self._make_request(
                url=url,
                method="GET",
                params=params,
                headers=self._get_bp_headers(),
            )

            records = self._safe_parse_response(response, f"streaks/{market_name}")
            if records is not None:
                results[market_name] = records
                if self.verbose:
                    count = len(records) if isinstance(records, list) else 1
                    logger.info(f"  {market_name}: {count} streak records")

        total = sum(len(v) for v in results.values() if isinstance(v, list))
        logger.info(f"[{self.source_name}] Streaks: {total} records across {len(results)} markets")

        return {
            "endpoint": "streaks",
            "date": self.date,
            "fetched_at": datetime.now(_EST).isoformat(),
            "markets": results,
            "total_records": total,
        }

    # ------------------------------------------------------------------
    # Injuries endpoint
    # ------------------------------------------------------------------

    def fetch_injuries(self) -> Dict[str, Any]:
        """Fetch injury reports from /v3/injuries.

        Unlike trends/streaks, this endpoint is not market-specific — it
        returns all NBA injuries in a single request.

        Returns:
            Dict with 'injuries' key containing list of injury records.
        """
        logger.info(f"[{self.source_name}] Fetching injuries for {self.date}")
        url = f"{self.API_BASE}{self.ENDPOINTS['injuries']}"

        params: Dict[str, Any] = {
            "sport": "NBA",
        }

        response = self._make_request(
            url=url,
            method="GET",
            params=params,
            headers=self._get_bp_headers(),
        )

        records = self._safe_parse_response(response, "injuries")

        if records is None:
            records = []

        count = len(records) if isinstance(records, list) else 1
        logger.info(f"[{self.source_name}] Injuries: {count} records")

        return {
            "endpoint": "injuries",
            "date": self.date,
            "fetched_at": datetime.now(_EST).isoformat(),
            "injuries": records,
            "total_records": count if isinstance(records, list) else 1,
        }

    # ------------------------------------------------------------------
    # Response parsing helper
    # ------------------------------------------------------------------

    def _safe_parse_response(
        self,
        response: Optional[Any],
        label: str,
    ) -> Optional[List[Dict[str, Any]]]:
        """Safely parse a BP API response, tolerating errors.

        Args:
            response: The requests.Response object (or None if request failed).
            label: Human-readable label for log messages (e.g. 'trends/POINTS').

        Returns:
            Parsed JSON data (list or dict) or None if parsing failed.
        """
        if response is None:
            logger.warning(f"[{self.source_name}] {label}: No response (request failed)")
            return None

        # Check for non-JSON responses (HTML error pages, rate limit pages)
        content_type = response.headers.get("content-type", "").lower()
        if "application/json" not in content_type:
            logger.warning(
                f"[{self.source_name}] {label}: Unexpected content-type '{content_type}' "
                f"({len(response.content)} bytes). Endpoint may not exist."
            )
            return None

        # Check for suspiciously small responses
        if len(response.content) < 10:
            logger.warning(
                f"[{self.source_name}] {label}: Empty response ({len(response.content)} bytes)"
            )
            return None

        try:
            data = response.json()
        except (ValueError, TypeError) as e:
            logger.warning(f"[{self.source_name}] {label}: JSON parse error: {e}")
            return None

        # BP responses typically wrap data in a top-level key.
        # Extract the most likely data payload.
        if isinstance(data, list):
            return data

        if isinstance(data, dict):
            # Try common BP response keys
            for key in ("props", "trends", "streaks", "injuries", "players", "data", "results"):
                if key in data:
                    return data[key] if isinstance(data[key], list) else [data[key]]

            # If none of the known keys exist, return the whole dict as a single-item list
            # so downstream can still inspect the raw response structure
            logger.info(
                f"[{self.source_name}] {label}: Unknown response structure. "
                f"Keys: {list(data.keys())}"
            )
            return [data]

        logger.warning(f"[{self.source_name}] {label}: Unexpected response type: {type(data)}")
        return None

    # ------------------------------------------------------------------
    # Saving results
    # ------------------------------------------------------------------

    def _save_endpoint_result(self, endpoint_name: str, data: Dict[str, Any]) -> Path:
        """Save a single endpoint's results to a dated JSON file.

        Args:
            endpoint_name: 'trends', 'streaks', or 'injuries'.
            data: The result dict from fetch_trends/fetch_streaks/fetch_injuries.

        Returns:
            Path to the saved file.
        """
        output_dir = Path(__file__).parent.parent / "lines"
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"bp_{endpoint_name}_{self.date}.json"
        output_file = output_dir / filename

        with open(output_file, "w") as f:
            json.dump(data, f, indent=2, default=str)

        if self.verbose:
            total = data.get("total_records", 0)
            logger.info(f"[{self.source_name}] Saved {total} records to {output_file}")

        return output_file

    # ------------------------------------------------------------------
    # Main fetch() — required by BaseFetcher
    # ------------------------------------------------------------------

    def fetch(self) -> List[Dict[str, Any]]:
        """Fetch from selected endpoints and return combined results.

        BaseFetcher requires this to return List[Dict], but since this fetcher
        produces analytics data (not standardized props), we return a list
        containing one dict per endpoint result.

        Returns:
            List of endpoint result dicts (one per endpoint fetched).
        """
        results: List[Dict[str, Any]] = []
        endpoints_to_fetch = self._get_endpoints_to_fetch()

        if self.verbose:
            print(f"\n{'=' * 60}")
            print(f"BETTINGPROS ANALYTICS EXPLORER")
            print(f"{'=' * 60}")
            print(f"Date:      {self.date}")
            print(f"Endpoints: {', '.join(endpoints_to_fetch)}")
            print(f"{'=' * 60}\n", flush=True)

        for endpoint_name in endpoints_to_fetch:
            if self.verbose:
                print(f"--- {endpoint_name.upper()} ---", flush=True)

            try:
                if endpoint_name == "trends":
                    data = self.fetch_trends()
                elif endpoint_name == "streaks":
                    data = self.fetch_streaks()
                elif endpoint_name == "injuries":
                    data = self.fetch_injuries()
                else:
                    continue

                results.append(data)

                if self.verbose:
                    total = data.get("total_records", 0)
                    print(f"  Result: {total} records\n", flush=True)

            except Exception as e:
                logger.warning(
                    f"[{self.source_name}] {endpoint_name} fetch failed: {e}",
                    exc_info=True,
                )
                if self.verbose:
                    print(f"  FAILED: {e}\n", flush=True)

        # Summary
        if self.verbose:
            print(f"{'=' * 60}")
            print(f"ANALYTICS EXPLORER SUMMARY")
            print(f"{'=' * 60}")
            for r in results:
                ep = r.get("endpoint", "unknown")
                total = r.get("total_records", 0)
                print(f"  {ep:12s}: {total} records")
            print(f"{'=' * 60}\n", flush=True)

        # Atlas data registry
        try:
            from nba.core.data_registry import log_ingestion

            stats = self.get_registry_stats()
            total_records = sum(r.get("total_records", 0) for r in results)
            log_ingestion(
                "bettingpros_trends",
                "fetch",
                "success",
                records_fetched=total_records,
                api_calls_made=stats["api_calls_made"],
                bytes_transferred=stats["bytes_transferred"],
                error_count=stats["error_count"],
                error_message=stats["error_message"],
                metadata={
                    "game_date": self.date,
                    "endpoints": endpoints_to_fetch,
                },
            )
        except Exception:
            pass

        return results

    def fetch_and_save(self) -> Dict[str, Path]:
        """Fetch from selected endpoints and save each to its own JSON file.

        Returns:
            Dict mapping endpoint name -> saved file path.
        """
        results = self.fetch()
        saved_files: Dict[str, Path] = {}

        for result in results:
            endpoint_name = result.get("endpoint", "unknown")
            path = self._save_endpoint_result(endpoint_name, result)
            saved_files[endpoint_name] = path

        return saved_files

    def _get_endpoints_to_fetch(self) -> List[str]:
        """Return list of endpoint names based on the endpoint filter."""
        if self.endpoint_filter == "all":
            return ["trends", "streaks", "injuries"]
        return [self.endpoint_filter]


def main() -> None:
    """CLI entry point for exploratory BettingPros analytics fetching."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Fetch BettingPros analytics data (trends, streaks, injuries)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch all three endpoints
  python fetch_bettingpros_trends.py

  # Fetch trends only and save to JSON
  python fetch_bettingpros_trends.py --endpoint trends --save

  # Fetch streaks for a specific date
  python fetch_bettingpros_trends.py --endpoint streaks --date 2026-03-15

  # Quiet mode (less output)
  python fetch_bettingpros_trends.py --save --quiet

Endpoints:
  trends    /v3/props/trends   — Line movement data over time
  streaks   /v3/props/streaks  — Player hit/miss streak information
  injuries  /v3/injuries       — BettingPros injury reports

NOTE: These are exploratory endpoints. They may return 404 or unexpected
formats. The fetcher logs warnings and continues gracefully.
        """,
    )
    parser.add_argument(
        "--date",
        type=str,
        help="Date to fetch (YYYY-MM-DD). Defaults to today EST.",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        choices=["trends", "streaks", "injuries", "all"],
        default="all",
        help="Which endpoint to fetch (default: all)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save results to JSON files in nba/betting_xl/lines/",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce logging output",
    )

    args = parser.parse_args()

    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    with BettingProsTrendsFetcher(
        date=args.date,
        endpoint=args.endpoint,
        verbose=not args.quiet,
    ) as fetcher:
        if args.save:
            saved = fetcher.fetch_and_save()
            if saved:
                for ep, path in saved.items():
                    print(f"Saved {ep} -> {path}")
            else:
                print("No data fetched from any endpoint")
        else:
            results = fetcher.fetch()
            if not results:
                print("No data fetched from any endpoint")
                return

            # Print a sample of data from each endpoint
            for result in results:
                ep = result.get("endpoint", "unknown")
                total = result.get("total_records", 0)
                print(f"\n{ep.upper()}: {total} records")

                # Show structure preview
                if ep == "injuries":
                    injuries = result.get("injuries", [])
                    for inj in injuries[:5]:
                        if isinstance(inj, dict):
                            name = inj.get("player_name", inj.get("name", "unknown"))
                            status = inj.get("status", inj.get("injury_status", ""))
                            print(f"  {name}: {status}")
                    if len(injuries) > 5:
                        print(f"  ... and {len(injuries) - 5} more")
                else:
                    markets = result.get("markets", {})
                    for market_name, records in markets.items():
                        if isinstance(records, list):
                            print(f"  {market_name}: {len(records)} records")
                            if records:
                                # Show first record keys for structure discovery
                                first = records[0]
                                if isinstance(first, dict):
                                    keys = list(first.keys())[:10]
                                    print(f"    Keys: {keys}")


if __name__ == "__main__":
    main()
