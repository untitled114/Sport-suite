#!/usr/bin/env python3
"""BettingPros hit-rate fetcher for XL sandbox."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from nba.betting_xl.fetchers.base_fetcher import BaseFetcher


class BettingProsHitRateFetcher(BaseFetcher):
    """Fetch consensus props and derived hit rates from BettingPros."""

    API_URL = "https://api.bettingpros.com/v3/props"
    # FIXED 2026-01-03: assists was 160 (STEALS), correct is 151
    MARKETS = {
        "points": 156,
        "rebounds": 157,
        "assists": 151,  # FIXED: Was 160 (steals)
        "threes": 162,
    }
    HEADERS = {
        "x-api-key": os.getenv("BETTINGPROS_API_KEY"),
        "x-level": "cHJlbWl1bQ==",
        "accept": "application/json",
    }

    def __init__(self, date: Optional[str] = None, verbose: bool = True):
        super().__init__(
            source_name="bettingpros_hit_rates",
            rate_limit=1.0,
            max_retries=3,
            timeout=30,
            verbose=verbose,
        )
        self.date = date or datetime.now().strftime("%Y-%m-%d")

    def fetch_market(self, market_name: str, market_id: int) -> List[Dict[str, Any]]:
        page = 1
        records: List[Dict[str, Any]] = []

        while True:
            params = {
                "sport": "NBA",
                "date": self.date,
                "market_id": market_id,
                "limit": 500,
                "page": page,
                "include_markets": "true",
                "include_counts": "true",
            }

            response = self._make_request(
                url=self.API_URL,
                method="GET",
                params=params,
                headers=self.HEADERS,
            )

            if not response:
                break

            try:
                payload = response.json()
            except ValueError:
                break

            props = payload.get("props", [])
            if not props:
                break

            for prop in props:
                parsed = self._parse_prop(prop, market_name)
                if parsed:
                    records.append(parsed)

            pagination = payload.get("_pagination", {})
            total_pages = pagination.get("total_pages", 1)
            if page >= total_pages:
                break
            page += 1

        return records

    def _parse_prop(self, raw_prop: Dict[str, Any], market_name: str) -> Optional[Dict[str, Any]]:
        participant = raw_prop.get("participant", {})
        player = participant.get("player", {})
        player_name = self.normalize_player_name(participant.get("name"))
        if not player_name:
            return None

        performance = raw_prop.get("performance") or {}
        if not performance:
            return None

        stat_type = self.normalize_stat_type(market_name)
        line = raw_prop.get("over", {}).get("consensus_line")
        if line is None:
            line = raw_prop.get("under", {}).get("consensus_line")

        hit_rates, sample_sizes = self._build_hit_rates(performance)

        record = {
            "player_name": player_name,
            "player_slug": player.get("slug"),
            "player_id": participant.get("id"),
            "stat_type": stat_type,
            "market_id": raw_prop.get("market_id"),
            "event_id": raw_prop.get("event_id"),
            "game_date": self.date,
            "line": float(line) if line is not None else None,
            "hit_rates": hit_rates,
            "samples": sample_sizes,
            "raw_performance": performance,
        }
        return record

    def _build_hit_rates(
        self, performance: Dict[str, Dict[str, int]]
    ) -> Tuple[Dict[str, Any], Dict[str, int]]:
        hit_rates: Dict[str, Any] = {}
        samples: Dict[str, int] = {}

        for window, splits in performance.items():
            if not isinstance(splits, dict):
                continue
            over = splits.get("over", 0) or 0
            under = splits.get("under", 0) or 0
            push = splits.get("push", 0) or 0
            total = over + under + push
            samples[window] = total
            if total > 0:
                hit_rates[window] = {
                    "rate": round(over / total, 3),
                    "over": over,
                    "under": under,
                    "push": push,
                }
            else:
                hit_rates[window] = {
                    "rate": None,
                    "over": over,
                    "under": under,
                    "push": push,
                }

        return hit_rates, samples

    def fetch(self) -> List[Dict[str, Any]]:
        all_records: List[Dict[str, Any]] = []
        for market_name, market_id in self.MARKETS.items():
            market_records = self.fetch_market(market_name, market_id)
            all_records.extend(market_records)

        deduped: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for record in all_records:
            key = (record["player_name"], record["stat_type"])
            deduped[key] = record

        return list(deduped.values())

    def save_hit_rates(self, records: List[Dict[str, Any]]) -> Path:
        output_dir = Path(__file__).parent.parent / "lines"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"bettingpros_hit_rates_{self.date}.json"

        payload = {
            "date": self.date,
            "generated_at": datetime.now().isoformat(),
            "total_records": len(records),
            "records": records,
        }

        with open(output_file, "w") as f:
            json.dump(payload, f, indent=2)

        return output_file


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fetch BettingPros hit-rate data (consensus).")
    parser.add_argument("--date", type=str, help="Date (YYYY-MM-DD). Defaults to today.")
    parser.add_argument("--quiet", action="store_true", help="Reduce logging output")
    args = parser.parse_args()

    with BettingProsHitRateFetcher(date=args.date, verbose=not args.quiet) as fetcher:
        records = fetcher.fetch()
        if not records:
            print("⚠️  No hit-rate records fetched")
            return

        output_file = fetcher.save_hit_rates(records)
        print(f"✅ Saved {len(records)} hit-rate records to {output_file}")


if __name__ == "__main__":
    main()
