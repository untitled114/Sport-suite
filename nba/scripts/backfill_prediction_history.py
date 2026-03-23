#!/usr/bin/env python3
"""
Backfill nba_prediction_history from JSON pick files.

Reads all xl_picks_*.json files, inserts picks into nba_prediction_history
via write_picks(), then grades them via write_actuals().

Usage:
    python3 -m nba.scripts.backfill_prediction_history
    python3 -m nba.scripts.backfill_prediction_history --dry-run
"""

import argparse
import json
import re
import sys
from pathlib import Path

_PREDICTIONS_DIR = Path(__file__).resolve().parent.parent / "betting_xl" / "predictions"


def _extract_date(filename: str) -> str | None:
    """Extract date from filename like xl_picks_2026-02-27.json or xl_picks_20251110.json."""
    m = re.search(r"xl_picks_(\d{4}-\d{2}-\d{2})", filename)
    if m:
        return m.group(1)
    m = re.search(r"xl_picks_(\d{8})", filename)
    if m:
        d = m.group(1)
        return f"{d[:4]}-{d[4:6]}-{d[6:8]}"
    return None


def backfill(dry_run: bool = False):
    from nba.core.axiom_writer import write_actuals, write_picks

    files = sorted(_PREDICTIONS_DIR.glob("xl_picks_*.json"))
    print(f"Found {len(files)} pick files in {_PREDICTIONS_DIR}")

    total_inserted = 0
    total_graded = 0
    dates_processed = []

    for f in files:
        run_date = _extract_date(f.name)
        if not run_date:
            print(f"  SKIP {f.name} (can't parse date)")
            continue

        try:
            with open(f) as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  SKIP {f.name} ({e})")
            continue

        picks = data.get("picks", [])
        if not picks:
            continue

        timestamp = data.get("generated_at", f"{run_date}T12:00:00")

        if dry_run:
            print(f"  {run_date}: {len(picks)} picks (dry run)")
            dates_processed.append(run_date)
            continue

        inserted = write_picks(run_date, 1, timestamp, picks)
        total_inserted += inserted

        graded = write_actuals(run_date)
        total_graded += graded

        status = f"{inserted} inserted, {graded} graded"
        print(f"  {run_date}: {len(picks)} picks -> {status}")
        dates_processed.append(run_date)

    print(f"\nDone: {len(dates_processed)} dates, {total_inserted} inserted, {total_graded} graded")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    backfill(dry_run=args.dry_run)
