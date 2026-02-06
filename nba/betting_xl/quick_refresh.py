#!/usr/bin/env python3
"""
Quick Refresh - Lightweight pipeline for Discord bot.
Runs essential steps with minimal output and parallel fetching.

Usage:
    python3 -m nba.betting_xl.quick_refresh
    python3 nba/betting_xl/quick_refresh.py
"""

import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
PREDICTIONS_DIR = SCRIPT_DIR / "predictions"
DATE_STR = datetime.now().strftime("%Y-%m-%d")


def run_step(name, cmd, timeout=120):
    """Run a pipeline step, return (name, success, elapsed, output)."""
    t0 = time.time()
    try:
        result = subprocess.run(  # nosec B602 - commands are hardcoded, not user input
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(PROJECT_ROOT),
        )
        elapsed = time.time() - t0
        ok = result.returncode == 0
        output = result.stdout[-500:] if ok else (result.stderr[-500:] or result.stdout[-500:])
        return name, ok, elapsed, output
    except subprocess.TimeoutExpired:
        return name, False, time.time() - t0, "TIMEOUT"
    except Exception as e:
        return name, False, time.time() - t0, str(e)


def main():
    t_start = time.time()
    print(f"Quick Refresh - {DATE_STR}")
    print("=" * 50)

    # Phase 1: Parallel data fetching
    print("\n[1/4] Fetching data (parallel)...")
    fetch_steps = {
        "Props": f"python3 nba/betting_xl/fetchers/fetch_all.py",
        "PrizePicks": f"python3 nba/betting_xl/loaders/load_prizepicks_to_db.py --fetch --quiet",
        "Cheatsheet": f"python3 nba/betting_xl/fetchers/fetch_cheatsheet.py --platform underdog",
    }

    fetch_results = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(run_step, name, cmd, 180): name for name, cmd in fetch_steps.items()}
        for future in as_completed(futures):
            name, ok, elapsed, output = future.result()
            fetch_results[name] = ok
            status = "OK" if ok else "FAIL"
            print(f"  {name}: {status} ({elapsed:.0f}s)")

    # Load props to DB (needs props fetch done first)
    if fetch_results.get("Props"):
        import glob

        props_files = sorted(
            glob.glob(str(SCRIPT_DIR / "lines" / "all_sources_*.json")), reverse=True
        )
        if props_files:
            name, ok, elapsed, _ = run_step(
                "Load Props",
                f"python3 nba/betting_xl/loaders/load_props_to_db.py --file {props_files[0]} --skip-mongodb",
                timeout=60,
            )
            print(f"  Load Props: {'OK' if ok else 'FAIL'} ({elapsed:.0f}s)")
            if not ok:
                print("ERROR: Props load failed")
                sys.exit(1)
    else:
        print("ERROR: Props fetch failed")
        sys.exit(1)

    # Load cheatsheet to DB
    if fetch_results.get("Cheatsheet"):
        import glob

        cs_files = sorted(
            glob.glob(str(SCRIPT_DIR / "lines" / "cheatsheet_underdog_*.json")), reverse=True
        )
        if cs_files:
            run_step(
                "Load Cheatsheet",
                f"python3 nba/betting_xl/loaders/load_cheatsheet_to_db.py --file {cs_files[0]}",
                timeout=30,
            )

    # Phase 2: Parallel updates
    print("\n[2/4] Updating context (parallel)...")
    update_steps = {
        "Injuries": "python3 nba/scripts/update_injuries_NOW.py",
        "Vegas": f"python3 nba/betting_xl/fetchers/fetch_vegas_lines.py --date {DATE_STR} --save-to-db",
    }
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(run_step, name, cmd, 60): name for name, cmd in update_steps.items()}
        for future in as_completed(futures):
            name, ok, elapsed, _ = future.result()
            print(f"  {name}: {'OK' if ok else 'SKIP'} ({elapsed:.0f}s)")

    # Phase 3: Enrichment (sequential, needs props loaded)
    print("\n[3/4] Enriching matchups...")
    name, ok, elapsed, _ = run_step(
        "Enrich",
        f"python3 nba/betting_xl/enrich_props_with_matchups.py --date {DATE_STR}",
        timeout=60,
    )
    print(f"  Enrich: {'OK' if ok else 'FAIL'} ({elapsed:.0f}s)")

    # Phase 4: Parallel prediction generation
    print("\n[4/4] Generating picks (parallel)...")
    xl_file = PREDICTIONS_DIR / f"xl_picks_{DATE_STR}.json"
    pro_file = PREDICTIONS_DIR / f"pro_picks_{DATE_STR}.json"
    odds_file = PREDICTIONS_DIR / f"odds_api_picks_{DATE_STR.replace('-', '')}.json"
    energy_file = PREDICTIONS_DIR / f"two_energy_picks_{DATE_STR}.json"

    pred_steps = {
        "XL": f"python3 nba/betting_xl/generate_xl_predictions.py --output {xl_file} --underdog-only",
        "PRO": f"python3 nba/betting_xl/generate_cheatsheet_picks.py --output {pro_file}",
        "Odds API": f"python3 nba/betting_xl/generate_odds_api_picks.py --date {DATE_STR} --output {odds_file}",
        "Two Energy": f"python3 -m nba.betting_xl.generate_two_energy_picks --date {DATE_STR} --output {energy_file}",
    }

    pick_counts = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(run_step, name, cmd, 120): name for name, cmd in pred_steps.items()}
        for future in as_completed(futures):
            name, ok, elapsed, output = future.result()
            status = "OK" if ok else "FAIL"
            print(f"  {name}: {status} ({elapsed:.0f}s)")
            pick_counts[name] = 0

    # Count picks from files
    for name, filepath in [
        ("XL", xl_file),
        ("PRO", pro_file),
        ("Odds API", odds_file),
        ("Two Energy", energy_file),
    ]:
        try:
            if filepath.exists():
                data = json.loads(filepath.read_text())
                pick_counts[name] = data.get("total_picks", len(data.get("picks", [])))
        except Exception:
            pass

    # Summary
    total_elapsed = time.time() - t_start
    total_picks = sum(pick_counts.values())
    print(f"\n{'=' * 50}")
    print(f"Refresh Complete ({total_elapsed:.0f}s)")
    print(
        f"  XL: {pick_counts.get('XL', 0)} | PRO: {pick_counts.get('PRO', 0)} | "
        f"Odds API: {pick_counts.get('Odds API', 0)} | Two Energy: {pick_counts.get('Two Energy', 0)}"
    )
    print(f"  Total: {total_picks} picks")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
