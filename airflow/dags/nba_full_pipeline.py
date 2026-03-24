"""
NBA Full Pipeline DAG

Scheduled: Every 3 hours from 2 AM to 8 PM EST (7 runs/day)
Purpose: Complete data collection + predictions

Runs 7x daily: 2AM, 5AM, 8AM, 11AM, 2PM, 5PM, 8PM EST.
Each run fetches fresh lines from all sources, generates XL/V3 predictions,
and writes to nba_prediction_history for conviction scoring across runs.

Run 1 (2 AM) includes heavy one-time tasks: game results, team stats,
matchup history, prop history, minutes projections.
Runs 2-7 skip those and focus on line refresh + predictions.

All runs gate on T-120: if first game tip is within 120 minutes, the run
is skipped (picks are locked, no more changes).

Author: Claude Code
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from callbacks import dag_failure, on_failure, on_retry, on_success

from airflow.decorators import dag, task
from airflow.models import Variable
from airflow.timetables.trigger import CronTriggerTimetable

# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Variable.get("nba_project_root", default_var="/home/untitled/Sport-suite")
SCRIPT_DIR = f"{PROJECT_ROOT}/nba"
PREDICTIONS_DIR = f"{SCRIPT_DIR}/betting_xl/predictions"

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

default_args = {
    "owner": "nba_pipeline",
    "depends_on_past": False,
    "email": Variable.get("alert_email", default_var="alerts@example.com").split(","),
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=30),
    "execution_timeout": timedelta(minutes=30),
    "on_success_callback": on_success,
    "on_failure_callback": on_failure,
    "on_retry_callback": on_retry,
}

# T-120 gate: skip if first game is within this many minutes
_TIPOFF_CUTOFF_MINUTES = 120


def get_current_season() -> int:
    """Calculate NBA season (uses END year: 2024-25 season = 2025)."""
    now = datetime.now()
    return now.year + 1 if now.month >= 10 else now.year


def get_prizepicks_count(date_str: str) -> int:
    """Get count of PrizePicks props for a given date."""
    import psycopg2

    from nba.config.database import get_intelligence_db_config

    config = get_intelligence_db_config()
    conn = psycopg2.connect(**config)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT COUNT(*) FROM nba_props_xl
            WHERE game_date = %s AND book_name LIKE 'prizepicks%%'
            """,
            (date_str,),
        )
        return cursor.fetchone()[0]
    finally:
        cursor.close()
        conn.close()


def run_script(
    script_path: str,
    args: list[str] | None = None,
    timeout: int = 300,
    raise_on_error: bool = False,
) -> dict[str, Any]:
    """Helper to run a Python script with standard error handling."""
    import subprocess

    if not Path(script_path).exists():
        if raise_on_error:
            raise Exception(f"Script not found: {script_path}")
        return {"status": "skipped", "reason": f"Script not found: {script_path}"}

    cmd = ["python3", script_path] + (args or [])
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        env={**os.environ, "PYTHONPATH": PROJECT_ROOT},
        timeout=timeout,
    )

    if result.returncode != 0:
        error_output = result.stderr or result.stdout
        if raise_on_error:
            raise Exception(
                f"{Path(script_path).name} failed (exit {result.returncode}): "
                f"{error_output[-500:]}"
            )
        return {"status": "error", "error": result.stderr, "stdout": result.stdout}

    return {"status": "success", "stdout": result.stdout}


# ============================================================================
# DAG Definition
# ============================================================================


@dag(
    dag_id="nba_full_pipeline",
    description="NBA full pipeline — every 3hr from 2:30AM-8:30PM EST, T-120 gated",
    schedule=CronTriggerTimetable("30 2,5,8,11,14,17,20 * * *", timezone="America/New_York"),
    start_date=datetime(2025, 11, 7),
    catchup=False,
    tags=["nba", "predictions", "full", "data-collection"],
    default_args=default_args,
    max_active_runs=1,
    doc_md=__doc__,
    on_failure_callback=dag_failure,
)
def nba_full_pipeline():
    """
    NBA Full Pipeline — 7 runs/day (2AM-8PM EST, every 3 hours).

    Run 1 (2 AM): Full pipeline with heavy one-time tasks.
    Runs 2-7: Line refresh + predictions only.
    All runs: T-120 gate (skip if within 120 min of first tip).
    """

    # ========================================================================
    # T-120 Gate + Audit Start
    # ========================================================================

    @task(task_id="gate_and_audit")
    def gate_and_audit() -> dict[str, Any]:
        """T-120 gate: skip entire pipeline if first game is within 120 min.

        Also records pipeline start in axiom_pipeline_audit.
        """
        import urllib.request
        from zoneinfo import ZoneInfo

        from nba.core.axiom_writer import audit_run_start as _start
        from nba.core.axiom_writer import get_run_number

        est = ZoneInfo("America/New_York")
        now = datetime.now(est)
        run_date = now.strftime("%Y-%m-%d")
        run_number = get_run_number()
        is_first_run = run_number == 1

        # Fetch first tip time from ESPN
        minutes_to_tip = None
        tip_str = "unknown"
        try:
            date_compact = run_date.replace("-", "")
            url = (
                "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
                f"?dates={date_compact}"
            )
            with urllib.request.urlopen(url, timeout=10) as resp:  # nosec B310
                data = json.loads(resp.read())

            times = []
            for event in data.get("events", []):
                date_str = event.get("date", "")
                if date_str:
                    dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                    times.append(dt.astimezone(est))

            if not times:
                print(f"[INFO] No NBA games found for {run_date}")
                _start(run_date, run_number=run_number, run_type="full")
                return {
                    "run_date": run_date,
                    "run_number": run_number,
                    "is_first_run": is_first_run,
                    "status": "no_games",
                    "started_at": datetime.now().isoformat(),
                }

            first_tip = min(times)
            tip_str = first_tip.strftime("%-I:%M %p ET")
            minutes_to_tip = (first_tip - now).total_seconds() / 60
        except Exception as e:
            print(f"[WARN] ESPN tip time check failed: {e} — continuing anyway")

        # T-120 gate: skip if we're within 120 min of first tip
        if minutes_to_tip is not None and 0 < minutes_to_tip < _TIPOFF_CUTOFF_MINUTES:
            print(
                f"[GATE] {minutes_to_tip:.0f} min to first tip ({tip_str}) — "
                f"within T-{_TIPOFF_CUTOFF_MINUTES} cutoff. Picks are locked."
            )
            return {
                "run_date": run_date,
                "run_number": run_number,
                "is_first_run": is_first_run,
                "status": "gated",
                "minutes_to_tip": minutes_to_tip,
                "started_at": datetime.now().isoformat(),
            }

        # Also skip if games already started (negative minutes_to_tip)
        if minutes_to_tip is not None and minutes_to_tip < 0:
            print(
                f"[GATE] First game already tipped ({tip_str}, {abs(minutes_to_tip):.0f}m ago). Skipping."
            )
            return {
                "run_date": run_date,
                "run_number": run_number,
                "is_first_run": is_first_run,
                "status": "gated",
                "minutes_to_tip": minutes_to_tip,
                "started_at": datetime.now().isoformat(),
            }

        run_type = "full" if is_first_run else "refresh"
        print(
            f"[RUN {run_number}] {run_type} pipeline | "
            f"{'First tip: ' + tip_str + f' ({minutes_to_tip:.0f}m away)' if minutes_to_tip else 'Tip time unknown'}"
        )

        _start(run_date, run_number=run_number, run_type=run_type)
        return {
            "run_date": run_date,
            "run_number": run_number,
            "is_first_run": is_first_run,
            "status": "running",
            "minutes_to_tip": minutes_to_tip,
            "started_at": datetime.now().isoformat(),
        }

    # ========================================================================
    # Data Collection Tasks
    # ========================================================================

    @task(task_id="fetch_props")
    def fetch_props(gate: dict[str, Any]) -> dict[str, Any]:
        """Fetch props from 7 sportsbooks."""
        if gate["status"] in ("gated", "no_games"):
            return {"props_file": None, "status": gate["status"]}

        import glob

        result = run_script(f"{SCRIPT_DIR}/betting_xl/fetchers/fetch_all.py", timeout=600)
        if result["status"] != "success":
            raise Exception(f"fetch_props failed: {result.get('error')}")

        pattern = f"{SCRIPT_DIR}/betting_xl/lines/all_sources_*.json"
        files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)

        if not files:
            print("[INFO] No props file generated - likely no NBA games today")
            return {"props_file": None, "status": "no_games"}

        return {"props_file": files[0], "status": "success"}

    @task(task_id="load_props")
    def load_props(fetch_result: dict[str, Any]) -> dict[str, Any]:
        """Load props to database."""
        if fetch_result.get("status") in ("gated", "no_games"):
            return {"status": fetch_result["status"]}

        props_file = fetch_result.get("props_file")
        if not props_file:
            raise Exception("No props file to load")

        result = run_script(
            f"{SCRIPT_DIR}/betting_xl/loaders/load_props_to_db.py",
            ["--file", props_file, "--skip-mongodb"],
        )
        if result["status"] == "error":
            raise Exception(f"load_props failed: {result.get('error')}")
        return {"status": "success"}

    @task(task_id="fetch_cheatsheet")
    def fetch_cheatsheet(gate: dict[str, Any]) -> dict[str, Any]:
        """Fetch BettingPros cheatsheet (Underdog lines + BP analytics)."""
        if gate["status"] in ("gated", "no_games"):
            return {"status": gate["status"]}

        import glob

        result = run_script(
            f"{SCRIPT_DIR}/betting_xl/fetchers/fetch_cheatsheet.py",
            ["--platform", "underdog"],
        )
        if result["status"] in ("skipped", "error"):
            if result["status"] == "error":
                print(
                    f"[WARN] fetch_cheatsheet failed (non-critical): {result.get('error', '')[-300:]}"
                )
            return result

        pattern = f"{SCRIPT_DIR}/betting_xl/lines/cheatsheet_underdog_*.json"
        files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)

        if files:
            load_result = run_script(
                f"{SCRIPT_DIR}/betting_xl/loaders/load_cheatsheet_to_db.py",
                ["--file", files[0]],
            )
            if load_result["status"] == "error":
                print(
                    f"[WARN] load_cheatsheet failed (non-critical): {load_result.get('error', '')[-300:]}"
                )

        return {"status": "success"}

    @task(task_id="fetch_hit_rates")
    def fetch_hit_rates(gate: dict[str, Any]) -> dict[str, Any]:
        """Fetch BettingPros consensus hit rates, streaks, and projections."""
        if gate["status"] in ("gated", "no_games"):
            return {"status": gate["status"]}

        from zoneinfo import ZoneInfo

        from nba.betting_xl.fetchers.fetch_bettingpros_hit_rates import BettingProsHitRateFetcher

        date_str = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
        try:
            with BettingProsHitRateFetcher(date=date_str, verbose=False) as fetcher:
                records = fetcher.fetch()
                if records:
                    output_file = fetcher.save_hit_rates(records)
                    print(f"[OK] Hit rates: {len(records)} records -> {output_file.name}")
                    return {"status": "success", "records": len(records)}
                print("[WARN] fetch_hit_rates: no records returned")
                return {"status": "no_data", "records": 0}
        except Exception as exc:
            print(f"[WARN] fetch_hit_rates failed (non-critical): {exc}")
            return {"status": "error", "error": str(exc)}

    @task(task_id="fetch_pick_recs")
    def fetch_pick_recs(gate: dict[str, Any]) -> dict[str, Any]:
        """Fetch BettingPros session-gated pick recommendations."""
        if gate["status"] in ("gated", "no_games"):
            return {"status": gate["status"]}

        from zoneinfo import ZoneInfo

        from nba.betting_xl.fetchers.fetch_pick_recommendations import (
            fetch_pick_recommendations,
            save_recommendations,
        )

        date_str = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
        try:
            picks = fetch_pick_recommendations(date_str)
            if picks:
                output_file = save_recommendations(date_str, picks)
                print(f"[OK] Pick recommendations: {len(picks)} picks -> {output_file.name}")
                return {"status": "success", "picks": len(picks)}
            print("[WARN] fetch_pick_recs: no picks returned (session expired?)")
            return {"status": "no_data", "picks": 0}
        except Exception as exc:
            print(f"[WARN] fetch_pick_recs failed (non-critical): {exc}")
            return {"status": "error", "error": str(exc)}

    @task(task_id="fetch_prizepicks")
    def fetch_prizepicks(gate: dict[str, Any]) -> dict[str, Any]:
        """Fetch PrizePicks props (standard lines only)."""
        if gate["status"] in ("gated", "no_games"):
            return {"status": gate["status"]}

        from zoneinfo import ZoneInfo

        date_str = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")

        run_script(
            f"{SCRIPT_DIR}/betting_xl/loaders/load_prizepicks_to_db.py",
            ["--fetch", "--quiet"],
            timeout=300,
        )

        pp_count = get_prizepicks_count(date_str)
        if pp_count == 0:
            print(f"[WARN] No PrizePicks data for {date_str} — GOLDMINE picks unavailable")
            return {"status": "no_data", "prizepicks_count": 0}

        print(f"[OK] PrizePicks loaded: {pp_count} props for {date_str}")
        return {"status": "success", "prizepicks_count": pp_count}

    @task(task_id="fetch_direct_sportsbooks")
    def fetch_direct_sportsbooks(gate: dict[str, Any]) -> dict[str, Any]:
        """Fetch props directly from all Colorado-legal sportsbooks."""
        if gate["status"] in ("gated", "no_games"):
            return {"status": gate["status"]}

        import glob

        result = run_script(
            f"{SCRIPT_DIR}/betting_xl/fetchers/fetch_all.py",
            ["--direct-only"],
            timeout=600,
        )
        if result["status"] == "error":
            # Direct fetching is non-critical — log and continue
            print(
                f"[WARN] Direct sportsbook fetch failed (non-critical): {result.get('error', '')[-300:]}"
            )
            return {"status": "partial", "error": result.get("error", "")[-300:]}

        # Load any direct props JSON files
        pattern = f"{SCRIPT_DIR}/betting_xl/lines/all_sources_*.json"
        files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
        if files:
            load_result = run_script(
                f"{SCRIPT_DIR}/betting_xl/loaders/load_props_to_db.py",
                ["--file", files[0], "--skip-mongodb"],
            )
            if load_result["status"] == "error":
                print(f"[WARN] Direct props load failed: {load_result.get('error', '')[-300:]}")

        return {"status": "success"}

    @task(task_id="enrich_matchups")
    def enrich_matchups(load_result: dict[str, Any]) -> dict[str, Any]:
        """Enrich props with matchup context."""
        if load_result.get("status") in ("gated", "no_games"):
            return {"coverage": 0, "total": 0, "status": load_result["status"]}

        from zoneinfo import ZoneInfo

        import psycopg2

        from nba.config.database import get_intelligence_db_config

        date_str = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
        run_script(
            f"{SCRIPT_DIR}/betting_xl/enrich_props_with_matchups.py",
            ["--date", date_str],
        )

        min_coverage = float(Variable.get("nba_min_coverage_pct", default_var="90"))
        config = get_intelligence_db_config()
        conn = psycopg2.connect(**config)
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT COUNT(*) as total,
                       COUNT(CASE WHEN opponent_team <> '' AND opponent_team IS NOT NULL
                             AND is_home IS NOT NULL THEN 1 END) as enriched
                FROM nba_props_xl WHERE game_date = %s;
                """,
                (date_str,),
            )
            row = cursor.fetchone()
        finally:
            cursor.close()
            conn.close()

        total, enriched = row if row else (0, 0)
        coverage = round(100.0 * enriched / total, 1) if total > 0 else 0

        if total > 0 and coverage < min_coverage:
            raise Exception(f"Coverage {coverage}% below {min_coverage}% threshold")

        return {"coverage": coverage, "total": total, "status": "success"}

    # ========================================================================
    # Heavy One-Time Tasks (Run 1 only)
    # ========================================================================

    @task(task_id="fetch_game_results")
    def fetch_game_results(gate: dict[str, Any]) -> dict[str, Any]:
        """Fetch yesterday's game results. Run 1 only."""
        if gate["status"] in ("gated", "no_games"):
            return {"status": gate["status"]}
        if not gate.get("is_first_run"):
            print("[SKIP] Not first run — game results already fetched today")
            return {"status": "skipped"}

        from datetime import date

        import psycopg2

        from nba.config.database import get_players_db_config

        result = run_script(
            f"{SCRIPT_DIR}/scripts/fetch_daily_stats.py",
            ["--days", "1"],
            timeout=600,
            raise_on_error=True,
        )

        config = get_players_db_config()
        conn = psycopg2.connect(**config)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(game_date) FROM player_game_logs;")
            latest = cursor.fetchone()[0]
        finally:
            cursor.close()
            conn.close()

        if latest is None:
            raise Exception("player_game_logs is empty")

        days_stale = (date.today() - latest).days
        if days_stale > 3:
            raise Exception(f"player_game_logs is {days_stale} days stale (latest: {latest})")

        print(f"[OK] player_game_logs current: latest={latest} ({days_stale}d ago)")
        return result

    @task(task_id="populate_actuals")
    def populate_actuals(game_results: dict[str, Any]) -> dict[str, Any]:
        """Populate actual values for recent props. Run 1 only."""
        if game_results.get("status") in ("gated", "no_games", "skipped"):
            return {"status": game_results["status"]}
        return run_script(f"{SCRIPT_DIR}/betting_xl/populate_actual_values.py", ["--days", "7"])

    @task(task_id="update_injuries")
    def update_injuries(gate: dict[str, Any]) -> dict[str, Any]:
        """Update injury reports. Every run (game-time decisions matter)."""
        if gate["status"] in ("gated", "no_games"):
            return {"status": gate["status"]}
        return run_script(f"{SCRIPT_DIR}/scripts/update_injuries_NOW.py")

    @task(task_id="update_team_stats")
    def update_team_stats(gate: dict[str, Any]) -> dict[str, Any]:
        """Update team statistics. Run 1 only."""
        if gate["status"] in ("gated", "no_games"):
            return {"status": gate["status"]}
        if not gate.get("is_first_run"):
            print("[SKIP] Not first run — team stats already updated today")
            return {"status": "skipped"}

        season = str(get_current_season())
        run_script(f"{SCRIPT_DIR}/scripts/loaders/load_nba_games_incremental.py")
        run_script(f"{SCRIPT_DIR}/scripts/loaders/calculate_team_stats.py", ["--season", season])
        run_script(f"{SCRIPT_DIR}/scripts/loaders/load_team_advanced_stats.py")
        return {"status": "success"}

    @task(task_id="fetch_vegas")
    def fetch_vegas(gate: dict[str, Any]) -> dict[str, Any]:
        """Fetch vegas lines. Every run (lines move)."""
        if gate["status"] in ("gated", "no_games"):
            return {"status": gate["status"]}

        from zoneinfo import ZoneInfo

        date_str = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
        return run_script(
            f"{SCRIPT_DIR}/betting_xl/fetchers/fetch_vegas_lines.py",
            ["--date", date_str, "--save-to-db"],
        )

    @task(task_id="update_minutes")
    def update_minutes(gate: dict[str, Any]) -> dict[str, Any]:
        """Update minutes projections. Run 1 only."""
        if gate["status"] in ("gated", "no_games"):
            return {"status": gate["status"]}
        if not gate.get("is_first_run"):
            return {"status": "skipped"}
        return run_script(
            f"{SCRIPT_DIR}/scripts/loaders/calculate_minutes_projections.py", ["--update"]
        )

    @task(task_id="update_prop_history")
    def update_prop_history(gate: dict[str, Any]) -> dict[str, Any]:
        """Update prop performance history. Run 1 only."""
        if gate["status"] in ("gated", "no_games"):
            return {"status": gate["status"]}
        if not gate.get("is_first_run"):
            return {"status": "skipped"}

        season = str(get_current_season())
        return run_script(
            f"{SCRIPT_DIR}/scripts/compute_prop_history.py",
            ["--season", season, "--incremental", "--days", "7"],
            timeout=600,
        )

    @task(task_id="update_matchup_history")
    def update_matchup_history(game_results: dict[str, Any]) -> dict[str, Any]:
        """Recompute H2H matchup stats. Run 1 only."""
        if game_results.get("status") in ("gated", "no_games", "skipped"):
            return {"status": game_results["status"]}
        return run_script(
            f"{SCRIPT_DIR}/scripts/compute_matchup_history.py",
            ["--incremental", "--days", "7"],
            timeout=600,
        )

    # ========================================================================
    # Prediction Tasks
    # ========================================================================

    @task(task_id="generate_xl_predictions")
    def generate_xl_predictions(enrichment: dict[str, Any]) -> dict[str, Any]:
        """Generate XL model predictions."""
        if enrichment.get("status") in ("gated", "no_games"):
            return {"output_file": None, "total_picks": 0, "status": enrichment["status"]}

        from zoneinfo import ZoneInfo

        date_str = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
        output_file = f"{PREDICTIONS_DIR}/xl_picks_{date_str}.json"

        pp_count = get_prizepicks_count(date_str)
        if pp_count == 0:
            print(f"[WARN] No PrizePicks data for {date_str} — GOLDMINE picks unavailable")
        else:
            print(f"[OK] Pre-flight: {pp_count} PrizePicks props available")

        result = run_script(
            f"{SCRIPT_DIR}/betting_xl/generate_xl_predictions.py",
            ["--output", output_file, "--underdog-only"],
            timeout=600,
        )

        if result["status"] == "error":
            raise Exception(f"XL prediction failed: {result.get('error')}")

        picks_count = 0
        if Path(output_file).exists():
            with open(output_file) as f:
                picks_count = json.load(f).get("total_picks", 0)

        return {"output_file": output_file, "total_picks": picks_count, "status": "success"}

    @task(task_id="generate_pro_picks")
    def generate_pro_picks(enrichment: dict[str, Any]) -> dict[str, Any]:
        """Generate Pro tier picks."""
        if enrichment.get("status") in ("gated", "no_games"):
            return {"output_file": None, "total_picks": 0, "status": enrichment["status"]}

        from zoneinfo import ZoneInfo

        date_str = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
        output_file = f"{PREDICTIONS_DIR}/pro_picks_{date_str}.json"

        result = run_script(
            f"{SCRIPT_DIR}/betting_xl/generate_cheatsheet_picks.py",
            ["--output", output_file],
        )

        picks_count = 0
        if Path(output_file).exists():
            with open(output_file) as f:
                picks_count = json.load(f).get("total_picks", 0)

        return {"output_file": output_file, "total_picks": picks_count, "status": result["status"]}

    @task(task_id="output_summary")
    def output_summary(
        xl_result: dict[str, Any],
        pro_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Output final summary."""
        from zoneinfo import ZoneInfo

        date_str = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")

        if all(r.get("status") in ("gated", "no_games") for r in [xl_result, pro_result]):
            status = xl_result.get("status", "gated")
            print(f"[INFO] Pipeline {status} on {date_str}")
            return {"date": date_str, "status": status, "total": 0}

        summary = {
            "date": date_str,
            "generated_at": datetime.now().isoformat(),
            "xl_picks": xl_result.get("total_picks", 0),
            "pro_picks": pro_result.get("total_picks", 0),
            "total": xl_result.get("total_picks", 0) + pro_result.get("total_picks", 0),
        }

        summary_file = f"{PREDICTIONS_DIR}/daily_summary_{date_str}.json"
        Path(PREDICTIONS_DIR).mkdir(parents=True, exist_ok=True)
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        print(
            f"Pipeline complete: {summary['total']} total picks (XL: {summary['xl_picks']}, Pro: {summary['pro_picks']})"
        )
        return summary

    @task(task_id="write_to_axiom")
    def write_to_axiom(
        gate: dict[str, Any],
        xl_result: dict[str, Any],
        summary: dict[str, Any],
    ) -> dict[str, Any]:
        """Write picks to nba_prediction_history and complete pipeline audit.

        Never raises — axiom is an observer. Pipeline success is independent.
        """
        if gate["status"] in ("gated",):
            return {"picks_written": 0, "status": "gated"}

        from nba.core.axiom_writer import audit_run_complete, count_todays_props, write_picks

        run_date = gate["run_date"]
        run_number = gate["run_number"]
        started_at = gate["started_at"]
        duration = int((datetime.now() - datetime.fromisoformat(started_at)).total_seconds())

        props_fetched, books_available = count_todays_props(run_date)

        picks_written = 0
        v5_count = 0
        if xl_result.get("output_file") and Path(xl_result["output_file"]).exists():
            with open(xl_result["output_file"]) as f:
                data = json.load(f)
            picks = data.get("picks", [])
            picks_written = write_picks(run_date, run_number, datetime.now().isoformat(), picks)
            v5_count = sum(1 for p in picks if p.get("model_version") == "v5")

        status = "no_games" if summary.get("status") == "no_games" else "success"
        audit_run_complete(
            run_date,
            run_number,
            status=status,
            props_fetched=props_fetched,
            books_available=books_available,
            games_found=summary.get("total", 0) if status != "no_games" else 0,
            duration_seconds=duration,
            picks_generated=picks_written,
            xl_picks=v5_count,
            v3_picks=0,
        )

        return {"picks_written": picks_written, "duration_seconds": duration}

    @task(task_id="compute_conviction")
    def compute_conviction(axiom_result: dict[str, Any]) -> dict[str, Any]:
        """Compute conviction scores across all of today's runs. Never fails."""
        if axiom_result.get("status") == "gated":
            return {"conviction_rows": 0, "status": "gated"}

        from zoneinfo import ZoneInfo

        from nba.core.axiom_writer import get_run_number
        from nba.core.conviction_engine import compute_conviction as _compute

        run_date = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
        run_number = get_run_number()
        count = _compute(run_date, run_number=run_number)
        return {"conviction_rows": count}

    @task(task_id="check_feature_store", retries=0)
    def check_feature_store(xl_result: dict[str, Any]) -> dict[str, Any]:
        """Check feature store coverage after predictions wrote features inline.

        Features are written during extract_features() in generate_xl_predictions.
        This task verifies coverage and logs gaps.
        """
        try:
            from zoneinfo import ZoneInfo

            from nba.features.feature_store import FeatureStore

            run_date = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
            store = FeatureStore()
            coverage = store.get_coverage("xl_v1", stat_type=None)
            today = store.get_coverage("xl_v1")
            store.close()

            return {
                "total_features": coverage["count"],
                "date_range": f"{coverage['min_date']} to {coverage['max_date']}",
                "today_count": today["count"],
            }
        except Exception as e:
            return {"total_features": 0, "error": str(e)}

    # ========================================================================
    # Task Dependencies
    # ========================================================================

    # Gate + audit (must run first)
    gate = gate_and_audit()

    # Data collection — props chain
    props = fetch_props(gate)
    loaded = load_props(props)
    enriched = enrich_matchups(loaded)

    # Parallel data tasks (all receive gate for status checks)
    cheatsheet = fetch_cheatsheet(gate)
    hit_rates = fetch_hit_rates(gate)
    pick_recs = fetch_pick_recs(gate)
    prizepicks = fetch_prizepicks(gate)
    injuries = update_injuries(gate)
    vegas = fetch_vegas(gate)
    direct_books = fetch_direct_sportsbooks(gate)

    # Heavy one-time tasks (run 1 only, self-skip on runs 2+)
    game_results_task = fetch_game_results(gate)
    actuals = populate_actuals(game_results_task)
    matchup_history = update_matchup_history(game_results_task)
    team_stats = update_team_stats(gate)
    minutes = update_minutes(gate)
    prop_history = update_prop_history(gate)

    # All data must complete before enrichment -> predictions
    [
        cheatsheet,
        hit_rates,
        pick_recs,
        prizepicks,
        actuals,
        matchup_history,
        injuries,
        team_stats,
        vegas,
        minutes,
        prop_history,
        direct_books,
    ] >> enriched

    # Predictions (parallel)
    xl = generate_xl_predictions(enriched)
    pro = generate_pro_picks(enriched)

    # Summary
    summary = output_summary(xl, pro)

    # Axiom write + conviction — after summary, never block pipeline
    axiom = write_to_axiom(gate, xl, summary)
    compute_conviction(axiom)

    # Feature store — verify today's features were written during prediction (after XL)
    check_feature_store(xl)


dag = nba_full_pipeline()
