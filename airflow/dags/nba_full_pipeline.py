"""
NBA Full Pipeline DAG

Scheduled: Daily at 09:00 AM EST (14:00 UTC)
Purpose: Complete data collection + predictions

This is the main daily workflow that:
1. Fetches all data (props, cheatsheet, vegas, injuries, team stats)
2. Enriches matchups
3. Generates predictions (XL, Pro, Odds API)

Run once daily at 9am EST when all data sources (including PrizePicks/DFS)
are available. Earlier runs miss soft lines that create high-spread opportunities.

Use nba_refresh_pipeline for line movement updates during the day.

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


def get_current_season() -> int:
    """Calculate NBA season (uses END year: 2024-25 season = 2025)."""
    now = datetime.now()
    return now.year + 1 if now.month >= 10 else now.year


def get_prizepicks_count(date_str: str) -> int:
    """Get count of PrizePicks props for a given date.

    Used by both fetch_prizepicks and generate_xl_predictions to verify
    DFS data availability for GOLDMINE picks.
    """
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
    """Helper to run a Python script with standard error handling.

    Args:
        raise_on_error: If True, raises Exception on non-zero exit instead of
                        returning {"status": "error"}. Use for critical tasks
                        where silent failure is unacceptable.
    """
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
    description="NBA complete data collection + predictions (run once daily at 2AM EST)",
    schedule=CronTriggerTimetable("0 2 * * *", timezone="America/New_York"),  # 2:00 AM EST
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
    NBA Full Pipeline - Complete daily workflow. Run #1 of 6 (2AM EST).

    Equivalent to: ./nba-predictions.sh (or ./nba-predictions.sh full)
    """

    # ========================================================================
    # Axiom Audit — Start
    # ========================================================================

    @task(task_id="audit_run_start")
    def audit_run_start() -> dict[str, Any]:
        """Record pipeline start in axiom_pipeline_audit. Never fails."""
        from zoneinfo import ZoneInfo

        from nba.core.axiom_writer import audit_run_start as _start

        run_date = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
        _start(run_date, run_number=1, run_type="full")
        return {"run_date": run_date, "run_number": 1, "started_at": datetime.now().isoformat()}

    # ========================================================================
    # Data Collection Tasks
    # ========================================================================

    @task(task_id="fetch_props")
    def fetch_props() -> dict[str, Any]:
        """Fetch props from 7 sportsbooks."""
        import glob

        result = run_script(f"{SCRIPT_DIR}/betting_xl/fetchers/fetch_all.py", timeout=600)
        if result["status"] != "success":
            raise Exception(f"fetch_props failed: {result.get('error')}")

        pattern = f"{SCRIPT_DIR}/betting_xl/lines/all_sources_*.json"
        files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)

        # Check if no games today (fetch succeeded but produced no props file)
        if not files:
            print("[INFO] No props file generated - likely no NBA games today")
            return {"props_file": None, "status": "no_games"}

        return {"props_file": files[0], "status": "success"}

    @task(task_id="load_props")
    def load_props(fetch_result: dict[str, Any]) -> dict[str, Any]:
        """Load props to database."""
        if fetch_result.get("status") == "no_games":
            print("[INFO] No games today - skipping load_props")
            return {"status": "no_games"}

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
    def fetch_cheatsheet() -> dict[str, Any]:
        """Fetch BettingPros cheatsheet (Underdog lines + BP analytics enrichment)."""
        import glob

        result = run_script(
            f"{SCRIPT_DIR}/betting_xl/fetchers/fetch_cheatsheet.py",
            ["--platform", "underdog"],
        )
        if result["status"] == "skipped":
            return result
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
                    f"[WARN] load_cheatsheet_to_db failed (non-critical): {load_result.get('error', '')[-300:]}"
                )
        else:
            print("[WARN] fetch_cheatsheet: no JSON file found after fetch")

        return {"status": "success"}

    @task(task_id="fetch_hit_rates")
    def fetch_hit_rates() -> dict[str, Any]:
        """Fetch BettingPros consensus hit rates, streaks, and BP projections.

        Non-critical: if missing, predictions still run but without hit-rate enrichment.
        """
        from zoneinfo import ZoneInfo

        from nba.betting_xl.fetchers.fetch_bettingpros_hit_rates import BettingProsHitRateFetcher

        date_str = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
        try:
            with BettingProsHitRateFetcher(date=date_str, verbose=False) as fetcher:
                records = fetcher.fetch()
                if records:
                    output_file = fetcher.save_hit_rates(records)
                    print(f"[OK] Hit rates: {len(records)} records → {output_file.name}")
                    return {"status": "success", "records": len(records)}
                print("[WARN] fetch_hit_rates: no records returned")
                return {"status": "no_data", "records": 0}
        except Exception as exc:
            print(f"[WARN] fetch_hit_rates failed (non-critical): {exc}")
            return {"status": "error", "error": str(exc)}

    @task(task_id="fetch_pick_recs")
    def fetch_pick_recs() -> dict[str, Any]:
        """Fetch BettingPros session-gated pick recommendations.

        Requires BETTINGPROS_SESSION_ID + BETTINGPROS_WEB_API_KEY (from refresh_bp_session.py).
        Non-critical: silently skips if session is expired or credentials missing.
        """
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
                print(f"[OK] Pick recommendations: {len(picks)} picks → {output_file.name}")
                return {"status": "success", "picks": len(picks)}
            print("[WARN] fetch_pick_recs: no picks returned (session expired?)")
            return {"status": "no_data", "picks": 0}
        except Exception as exc:
            print(f"[WARN] fetch_pick_recs failed (non-critical): {exc}")
            return {"status": "error", "error": str(exc)}

    @task(task_id="fetch_prizepicks")
    def fetch_prizepicks() -> dict[str, Any]:
        """Fetch PrizePicks props (standard lines only).

        PrizePicks provides DFS lines that are often softer than sportsbooks,
        creating high-spread opportunities for GOLDMINE picks.
        """
        from zoneinfo import ZoneInfo

        date_str = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")

        # Fetch and load PrizePicks data. Script errors are intentionally not
        # raised here — PrizePicks is a non-critical DFS source. If it fails,
        # GOLDMINE picks are skipped but the rest of the pipeline continues.
        run_script(
            f"{SCRIPT_DIR}/betting_xl/loaders/load_prizepicks_to_db.py",
            ["--fetch", "--quiet"],
            timeout=300,
        )

        # Verify data was loaded
        pp_count = get_prizepicks_count(date_str)

        if pp_count == 0:
            print(
                f"[WARN] No PrizePicks data for {date_str} "
                "(no games today or API unavailable). "
                "GOLDMINE picks will be skipped."
            )
            return {"status": "no_data", "prizepicks_count": 0}

        print(f"[OK] PrizePicks loaded: {pp_count} props for {date_str}")
        return {"status": "success", "prizepicks_count": pp_count}

    @task(task_id="enrich_matchups")
    def enrich_matchups(load_result: dict[str, Any]) -> dict[str, Any]:
        """Enrich props with matchup context."""
        if load_result.get("status") == "no_games":
            print("[INFO] No games today - skipping enrich_matchups")
            return {"coverage": 0, "total": 0, "status": "no_games"}

        from zoneinfo import ZoneInfo

        import psycopg2

        from nba.config.database import get_intelligence_db_config

        date_str = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
        run_script(
            f"{SCRIPT_DIR}/betting_xl/enrich_props_with_matchups.py",
            ["--date", date_str],
        )

        # Verify coverage
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

    @task(task_id="fetch_game_results")
    def fetch_game_results() -> dict[str, Any]:
        """Fetch yesterday's game results and verify game logs were updated."""
        from datetime import date, timedelta

        import psycopg2

        from nba.config.database import get_players_db_config

        result = run_script(
            f"{SCRIPT_DIR}/scripts/fetch_daily_stats.py",
            ["--days", "1"],
            timeout=600,
            raise_on_error=True,
        )

        # Verify game logs are actually fresh (max 3 days stale)
        # This catches ESPN API silent failures that previously went undetected
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
            raise Exception("player_game_logs is empty — game log fetch failed silently")

        days_stale = (date.today() - latest).days
        # NBA has occasional off days; allow up to 3 days before alarming
        if days_stale > 3:
            raise Exception(
                f"player_game_logs is {days_stale} days stale (latest: {latest}). "
                "ESPN API may be down or fetch_daily_stats failed."
            )

        print(f"[OK] player_game_logs current: latest={latest} ({days_stale}d ago)")
        return result

    @task(task_id="populate_actuals")
    def populate_actuals(game_results: dict[str, Any]) -> dict[str, Any]:
        """Populate actual values for recent props."""
        return run_script(f"{SCRIPT_DIR}/betting_xl/populate_actual_values.py", ["--days", "7"])

    @task(task_id="update_injuries")
    def update_injuries() -> dict[str, Any]:
        """Update injury reports."""
        return run_script(f"{SCRIPT_DIR}/scripts/update_injuries_NOW.py")

    @task(task_id="update_team_stats")
    def update_team_stats() -> dict[str, Any]:
        """Update team statistics."""
        season = str(get_current_season())
        run_script(f"{SCRIPT_DIR}/scripts/loaders/load_nba_games_incremental.py")
        run_script(f"{SCRIPT_DIR}/scripts/loaders/calculate_team_stats.py", ["--season", season])
        run_script(f"{SCRIPT_DIR}/scripts/loaders/load_team_advanced_stats.py")
        return {"status": "success"}

    @task(task_id="fetch_vegas")
    def fetch_vegas() -> dict[str, Any]:
        """Fetch vegas lines."""
        from zoneinfo import ZoneInfo

        date_str = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
        return run_script(
            f"{SCRIPT_DIR}/betting_xl/fetchers/fetch_vegas_lines.py",
            ["--date", date_str, "--save-to-db"],
        )

    @task(task_id="update_minutes")
    def update_minutes() -> dict[str, Any]:
        """Update minutes projections."""
        return run_script(
            f"{SCRIPT_DIR}/scripts/loaders/calculate_minutes_projections.py", ["--update"]
        )

    @task(task_id="update_prop_history")
    def update_prop_history() -> dict[str, Any]:
        """Update prop performance history."""
        season = str(get_current_season())
        return run_script(
            f"{SCRIPT_DIR}/scripts/compute_prop_history.py",
            ["--season", season, "--incremental", "--days", "7"],
            timeout=600,
        )

    # ========================================================================
    # Prediction Tasks
    # ========================================================================

    @task(task_id="generate_xl_predictions")
    def generate_xl_predictions(enrichment: dict[str, Any]) -> dict[str, Any]:
        """Generate XL model predictions."""
        if enrichment.get("status") == "no_games":
            print("[INFO] No games today - skipping XL predictions")
            return {"output_file": None, "total_picks": 0, "status": "no_games"}

        from zoneinfo import ZoneInfo

        date_str = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
        output_file = f"{PREDICTIONS_DIR}/xl_picks_{date_str}.json"

        # Pre-flight check: verify PrizePicks data for GOLDMINE picks
        pp_count = get_prizepicks_count(date_str)

        if pp_count == 0:
            print(
                f"[WARN] No PrizePicks data for {date_str} - " "GOLDMINE picks will be unavailable"
            )
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
        if enrichment.get("status") == "no_games":
            print("[INFO] No games today - skipping Pro picks")
            return {"output_file": None, "total_picks": 0, "status": "no_games"}

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

        if all(r.get("status") == "no_games" for r in [xl_result, pro_result]):
            print(f"[INFO] No NBA games on {date_str} - pipeline completed with nothing to do")
            return {"date": date_str, "status": "no_games", "total": 0}

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

        print(f"Full pipeline complete: {summary['total']} total picks")
        print(f"  XL: {summary['xl_picks']}, Pro: {summary['pro_picks']}")
        return summary

    @task(task_id="write_to_axiom")
    def write_to_axiom(
        audit_info: dict[str, Any],
        xl_result: dict[str, Any],
        summary: dict[str, Any],
    ) -> dict[str, Any]:
        """Write picks to nba_prediction_history and complete pipeline audit.

        Never raises — axiom is an observer. Pipeline success is independent.
        """
        from nba.core.axiom_writer import audit_run_complete, count_todays_props, write_picks

        run_date = audit_info["run_date"]
        run_number = audit_info["run_number"]
        started_at = audit_info["started_at"]
        duration = int((datetime.now() - datetime.fromisoformat(started_at)).total_seconds())

        # Count props fetched from intel DB
        props_fetched, books_available = count_todays_props(run_date)

        # Write picks to history
        picks_written = 0
        xl_count = 0
        v3_count = 0
        if xl_result.get("output_file") and Path(xl_result["output_file"]).exists():
            with open(xl_result["output_file"]) as f:
                data = json.load(f)
            picks = data.get("picks", [])
            picks_written = write_picks(run_date, run_number, datetime.now().isoformat(), picks)
            xl_count = sum(1 for p in picks if p.get("model_version") == "xl")
            v3_count = sum(1 for p in picks if p.get("model_version") == "v3")

        # Complete the audit record
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
            xl_picks=xl_count,
            v3_picks=v3_count,
        )

        return {"picks_written": picks_written, "duration_seconds": duration}

    @task(task_id="compute_conviction")
    def compute_conviction(axiom_result: dict[str, Any]) -> dict[str, Any]:
        """Compute conviction scores across all of today's runs. Never fails."""
        from zoneinfo import ZoneInfo

        from nba.core.conviction_engine import compute_conviction as _compute

        run_date = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
        count = _compute(run_date, run_number=1)
        return {"conviction_rows": count}

    # ========================================================================
    # Task Dependencies
    # ========================================================================

    # Audit start (before anything else)
    audit_start = audit_run_start()

    # Data collection chain
    props = fetch_props()
    loaded = load_props(props)
    enriched = enrich_matchups(loaded)

    # Parallel data tasks
    cheatsheet = fetch_cheatsheet()
    hit_rates = fetch_hit_rates()
    pick_recs = fetch_pick_recs()
    prizepicks = fetch_prizepicks()  # DFS soft lines for GOLDMINE picks
    game_results_task = fetch_game_results()
    actuals = populate_actuals(game_results_task)
    injuries = update_injuries()
    team_stats = update_team_stats()
    vegas = fetch_vegas()
    minutes = update_minutes()
    prop_history = update_prop_history()

    # All data must complete before predictions
    [
        cheatsheet,
        hit_rates,
        pick_recs,
        prizepicks,
        actuals,
        injuries,
        team_stats,
        vegas,
        minutes,
        prop_history,
    ] >> enriched

    # Predictions (parallel)
    xl = generate_xl_predictions(enriched)
    pro = generate_pro_picks(enriched)

    # Summary
    summary = output_summary(xl, pro)

    # Axiom write + conviction — after summary, never block pipeline
    axiom = write_to_axiom(audit_start, xl, summary)
    compute_conviction(axiom)


dag = nba_full_pipeline()
