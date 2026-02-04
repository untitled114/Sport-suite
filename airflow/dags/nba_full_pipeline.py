"""
NBA Full Pipeline DAG

Scheduled: Daily at 05:00 PM EST (22:00 UTC)
Purpose: Complete data collection + predictions

This is the main daily workflow that:
1. Fetches all data (props, cheatsheet, vegas, injuries, team stats)
2. Enriches matchups
3. Generates predictions (XL, Pro, Odds API)

Run once daily at 5pm EST when all data sources (including PrizePicks/DFS)
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

from callbacks import on_failure, on_retry, on_success

from airflow.decorators import dag, task
from airflow.models import Variable
from airflow.timetables.trigger import CronTriggerTimetable
from airflow.utils.email import send_email

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
    script_path: str, args: list[str] | None = None, timeout: int = 300
) -> dict[str, Any]:
    """Helper to run a Python script with standard error handling."""
    import subprocess

    if not Path(script_path).exists():
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
        return {"status": "error", "error": result.stderr, "stdout": result.stdout}

    return {"status": "success", "stdout": result.stdout}


def alert_on_failure(context: dict[str, Any]) -> None:
    """Send alert on task failure."""
    task_instance = context.get("task_instance")
    dag_id = context.get("dag").dag_id
    task_id = task_instance.task_id

    subject = f"[AIRFLOW] NBA Full Pipeline Failed: {task_id}"
    body = f"""
    <h3>Task Failure Alert</h3>
    <p><b>DAG:</b> {dag_id}</p>
    <p><b>Task:</b> {task_id}</p>
    <p><b>Log URL:</b> <a href="{task_instance.log_url}">{task_instance.log_url}</a></p>
    """

    try:
        send_email(to=default_args["email"], subject=subject, html_content=body)
    except Exception as e:
        print(f"Failed to send alert email: {e}")


# ============================================================================
# DAG Definition
# ============================================================================


@dag(
    dag_id="nba_full_pipeline",
    description="NBA complete data collection + predictions (run once daily)",
    schedule=CronTriggerTimetable("0 22 * * *", timezone="UTC"),  # 05:00 PM EST daily
    start_date=datetime(2025, 11, 7),
    catchup=False,
    tags=["nba", "predictions", "full", "data-collection"],
    default_args=default_args,
    max_active_runs=1,
    doc_md=__doc__,
    on_failure_callback=alert_on_failure,
)
def nba_full_pipeline():
    """
    NBA Full Pipeline - Complete daily workflow.

    Equivalent to: ./nba-predictions.sh (or ./nba-predictions.sh full)
    """

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
        return {"props_file": files[0] if files else None, "status": "success"}

    @task(task_id="load_props")
    def load_props(fetch_result: dict[str, Any]) -> dict[str, Any]:
        """Load props to database."""
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
        """Fetch BettingPros cheatsheet."""
        import glob

        result = run_script(
            f"{SCRIPT_DIR}/betting_xl/fetchers/fetch_cheatsheet.py",
            ["--platform", "underdog"],
        )
        if result["status"] == "skipped":
            return result

        pattern = f"{SCRIPT_DIR}/betting_xl/lines/cheatsheet_underdog_*.json"
        files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)

        if files:
            run_script(
                f"{SCRIPT_DIR}/betting_xl/loaders/load_cheatsheet_to_db.py",
                ["--file", files[0]],
            )

        return {"status": "success"}

    @task(task_id="fetch_prizepicks")
    def fetch_prizepicks() -> dict[str, Any]:
        """Fetch PrizePicks props (standard/goblin/demon lines).

        PrizePicks provides DFS lines that are often softer than sportsbooks,
        creating high-spread opportunities for GOLDMINE picks.
        """
        date_str = datetime.now().strftime("%Y-%m-%d")

        # Fetch and load PrizePicks data
        run_script(
            f"{SCRIPT_DIR}/betting_xl/loaders/load_prizepicks_to_db.py",
            ["--fetch", "--quiet"],
            timeout=300,
        )

        # Verify data was loaded (this is critical for GOLDMINE picks)
        pp_count = get_prizepicks_count(date_str)

        if pp_count == 0:
            raise Exception(
                f"PrizePicks data not loaded for {date_str}. "
                "GOLDMINE picks require PrizePicks goblin/demon lines."
            )

        print(f"[OK] PrizePicks loaded: {pp_count} props for {date_str}")
        return {"status": "success", "prizepicks_count": pp_count}

    @task(task_id="enrich_matchups")
    def enrich_matchups(load_result: dict[str, Any]) -> dict[str, Any]:
        """Enrich props with matchup context."""
        import psycopg2

        from nba.config.database import get_intelligence_db_config

        date_str = datetime.now().strftime("%Y-%m-%d")
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
        """Fetch yesterday's game results."""
        return run_script(
            f"{SCRIPT_DIR}/scripts/fetch_daily_stats.py", ["--days", "1"], timeout=600
        )

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
        date_str = datetime.now().strftime("%Y-%m-%d")
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
        date_str = datetime.now().strftime("%Y-%m-%d")
        output_file = f"{PREDICTIONS_DIR}/xl_picks_{date_str}.json"

        # Pre-flight check: verify we have PrizePicks data for GOLDMINE picks
        pp_count = get_prizepicks_count(date_str)

        if pp_count == 0:
            raise Exception(
                f"Pre-flight failed: No PrizePicks data for {date_str}. "
                "Cannot generate GOLDMINE picks without soft lines."
            )

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
        date_str = datetime.now().strftime("%Y-%m-%d")
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

    @task(task_id="generate_odds_api_picks")
    def generate_odds_api_picks(enrichment: dict[str, Any]) -> dict[str, Any]:
        """Generate Odds API picks."""
        date_str = datetime.now().strftime("%Y-%m-%d")
        output_file = f"{PREDICTIONS_DIR}/odds_api_picks_{date_str.replace('-', '')}.json"

        result = run_script(
            f"{SCRIPT_DIR}/betting_xl/generate_odds_api_picks.py",
            ["--date", date_str, "--output", output_file],
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
        odds_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Output final summary."""
        date_str = datetime.now().strftime("%Y-%m-%d")

        summary = {
            "date": date_str,
            "generated_at": datetime.now().isoformat(),
            "xl_picks": xl_result.get("total_picks", 0),
            "pro_picks": pro_result.get("total_picks", 0),
            "odds_api_picks": odds_result.get("total_picks", 0),
            "total": (
                xl_result.get("total_picks", 0)
                + pro_result.get("total_picks", 0)
                + odds_result.get("total_picks", 0)
            ),
        }

        summary_file = f"{PREDICTIONS_DIR}/daily_summary_{date_str}.json"
        Path(PREDICTIONS_DIR).mkdir(parents=True, exist_ok=True)
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"Full pipeline complete: {summary['total']} total picks")
        return summary

    # ========================================================================
    # Task Dependencies
    # ========================================================================

    # Data collection chain
    props = fetch_props()
    loaded = load_props(props)
    enriched = enrich_matchups(loaded)

    # Parallel data tasks
    cheatsheet = fetch_cheatsheet()
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
    odds = generate_odds_api_picks(enriched)

    # Final summary
    output_summary(xl, pro, odds)


dag = nba_full_pipeline()
