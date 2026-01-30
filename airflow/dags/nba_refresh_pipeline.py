"""
NBA Refresh Pipeline DAG

Scheduled: Manual trigger (or multiple times daily)
Purpose: Quick refresh of lines + injuries + regenerate predictions

This is a lightweight workflow for capturing line movements:
1. Refresh props (line movements, late props)
2. Refresh cheatsheet (updated projections)
3. Update injuries (game-time decisions)
4. Refresh vegas lines
5. Re-enrich matchups
6. Regenerate predictions

Use this after the full pipeline to capture line movements before game time.

Equivalent to: ./nba-predictions.sh refresh

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
    "retries": 2,
    "retry_delay": timedelta(minutes=3),
    "execution_timeout": timedelta(minutes=20),
    "on_success_callback": on_success,
    "on_failure_callback": on_failure,
    "on_retry_callback": on_retry,
}


def run_script(
    script_path: str, args: list[str] | None = None, timeout: int = 300
) -> dict[str, Any]:
    """Helper to run a Python script."""
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
    subject = f"[AIRFLOW] NBA Refresh Pipeline Failed: {task_instance.task_id}"
    body = f"<p><b>Log:</b> <a href='{task_instance.log_url}'>{task_instance.log_url}</a></p>"

    try:
        send_email(to=default_args["email"], subject=subject, html_content=body)
    except Exception as e:
        print(f"Failed to send alert: {e}")


# ============================================================================
# DAG Definition
# ============================================================================


@dag(
    dag_id="nba_refresh_pipeline",
    description="NBA quick refresh - line movements + predictions (run anytime)",
    schedule=None,  # Manual trigger only
    start_date=datetime(2025, 11, 7),
    catchup=False,
    tags=["nba", "predictions", "refresh", "line-movements"],
    default_args=default_args,
    max_active_runs=1,
    doc_md=__doc__,
    on_failure_callback=alert_on_failure,
)
def nba_refresh_pipeline():
    """
    NBA Refresh Pipeline - Quick line movement capture.

    Equivalent to: ./nba-predictions.sh refresh
    """

    # ========================================================================
    # Pre-flight Checks
    # ========================================================================

    @task(task_id="health_check")
    def health_check() -> dict[str, Any]:
        """Quick health check."""
        result = run_script(f"{SCRIPT_DIR}/betting_xl/health_check.py", ["--quick"], timeout=60)
        if result["status"] == "error":
            print(f"Health check warning: {result.get('error')}")
        return {"status": "ok"}

    @task(task_id="stop_loss_check")
    def stop_loss_check() -> dict[str, Any]:
        """Check performance thresholds."""
        result = run_script(f"{SCRIPT_DIR}/betting_xl/monitor.py", timeout=60)
        if result["status"] == "error":
            raise Exception(f"Stop-loss triggered: {result.get('error')}")
        return {"status": "ok"}

    # ========================================================================
    # Refresh Tasks
    # ========================================================================

    @task(task_id="refresh_props")
    def refresh_props() -> dict[str, Any]:
        """Refresh props to capture line movements."""
        import glob

        result = run_script(f"{SCRIPT_DIR}/betting_xl/fetchers/fetch_all.py", timeout=600)
        if result["status"] == "error":
            raise Exception(f"Props refresh failed: {result.get('error')}")

        pattern = f"{SCRIPT_DIR}/betting_xl/lines/all_sources_*.json"
        files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)

        if files:
            run_script(
                f"{SCRIPT_DIR}/betting_xl/loaders/load_props_to_db.py",
                ["--file", files[0], "--skip-mongodb"],
            )

        return {"props_file": files[0] if files else None, "status": "success"}

    @task(task_id="refresh_cheatsheet")
    def refresh_cheatsheet() -> dict[str, Any]:
        """Refresh cheatsheet data."""
        import glob

        result = run_script(
            f"{SCRIPT_DIR}/betting_xl/fetchers/fetch_cheatsheet.py",
            ["--platform", "underdog"],
        )

        pattern = f"{SCRIPT_DIR}/betting_xl/lines/cheatsheet_underdog_*.json"
        files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)

        if files:
            run_script(
                f"{SCRIPT_DIR}/betting_xl/loaders/load_cheatsheet_to_db.py",
                ["--file", files[0]],
            )

        return {"status": result["status"]}

    @task(task_id="refresh_injuries")
    def refresh_injuries() -> dict[str, Any]:
        """Update injury reports (game-time decisions)."""
        return run_script(f"{SCRIPT_DIR}/scripts/update_injuries_NOW.py")

    @task(task_id="refresh_vegas")
    def refresh_vegas() -> dict[str, Any]:
        """Refresh vegas lines."""
        date_str = datetime.now().strftime("%Y-%m-%d")
        return run_script(
            f"{SCRIPT_DIR}/betting_xl/fetchers/fetch_vegas_lines.py",
            ["--date", date_str, "--save-to-db"],
        )

    @task(task_id="enrich_matchups")
    def enrich_matchups(props_result: dict[str, Any]) -> dict[str, Any]:
        """Re-enrich matchups after refresh."""
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
        cursor.close()
        conn.close()

        total, enriched = row if row else (0, 0)
        coverage = round(100.0 * enriched / total, 1) if total > 0 else 0

        if total > 0 and coverage < min_coverage:
            raise Exception(f"Coverage {coverage}% below threshold")

        return {"coverage": coverage, "status": "success"}

    # ========================================================================
    # Prediction Tasks
    # ========================================================================

    @task(task_id="generate_xl_predictions")
    def generate_xl_predictions(enrichment: dict[str, Any]) -> dict[str, Any]:
        """Generate XL predictions."""
        date_str = datetime.now().strftime("%Y-%m-%d")
        output_file = f"{PREDICTIONS_DIR}/xl_picks_{date_str}.json"

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

        return {"total_picks": picks_count, "status": "success"}

    @task(task_id="generate_pro_picks")
    def generate_pro_picks(enrichment: dict[str, Any]) -> dict[str, Any]:
        """Generate Pro picks."""
        date_str = datetime.now().strftime("%Y-%m-%d")
        output_file = f"{PREDICTIONS_DIR}/pro_picks_{date_str}.json"

        run_script(
            f"{SCRIPT_DIR}/betting_xl/generate_cheatsheet_picks.py",
            ["--output", output_file],
        )

        picks_count = 0
        if Path(output_file).exists():
            with open(output_file) as f:
                picks_count = json.load(f).get("total_picks", 0)

        return {"total_picks": picks_count, "status": "success"}

    @task(task_id="generate_odds_api_picks")
    def generate_odds_api_picks(enrichment: dict[str, Any]) -> dict[str, Any]:
        """Generate Odds API picks."""
        date_str = datetime.now().strftime("%Y-%m-%d")
        output_file = f"{PREDICTIONS_DIR}/odds_api_picks_{date_str.replace('-', '')}.json"

        run_script(
            f"{SCRIPT_DIR}/betting_xl/generate_odds_api_picks.py",
            ["--date", date_str, "--output", output_file],
        )

        picks_count = 0
        if Path(output_file).exists():
            with open(output_file) as f:
                picks_count = json.load(f).get("total_picks", 0)

        return {"total_picks": picks_count, "status": "success"}

    @task(task_id="output_summary")
    def output_summary(
        xl_result: dict[str, Any],
        pro_result: dict[str, Any],
        odds_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Output refresh summary."""
        total = (
            xl_result.get("total_picks", 0)
            + pro_result.get("total_picks", 0)
            + odds_result.get("total_picks", 0)
        )
        print(f"Refresh complete: {total} total picks")
        print(f"  XL: {xl_result.get('total_picks', 0)}")
        print(f"  Pro: {pro_result.get('total_picks', 0)}")
        print(f"  Odds API: {odds_result.get('total_picks', 0)}")
        return {"total_picks": total, "status": "complete"}

    # ========================================================================
    # Task Dependencies
    # ========================================================================

    # Pre-flight
    health = health_check()
    stop_loss = stop_loss_check()
    health >> stop_loss

    # Refresh data (parallel)
    props = refresh_props()
    cheatsheet = refresh_cheatsheet()
    injuries = refresh_injuries()
    vegas = refresh_vegas()

    stop_loss >> [props, cheatsheet, injuries, vegas]

    # Enrich after props loaded
    enriched = enrich_matchups(props)
    [cheatsheet, injuries, vegas] >> enriched

    # Predictions (parallel)
    xl = generate_xl_predictions(enriched)
    pro = generate_pro_picks(enriched)
    odds = generate_odds_api_picks(enriched)

    # Summary
    output_summary(xl, pro, odds)


dag = nba_refresh_pipeline()
