"""
NBA Evening Pipeline DAG

Scheduled: Daily at 5:00 PM EST
Purpose: Generate predictions using XL models with line shopping

Tasks:
1. health_check - Quick system readiness verification
2. validate_data_freshness - Pre-flight data quality checks
3. check_performance_thresholds - Stop-loss monitoring
4. refresh_props - Capture line movements
5. refresh_vegas_lines - Update spreads/totals
6. refresh_cheatsheet - Latest projections
7. enrich_matchups - Update matchup context
8. generate_xl_predictions - XL model predictions (Tier X, Tier A)
9. generate_pro_picks - Pro tier picks (cheatsheet-based)
10. generate_odds_api_picks - Pick6 multiplier picks
11. output_final_picks - Combine and output all picks

Author: Claude Code
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from airflow.decorators import dag, task
from airflow.models import Variable
from airflow.utils.email import send_email

# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Variable.get("nba_project_root", default_var="/home/untitled/Sport-suite")
SCRIPT_DIR = f"{PROJECT_ROOT}/nba"
LOG_DIR = f"{SCRIPT_DIR}/betting_xl/logs"
PREDICTIONS_DIR = f"{SCRIPT_DIR}/betting_xl/predictions"

# Add project to path for imports
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
}


def alert_on_failure(context: dict[str, Any]) -> None:
    """Send alert on task failure."""
    task_instance = context.get("task_instance")
    dag_id = context.get("dag").dag_id
    task_id = task_instance.task_id
    execution_date = context.get("execution_date")
    log_url = task_instance.log_url

    subject = f"[AIRFLOW] NBA Evening Pipeline Failed: {task_id}"
    body = f"""
    <h3>Task Failure Alert</h3>
    <p><b>DAG:</b> {dag_id}</p>
    <p><b>Task:</b> {task_id}</p>
    <p><b>Execution Date:</b> {execution_date}</p>
    <p><b>Log URL:</b> <a href="{log_url}">{log_url}</a></p>
    <p><b>Exception:</b> {context.get('exception', 'N/A')}</p>
    """

    try:
        send_email(
            to=default_args["email"],
            subject=subject,
            html_content=body,
        )
    except Exception as e:
        print(f"Failed to send alert email: {e}")


# ============================================================================
# DAG Definition
# ============================================================================


@dag(
    dag_id="nba_evening_pipeline",
    description="NBA betting predictions generation with XL models",
    schedule="0 17 * * *",  # 5:00 PM EST daily
    start_date=datetime(2025, 11, 7),
    catchup=False,
    tags=["nba", "predictions", "evening", "ml-models"],
    default_args=default_args,
    max_active_runs=1,
    doc_md=__doc__,
    on_failure_callback=alert_on_failure,
)
def nba_evening_pipeline():
    """
    NBA Evening Predictions Pipeline

    Generates betting predictions using:
    - XL Models: Tier X (POINTS) + Tier A (REBOUNDS)
    - Pro Tier: Cheatsheet-based high-confidence picks
    - Odds API: Pick6 multiplier integration

    Line shopping strategy identifies softest lines across 7 sportsbooks.
    """

    @task(
        task_id="health_check",
        doc_md="### Quick Health Check\n\nVerify system readiness before generating predictions.",
        sla=timedelta(minutes=5),
    )
    def health_check() -> dict[str, Any]:
        """Quick system health verification."""
        import subprocess

        script_path = f"{SCRIPT_DIR}/betting_xl/health_check.py"

        if not Path(script_path).exists():
            print("Health check script not found, proceeding with caution")
            return {"status": "skipped", "reason": "Script not found"}

        result = subprocess.run(
            ["python3", script_path, "--quick"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            env={**os.environ, "PYTHONPATH": PROJECT_ROOT},
            timeout=60,
        )

        if result.returncode != 0:
            print(f"Health check warning: {result.stderr}")
            return {"status": "warning", "error": result.stderr}

        return {"status": "passed"}

    @task(
        task_id="validate_data_freshness",
        doc_md="### Validate Data Freshness\n\nPre-flight data quality validation.",
    )
    def validate_data_freshness() -> dict[str, Any]:
        """Validate data freshness before prediction."""
        import subprocess

        script_path = f"{SCRIPT_DIR}/betting_xl/config/data_freshness_validator.py"

        if not Path(script_path).exists():
            return {"status": "skipped", "reason": "Validator not configured"}

        result = subprocess.run(
            ["python3", script_path],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            env={**os.environ, "PYTHONPATH": PROJECT_ROOT},
            timeout=120,
        )

        if result.returncode != 0:
            # Combine stdout and stderr for full context
            full_output = f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
            # Extract failure lines from output
            failures = [
                line for line in result.stdout.split("\n") if "❌" in line or "FAILED" in line
            ]
            failure_summary = "; ".join(failures) if failures else "Check logs for details"
            raise Exception(f"Data freshness validation failed: {failure_summary}")

        return {"status": "validated"}

    @task(
        task_id="check_performance_thresholds",
        doc_md="### Check Performance Thresholds\n\nStop-loss monitoring to pause if performance degrades.",
    )
    def check_performance_thresholds() -> dict[str, Any]:
        """Check performance thresholds and stop-loss conditions."""
        import subprocess

        script_path = f"{SCRIPT_DIR}/betting_xl/monitor.py"

        if not Path(script_path).exists():
            return {"status": "skipped", "reason": "Monitor not configured"}

        result = subprocess.run(
            ["python3", script_path],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            env={**os.environ, "PYTHONPATH": PROJECT_ROOT},
            timeout=120,
        )

        if result.returncode != 0:
            raise Exception(f"Stop-loss triggered - system paused: {result.stderr}")

        return {"status": "ok"}

    @task(
        task_id="refresh_props",
        doc_md="### Refresh Props\n\nCapture latest line movements from all sportsbooks.",
    )
    def refresh_props() -> dict[str, Any]:
        """Refresh props to capture line movements."""
        import glob
        import subprocess

        script_path = f"{SCRIPT_DIR}/betting_xl/fetchers/fetch_all.py"

        result = subprocess.run(
            ["python3", script_path],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            env={**os.environ, "PYTHONPATH": PROJECT_ROOT},
            timeout=600,
        )

        if result.returncode != 0:
            raise Exception(f"Props refresh failed: {result.stderr}")

        # Find latest props file
        pattern = f"{SCRIPT_DIR}/betting_xl/lines/all_sources_*.json"
        files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)

        return {"props_file": files[0] if files else None, "status": "success"}

    @task(
        task_id="load_refreshed_props",
        doc_md="### Load Refreshed Props\n\nStore refreshed props in database.",
    )
    def load_refreshed_props(props_result: dict[str, Any]) -> dict[str, Any]:
        """Load refreshed props to database."""
        import subprocess

        props_file = props_result.get("props_file")
        if not props_file:
            return {"status": "skipped", "reason": "No props file"}

        script_path = f"{SCRIPT_DIR}/betting_xl/loaders/load_props_to_db.py"

        result = subprocess.run(
            ["python3", script_path, "--file", props_file, "--skip-mongodb"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            env={**os.environ, "PYTHONPATH": PROJECT_ROOT},
            timeout=300,
        )

        return {"status": "success" if result.returncode == 0 else "warning"}

    @task(
        task_id="refresh_vegas_lines",
        doc_md="### Refresh Vegas Lines\n\nCapture spread/total movements.",
    )
    def refresh_vegas_lines() -> dict[str, Any]:
        """Refresh Vegas lines for line movements."""
        import subprocess

        date_str = datetime.now().strftime("%Y-%m-%d")
        script_path = f"{SCRIPT_DIR}/betting_xl/fetchers/fetch_vegas_lines.py"

        if not Path(script_path).exists():
            return {"status": "skipped", "reason": "Vegas fetcher not configured"}

        result = subprocess.run(
            ["python3", script_path, "--date", date_str, "--save-to-db"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            env={**os.environ, "PYTHONPATH": PROJECT_ROOT},
            timeout=300,
        )

        return {"status": "success" if result.returncode == 0 else "warning"}

    @task(
        task_id="refresh_cheatsheet",
        doc_md="### Refresh Cheatsheet\n\nGet latest projections and hit rates.",
    )
    def refresh_cheatsheet() -> dict[str, Any]:
        """Refresh cheatsheet data."""
        import glob
        import subprocess

        script_path = f"{SCRIPT_DIR}/betting_xl/fetchers/fetch_cheatsheet.py"

        if not Path(script_path).exists():
            return {"status": "skipped", "reason": "Cheatsheet fetcher not configured"}

        result = subprocess.run(
            ["python3", script_path, "--platform", "underdog"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            env={**os.environ, "PYTHONPATH": PROJECT_ROOT},
            timeout=300,
        )

        if result.returncode != 0:
            return {"status": "warning", "error": result.stderr}

        # Find and load to database
        pattern = f"{SCRIPT_DIR}/betting_xl/lines/cheatsheet_underdog_*.json"
        files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)

        if files:
            loader_path = f"{SCRIPT_DIR}/betting_xl/loaders/load_cheatsheet_to_db.py"
            subprocess.run(
                ["python3", loader_path, "--file", files[0]],
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
                env={**os.environ, "PYTHONPATH": PROJECT_ROOT},
                timeout=300,
            )

        return {"cheatsheet_file": files[0] if files else None, "status": "success"}

    @task(
        task_id="enrich_matchups",
        doc_md="### Enrich Matchups\n\nUpdate matchup context after line refresh.",
    )
    def enrich_matchups(props_loaded: dict[str, Any]) -> dict[str, Any]:
        """Re-enrich matchups after line refresh."""
        import subprocess

        date_str = datetime.now().strftime("%Y-%m-%d")
        script_path = f"{SCRIPT_DIR}/betting_xl/enrich_props_with_matchups.py"

        result = subprocess.run(
            ["python3", script_path, "--date", date_str],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            env={**os.environ, "PYTHONPATH": PROJECT_ROOT},
            timeout=300,
        )

        # Verify coverage using env-based config
        import psycopg2

        from nba.config.database import get_intelligence_db_config

        # Coverage threshold - configurable via Airflow Variable
        min_coverage = float(Variable.get("nba_min_coverage_pct", default_var="90"))

        config = get_intelligence_db_config()
        conn = psycopg2.connect(**config)
        cursor = conn.cursor()
        # Use parameterized query to prevent SQL injection
        cursor.execute(
            """
            SELECT
                COUNT(*) as total,
                COUNT(CASE WHEN opponent_team <> ''
                    AND opponent_team IS NOT NULL
                    AND is_home IS NOT NULL THEN 1 END) as enriched
            FROM nba_props_xl WHERE game_date = %s;
        """,
            (date_str,),
        )
        result_row = cursor.fetchone()
        total = result_row[0] if result_row else 0
        enriched = result_row[1] if result_row else 0
        coverage = round(100.0 * enriched / total, 1) if total > 0 else 0
        cursor.close()
        conn.close()

        if total == 0:
            print(f"Warning: No props found for {date_str}")
            return {"coverage": 0, "total": 0, "status": "no_data"}

        if coverage < min_coverage:
            raise Exception(
                f"Coverage {coverage}% below {min_coverage}% threshold ({enriched}/{total} props)"
            )

        return {"coverage": coverage, "total": total, "enriched": enriched, "status": "success"}

    @task(
        task_id="generate_xl_predictions",
        doc_md="""
        ### Generate XL Predictions

        Run XL models for POINTS and REBOUNDS:
        - POINTS: Tier X (74% expected WR)
        - REBOUNDS: Tier A (69% expected WR)

        Output: `predictions/xl_picks_YYYY-MM-DD.json`
        """,
        sla=timedelta(minutes=15),
    )
    def generate_xl_predictions(enrichment: dict[str, Any]) -> dict[str, Any]:
        """Generate predictions using XL models."""
        import subprocess

        date_str = datetime.now().strftime("%Y-%m-%d")
        output_file = f"{PREDICTIONS_DIR}/xl_picks_{date_str}.json"
        script_path = f"{SCRIPT_DIR}/betting_xl/generate_xl_predictions.py"

        if not Path(script_path).exists():
            raise Exception("XL prediction system not found")

        result = subprocess.run(
            ["python3", script_path, "--output", output_file, "--underdog-only"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            env={**os.environ, "PYTHONPATH": PROJECT_ROOT},
            timeout=600,
        )

        if result.returncode != 0:
            raise Exception(f"XL prediction failed: {result.stderr}")

        # Parse results
        picks_data = {"total_picks": 0, "picks": []}
        if Path(output_file).exists():
            with open(output_file) as f:
                picks_data = json.load(f)

        return {
            "output_file": output_file,
            "total_picks": picks_data.get("total_picks", 0),
            "picks_by_market": {
                "POINTS": len(
                    [p for p in picks_data.get("picks", []) if p.get("stat_type") == "POINTS"]
                ),
                "REBOUNDS": len(
                    [p for p in picks_data.get("picks", []) if p.get("stat_type") == "REBOUNDS"]
                ),
            },
            "status": "success",
        }

    @task(
        task_id="generate_pro_picks",
        doc_md="""
        ### Generate Pro Picks

        Cheatsheet-based high-confidence picks:
        - POINTS: 88% expected WR
        - ASSISTS: 73% expected WR
        - REBOUNDS: 80% expected WR
        - COMBO: 71-86% expected WR

        Output: `predictions/pro_picks_YYYY-MM-DD.json`
        """,
    )
    def generate_pro_picks(enrichment: dict[str, Any]) -> dict[str, Any]:
        """Generate Pro tier picks."""
        import subprocess

        date_str = datetime.now().strftime("%Y-%m-%d")
        output_file = f"{PREDICTIONS_DIR}/pro_picks_{date_str}.json"
        script_path = f"{SCRIPT_DIR}/betting_xl/generate_cheatsheet_picks.py"

        if not Path(script_path).exists():
            return {"status": "skipped", "reason": "Pro generator not configured"}

        result = subprocess.run(
            ["python3", script_path, "--output", output_file],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            env={**os.environ, "PYTHONPATH": PROJECT_ROOT},
            timeout=300,
        )

        if result.returncode != 0:
            print(f"Pro picks warning: {result.stderr}")
            return {"status": "warning", "error": result.stderr}

        # Parse results
        picks_data = {"total_picks": 0}
        if Path(output_file).exists():
            with open(output_file) as f:
                picks_data = json.load(f)

        return {
            "output_file": output_file,
            "total_picks": picks_data.get("total_picks", 0),
            "status": "success",
        }

    @task(
        task_id="generate_odds_api_picks",
        doc_md="""
        ### Generate Odds API Picks

        Pick6 multiplier + BettingPros features:
        - TIER_1: 100% WR (mult<1.1 + R>=3 + opp>=11 + L5=100%)
        - TIER_2: 100% WR (mult<1.2 + R>=4 + opp>=16 + L5>=80% + L15>=60%)

        Output: `predictions/odds_api_picks_YYYYMMDD.json`
        """,
    )
    def generate_odds_api_picks(enrichment: dict[str, Any]) -> dict[str, Any]:
        """Generate Odds API picks with Pick6 multipliers."""
        import subprocess

        date_str = datetime.now().strftime("%Y-%m-%d")
        output_file = f"{PREDICTIONS_DIR}/odds_api_picks_{date_str.replace('-', '')}.json"
        script_path = f"{SCRIPT_DIR}/betting_xl/generate_odds_api_picks.py"

        if not Path(script_path).exists():
            return {"status": "skipped", "reason": "Odds API generator not configured"}

        result = subprocess.run(
            ["python3", script_path, "--date", date_str, "--output", output_file],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            env={**os.environ, "PYTHONPATH": PROJECT_ROOT},
            timeout=300,
        )

        if result.returncode != 0:
            print(f"Odds API picks warning: {result.stderr}")
            return {"status": "warning", "error": result.stderr}

        # Parse results
        picks_data = {"total_picks": 0}
        if Path(output_file).exists():
            with open(output_file) as f:
                picks_data = json.load(f)

        return {
            "output_file": output_file,
            "total_picks": picks_data.get("total_picks", 0),
            "status": "success",
        }

    @task(
        task_id="output_final_picks",
        doc_md="""
        ### Output Final Picks

        Combine all prediction sources and output final picks.

        Summary includes:
        - Total picks count
        - Picks by tier and market
        - Expected win rates
        """,
    )
    def output_final_picks(
        xl_result: dict[str, Any],
        pro_result: dict[str, Any],
        odds_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Combine and output final picks summary."""
        date_str = datetime.now().strftime("%Y-%m-%d")

        summary = {
            "date": date_str,
            "generated_at": datetime.now().isoformat(),
            "xl_predictions": {
                "file": xl_result.get("output_file"),
                "total": xl_result.get("total_picks", 0),
                "by_market": xl_result.get("picks_by_market", {}),
                "status": xl_result.get("status"),
            },
            "pro_picks": {
                "file": pro_result.get("output_file"),
                "total": pro_result.get("total_picks", 0),
                "status": pro_result.get("status"),
            },
            "odds_api_picks": {
                "file": odds_result.get("output_file"),
                "total": odds_result.get("total_picks", 0),
                "status": odds_result.get("status"),
            },
            "total_all_sources": (
                xl_result.get("total_picks", 0)
                + pro_result.get("total_picks", 0)
                + odds_result.get("total_picks", 0)
            ),
        }

        # Save summary
        summary_file = f"{PREDICTIONS_DIR}/daily_summary_{date_str}.json"
        Path(PREDICTIONS_DIR).mkdir(parents=True, exist_ok=True)

        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        # Show picks using show_picks.py for consistent, detailed output
        import subprocess

        show_picks_script = f"{SCRIPT_DIR}/betting_xl/show_picks.py"
        if Path(show_picks_script).exists():
            result = subprocess.run(
                ["python3", show_picks_script],
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
            )
            print(result.stdout)
            if result.stderr:
                print(result.stderr)

        # Check for low volume
        if summary["xl_predictions"]["total"] < 2:
            print("WARNING: Low XL prediction volume - filtering may be too strict")

        return {
            "summary_file": summary_file,
            "total_picks": summary["total_all_sources"],
            "status": "completed",
        }

    # ========================================================================
    # Task Dependencies
    # ========================================================================

    # Pre-flight checks (sequential)
    health = health_check()
    freshness = validate_data_freshness()
    performance = check_performance_thresholds()

    health >> freshness >> performance

    # Refresh data (parallel after pre-flight)
    props = refresh_props()
    vegas = refresh_vegas_lines()
    cheatsheet = refresh_cheatsheet()

    performance >> [props, vegas, cheatsheet]

    # Load and enrich
    props_loaded = load_refreshed_props(props)
    enrichment = enrich_matchups(props_loaded)

    # Wait for all refreshes before enrichment
    [vegas, cheatsheet] >> enrichment

    # Generate predictions (parallel after enrichment)
    xl_predictions = generate_xl_predictions(enrichment)
    pro_picks = generate_pro_picks(enrichment)
    odds_picks = generate_odds_api_picks(enrichment)

    # Final output
    final = output_final_picks(xl_predictions, pro_picks, odds_picks)


# Instantiate the DAG
dag = nba_evening_pipeline()
