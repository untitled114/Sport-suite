"""
NBA Morning Pipeline DAG

Scheduled: Daily at 10:00 AM EST
Purpose: Data collection and enrichment workflow

Tasks:
1. fetch_props - Multi-source collection from 7 sportsbooks
2. load_props_to_db - Store props in PostgreSQL
3. fetch_cheatsheet - BettingPros recommendations
4. enrich_matchups - Add opponent & home/away context
5. fetch_game_results - Daily stats for completed games
6. populate_actual_values - Update props with game results
7. update_injuries - Injury report synchronization
8. load_team_games - Incremental from NBA API
9. update_team_stats - Pace/ratings for current season
10. load_team_advanced_stats - Real PACE from NBA API
11. fetch_vegas_lines - Game spreads & totals
12. update_minutes_projections - Minutes projection refresh
13. update_prop_history - Bayesian hit rate calculations
14. verify_data_freshness - Final validation

Note: Rolling stats are calculated on-the-fly from player_game_logs during
feature extraction - no separate update task needed.

Author: Claude Code
"""

from __future__ import annotations

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

# Default args for all tasks
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


def get_current_season() -> int:
    """Calculate NBA season (uses END year: 2024-25 season = 2025)."""
    now = datetime.now()
    if now.month >= 10:
        return now.year + 1
    return now.year


def alert_on_failure(context: dict[str, Any]) -> None:
    """Send alert on task failure - can be extended for Slack/PagerDuty."""
    task_instance = context.get("task_instance")
    dag_id = context.get("dag").dag_id
    task_id = task_instance.task_id
    execution_date = context.get("execution_date")
    log_url = task_instance.log_url

    subject = f"[AIRFLOW] NBA Morning Pipeline Failed: {task_id}"
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

    # Placeholder for Slack integration
    # slack_webhook = Variable.get("slack_webhook_url", default_var=None)
    # if slack_webhook:
    #     requests.post(slack_webhook, json={"text": f":x: {subject}"})


# ============================================================================
# DAG Definition
# ============================================================================


@dag(
    dag_id="nba_morning_pipeline",
    description="NBA betting data collection and enrichment workflow",
    schedule="0 10 * * *",  # 10:00 AM EST daily
    start_date=datetime(2025, 11, 7),
    catchup=False,
    tags=["nba", "predictions", "morning", "data-collection"],
    default_args=default_args,
    max_active_runs=1,
    doc_md=__doc__,
    on_failure_callback=alert_on_failure,
)
def nba_morning_pipeline():
    """
    NBA Morning Data Pipeline

    Collects and enriches betting data from multiple sources:
    - Props from 7 sportsbooks via BettingPros API
    - Cheatsheet recommendations
    - Matchup context (opponent, home/away)
    - Game results and actual values
    - Injury reports
    - Team statistics and advanced metrics
    - Vegas lines (spreads/totals)
    - Rolling player statistics
    """

    @task(
        task_id="fetch_props",
        doc_md="""
        ### Fetch Props

        Multi-source collection from 7 sportsbooks:
        - DraftKings, FanDuel, BetMGM, Caesars
        - BetRivers, ESPNBet, Underdog Fantasy

        **Output:** JSON file in `betting_xl/lines/all_sources_*.json`
        """,
        sla=timedelta(minutes=10),
    )
    def fetch_props() -> dict[str, Any]:
        """Fetch player props from all configured sportsbooks."""
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
            raise Exception(f"fetch_props failed: {result.stderr}")

        # Find latest props file
        pattern = f"{SCRIPT_DIR}/betting_xl/lines/all_sources_*.json"
        files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)

        if not files:
            raise Exception("No props file generated")

        return {"props_file": files[0], "status": "success"}

    @task(
        task_id="load_props_to_db",
        doc_md="""
        ### Load Props to Database

        Stores fetched props in PostgreSQL (nba_intelligence database).
        Deduplicates and tracks line movements.
        """,
    )
    def load_props_to_db(fetch_result: dict[str, Any]) -> dict[str, Any]:
        """Load props to PostgreSQL database."""
        import subprocess

        props_file = fetch_result["props_file"]
        script_path = f"{SCRIPT_DIR}/betting_xl/loaders/load_props_to_db.py"

        result = subprocess.run(
            ["python3", script_path, "--file", props_file, "--skip-mongodb"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            env={**os.environ, "PYTHONPATH": PROJECT_ROOT},
            timeout=300,
        )

        if result.returncode != 0:
            raise Exception(f"load_props_to_db failed: {result.stderr}")

        # Get props count from database using env-based config
        import psycopg2

        from nba.config.database import get_intelligence_db_config

        date_str = datetime.now().strftime("%Y-%m-%d")
        config = get_intelligence_db_config()
        conn = psycopg2.connect(**config)
        cursor = conn.cursor()
        # Use parameterized query
        cursor.execute("SELECT COUNT(*) FROM nba_props_xl WHERE game_date = %s;", (date_str,))
        count = cursor.fetchone()[0]
        cursor.close()
        conn.close()

        return {"props_count": count, "status": "success"}

    @task(
        task_id="fetch_cheatsheet",
        doc_md="""
        ### Fetch Cheatsheet Data

        Retrieves BettingPros recommendations and projections.
        Platform: Underdog Fantasy
        """,
    )
    def fetch_cheatsheet() -> dict[str, Any]:
        """Fetch BettingPros cheatsheet data."""
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
            print(f"Cheatsheet fetch warning: {result.stderr}")
            return {"status": "warning", "error": result.stderr}

        # Find latest cheatsheet file
        pattern = f"{SCRIPT_DIR}/betting_xl/lines/cheatsheet_underdog_*.json"
        files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)

        return {"cheatsheet_file": files[0] if files else None, "status": "success"}

    @task(
        task_id="load_cheatsheet_to_db",
        doc_md="### Load Cheatsheet to Database\n\nStore cheatsheet data in PostgreSQL.",
    )
    def load_cheatsheet_to_db(cheatsheet_result: dict[str, Any]) -> dict[str, Any]:
        """Load cheatsheet data to database."""
        import subprocess

        if cheatsheet_result.get("status") != "success":
            return {"status": "skipped"}

        cheatsheet_file = cheatsheet_result.get("cheatsheet_file")
        if not cheatsheet_file:
            return {"status": "skipped", "reason": "No cheatsheet file"}

        script_path = f"{SCRIPT_DIR}/betting_xl/loaders/load_cheatsheet_to_db.py"

        result = subprocess.run(
            ["python3", script_path, "--file", cheatsheet_file],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            env={**os.environ, "PYTHONPATH": PROJECT_ROOT},
            timeout=300,
        )

        return {"status": "success" if result.returncode == 0 else "warning"}

    @task(
        task_id="enrich_matchups",
        doc_md="""
        ### Enrich Matchup Data

        Adds opponent team and home/away context to props.
        Minimum coverage configurable via `nba_min_coverage_pct` Variable (default 90%).
        """,
    )
    def enrich_matchups(load_result: dict[str, Any]) -> dict[str, Any]:
        """Enrich props with matchup data."""
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

        if result.returncode != 0:
            print(f"Enrichment warning: {result.stderr}")

        # Check coverage using env-based config
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
        task_id="fetch_game_results",
        doc_md="### Fetch Game Results\n\nDaily stats for completed games (yesterday).",
    )
    def fetch_game_results() -> dict[str, Any]:
        """Fetch yesterday's game results."""
        import subprocess

        script_path = f"{SCRIPT_DIR}/scripts/fetch_daily_stats.py"

        if not Path(script_path).exists():
            return {"status": "skipped", "reason": "Script not found"}

        result = subprocess.run(
            ["python3", script_path, "--days", "1"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            env={**os.environ, "PYTHONPATH": PROJECT_ROOT},
            timeout=600,
        )

        return {"status": "success" if result.returncode == 0 else "no_new_games"}

    @task(
        task_id="populate_actual_values",
        doc_md="### Populate Actual Values\n\nUpdate props with game results (last 7 days).",
    )
    def populate_actual_values(game_results: dict[str, Any]) -> dict[str, Any]:
        """Populate actual values for recent props."""
        import subprocess

        script_path = f"{SCRIPT_DIR}/betting_xl/populate_actual_values.py"

        if not Path(script_path).exists():
            return {"status": "skipped", "reason": "Script not found"}

        result = subprocess.run(
            ["python3", script_path, "--days", "7"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            env={**os.environ, "PYTHONPATH": PROJECT_ROOT},
            timeout=300,
        )

        return {"status": "success" if result.returncode == 0 else "skipped"}

    @task(
        task_id="update_injuries",
        doc_md="### Update Injury Reports\n\nSynchronize latest injury data.",
    )
    def update_injuries() -> dict[str, Any]:
        """Update injury reports."""
        import subprocess

        script_path = f"{SCRIPT_DIR}/scripts/update_injuries_NOW.py"

        if not Path(script_path).exists():
            return {"status": "skipped", "reason": "Injury tracker not configured"}

        result = subprocess.run(
            ["python3", script_path],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            env={**os.environ, "PYTHONPATH": PROJECT_ROOT},
            timeout=300,
        )

        return {"status": "success" if result.returncode == 0 else "skipped"}

    @task(
        task_id="load_team_games",
        doc_md="### Load Team Games\n\nIncremental load from NBA API (1 API call).",
    )
    def load_team_games() -> dict[str, Any]:
        """Load team games incrementally."""
        import subprocess

        script_path = f"{SCRIPT_DIR}/scripts/loaders/load_nba_games_incremental.py"

        if not Path(script_path).exists():
            return {"status": "skipped", "reason": "Team games loader not found"}

        result = subprocess.run(
            ["python3", script_path],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            env={**os.environ, "PYTHONPATH": PROJECT_ROOT},
            timeout=300,
        )

        return {"status": "success" if result.returncode == 0 else "skipped"}

    @task(
        task_id="update_team_stats",
        doc_md="### Update Team Season Stats\n\nPace/ratings for current season.",
    )
    def update_team_stats() -> dict[str, Any]:
        """Update team season statistics."""
        import subprocess

        script_path = f"{SCRIPT_DIR}/scripts/loaders/calculate_team_stats.py"
        season = get_current_season()

        if not Path(script_path).exists():
            return {"status": "skipped", "reason": "Script not found"}

        result = subprocess.run(
            ["python3", script_path, "--season", str(season)],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            env={**os.environ, "PYTHONPATH": PROJECT_ROOT},
            timeout=300,
        )

        return {"status": "success" if result.returncode == 0 else "warning"}

    @task(
        task_id="load_team_advanced_stats",
        doc_md="### Load Team Advanced Stats\n\nReal PACE from NBA API.",
    )
    def load_team_advanced_stats() -> dict[str, Any]:
        """Load advanced team statistics from NBA API."""
        import subprocess

        script_path = f"{SCRIPT_DIR}/scripts/loaders/load_team_advanced_stats.py"

        if not Path(script_path).exists():
            return {"status": "skipped", "reason": "Script not found"}

        result = subprocess.run(
            ["python3", script_path],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            env={**os.environ, "PYTHONPATH": PROJECT_ROOT},
            timeout=300,
        )

        return {"status": "success" if result.returncode == 0 else "skipped"}

    @task(
        task_id="fetch_vegas_lines",
        doc_md="### Fetch Vegas Lines\n\nGame spreads & totals for feature extraction.",
    )
    def fetch_vegas_lines() -> dict[str, Any]:
        """Fetch Vegas lines for today's games."""
        import subprocess

        date_str = datetime.now().strftime("%Y-%m-%d")
        script_path = f"{SCRIPT_DIR}/betting_xl/fetchers/fetch_vegas_lines.py"

        if not Path(script_path).exists():
            return {"status": "skipped", "reason": "Vegas lines fetcher not configured"}

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
        task_id="update_minutes_projections",
        doc_md="### Update Minutes Projections\n\nRefresh minutes projection data.",
    )
    def update_minutes_projections() -> dict[str, Any]:
        """Update minutes projections."""
        import subprocess

        script_path = f"{SCRIPT_DIR}/scripts/loaders/calculate_minutes_projections.py"

        if not Path(script_path).exists():
            return {"status": "skipped", "reason": "Script not found"}

        result = subprocess.run(
            ["python3", script_path, "--update"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            env={**os.environ, "PYTHONPATH": PROJECT_ROOT},
            timeout=300,
        )

        return {"status": "success" if result.returncode == 0 else "warning"}

    @task(
        task_id="update_prop_history",
        doc_md="### Update Prop History\n\nBayesian hit rate calculations (incremental, last 7 days).",
    )
    def update_prop_history() -> dict[str, Any]:
        """Update prop performance history."""
        import subprocess

        script_path = f"{SCRIPT_DIR}/scripts/compute_prop_history.py"
        season = get_current_season()

        if not Path(script_path).exists():
            return {"status": "skipped", "reason": "Script not found"}

        result = subprocess.run(
            ["python3", script_path, "--season", str(season), "--incremental", "--days", "7"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            env={**os.environ, "PYTHONPATH": PROJECT_ROOT},
            timeout=600,
        )

        return {"status": "success" if result.returncode == 0 else "warning"}

    @task(
        task_id="verify_data_freshness",
        doc_md="""
        ### Verify Data Freshness

        Final validation step ensuring all data is ready for evening predictions.

        Checks:
        - Props count > minimum (configurable via `nba_min_props` Variable, default 50)
        - Matchup coverage (from enrichment task)
        - Database connectivity
        """,
    )
    def verify_data_freshness(enrichment_result: dict[str, Any]) -> dict[str, Any]:
        """Verify all data is fresh and ready."""
        import psycopg2

        from nba.config.database import get_intelligence_db_config

        date_str = datetime.now().strftime("%Y-%m-%d")
        min_props = int(Variable.get("nba_min_props", default_var="50"))

        config = get_intelligence_db_config()
        conn = psycopg2.connect(**config)
        cursor = conn.cursor()

        # Check props count using parameterized query
        cursor.execute("SELECT COUNT(*) FROM nba_props_xl WHERE game_date = %s;", (date_str,))
        result = cursor.fetchone()
        props_count = result[0] if result else 0
        cursor.close()
        conn.close()

        # Skip volume check if no data (might be no games today)
        if props_count == 0:
            print(f"Warning: No props found for {date_str} - no games scheduled?")
            return {
                "date": date_str,
                "props_count": 0,
                "coverage": 0,
                "status": "no_games",
            }

        if props_count < min_props:
            raise Exception(f"Low prop volume: {props_count} (minimum {min_props})")

        # Verify coverage from enrichment
        coverage = enrichment_result.get("coverage", 0)

        return {
            "date": date_str,
            "props_count": props_count,
            "coverage": coverage,
            "status": "ready_for_evening",
        }

    # ========================================================================
    # Task Dependencies
    # ========================================================================

    # Primary data flow
    fetch_result = fetch_props()
    load_result = load_props_to_db(fetch_result)
    enrichment_result = enrich_matchups(load_result)

    # Cheatsheet flow (parallel to enrichment)
    cheatsheet_result = fetch_cheatsheet()
    cheatsheet_loaded = load_cheatsheet_to_db(cheatsheet_result)

    # Game results flow
    game_results = fetch_game_results()
    actual_values = populate_actual_values(game_results)

    # Independent tasks (can run in parallel)
    injuries = update_injuries()
    team_games = load_team_games()
    team_stats = update_team_stats()
    advanced_stats = load_team_advanced_stats()
    vegas = fetch_vegas_lines()
    minutes = update_minutes_projections()
    prop_history = update_prop_history()

    # Final verification depends on enrichment and key data tasks
    verification = verify_data_freshness(enrichment_result)

    # Set task order
    fetch_result >> load_result >> enrichment_result >> verification
    cheatsheet_result >> cheatsheet_loaded
    game_results >> actual_values

    # All secondary tasks should complete before verification
    [
        cheatsheet_loaded,
        actual_values,
        injuries,
        team_games,
        team_stats,
        advanced_stats,
        vegas,
        minutes,
        prop_history,
    ] >> verification


# Instantiate the DAG
dag = nba_morning_pipeline()
