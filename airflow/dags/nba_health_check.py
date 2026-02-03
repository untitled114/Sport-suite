"""
NBA Health Check DAG

Scheduled: Every 4 hours
Purpose: System health monitoring and alerting

Tasks:
1. check_database_connectivity - Verify all 4 NBA databases are accessible
2. check_model_files - Verify XL model files exist (24 .pkl files)
3. check_data_freshness - Verify props and coverage data
4. check_disk_space - Monitor disk usage
5. check_api_health - Test API endpoints (if available)
6. send_health_report - Aggregate and report health status

Author: Claude Code
"""

from __future__ import annotations

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

# Project paths - use env var with fallback
PROJECT_ROOT = Variable.get("nba_project_root", default_var="/home/untitled/Sport-suite")
SCRIPT_DIR = f"{PROJECT_ROOT}/nba"
MODELS_DIR = f"{SCRIPT_DIR}/models/saved_xl"

# Add project to path for imports
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

default_args = {
    "owner": "nba_pipeline",
    "depends_on_past": False,
    "email": Variable.get("alert_email", default_var="alerts@example.com").split(","),
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
    "execution_timeout": timedelta(minutes=10),
    "on_success_callback": on_success,
    "on_failure_callback": on_failure,
    "on_retry_callback": on_retry,
}


def send_health_alert(subject: str, body: str, is_critical: bool = False) -> None:
    """Send health alert via email."""
    priority_prefix = "[CRITICAL]" if is_critical else "[WARNING]"

    try:
        send_email(
            to=default_args["email"],
            subject=f"{priority_prefix} {subject}",
            html_content=body,
        )
    except Exception as e:
        print(f"Failed to send health alert: {e}")


def alert_on_health_failure(context: dict[str, Any]) -> None:
    """Alert callback for health check failures."""
    task_instance = context.get("task_instance")
    task_id = task_instance.task_id

    subject = f"NBA Health Check Failed: {task_id}"
    body = f"""
    <h3>Health Check Failure</h3>
    <p><b>Check:</b> {task_id}</p>
    <p><b>Time:</b> {datetime.now().isoformat()}</p>
    <p><b>Error:</b> {context.get('exception', 'Unknown error')}</p>
    <p>Please investigate immediately.</p>
    """

    send_health_alert(subject, body, is_critical=True)


# ============================================================================
# DAG Definition
# ============================================================================


@dag(
    dag_id="nba_health_check",
    description="NBA system health monitoring (every 4 hours)",
    schedule=CronTriggerTimetable("0 */4 * * *", timezone="UTC"),  # Every 4 hours
    start_date=datetime(2025, 11, 7),
    catchup=False,
    tags=["nba", "health", "monitoring"],
    default_args=default_args,
    max_active_runs=1,
    doc_md=__doc__,
    on_failure_callback=alert_on_health_failure,
)
def nba_health_check():
    """
    NBA System Health Check Pipeline

    Monitors critical system components:
    - Database connectivity (4 NBA databases)
    - XL model files integrity (24 .pkl files)
    - Data freshness (props, coverage)
    - Disk space utilization
    - API health (optional)

    Alerts are sent on any check failure.
    """

    @task(
        task_id="check_database_connectivity",
        doc_md="""
        ### Check Database Connectivity

        Verifies all 4 NBA databases are accessible:
        - nba_players (port 5536)
        - nba_games (port 5537)
        - nba_team (port 5538)
        - nba_intelligence (port 5539)

        Uses centralized config from `nba.config.database` with env vars.
        """,
    )
    def check_database_connectivity() -> dict[str, Any]:
        """Check connectivity to all NBA databases using env-based config."""
        import psycopg2

        from nba.config.database import (
            get_games_db_config,
            get_intelligence_db_config,
            get_players_db_config,
            get_team_db_config,
        )

        databases = [
            {"name": "nba_players", "config_fn": get_players_db_config},
            {"name": "nba_games", "config_fn": get_games_db_config},
            {"name": "nba_team", "config_fn": get_team_db_config},
            {"name": "nba_intelligence", "config_fn": get_intelligence_db_config},
        ]

        results = {}
        failures = []

        for db in databases:
            try:
                config = db["config_fn"]()
                conn = psycopg2.connect(**config)
                cursor = conn.cursor()
                cursor.execute("SELECT 1;")
                cursor.fetchone()
                cursor.close()
                conn.close()
                results[db["name"]] = "ok"
                print(f"[OK] {db['name']} (port {config['port']})")
            except Exception as e:
                results[db["name"]] = f"failed: {str(e)}"
                failures.append(db["name"])
                print(f"[FAIL] {db['name']}: {e}")

        if failures:
            raise Exception(f"Database connectivity failures: {', '.join(failures)}")

        return {"databases": results, "status": "all_connected"}

    @task(
        task_id="check_model_files",
        doc_md="""
        ### Check Model Files

        Verifies XL model files exist:
        - Expected: 24 .pkl files (6 per market x 4 markets)
        - 4 .json metadata files

        Model location: `nba/models/saved_xl/`
        """,
    )
    def check_model_files() -> dict[str, Any]:
        """Verify XL model files exist and are valid."""
        models_path = Path(MODELS_DIR)

        if not models_path.exists():
            raise Exception(f"Models directory not found: {MODELS_DIR}")

        # Count pkl files
        pkl_files = list(models_path.glob("*.pkl"))
        pkl_count = len(pkl_files)

        # Count json metadata files
        json_files = list(models_path.glob("*.json"))
        json_count = len(json_files)

        # Check minimum requirements (24 pkl files, 4 json metadata)
        expected_pkl = 24

        if pkl_count < expected_pkl:
            raise Exception(f"Missing model files: {pkl_count}/{expected_pkl} .pkl files")

        # Verify key model files exist
        required_markets = ["points", "rebounds", "assists", "threes"]
        required_components = [
            "regressor",
            "classifier",
            "calibrator",
            "imputer",
            "scaler",
            "features",
        ]

        missing_files = []
        for market in required_markets:
            for component in required_components:
                expected_file = models_path / f"{market}_xl_{component}.pkl"
                if not expected_file.exists():
                    missing_files.append(expected_file.name)

        if missing_files:
            raise Exception(f"Missing model components: {missing_files[:5]}...")

        # Check file ages (warn if older than 90 days)
        oldest_file = min(pkl_files, key=lambda f: f.stat().st_mtime)
        oldest_age_days = (datetime.now().timestamp() - oldest_file.stat().st_mtime) / 86400

        print(f"[OK] Model files: {pkl_count} .pkl, {json_count} .json")
        print(f"[INFO] Oldest model file: {oldest_file.name} ({oldest_age_days:.0f} days old)")

        return {
            "pkl_count": pkl_count,
            "json_count": json_count,
            "oldest_file_age_days": oldest_age_days,
            "status": "ok",
        }

    @task(
        task_id="check_data_freshness",
        doc_md="""
        ### Check Data Freshness

        Verifies:
        - Props available for today (minimum 50)
        - Matchup coverage >= 95%
        - Vegas lines present for scheduled games
        - Cheatsheet data available

        Uses centralized config from `nba.config.database` with env vars.
        """,
    )
    def check_data_freshness() -> dict[str, Any]:
        """Check data freshness and availability."""
        import psycopg2

        from nba.config.database import get_games_db_config, get_intelligence_db_config

        date_str = datetime.now().strftime("%Y-%m-%d")
        results = {}
        warnings = []

        # Check props count
        try:
            config = get_intelligence_db_config()
            conn = psycopg2.connect(**config)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM nba_props_xl WHERE game_date = %s;", (date_str,))
            result = cursor.fetchone()
            props_count = result[0] if result else 0
            cursor.close()
            conn.close()
            results["props_count"] = props_count

            if props_count < 50:
                warnings.append(f"Low prop volume: {props_count}")
                print(f"[WARN] Props: {props_count} (minimum 50)")
            else:
                print(f"[OK] Props: {props_count}")
        except Exception as e:
            warnings.append(f"Props check failed: {e}")
            results["props_count"] = 0

        # Check matchup coverage
        try:
            config = get_intelligence_db_config()
            conn = psycopg2.connect(**config)
            cursor = conn.cursor()
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
            result = cursor.fetchone()
            total = result[0] if result else 0
            enriched = result[1] if result else 0
            coverage = round(100.0 * enriched / total, 1) if total > 0 else 0
            cursor.close()
            conn.close()
            results["coverage"] = coverage

            if coverage < 95:
                warnings.append(f"Coverage below target: {coverage}%")
                print(f"[WARN] Coverage: {coverage}% (target 95%)")
            else:
                print(f"[OK] Coverage: {coverage}%")
        except Exception as e:
            warnings.append(f"Coverage check failed: {e}")
            results["coverage"] = 0

        # Check vegas data
        try:
            config = get_games_db_config()
            conn = psycopg2.connect(**config)
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT
                    COUNT(*) as total,
                    COUNT(CASE WHEN vegas_spread IS NOT NULL THEN 1 END) as with_vegas
                FROM games WHERE game_date = %s;
            """,
                (date_str,),
            )
            vegas_result = cursor.fetchone()
            cursor.close()
            conn.close()

            total_games = vegas_result[0] if vegas_result else 0
            vegas_count = vegas_result[1] if vegas_result else 0
            results["vegas_games"] = vegas_count
            results["total_games"] = total_games

            if total_games > 0 and vegas_count < total_games:
                warnings.append(f"Vegas coverage: {vegas_count}/{total_games}")
                print(f"[WARN] Vegas: {vegas_count}/{total_games} games")
            elif total_games > 0:
                print(f"[OK] Vegas: {vegas_count}/{total_games} games")
            else:
                print("[INFO] No games scheduled today")
        except Exception as e:
            warnings.append(f"Vegas check failed: {e}")

        # Check cheatsheet data
        try:
            config = get_intelligence_db_config()
            conn = psycopg2.connect(**config)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM cheatsheet_data WHERE game_date = %s;", (date_str,)
            )
            result = cursor.fetchone()
            cheatsheet_count = result[0] if result else 0
            cursor.close()
            conn.close()
            results["cheatsheet_count"] = cheatsheet_count

            if cheatsheet_count < 50:
                warnings.append(f"Cheatsheet data: {cheatsheet_count} props")
                print(f"[WARN] Cheatsheet: {cheatsheet_count} props")
            else:
                print(f"[OK] Cheatsheet: {cheatsheet_count} props")
        except Exception as e:
            # Cheatsheet might not exist, this is optional
            print(f"[INFO] Cheatsheet check: {e}")
            results["cheatsheet_count"] = 0

        results["warnings"] = warnings
        results["status"] = "ok" if not warnings else "warnings"

        return results

    @task(
        task_id="check_disk_space",
        doc_md="### Check Disk Space\n\nMonitor disk usage on project directory.",
    )
    def check_disk_space() -> dict[str, Any]:
        """Check disk space utilization."""
        import shutil

        total, used, free = shutil.disk_usage(PROJECT_ROOT)

        usage_percent = (used / total) * 100
        free_gb = free / (1024**3)

        if usage_percent >= 90:
            raise Exception(f"Disk space critical: {usage_percent:.1f}% used, {free_gb:.1f}GB free")

        if usage_percent >= 80:
            print(f"[WARN] Disk space: {usage_percent:.1f}% used, {free_gb:.1f}GB free")
        else:
            print(f"[OK] Disk space: {usage_percent:.1f}% used, {free_gb:.1f}GB free")

        return {
            "usage_percent": usage_percent,
            "free_gb": free_gb,
            "status": "ok" if usage_percent < 90 else "critical",
        }

    @task(
        task_id="check_api_health",
        doc_md="""
        ### Check API Health

        Tests API endpoints if configured:
        - BettingPros API (requires API key)
        - ESPN API fallback (public endpoint)
        """,
    )
    def check_api_health() -> dict[str, Any]:
        """Check API endpoint health."""
        import urllib.error
        import urllib.request

        results = {}

        # Check ESPN API (public, no auth required)
        espn_url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"

        try:
            request = urllib.request.Request(espn_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=10) as response:
                if response.status == 200:
                    results["espn_api"] = "ok"
                    print("[OK] ESPN API: accessible")
                else:
                    results["espn_api"] = f"status {response.status}"
                    print(f"[WARN] ESPN API: status {response.status}")
        except urllib.error.URLError as e:
            results["espn_api"] = f"failed: {e}"
            print(f"[WARN] ESPN API: {e}")
        except Exception as e:
            results["espn_api"] = f"error: {e}"
            print(f"[WARN] ESPN API: {e}")

        # BettingPros API check would require auth, skip in health check
        results["bettingpros_api"] = "skipped (requires auth)"

        results["status"] = "ok"
        return results

    @task(
        task_id="send_health_report",
        doc_md="""
        ### Send Health Report

        Aggregate all health checks and send report.
        Only sends email if there are warnings or failures.
        """,
    )
    def send_health_report(
        db_result: dict[str, Any],
        model_result: dict[str, Any],
        data_result: dict[str, Any],
        disk_result: dict[str, Any],
        api_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Aggregate health checks and send report."""
        all_results = {
            "database": db_result,
            "models": model_result,
            "data": data_result,
            "disk": disk_result,
            "api": api_result,
        }

        # Determine overall status
        has_failures = any(r.get("status") in ["failed", "critical"] for r in all_results.values())

        has_warnings = (
            data_result.get("warnings", [])
            or disk_result.get("usage_percent", 0) >= 80
            or model_result.get("oldest_file_age_days", 0) > 90
        )

        overall_status = "healthy"
        if has_failures:
            overall_status = "critical"
        elif has_warnings:
            overall_status = "warnings"

        # Log summary
        print("\n" + "=" * 60)
        print("NBA HEALTH CHECK SUMMARY")
        print("=" * 60)
        print(f"Database: {db_result.get('status', 'unknown')}")
        model_status = model_result.get("status", "unknown")
        model_files = model_result.get("pkl_count", 0)
        print(f"Models: {model_status} ({model_files} files)")
        data_status = data_result.get("status", "unknown")
        props_count = data_result.get("props_count", 0)
        print(f"Data: {data_status} ({props_count} props)")
        disk_status = disk_result.get("status", "unknown")
        disk_pct = disk_result.get("usage_percent", 0)
        print(f"Disk: {disk_status} ({disk_pct:.1f}%)")
        print(f"API: {api_result.get('status', 'unknown')}")
        print("=" * 60)
        print(f"OVERALL: {overall_status.upper()}")
        print("=" * 60 + "\n")

        # Send email only on issues
        if has_failures or has_warnings:
            warnings_html = ""
            if data_result.get("warnings"):
                warnings_html = (
                    "<ul>" + "".join(f"<li>{w}</li>" for w in data_result["warnings"]) + "</ul>"
                )

            body = f"""
            <h3>NBA System Health Report</h3>
            <p><b>Status:</b> {overall_status.upper()}</p>
            <p><b>Time:</b> {datetime.now().isoformat()}</p>

            <h4>Check Results:</h4>
            <table border="1" cellpadding="5">
                <tr><th>Component</th><th>Status</th><th>Details</th></tr>
                <tr>
                    <td>Database</td>
                    <td>{db_result.get('status')}</td>
                    <td>4/4 connected</td>
                </tr>
                <tr>
                    <td>Models</td>
                    <td>{model_result.get('status')}</td>
                    <td>{model_result.get('pkl_count')} files</td>
                </tr>
                <tr>
                    <td>Data</td>
                    <td>{data_result.get('status')}</td>
                    <td>{data_result.get('props_count')} props</td>
                </tr>
                <tr>
                    <td>Disk</td>
                    <td>{disk_result.get('status')}</td>
                    <td>{disk_result.get('usage_percent'):.1f}% used</td>
                </tr>
                <tr>
                    <td>API</td>
                    <td>{api_result.get('status')}</td>
                    <td>ESPN: {api_result.get('espn_api')}</td>
                </tr>
            </table>

            {f"<h4>Warnings:</h4>{warnings_html}" if warnings_html else ""}
            """

            send_health_alert(
                f"NBA System Health: {overall_status.upper()}",
                body,
                is_critical=has_failures,
            )

        return {
            "overall_status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "results": all_results,
        }

    # ========================================================================
    # Task Dependencies
    # ========================================================================

    # All checks run in parallel
    db_check = check_database_connectivity()
    model_check = check_model_files()
    data_check = check_data_freshness()
    disk_check = check_disk_space()
    api_check = check_api_health()

    # Aggregate results (terminal task)
    send_health_report(
        db_check,
        model_check,
        data_check,
        disk_check,
        api_check,
    )


# Instantiate the DAG
dag = nba_health_check()
