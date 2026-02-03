"""
NBA Validation Pipeline DAG

Scheduled: Daily at 09:30 AM EST (14:30 UTC)
Purpose: Validate pick performance and track win rates

Tasks:
1. validate_yesterday - Validate yesterday's picks against actuals
2. validate_rolling_7d - Rolling 7-day performance by system
3. validate_rolling_30d - Rolling 30-day performance trends
4. check_performance_alerts - Alert if win rate drops below thresholds
5. save_validation_results - Store results for historical tracking
6. generate_performance_report - Create daily performance summary

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
VALIDATION_DIR = f"{SCRIPT_DIR}/betting_xl/validation_results"

# Add project to path for imports
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Performance thresholds for alerts
MIN_WIN_RATE = float(Variable.get("nba_min_win_rate", default_var="52.0"))
MIN_ROI = float(Variable.get("nba_min_roi", default_var="-5.0"))

default_args = {
    "owner": "nba_pipeline",
    "depends_on_past": False,
    "email": Variable.get("alert_email", default_var="alerts@example.com").split(","),
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(minutes=15),
    "on_success_callback": on_success,
    "on_failure_callback": on_failure,
    "on_retry_callback": on_retry,
}


def send_performance_alert(subject: str, body: str, is_critical: bool = False) -> None:
    """Send performance alert via email."""
    priority_prefix = "[CRITICAL]" if is_critical else "[WARNING]"
    try:
        send_email(
            to=default_args["email"],
            subject=f"{priority_prefix} {subject}",
            html_content=body,
        )
    except Exception as e:
        print(f"Failed to send alert: {e}")


# ============================================================================
# DAG Definition
# ============================================================================


@dag(
    dag_id="nba_validation_pipeline",
    description="NBA pick performance validation and tracking",
    schedule=CronTriggerTimetable("30 14 * * *", timezone="UTC"),  # 09:30 AM EST daily
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["nba", "validation", "performance", "tracking"],
    default_args=default_args,
    max_active_runs=1,
    doc_md=__doc__,
)
def nba_validation_pipeline():
    """
    NBA Pick Validation Pipeline

    Validates betting picks against actual results:
    - XL picks (ML model predictions)
    - PRO picks (BettingPros cheatsheet filters)
    - ODDS_API picks (Pick6 multiplier strategy)

    Tracks performance over time and alerts on degradation.
    """

    @task(
        task_id="validate_yesterday",
        doc_md="""
        ### Validate Yesterday's Picks

        Runs validation for yesterday's picks against actual game results.
        Returns detailed breakdown by system, market, and filter.
        """,
    )
    def validate_yesterday() -> dict[str, Any]:
        """Validate yesterday's picks."""
        import subprocess

        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        script_path = f"{SCRIPT_DIR}/betting_xl/validate_predictions.py"

        result = subprocess.run(
            [
                "python3",
                script_path,
                "--date",
                yesterday,
            ],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            env={**os.environ, "PYTHONPATH": PROJECT_ROOT},
            timeout=300,
        )

        # Parse output for key metrics
        output = result.stdout
        metrics = {
            "date": yesterday,
            "raw_output": output,
            "returncode": result.returncode,
        }

        # Extract system results from output
        for system in ["XL", "PRO", "ODDS_API"]:
            # Look for lines like "XL           44       26     18     0      59.1%"
            for line in output.split("\n"):
                if line.strip().startswith(system):
                    parts = line.split()
                    if len(parts) >= 6:
                        try:
                            metrics[f"{system.lower()}_graded"] = int(parts[1])
                            metrics[f"{system.lower()}_wins"] = int(parts[2])
                            metrics[f"{system.lower()}_losses"] = int(parts[3])
                            metrics[f"{system.lower()}_wr"] = float(parts[5].replace("%", ""))
                        except (ValueError, IndexError):
                            pass

        print(output)
        return metrics

    @task(
        task_id="validate_rolling_7d",
        doc_md="""
        ### Validate Rolling 7-Day Performance

        Calculates rolling 7-day win rates for trend analysis.
        """,
    )
    def validate_rolling_7d() -> dict[str, Any]:
        """Validate rolling 7-day performance."""
        import subprocess

        end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

        script_path = f"{SCRIPT_DIR}/betting_xl/validate_predictions.py"

        result = subprocess.run(
            [
                "python3",
                script_path,
                "--start-date",
                start_date,
                "--end-date",
                end_date,
            ],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            env={**os.environ, "PYTHONPATH": PROJECT_ROOT},
            timeout=300,
        )

        output = result.stdout
        metrics = {
            "period": "7d",
            "start_date": start_date,
            "end_date": end_date,
            "raw_output": output,
        }

        # Extract TOTAL line
        for line in output.split("\n"):
            if line.strip().startswith("TOTAL"):
                parts = line.split()
                if len(parts) >= 6:
                    try:
                        metrics["total_graded"] = int(parts[1])
                        metrics["total_wins"] = int(parts[2])
                        metrics["total_losses"] = int(parts[3])
                        metrics["total_wr"] = float(parts[5].replace("%", ""))
                        metrics["total_roi"] = float(parts[6].replace("%", "").replace("+", ""))
                    except (ValueError, IndexError):
                        pass

        # Extract by system
        for system in ["XL", "PRO", "ODDS_API"]:
            for line in output.split("\n"):
                if line.strip().startswith(system) and "SYSTEM" not in line:
                    parts = line.split()
                    if len(parts) >= 6:
                        try:
                            metrics[f"{system.lower()}_wr"] = float(parts[5].replace("%", ""))
                            metrics[f"{system.lower()}_roi"] = float(
                                parts[6].replace("%", "").replace("+", "")
                            )
                        except (ValueError, IndexError):
                            pass

        wr = metrics.get("total_wr", 0)
        roi = metrics.get("total_roi", 0)
        print(f"7-Day Rolling: {wr:.1f}% WR, {roi:+.1f}% ROI")
        return metrics

    @task(
        task_id="validate_rolling_30d",
        doc_md="""
        ### Validate Rolling 30-Day Performance

        Calculates rolling 30-day win rates for long-term trend analysis.
        """,
    )
    def validate_rolling_30d() -> dict[str, Any]:
        """Validate rolling 30-day performance."""
        import subprocess

        end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        script_path = f"{SCRIPT_DIR}/betting_xl/validate_predictions.py"

        result = subprocess.run(
            [
                "python3",
                script_path,
                "--start-date",
                start_date,
                "--end-date",
                end_date,
            ],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            env={**os.environ, "PYTHONPATH": PROJECT_ROOT},
            timeout=300,
        )

        output = result.stdout
        metrics = {
            "period": "30d",
            "start_date": start_date,
            "end_date": end_date,
        }

        # Extract TOTAL line
        for line in output.split("\n"):
            if line.strip().startswith("TOTAL"):
                parts = line.split()
                if len(parts) >= 6:
                    try:
                        metrics["total_graded"] = int(parts[1])
                        metrics["total_wins"] = int(parts[2])
                        metrics["total_losses"] = int(parts[3])
                        metrics["total_wr"] = float(parts[5].replace("%", ""))
                        metrics["total_roi"] = float(parts[6].replace("%", "").replace("+", ""))
                    except (ValueError, IndexError):
                        pass

        # Extract by system
        for system in ["XL", "PRO", "ODDS_API"]:
            for line in output.split("\n"):
                if line.strip().startswith(system) and "SYSTEM" not in line:
                    parts = line.split()
                    if len(parts) >= 6:
                        try:
                            metrics[f"{system.lower()}_wr"] = float(parts[5].replace("%", ""))
                            metrics[f"{system.lower()}_roi"] = float(
                                parts[6].replace("%", "").replace("+", "")
                            )
                        except (ValueError, IndexError):
                            pass

        wr = metrics.get("total_wr", 0)
        roi = metrics.get("total_roi", 0)
        print(f"30-Day Rolling: {wr:.1f}% WR, {roi:+.1f}% ROI")
        return metrics

    @task(
        task_id="check_performance_alerts",
        doc_md="""
        ### Check Performance Alerts

        Monitors performance metrics and sends alerts if thresholds are breached:
        - Win rate below minimum (default 52%)
        - ROI below minimum (default -5%)
        - Individual system degradation
        """,
    )
    def check_performance_alerts(
        yesterday: dict[str, Any],
        rolling_7d: dict[str, Any],
        rolling_30d: dict[str, Any],
    ) -> dict[str, Any]:
        """Check performance thresholds and send alerts."""
        alerts = []

        # Check 7-day rolling performance
        wr_7d = rolling_7d.get("total_wr", 0)
        roi_7d = rolling_7d.get("total_roi", 0)

        if wr_7d > 0 and wr_7d < MIN_WIN_RATE:
            alerts.append(f"7-Day Win Rate ({wr_7d:.1f}%) below threshold ({MIN_WIN_RATE}%)")

        if roi_7d < MIN_ROI:
            alerts.append(f"7-Day ROI ({roi_7d:+.1f}%) below threshold ({MIN_ROI:+.1f}%)")

        # Check individual system performance (7-day)
        for system in ["xl", "pro", "odds_api"]:
            sys_wr = rolling_7d.get(f"{system}_wr", 0)
            sys_roi = rolling_7d.get(f"{system}_roi", 0)

            if sys_wr > 0 and sys_wr < 50.0:
                alerts.append(f"{system.upper()} 7-Day Win Rate ({sys_wr:.1f}%) below 50%")

            if sys_roi < -20.0:
                alerts.append(f"{system.upper()} 7-Day ROI ({sys_roi:+.1f}%) severely negative")

        # Check 30-day trends
        wr_30d = rolling_30d.get("total_wr", 0)
        if wr_30d > 0 and wr_30d < MIN_WIN_RATE:
            alerts.append(
                f"30-Day Win Rate ({wr_30d:.1f}%) below threshold - consider model review"
            )

        # Send alert if any issues
        if alerts:
            body = f"""
            <h3>NBA Pick Performance Alert</h3>
            <p><b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>

            <h4>Alerts:</h4>
            <ul>
                {"".join(f"<li>{a}</li>" for a in alerts)}
            </ul>

            <h4>7-Day Performance:</h4>
            <table border="1" cellpadding="5">
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Total Win Rate</td><td>{wr_7d:.1f}%</td></tr>
                <tr><td>Total ROI</td><td>{roi_7d:+.1f}%</td></tr>
                <tr><td>XL Win Rate</td><td>{rolling_7d.get('xl_wr', 0):.1f}%</td></tr>
                <tr><td>PRO Win Rate</td><td>{rolling_7d.get('pro_wr', 0):.1f}%</td></tr>
                <tr><td>ODDS_API Win Rate</td><td>{rolling_7d.get('odds_api_wr', 0):.1f}%</td></tr>
            </table>

            <h4>30-Day Performance:</h4>
            <table border="1" cellpadding="5">
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Total Win Rate</td><td>{wr_30d:.1f}%</td></tr>
                <tr><td>Total ROI</td><td>{rolling_30d.get('total_roi', 0):+.1f}%</td></tr>
            </table>

            <p>Review performance and consider adjusting filters or thresholds.</p>
            """

            is_critical = wr_7d < 48.0 or roi_7d < -15.0
            send_performance_alert(
                "NBA Pick Performance Degradation",
                body,
                is_critical=is_critical,
            )

            print(f"[ALERT] {len(alerts)} performance alerts triggered")
            for alert in alerts:
                print(f"  - {alert}")
        else:
            print("[OK] All performance metrics within thresholds")

        return {
            "alerts_count": len(alerts),
            "alerts": alerts,
            "status": "alert" if alerts else "ok",
        }

    @task(
        task_id="save_validation_results",
        doc_md="""
        ### Save Validation Results

        Stores validation results to JSON for historical tracking.
        """,
    )
    def save_validation_results(
        yesterday: dict[str, Any],
        rolling_7d: dict[str, Any],
        rolling_30d: dict[str, Any],
        alerts: dict[str, Any],
    ) -> dict[str, Any]:
        """Save validation results to file."""
        Path(VALIDATION_DIR).mkdir(parents=True, exist_ok=True)

        date_str = datetime.now().strftime("%Y-%m-%d")

        results = {
            "validation_date": date_str,
            "generated_at": datetime.now().isoformat(),
            "yesterday": {
                "date": yesterday.get("date"),
                "xl": {
                    "graded": yesterday.get("xl_graded", 0),
                    "wins": yesterday.get("xl_wins", 0),
                    "losses": yesterday.get("xl_losses", 0),
                    "win_rate": yesterday.get("xl_wr", 0),
                },
                "pro": {
                    "graded": yesterday.get("pro_graded", 0),
                    "wins": yesterday.get("pro_wins", 0),
                    "losses": yesterday.get("pro_losses", 0),
                    "win_rate": yesterday.get("pro_wr", 0),
                },
                "odds_api": {
                    "graded": yesterday.get("odds_api_graded", 0),
                    "wins": yesterday.get("odds_api_wins", 0),
                    "losses": yesterday.get("odds_api_losses", 0),
                    "win_rate": yesterday.get("odds_api_wr", 0),
                },
            },
            "rolling_7d": {
                "period": rolling_7d.get("period"),
                "start_date": rolling_7d.get("start_date"),
                "end_date": rolling_7d.get("end_date"),
                "total_graded": rolling_7d.get("total_graded", 0),
                "total_wins": rolling_7d.get("total_wins", 0),
                "total_losses": rolling_7d.get("total_losses", 0),
                "total_win_rate": rolling_7d.get("total_wr", 0),
                "total_roi": rolling_7d.get("total_roi", 0),
                "xl_win_rate": rolling_7d.get("xl_wr", 0),
                "pro_win_rate": rolling_7d.get("pro_wr", 0),
                "odds_api_win_rate": rolling_7d.get("odds_api_wr", 0),
            },
            "rolling_30d": {
                "period": rolling_30d.get("period"),
                "start_date": rolling_30d.get("start_date"),
                "end_date": rolling_30d.get("end_date"),
                "total_graded": rolling_30d.get("total_graded", 0),
                "total_win_rate": rolling_30d.get("total_wr", 0),
                "total_roi": rolling_30d.get("total_roi", 0),
            },
            "alerts": alerts,
        }

        # Save daily result
        output_file = f"{VALIDATION_DIR}/validation_{date_str}.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"[OK] Saved validation results to {output_file}")

        return {"output_file": output_file, "status": "saved"}

    @task(
        task_id="generate_performance_report",
        doc_md="""
        ### Generate Performance Report

        Creates a summary report of pick performance.
        """,
    )
    def generate_performance_report(
        yesterday: dict[str, Any],
        rolling_7d: dict[str, Any],
        rolling_30d: dict[str, Any],
        saved: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate and print performance report."""
        print("\n" + "=" * 70)
        print("NBA PICK VALIDATION SUMMARY")
        print("=" * 70)

        print(f"\nValidation Date: {datetime.now().strftime('%Y-%m-%d')}")

        # Yesterday's results
        print(f"\n--- YESTERDAY ({yesterday.get('date', 'N/A')}) ---")
        for system in ["xl", "pro", "odds_api"]:
            graded = yesterday.get(f"{system}_graded", 0)
            wins = yesterday.get(f"{system}_wins", 0)
            losses = yesterday.get(f"{system}_losses", 0)
            wr = yesterday.get(f"{system}_wr", 0)
            if graded > 0:
                print(f"  {system.upper():10s}: {wins}W-{losses}L ({wr:.1f}%)")

        # 7-day rolling
        print("\n--- 7-DAY ROLLING ---")
        print(
            f"  Total: {rolling_7d.get('total_wins', 0)}W-{rolling_7d.get('total_losses', 0)}L "
            f"({rolling_7d.get('total_wr', 0):.1f}%) | ROI: {rolling_7d.get('total_roi', 0):+.1f}%"
        )
        for system in ["xl", "pro", "odds_api"]:
            wr = rolling_7d.get(f"{system}_wr", 0)
            roi = rolling_7d.get(f"{system}_roi", 0)
            if wr > 0:
                print(f"  {system.upper():10s}: {wr:.1f}% WR | {roi:+.1f}% ROI")

        # 30-day rolling
        print("\n--- 30-DAY ROLLING ---")
        print(
            f"  Total: {rolling_30d.get('total_graded', 0)} picks | "
            f"{rolling_30d.get('total_wr', 0):.1f}% WR | "
            f"{rolling_30d.get('total_roi', 0):+.1f}% ROI"
        )

        print("\n" + "=" * 70)

        return {
            "status": "completed",
            "results_file": saved.get("output_file"),
        }

    # ========================================================================
    # Task Dependencies
    # ========================================================================

    # Run validations in parallel
    yesterday_result = validate_yesterday()
    rolling_7d_result = validate_rolling_7d()
    rolling_30d_result = validate_rolling_30d()

    # Check alerts after all validations complete
    alerts_result = check_performance_alerts(
        yesterday_result,
        rolling_7d_result,
        rolling_30d_result,
    )

    # Save results
    saved_result = save_validation_results(
        yesterday_result,
        rolling_7d_result,
        rolling_30d_result,
        alerts_result,
    )

    # Generate final report (terminal task)
    generate_performance_report(
        yesterday_result,
        rolling_7d_result,
        rolling_30d_result,
        saved_result,
    )


# Instantiate the DAG
dag = nba_validation_pipeline()
