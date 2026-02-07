"""
Airflow DAG Callbacks - Notifications
=====================================
Sends notifications on DAG/task events.

Desktop notifications (notify-send) for local development.
Logging fallback for headless servers.
"""

import logging
import os
import shutil
import subprocess

logger = logging.getLogger("airflow.callbacks")


def _notify(title: str, message: str, urgency: str = "normal"):
    """Send notification via desktop or logging fallback.

    On desktop: Uses notify-send for visual notifications.
    On server: Logs to Airflow logger (visible in task logs).
    """
    # Check if we have a display and notify-send available
    has_display = os.environ.get("DISPLAY") is not None
    has_notify_send = shutil.which("notify-send") is not None

    if has_display and has_notify_send:
        try:
            subprocess.run(
                [
                    "notify-send",
                    "-u",
                    urgency,
                    "-t",
                    "10000",  # 10 seconds
                    "-a",
                    "Airflow",
                    title,
                    message,
                ],
                check=False,
                capture_output=True,
            )
            return
        except OSError as e:
            logger.debug(f"Desktop notification unavailable: {e}")
            # Fall through to logging

    # Fallback: log the notification
    log_level = logging.WARNING if urgency == "critical" else logging.INFO
    logger.log(log_level, f"{title}: {message}")


def on_success(context):
    """Called when DAG/task succeeds."""
    dag_id = context.get("dag").dag_id
    task_id = context.get("task_instance").task_id if context.get("task_instance") else "DAG"
    _notify(
        f"✅ {dag_id}",
        f"Task '{task_id}' completed successfully",
        "normal",
    )


def on_failure(context):
    """Called when DAG/task fails."""
    dag_id = context.get("dag").dag_id
    task_id = context.get("task_instance").task_id if context.get("task_instance") else "DAG"
    exception = context.get("exception", "Unknown error")
    _notify(
        f"❌ {dag_id} FAILED",
        f"Task '{task_id}' failed: {str(exception)[:100]}",
        "critical",
    )


def on_retry(context):
    """Called when task retries."""
    dag_id = context.get("dag").dag_id
    task_id = context.get("task_instance").task_id
    _notify(
        f"🔄 {dag_id}",
        f"Task '{task_id}' is retrying...",
        "normal",
    )


def dag_success(context):
    """Called when entire DAG succeeds."""
    dag_id = context.get("dag").dag_id
    _notify(
        f"🎉 {dag_id} Complete",
        "All tasks finished successfully!",
        "normal",
    )


def dag_failure(context):
    """Called when DAG fails."""
    dag_id = context.get("dag").dag_id
    _notify(
        f"💥 {dag_id} FAILED",
        "DAG execution failed - check Airflow UI",
        "critical",
    )
