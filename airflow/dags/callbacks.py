"""
Airflow DAG Callbacks - Notifications
=====================================
Sends notifications on DAG/task events.

Discord DMs via Cephalon Axiom (primary).
Desktop notify-send (local dev fallback).
Logging fallback for headless servers without Axiom token.
"""

import logging
import os
import shutil
import subprocess

logger = logging.getLogger("airflow.callbacks")


def _discord_failure(dag_id: str, task_id: str, error: str, run_id: str = "") -> None:
    """Send Discord DM for failures. Import is local to avoid DAG parse errors."""
    try:
        from discord_notify import alert_pipeline_failure

        alert_pipeline_failure(dag_id, task_id, error, run_id)
    except Exception as e:
        logger.warning(f"Discord notification failed: {e}")


def _notify(title: str, message: str, urgency: str = "normal"):
    """Send notification via desktop or logging fallback."""
    has_display = os.environ.get("DISPLAY") is not None
    has_notify_send = shutil.which("notify-send") is not None

    if has_display and has_notify_send:
        try:
            subprocess.run(
                ["notify-send", "-u", urgency, "-t", "10000", "-a", "Airflow", title, message],
                check=False,
                capture_output=True,
            )
            return
        except OSError as e:
            logger.debug(f"Desktop notification unavailable: {e}")

    log_level = logging.WARNING if urgency == "critical" else logging.INFO
    logger.log(log_level, f"{title}: {message}")


def on_success(context):
    """Called when DAG/task succeeds."""
    dag_id = context.get("dag").dag_id
    task_id = context.get("task_instance").task_id if context.get("task_instance") else "DAG"
    _notify(f"✅ {dag_id}", f"Task '{task_id}' completed successfully", "normal")


def on_failure(context):
    """Called when DAG/task fails."""
    dag_id = context.get("dag").dag_id
    task_instance = context.get("task_instance")
    task_id = task_instance.task_id if task_instance else "DAG"
    run_id = task_instance.run_id if task_instance else ""
    exception = context.get("exception", "Unknown error")
    error_str = str(exception)

    _notify(f"❌ {dag_id} FAILED", f"Task '{task_id}' failed: {error_str[:100]}", "critical")
    _discord_failure(dag_id, task_id, error_str, run_id)


def on_retry(context):
    """Called when task retries."""
    dag_id = context.get("dag").dag_id
    task_instance = context.get("task_instance")
    task_id = task_instance.task_id if task_instance else "DAG"
    try_number = task_instance.try_number if task_instance else 0
    exception = context.get("exception", "Unknown error")

    _notify(f"🔄 {dag_id}", f"Task '{task_id}' is retrying (attempt {try_number})...", "normal")

    # Only DM on the last retry attempt to avoid noise
    max_tries = task_instance.max_tries if task_instance else 0
    if try_number >= max_tries:
        _discord_failure(dag_id, task_id, f"Final retry #{try_number}: {exception}")


def dag_failure(context):
    """Called when entire DAG fails. Sends Discord DM + desktop notification."""
    dag_id = context.get("dag").dag_id
    task_instance = context.get("task_instance")
    task_id = task_instance.task_id if task_instance else "DAG"
    run_id = task_instance.run_id if task_instance else ""
    exception = context.get("exception", "DAG execution failed")
    error_str = str(exception)

    _notify(f"💥 {dag_id} FAILED", f"Task '{task_id}': {error_str[:100]}", "critical")
    _discord_failure(dag_id, task_id, error_str, run_id)
