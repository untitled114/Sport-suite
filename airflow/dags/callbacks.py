"""
Airflow DAG Callbacks - Desktop Notifications
==============================================
Sends desktop notifications on DAG/task events.
"""

import subprocess


def _notify(title: str, message: str, urgency: str = "normal"):
    """Send desktop notification."""
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
            env={"DISPLAY": ":0", "DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/user/1000/bus"},
        )
    except Exception:
        pass  # Don't fail DAG if notification fails


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
