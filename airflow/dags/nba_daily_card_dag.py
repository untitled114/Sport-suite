"""
NBA Daily Card DAG

Scheduled: Every 30 minutes, 8 AM–10:30 PM EST
Purpose: Send Cephalon Axiom's autonomous conviction card at T-2hr before first tip

Polls every 30 minutes throughout the day. Each run fetches today's first game time
from ESPN and sends the card only when the current time is within 120 minutes of
tip-off. Covers early noon games (late-season push) as well as evening slates.
The dedup check in send_daily_card() ensures it sends exactly once per day regardless
of how many polls land in the window.

Card always sends — if nothing meets threshold, a "nothing tonight" card goes out.

Author: Claude Code
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from callbacks import on_failure, on_retry, on_success

from airflow.decorators import dag, task
from airflow.models import Variable
from airflow.timetables.trigger import CronTriggerTimetable

PROJECT_ROOT = Variable.get("nba_project_root", default_var="/home/untitled/Sport-suite")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

_EST = ZoneInfo("America/New_York")

default_args = {
    "owner": "nba_pipeline",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(minutes=5),
    "on_success_callback": on_success,
    "on_failure_callback": on_failure,
    "on_retry_callback": on_retry,
}

# Send window: T-120min to T-5min before first tip
_SEND_WINDOW_EARLY_MIN = 120
_SEND_WINDOW_LATE_MIN = 5


def _parse_tip_datetime(tip_str: str, run_date: str) -> datetime:
    """Parse '7:30 PM ET' + run_date into a timezone-aware EST datetime."""
    try:
        naive = datetime.strptime(f"{run_date} {tip_str.replace(' ET', '')}", "%Y-%m-%d %I:%M %p")
        return naive.replace(tzinfo=_EST)
    except Exception:
        # Fallback: 7 PM same day
        return datetime.strptime(run_date, "%Y-%m-%d").replace(hour=19, tzinfo=_EST)


@dag(
    dag_id="nba_daily_card",
    description="Axiom conviction card — polls every 30 min 8AM-10:30PM, sends at T-2hr before first tip",
    schedule=CronTriggerTimetable("*/30 8-22 * * *", timezone="America/New_York"),
    start_date=datetime(2025, 11, 7),
    catchup=False,
    tags=["nba", "axiom", "discord", "daily-card"],
    default_args=default_args,
    max_active_runs=1,
    doc_md=__doc__,
)
def nba_daily_card():
    """
    Axiom Daily Card — polls every 30 min from 8 AM to 10:30 PM EST.

    Each run fetches first tip time from ESPN and sends only when within the
    T-120min to T-5min window. Covers noon games and evening slates.
    Dedup prevents double-sends.
    """

    @task(task_id="send_daily_card")
    def send_daily_card_task() -> dict[str, Any]:
        from nba.core.daily_card import get_first_tip_time, send_daily_card

        now = datetime.now(_EST)
        run_date = now.strftime("%Y-%m-%d")

        tip_str = get_first_tip_time(run_date)
        tip_dt = _parse_tip_datetime(tip_str, run_date)
        minutes_to_tip = (tip_dt - now).total_seconds() / 60

        if minutes_to_tip > _SEND_WINDOW_EARLY_MIN or minutes_to_tip < -_SEND_WINDOW_LATE_MIN:
            print(
                f"[SKIP] {minutes_to_tip:.0f}m to first tip ({tip_str}) — "
                f"outside send window (T-{_SEND_WINDOW_EARLY_MIN}m to T+{_SEND_WINDOW_LATE_MIN}m)"
            )
            return {"sent": False, "reason": "outside_window", "minutes_to_tip": minutes_to_tip}

        print(
            f"[SEND] {minutes_to_tip:.0f}m to first tip ({tip_str}) — within window, sending card"
        )
        result = send_daily_card(run_date, trigger="airflow")

        # already_sent is fine on retry — not a failure
        if not result["sent"] and result.get("reason") not in ("already_sent", "outside_window"):
            raise Exception(f"Daily card failed: {result.get('reason')}")

        return result

    send_daily_card_task()


dag = nba_daily_card()
