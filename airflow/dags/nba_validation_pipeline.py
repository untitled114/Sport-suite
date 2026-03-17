"""
NBA Validation Pipeline DAG

Scheduled: Daily at 3:30 AM EST (AFTER full pipeline at 2:30 AM loads game results)
Purpose: Grade yesterday's picks, validate performance, send report card

Flow: Full pipeline (2:30) → loads game logs → Validation (3:30) → grade + report
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from callbacks import on_failure, on_retry, on_success

from airflow.decorators import dag, task
from airflow.models import Variable
from airflow.timetables.trigger import CronTriggerTimetable

# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Variable.get("nba_project_root", default_var="/home/untitled/Sport-suite")
VALIDATION_DIR = f"{PROJECT_ROOT}/nba/betting_xl/validation_results"
_EST = ZoneInfo("America/New_York")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

MIN_WIN_RATE = float(Variable.get("nba_min_win_rate", default_var="52.0"))
MIN_ROI = float(Variable.get("nba_min_roi", default_var="-5.0"))

default_args = {
    "owner": "nba_pipeline",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(minutes=15),
    "on_success_callback": on_success,
    "on_failure_callback": on_failure,
    "on_retry_callback": on_retry,
}


def _wr(w, losses):
    t = w + losses
    return f"{w}W-{losses}L ({w / t * 100:.0f}%)" if t else "—"


def _bucket_line(name, s):
    w, losses = s.get("w", 0), s.get("l", 0)
    t = w + losses
    if t == 0:
        return None
    roi = s.get("profit", 0) / t * 100
    return f"{name}: {_wr(w, losses)} {roi:+.0f}% ROI"


def _build_validation_embed(perf: dict, alert_list: list) -> dict:
    """Build rich Discord embed from validation performance data."""
    r7 = perf.get("rolling_7d", {})
    r30 = perf.get("rolling_30d", {})
    y = perf.get("yesterday", {})

    # Header
    t7 = r7.get("total", 0)
    if t7:
        title = f"Validation — {r7['wins']}W-{r7['losses']}L ({r7['win_rate']:.0f}% WR, {r7['roi']:+.1f}% ROI)"
    else:
        title = "Validation Report"

    # Color based on 7-day WR
    wr7 = r7.get("win_rate", 0)
    color = 0x2ECC71 if wr7 >= 55 else (0xE67E22 if wr7 >= 50 else 0xE74C3C)

    fields = []

    # Yesterday
    yt = y.get("total", 0)
    if yt:
        details = y.get("details", [])
        lines = []
        for p in sorted(details, key=lambda x: x.get("outcome", "")):
            icon = "W" if p["outcome"] == "WIN" else "L"
            conv = f"{p['conviction']:.0%}" if p.get("conviction") else "—"
            lines.append(
                f"`{icon}` {p['player'][:18]} {p['market'][:3]} O{p['line']} → {p['actual']} [{conv}]"
            )
        fields.append(
            {
                "name": f"Yesterday ({y['date']}): {_wr(y['wins'], yt - y['wins'])}",
                "value": "\n".join(lines[:10]) or "—",
                "inline": False,
            }
        )
    else:
        fields.append({"name": "Yesterday", "value": "No graded picks", "inline": False})

    # By Model (7d)
    by_model = r7.get("by_model", {})
    if by_model:
        lines = [
            line
            for line in (_bucket_line(m.upper(), s) for m, s in sorted(by_model.items()))
            if line
        ]
        if lines:
            fields.append({"name": "By Model (7d)", "value": "\n".join(lines), "inline": True})

    # By Market (7d)
    by_market = r7.get("by_market", {})
    if by_market:
        lines = [
            line for line in (_bucket_line(m, s) for m, s in sorted(by_market.items())) if line
        ]
        if lines:
            fields.append({"name": "By Market (7d)", "value": "\n".join(lines), "inline": True})

    # By Conviction (7d)
    by_conv = r7.get("by_conviction_band", {})
    if by_conv:
        lines = [line for line in (_bucket_line(b, s) for b, s in sorted(by_conv.items())) if line]
        if lines:
            fields.append(
                {"name": "By Conviction (7d)", "value": "\n".join(lines), "inline": False}
            )

    # By BP Signal (7d)
    by_bp = r7.get("by_bp_signal", {})
    if by_bp:
        lines = [
            line
            for line in (
                _bucket_line(b, s)
                for b, s in sorted(by_bp.items())
                if (s.get("w", 0) + s.get("l", 0)) >= 2
            )
            if line
        ]
        if lines:
            fields.append({"name": "By BP Signal (7d)", "value": "\n".join(lines), "inline": False})

    # 30-day summary
    t30 = r30.get("total", 0)
    if t30:
        fields.append(
            {
                "name": "30-Day",
                "value": f"{r30['wins']}W-{r30['losses']}L ({r30['win_rate']:.0f}% WR, {r30['roi']:+.1f}% ROI)",
                "inline": False,
            }
        )

    # Alerts
    if alert_list:
        fields.append(
            {
                "name": "Alerts",
                "value": "\n".join(f"- {a}" for a in alert_list[:5]),
                "inline": False,
            }
        )

    return {
        "title": title,
        "color": color,
        "fields": fields,
        "footer": {"text": datetime.now(_EST).strftime("%Y-%m-%d %I:%M %p EST")},
    }


@dag(
    dag_id="nba_validation_pipeline",
    description="NBA pick performance validation and tracking",
    schedule=CronTriggerTimetable("30 3 * * *", timezone="America/New_York"),
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["nba", "validation", "performance"],
    default_args=default_args,
    max_active_runs=1,
    doc_md=__doc__,
)
def nba_validation_pipeline():

    @task(task_id="backfill_actuals")
    def backfill_actuals() -> dict[str, Any]:
        """Grade picks from last 3 days against player_game_logs.

        Covers yesterday + 2 prior days to catch any missed grading.
        """
        from nba.core.axiom_writer import write_actuals

        now = datetime.now(_EST)
        total = 0
        results = {}

        for days_ago in range(1, 4):
            d = (now - timedelta(days=days_ago)).strftime("%Y-%m-%d")
            updated = write_actuals(d)
            results[d] = updated
            total += updated
            if updated:
                print(f"  {d}: graded {updated} picks")

        print(f"[axiom] Backfill complete: {total} picks graded")
        return {"total_graded": total, "by_date": results}

    @task(task_id="validate_performance")
    def validate_performance(backfill: dict[str, Any]) -> dict[str, Any]:
        """Run DB-based validation for 1d, 7d, and 30d windows."""
        from nba.betting_xl.validate_from_db import run_validation

        now = datetime.now(_EST)
        yesterday = (now - timedelta(days=1)).strftime("%Y-%m-%d")

        # Yesterday
        print("\n=== YESTERDAY ===")
        r_1d = run_validation(yesterday, yesterday)

        # 7-day rolling
        start_7d = (now - timedelta(days=7)).strftime("%Y-%m-%d")
        print("\n=== 7-DAY ROLLING ===")
        r_7d = run_validation(start_7d, yesterday)

        # 30-day rolling
        start_30d = (now - timedelta(days=30)).strftime("%Y-%m-%d")
        print("\n=== 30-DAY ROLLING ===")
        r_30d = run_validation(start_30d, yesterday)

        def _extract(r):
            return {
                "total": r.get("total", 0),
                "wins": r.get("wins", 0),
                "losses": r.get("losses", 0),
                "win_rate": r.get("win_rate", 0),
                "roi": r.get("roi", 0),
                "profit": r.get("profit", 0),
                "by_model": r.get("by_model", {}),
                "by_market": r.get("by_market", {}),
                "by_conviction_band": r.get("by_conviction_band", {}),
                "by_bp_signal": r.get("by_bp_signal", {}),
                "details": r.get("details", []),
            }

        return {
            "yesterday": {"date": yesterday, **_extract(r_1d)},
            "rolling_7d": {"start": start_7d, "end": yesterday, **_extract(r_7d)},
            "rolling_30d": {"start": start_30d, "end": yesterday, **_extract(r_30d)},
        }

    @task(task_id="check_alerts")
    def check_alerts(perf: dict[str, Any]) -> dict[str, Any]:
        """Check performance thresholds and log alerts."""
        alerts = []

        wr_7d = perf.get("rolling_7d", {}).get("win_rate", 0)
        roi_7d = perf.get("rolling_7d", {}).get("roi", 0)
        wr_30d = perf.get("rolling_30d", {}).get("win_rate", 0)
        total_7d = perf.get("rolling_7d", {}).get("total", 0)

        if total_7d >= 5 and wr_7d < MIN_WIN_RATE:
            alerts.append(f"7-Day Win Rate ({wr_7d:.1f}%) below threshold ({MIN_WIN_RATE}%)")

        if total_7d >= 5 and roi_7d < MIN_ROI:
            alerts.append(f"7-Day ROI ({roi_7d:+.1f}%) below threshold ({MIN_ROI:+.1f}%)")

        total_30d = perf.get("rolling_30d", {}).get("total", 0)
        if total_30d >= 10 and wr_30d < MIN_WIN_RATE:
            alerts.append(
                f"30-Day Win Rate ({wr_30d:.1f}%) below threshold — consider model review"
            )

        # Check model-specific performance
        by_model = perf.get("rolling_7d", {}).get("by_model", {})
        for model_name, stats in by_model.items():
            t = stats.get("w", 0) + stats.get("l", 0)
            if t >= 3:
                model_wr = stats["w"] / t * 100
                if model_wr < 45.0:
                    alerts.append(
                        f"Model {model_name} 7-Day Win Rate ({model_wr:.1f}%) critically low"
                    )

        if alerts:
            print(f"[ALERT] {len(alerts)} performance alerts:")
            for a in alerts:
                print(f"  - {a}")
        else:
            print("[OK] All performance metrics within thresholds")

        return {"alerts": alerts, "status": "alert" if alerts else "ok"}

    @task(task_id="save_results")
    def save_results(
        perf: dict[str, Any],
        alerts: dict[str, Any],
    ) -> dict[str, Any]:
        """Save validation results to JSON."""
        Path(VALIDATION_DIR).mkdir(parents=True, exist_ok=True)

        date_str = datetime.now(_EST).strftime("%Y-%m-%d")
        results = {
            "validation_date": date_str,
            "generated_at": datetime.now(_EST).isoformat(),
            "performance": perf,
            "alerts": alerts,
        }

        output_file = f"{VALIDATION_DIR}/validation_{date_str}.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"[OK] Saved to {output_file}")
        return {"output_file": output_file}

    @task(task_id="send_validation_card")
    def send_validation_card(perf: dict[str, Any], alerts: dict[str, Any]) -> dict[str, Any]:
        """Send validation summary to Discord via Axiom DM."""
        from nba.core.daily_card import _post_dm_embed

        if not perf:
            print("[SKIP] No performance data")
            return {"sent": False, "reason": "no_data"}

        embed = _build_validation_embed(perf, (alerts or {}).get("alerts", []))
        msg_id = _post_dm_embed(embed)
        if msg_id:
            print(f"[OK] Validation card sent (msg={msg_id})")
            return {"sent": True, "message_id": msg_id}
        else:
            print("[WARN] Validation card send failed")
            return {"sent": False, "reason": "discord_error"}

    # ── Task Dependencies ──────────────────────────────────────────
    backfill = backfill_actuals()
    perf = validate_performance(backfill)
    alerts = check_alerts(perf)
    save_results(perf, alerts)
    send_validation_card(perf, alerts)


dag = nba_validation_pipeline()
