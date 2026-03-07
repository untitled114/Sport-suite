"""
Discord DM notification utility for Airflow.
Sends alerts to the owner via Cephalon Axiom bot.

Uses Discord REST API directly — no discord.py, no event loop conflicts.
Token read from AXIOM_BOT_TOKEN env var (set in cephalon-axiom.service).
"""

import json
import logging
import os
import urllib.error
import urllib.request
from datetime import datetime
from zoneinfo import ZoneInfo

logger = logging.getLogger("airflow.discord_notify")

_EST = ZoneInfo("America/New_York")
DISCORD_API = "https://discord.com/api/v10"


def _get_token() -> str | None:
    token = os.environ.get("AXIOM_BOT_TOKEN")
    if not token:
        # Try reading from .env file on server
        env_path = os.path.join(os.environ.get("HOME", "/home/sportsuite"), "sport-suite", ".env")
        try:
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("AXIOM_BOT_TOKEN="):
                        token = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break
        except OSError:
            pass
    return token


def send_dm(message: str) -> bool:
    """
    Send a direct message to the owner via Axiom.
    Returns True on success, False on failure (never raises).
    """
    token = _get_token()
    if not token:
        logger.warning("AXIOM_BOT_TOKEN not available — Discord alert skipped")
        return False

    headers = {
        "Authorization": f"Bot {token}",
        "Content-Type": "application/json",
        "User-Agent": "CephalonAxiom/1.0",
    }

    try:
        # Open DM channel
        req = urllib.request.Request(
            f"{DISCORD_API}/users/@me/channels",
            data=json.dumps({"recipient_id": os.environ.get("DISCORD_OWNER_ID", "")}).encode(),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            channel_id = json.loads(resp.read())["id"]

        # Send message (cap at 2000 chars)
        req = urllib.request.Request(
            f"{DISCORD_API}/channels/{channel_id}/messages",
            data=json.dumps({"content": message[:2000]}).encode(),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read()

        logger.info("Discord DM sent to owner")
        return True

    except urllib.error.HTTPError as e:
        logger.error(f"Discord DM HTTP error {e.code}: {e.read().decode()[:200]}")
        return False
    except Exception as e:
        logger.error(f"Discord DM failed: {e}")
        return False


def alert_pipeline_failure(dag_id: str, task_id: str, error: str, run_id: str = "") -> bool:
    """Format and send a pipeline failure alert."""
    ts = datetime.now(_EST).strftime("%Y-%m-%d %H:%M EST")
    msg = (
        f"🚨 **Pipeline Failure**\n"
        f"**DAG:** `{dag_id}`\n"
        f"**Task:** `{task_id}`\n"
        f"**Time:** {ts}\n"
        f"**Error:** {str(error)[:300]}"
    )
    if run_id:
        msg += f"\n**Run:** `{run_id}`"
    return send_dm(msg)


def alert_data_stale(check_name: str, detail: str) -> bool:
    """Format and send a data staleness alert."""
    ts = datetime.now(_EST).strftime("%Y-%m-%d %H:%M EST")
    msg = (
        f"⚠️ **Stale Data Detected**\n"
        f"**Check:** {check_name}\n"
        f"**Detail:** {detail}\n"
        f"**Time:** {ts}"
    )
    return send_dm(msg)


def alert_health_warning(status: str, warnings: list[str]) -> bool:
    """Format and send a health check warning."""
    ts = datetime.now(_EST).strftime("%Y-%m-%d %H:%M EST")
    warn_lines = "\n".join(f"• {w}" for w in warnings[:10])
    msg = (
        f"{'🚨' if status == 'critical' else '⚠️'} **Health Check: {status.upper()}**\n"
        f"{warn_lines}\n"
        f"**Time:** {ts}"
    )
    return send_dm(msg)
