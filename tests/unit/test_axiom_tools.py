"""
Unit tests for cephalon.axiom_tools — new action tool handlers.

Skipped automatically in CI where the cephalon module is not available.
"""

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Skip entire file if cephalon isn't on the path (CI environment)
pytest.importorskip("cephalon.axiom_tools")

from cephalon.axiom_tools import (  # noqa: E402
    _check_admin,
    _handle_get_picks,
    _handle_pipeline_status,
    _handle_run_full_pipeline,
    _handle_run_refresh,
    _sport_suite_dir,
    handle_tool,
)

# ─────────────────────────────────────────────────────────────────
# _check_admin
# ─────────────────────────────────────────────────────────────────


class TestCheckAdmin:
    def test_no_admin_ids_allows_all(self):
        with patch("cephalon.axiom_tools._ADMIN_IDS", set()):
            assert _check_admin(999) is True

    def test_known_admin_allowed(self):
        with patch("cephalon.axiom_tools._ADMIN_IDS", {111, 222}):
            assert _check_admin(111) is True

    def test_unknown_user_blocked(self):
        with patch("cephalon.axiom_tools._ADMIN_IDS", {111, 222}):
            assert _check_admin(999) is False


# ─────────────────────────────────────────────────────────────────
# handle_tool dispatch
# ─────────────────────────────────────────────────────────────────


class TestHandleTool:
    def test_unknown_tool_returns_error(self):
        result = handle_tool("does_not_exist", {}, user_id=0)
        assert "Unknown tool" in result

    def test_action_tool_unauthorized(self):
        with patch("cephalon.axiom_tools._ADMIN_IDS", {999}):
            result = handle_tool("run_nba_refresh", {}, user_id=1)
        assert "Unauthorized" in result

    def test_action_tool_dispatches_to_handler(self):
        with (
            patch("cephalon.axiom_tools._ADMIN_IDS", {42}),
            patch(
                "cephalon.axiom_tools._handle_run_refresh",
                return_value="started PID 1234",
            ) as mock_handler,
        ):
            result = handle_tool("run_nba_refresh", {}, user_id=42)
        mock_handler.assert_called_once_with({}, 42)
        assert result == "started PID 1234"

    def test_db_tool_dispatches_without_admin(self):
        with patch("cephalon.axiom_tools._handle_get_picks", return_value="picks") as mock_handler:
            result = handle_tool("get_current_picks", {"date": "2026-03-07"}, user_id=0)
        mock_handler.assert_called_once_with({"date": "2026-03-07"})
        assert result == "picks"


# ─────────────────────────────────────────────────────────────────
# _handle_get_picks
# ─────────────────────────────────────────────────────────────────


class TestHandleGetPicks:
    def _make_xl_data(self):
        return {
            "picks": [
                {
                    "player_name": "Anthony Edwards",
                    "stat_type": "POINTS",
                    "best_line": 30.5,
                    "best_book": "prizepicks_goblin",
                    "edge": 5.4,
                    "filter_tier": "GOLDMINE",
                    "model_version": "xl",
                    "opponent_team": "ORL",
                }
            ]
        }

    def test_returns_picks_from_file(self, tmp_path):
        picks_file = tmp_path / "xl_picks_2026-03-07.json"
        picks_file.write_text(json.dumps(self._make_xl_data()))

        with patch("cephalon.axiom_tools._PREDICTIONS_DIRS", [str(tmp_path)]):
            result = _handle_get_picks({"date": "2026-03-07"})

        assert "Anthony Edwards" in result
        assert "POINTS OVER 30.5" in result
        assert "XL/GOLDMINE" in result
        assert "+5.4%" in result

    def test_no_file_returns_not_found(self, tmp_path):
        with patch("cephalon.axiom_tools._PREDICTIONS_DIRS", [str(tmp_path)]):
            result = _handle_get_picks({"date": "2026-03-07"})
        assert "No picks found" in result

    def test_uses_today_by_default(self, tmp_path):
        from datetime import datetime
        from zoneinfo import ZoneInfo

        today = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
        picks_file = tmp_path / f"xl_picks_{today}.json"
        picks_file.write_text(json.dumps(self._make_xl_data()))

        with patch("cephalon.axiom_tools._PREDICTIONS_DIRS", [str(tmp_path)]):
            result = _handle_get_picks({})

        assert "Anthony Edwards" in result

    def test_pro_picks_shown_separately(self, tmp_path):
        xl_file = tmp_path / "xl_picks_2026-03-07.json"
        xl_file.write_text(json.dumps(self._make_xl_data()))
        pro_file = tmp_path / "pro_picks_2026-03-07.json"
        pro_file.write_text(
            json.dumps(
                {
                    "picks": [
                        {
                            "player_name": "Dosunmu",
                            "stat_type": "POINTS",
                            "line": 10.5,
                            "ev_pct": 30.6,
                            "confidence": "MEDIUM",
                        }
                    ]
                }
            )
        )

        with patch("cephalon.axiom_tools._PREDICTIONS_DIRS", [str(tmp_path)]):
            result = _handle_get_picks({"date": "2026-03-07"})

        assert "XL Model" in result
        assert "PRO Cheatsheet" in result
        assert "Dosunmu" in result


# ─────────────────────────────────────────────────────────────────
# _handle_pipeline_status
# ─────────────────────────────────────────────────────────────────


class TestHandlePipelineStatus:
    def _mock_rows(self):
        return [
            {
                "run_date": "2026-03-07",
                "run_number": 4,
                "run_type": "refresh",
                "status": "success",
                "picks_generated": 3,
                "xl_picks": 2,
                "v3_picks": 1,
                "duration_seconds": 154,
                "error_message": None,
                "run_timestamp": "2026-03-07T16:23:00",
            },
            {
                "run_date": "2026-03-07",
                "run_number": 1,
                "run_type": "refresh",
                "status": "success",
                "picks_generated": 3,
                "xl_picks": 1,
                "v3_picks": 2,
                "duration_seconds": 150,
                "error_message": None,
                "run_timestamp": "2026-03-07T07:01:00",
            },
        ]

    def test_formats_run_rows(self):
        with patch("cephalon.axiom_db.execute_query", return_value=self._mock_rows()):
            result = _handle_pipeline_status({})

        assert "2026-03-07" in result
        assert "Run 4" in result
        assert "success" in result
        assert "3 picks" in result
        assert "154s" in result

    def test_no_rows_returns_not_found(self):
        with patch("cephalon.axiom_db.execute_query", return_value=[]):
            result = _handle_pipeline_status({})
        assert "No pipeline runs" in result

    def test_error_message_shown(self):
        rows = self._mock_rows()
        rows[0]["status"] = "failed"
        rows[0]["error_message"] = "DB connection refused"
        rows[0]["picks_generated"] = None

        with patch("cephalon.axiom_db.execute_query", return_value=rows):
            result = _handle_pipeline_status({})

        assert "DB connection refused" in result
        assert "❌" in result

    def test_days_back_capped_at_7(self):
        with patch("cephalon.axiom_db.execute_query", return_value=[]) as mock_q:
            _handle_pipeline_status({"days_back": 999})
        # The query should use 7, not 999
        call_args = mock_q.call_args[0]
        assert "7 days" in call_args[2][0]


# ─────────────────────────────────────────────────────────────────
# _handle_run_refresh
# ─────────────────────────────────────────────────────────────────


class TestHandleRunRefresh:
    def test_launches_subprocess_and_returns_pid(self, tmp_path):
        venv = tmp_path / "venv" / "bin"
        venv.mkdir(parents=True)
        python = venv / "python3"
        python.touch()
        script_dir = tmp_path / "nba" / "betting_xl"
        script_dir.mkdir(parents=True)
        (script_dir / "quick_refresh.py").touch()

        mock_proc = MagicMock()
        mock_proc.pid = 9999

        with (
            patch("cephalon.axiom_tools._sport_suite_dir", return_value=str(tmp_path)),
            patch("subprocess.Popen", return_value=mock_proc) as mock_popen,
        ):
            result = _handle_run_refresh({}, user_id=42)

        assert "9999" in result
        assert "started" in result.lower()
        mock_popen.assert_called_once()

    def test_missing_project_dir(self):
        with patch("cephalon.axiom_tools._sport_suite_dir", return_value=None):
            result = _handle_run_refresh({}, user_id=42)
        assert "not found" in result.lower()

    def test_missing_venv(self, tmp_path):
        # Project dir exists but no venv
        with patch("cephalon.axiom_tools._sport_suite_dir", return_value=str(tmp_path)):
            result = _handle_run_refresh({}, user_id=42)
        assert "Venv not found" in result

    def test_missing_script(self, tmp_path):
        venv = tmp_path / "venv" / "bin"
        venv.mkdir(parents=True)
        (venv / "python3").touch()
        # Don't create quick_refresh.py

        with patch("cephalon.axiom_tools._sport_suite_dir", return_value=str(tmp_path)):
            result = _handle_run_refresh({}, user_id=42)
        assert "not found" in result.lower()

    def test_popen_failure_returns_error(self, tmp_path):
        venv = tmp_path / "venv" / "bin"
        venv.mkdir(parents=True)
        (venv / "python3").touch()
        script_dir = tmp_path / "nba" / "betting_xl"
        script_dir.mkdir(parents=True)
        (script_dir / "quick_refresh.py").touch()

        with (
            patch("cephalon.axiom_tools._sport_suite_dir", return_value=str(tmp_path)),
            patch("subprocess.Popen", side_effect=OSError("permission denied")),
        ):
            result = _handle_run_refresh({}, user_id=42)
        assert "Failed" in result


# ─────────────────────────────────────────────────────────────────
# _handle_run_full_pipeline
# ─────────────────────────────────────────────────────────────────


class TestHandleRunFullPipeline:
    def test_launches_subprocess_and_returns_pid(self, tmp_path):
        script = tmp_path / "nba" / "nba-predictions.sh"
        script.parent.mkdir(parents=True)
        script.touch()

        mock_proc = MagicMock()
        mock_proc.pid = 5555

        with (
            patch("cephalon.axiom_tools._sport_suite_dir", return_value=str(tmp_path)),
            patch("subprocess.Popen", return_value=mock_proc) as mock_popen,
        ):
            result = _handle_run_full_pipeline({}, user_id=42)

        assert "5555" in result
        mock_popen.assert_called_once()
        args = mock_popen.call_args[0][0]
        assert args[0] == "bash"

    def test_missing_project_dir(self):
        with patch("cephalon.axiom_tools._sport_suite_dir", return_value=None):
            result = _handle_run_full_pipeline({}, user_id=42)
        assert "not found" in result.lower()

    def test_missing_script(self, tmp_path):
        with patch("cephalon.axiom_tools._sport_suite_dir", return_value=str(tmp_path)):
            result = _handle_run_full_pipeline({}, user_id=42)
        assert "not found" in result.lower()
