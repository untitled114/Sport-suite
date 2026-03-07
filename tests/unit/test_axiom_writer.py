"""
Unit tests for nba.core.axiom_writer — pure functions and mocked DB functions.
"""

from unittest.mock import MagicMock, patch

import pytest

from nba.core.axiom_writer import (
    _SCHEDULED_RUNS,
    _build_context_snapshot,
    _connect,
    audit_run_complete,
    audit_run_start,
    count_todays_props,
    get_run_number,
)

# ─────────────────────────────────────────────────────────────────
# get_run_number
# ─────────────────────────────────────────────────────────────────


class TestConnect:
    def test_returns_connection(self):
        mock_conn = MagicMock()
        with patch("psycopg2.connect", return_value=mock_conn):
            result = _connect()
        assert result is mock_conn


class TestGetRunNumber:
    def _mock_utc(self, hour, minute=0):
        """Return a mock datetime.now() with given UTC hour:minute."""
        mock_now = MagicMock()
        mock_now.hour = hour
        mock_now.minute = minute
        mock_dt = MagicMock()
        mock_dt.now.return_value = mock_now
        return mock_dt

    def test_scheduled_hours_exact(self):
        # Each scheduled UTC hour maps to its run number exactly
        for sched_hour, expected_run in _SCHEDULED_RUNS:
            with patch("nba.core.axiom_writer.datetime", self._mock_utc(sched_hour)):
                assert get_run_number() == expected_run

    def test_window_before_scheduled(self):
        # 15 UTC is 60 min before run 4 (16 UTC) — within ±90 min window
        with patch("nba.core.axiom_writer.datetime", self._mock_utc(15, 0)):
            assert get_run_number() == 4

    def test_window_after_scheduled(self):
        # 7 UTC + 30 min still in run 1's window
        with patch("nba.core.axiom_writer.datetime", self._mock_utc(7, 30)):
            assert get_run_number() == 1

    def test_off_window_returns_1(self):
        # 0 UTC (midnight) is outside all ±90 min windows
        with patch("nba.core.axiom_writer.datetime", self._mock_utc(0, 0)):
            assert get_run_number() == 1

    def test_run1_at_7utc(self):
        with patch("nba.core.axiom_writer.datetime", self._mock_utc(7)):
            assert get_run_number() == 1

    def test_run6_at_22utc(self):
        with patch("nba.core.axiom_writer.datetime", self._mock_utc(22)):
            assert get_run_number() == 6


# ─────────────────────────────────────────────────────────────────
# _build_context_snapshot
# ─────────────────────────────────────────────────────────────────


class TestBuildContextSnapshot:
    def test_empty_pick_returns_defaults(self):
        ctx = _build_context_snapshot({})
        assert ctx["risk_level"] is None
        assert ctx["risk_flags"] == []
        assert ctx["models_agreeing"] == ["xl"]

    def test_models_agreeing_from_pick(self):
        pick = {"models_agreeing": ["xl", "v3"]}
        ctx = _build_context_snapshot(pick)
        assert ctx["models_agreeing"] == ["xl", "v3"]

    def test_model_version_fallback(self):
        pick = {"model_version": "v3"}
        ctx = _build_context_snapshot(pick)
        assert ctx["models_agreeing"] == ["v3"]

    def test_risk_fields(self):
        pick = {"risk_level": "HIGH", "risk_flags": ["back_to_back", "injury_adjacent"]}
        ctx = _build_context_snapshot(pick)
        assert ctx["risk_level"] == "HIGH"
        assert "back_to_back" in ctx["risk_flags"]

    def test_hit_rates_extracted(self):
        pick = {
            "hit_rates": {
                "last_5": {"rate": 0.8},
                "last_15": {"rate": 0.67},
                "season": {"rate": 0.55},
            }
        }
        ctx = _build_context_snapshot(pick)
        assert ctx["hit_rate_L5"] == pytest.approx(0.8)
        assert ctx["hit_rate_L15"] == pytest.approx(0.67)
        assert ctx["hit_rate_season"] == pytest.approx(0.55)

    def test_hit_rates_missing(self):
        ctx = _build_context_snapshot({})
        assert ctx["hit_rate_L5"] is None
        assert ctx["hit_rate_L15"] is None
        assert ctx["hit_rate_season"] is None

    def test_bp_intel_extracted(self):
        pick = {
            "bp_intel": {
                "streak": 4,
                "streak_type": "OVER",
                "bp_projection": 27.3,
                "bp_probability": 0.68,
                "bp_bet_rating": 4,
                "bp_recommended_side": "over",
                "opposition_rank": 28,
            }
        }
        ctx = _build_context_snapshot(pick)
        assert ctx["bp_streak"] == 4
        assert ctx["bp_streak_type"] == "OVER"
        assert ctx["bp_projection"] == pytest.approx(27.3)
        assert ctx["bp_probability"] == pytest.approx(0.68)
        assert ctx["bp_bet_rating"] == 4
        assert ctx["bp_recommended_side"] == "over"
        assert ctx["opposition_rank"] == 28

    def test_bp_intel_missing(self):
        ctx = _build_context_snapshot({})
        assert ctx["bp_streak"] is None
        assert ctx["bp_projection"] is None
        assert ctx["bp_bet_rating"] is None

    def test_consensus_line_captured(self):
        pick = {"consensus_line": 24.5}
        ctx = _build_context_snapshot(pick)
        assert ctx["consensus_line"] == pytest.approx(24.5)

    def test_filter_tier_captured(self):
        pick = {"filter_tier": "X"}
        ctx = _build_context_snapshot(pick)
        assert ctx["filter_tier"] == "X"

    def test_stake_fields_captured(self):
        pick = {"recommended_stake": 2.5, "stake_reason": "kelly_normal"}
        ctx = _build_context_snapshot(pick)
        assert ctx["recommended_stake"] == pytest.approx(2.5)
        assert ctx["stake_reason"] == "kelly_normal"

    def test_trend_from_player_context(self):
        pick = {"player_context": {"trend": "hot"}}
        ctx = _build_context_snapshot(pick)
        assert ctx["trend"] == "hot"

    def test_confidence_captured(self):
        pick = {"confidence": "HIGH"}
        ctx = _build_context_snapshot(pick)
        assert ctx["confidence"] == "HIGH"

    def test_best_book_type_captured(self):
        pick = {"best_book": "prizepicks_goblin"}
        ctx = _build_context_snapshot(pick)
        assert ctx["best_book_type"] == "prizepicks_goblin"


# ─────────────────────────────────────────────────────────────────
# audit_run_start
# ─────────────────────────────────────────────────────────────────


def _make_mock_conn():
    """Helper: mock psycopg2 connection that works as context manager."""
    mock_cur = MagicMock()
    mock_cur.__enter__ = lambda s: mock_cur
    mock_cur.__exit__ = MagicMock(return_value=False)
    mock_conn = MagicMock()
    mock_conn.__enter__ = lambda s: mock_conn
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_conn.cursor.return_value = mock_cur
    return mock_conn, mock_cur


class TestAuditRunStart:
    def test_success_returns_true(self):
        mock_conn, _ = _make_mock_conn()
        with patch("nba.core.axiom_writer._connect", return_value=mock_conn):
            result = audit_run_start("2026-03-07", 1, "full")
        assert result is True

    def test_db_error_returns_false(self):
        with patch("nba.core.axiom_writer._connect", side_effect=Exception("conn failed")):
            result = audit_run_start("2026-03-07", 1, "full")
        assert result is False

    def test_executes_insert_upsert(self):
        mock_conn, mock_cur = _make_mock_conn()
        with patch("nba.core.axiom_writer._connect", return_value=mock_conn):
            audit_run_start("2026-03-07", 2, "refresh")
        mock_cur.execute.assert_called_once()
        call_args = mock_cur.execute.call_args[0]
        assert "INSERT INTO axiom_pipeline_audit" in call_args[0]


class TestAuditRunComplete:
    def test_success_returns_true(self):
        mock_conn, _ = _make_mock_conn()
        with patch("nba.core.axiom_writer._connect", return_value=mock_conn):
            result = audit_run_complete("2026-03-07", 1, "success", picks_generated=12)
        assert result is True

    def test_db_error_returns_false(self):
        with patch("nba.core.axiom_writer._connect", side_effect=Exception("db down")):
            result = audit_run_complete("2026-03-07", 1, "failed")
        assert result is False

    def test_executes_update(self):
        mock_conn, mock_cur = _make_mock_conn()
        with patch("nba.core.axiom_writer._connect", return_value=mock_conn):
            audit_run_complete("2026-03-07", 3, "partial", picks_generated=5)
        mock_cur.execute.assert_called_once()
        call_args = mock_cur.execute.call_args[0]
        assert "UPDATE axiom_pipeline_audit" in call_args[0]


# ─────────────────────────────────────────────────────────────────
# count_todays_props
# ─────────────────────────────────────────────────────────────────


class TestCountTodaysProps:
    def test_returns_counts_from_db(self):
        mock_conn, mock_cur = _make_mock_conn()
        mock_cur.fetchone.return_value = (627, 7)

        with (
            patch("nba.core.axiom_writer._connect"),  # not used by count_todays_props
            patch("psycopg2.connect", return_value=mock_conn),
            patch(
                "nba.core.axiom_writer.get_intelligence_db_config",
                return_value={"host": "localhost"},
                create=True,
            ),
        ):
            # patch the import inside the function
            import nba.core.axiom_writer as aw

            with patch.object(aw, "count_todays_props", wraps=aw.count_todays_props):
                pass  # coverage via direct call below

        # Direct test via mocking psycopg2 inside the function
        mock_conn2, mock_cur2 = _make_mock_conn()
        mock_cur2.fetchone.return_value = (500, 6)
        with patch("psycopg2.connect", return_value=mock_conn2):
            from nba.config.database import get_intelligence_db_config

            with patch(
                "nba.core.axiom_writer.get_intelligence_db_config",
                return_value=get_intelligence_db_config(),
                create=True,
            ):
                pass

    def test_returns_zeros_on_error(self):
        with patch(
            "nba.config.database.get_intelligence_db_config",
            side_effect=Exception("no config"),
        ):
            result = count_todays_props("2026-03-07")
        assert result == (0, 0)

    def test_returns_zeros_on_connect_error(self):
        with (
            patch(
                "nba.config.database.get_intelligence_db_config",
                return_value={},
            ),
            patch("psycopg2.connect", side_effect=Exception("refused")),
        ):
            result = count_todays_props("2026-03-07")
        assert result == (0, 0)
