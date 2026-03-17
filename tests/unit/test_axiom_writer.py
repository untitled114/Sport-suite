"""
Unit tests for nba.core.axiom_writer — pure functions and mocked DB functions.
"""

import json
from unittest.mock import MagicMock, call, patch

import pytest

from nba.core.axiom_writer import (
    _SCHEDULED_RUNS,
    _build_context_snapshot,
    _connect,
    audit_run_complete,
    audit_run_start,
    count_todays_props,
    get_run_number,
    write_actuals,
    write_picks,
)

# ─────────────────────────────────────────────────────────────────
# Helpers
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


def _sample_pick(**overrides):
    """Build a realistic pick dict with optional overrides."""
    pick = {
        "player_name": "LeBron James",
        "stat_type": "POINTS",
        "model_version": "xl",
        "filter_tier": "X",
        "best_line": 24.5,
        "p_over": 0.87,
        "edge": 4.1,
        "line_spread": 2.0,
        "best_book": "underdog",
        "opponent_team": "GSW",
        "is_home": True,
        "prediction": 28.5,
        "consensus_line": 25.7,
        "confidence": "HIGH",
        "recommended_stake": 2.5,
        "stake_reason": "kelly_normal",
        "risk_level": "LOW",
        "risk_flags": [],
        "hit_rates": {
            "last_5": {"rate": 0.8},
            "last_15": {"rate": 0.67},
            "season": {"rate": 0.55},
        },
        "bp_intel": {
            "streak": 4,
            "streak_type": "OVER",
            "bp_projection": 27.3,
            "bp_probability": 0.68,
            "bp_bet_rating": 4,
            "bp_recommended_side": "over",
            "opposition_rank": 28,
        },
    }
    pick.update(overrides)
    return pick


# ─────────────────────────────────────────────────────────────────
# _connect
# ─────────────────────────────────────────────────────────────────


class TestConnect:
    def test_returns_connection(self):
        mock_conn = MagicMock()
        with patch("psycopg2.connect", return_value=mock_conn):
            result = _connect()
        assert result is mock_conn

    def test_uses_env_vars(self):
        mock_conn = MagicMock()
        with (
            patch("psycopg2.connect", return_value=mock_conn) as mock_pg,
            patch.dict(
                "os.environ",
                {"DB_HOST": "my-host", "DB_USER": "my-user", "DB_PASSWORD": "secret"},
            ),
        ):
            _connect()
        mock_pg.assert_called_once_with(
            host="my-host",
            port=5541,
            dbname="cephalon_axiom",
            user="my-user",
            password="secret",
            connect_timeout=5,
        )

    def test_defaults_when_env_missing(self):
        mock_conn = MagicMock()
        with (
            patch("psycopg2.connect", return_value=mock_conn) as mock_pg,
            patch.dict("os.environ", {}, clear=True),
        ):
            _connect()
        kwargs = mock_pg.call_args[1]
        assert kwargs["host"] == "localhost"
        assert kwargs["user"] == "mlb_user"
        assert kwargs["password"] == ""


# ─────────────────────────────────────────────────────────────────
# get_run_number
# ─────────────────────────────────────────────────────────────────


class TestGetRunNumber:
    def _mock_est(self, hour, minute=0):
        """Return a mock datetime.now() with given EST hour:minute."""
        mock_now = MagicMock()
        mock_now.hour = hour
        mock_now.minute = minute
        mock_dt = MagicMock()
        mock_dt.now.return_value = mock_now
        return mock_dt

    def test_scheduled_hours_exact(self):
        for sched_hour, expected_run in _SCHEDULED_RUNS:
            with patch("nba.core.axiom_writer.datetime", self._mock_est(sched_hour)):
                assert get_run_number() == expected_run

    def test_window_before_scheduled(self):
        # 10 EST is 60 min before run 4 (11 EST) — within +/-90 min
        with patch("nba.core.axiom_writer.datetime", self._mock_est(10, 0)):
            assert get_run_number() == 4

    def test_window_after_scheduled(self):
        # 2 EST + 30 min still in run 1's window
        with patch("nba.core.axiom_writer.datetime", self._mock_est(2, 30)):
            assert get_run_number() == 1

    def test_off_window_returns_1(self):
        # 0 EST (midnight) is 120 min before run 1 (2 EST) — outside window
        with patch("nba.core.axiom_writer.datetime", self._mock_est(0, 0)):
            assert get_run_number() == 1

    def test_run1_at_2est(self):
        with patch("nba.core.axiom_writer.datetime", self._mock_est(2)):
            assert get_run_number() == 1

    def test_run6_at_17est(self):
        with patch("nba.core.axiom_writer.datetime", self._mock_est(17)):
            assert get_run_number() == 6

    def test_run7_at_20est(self):
        with patch("nba.core.axiom_writer.datetime", self._mock_est(20)):
            assert get_run_number() == 7

    def test_edge_of_window_boundary(self):
        # Exactly 90 min after the 5 EST slot = 6:30 EST -> should match run 2
        with patch("nba.core.axiom_writer.datetime", self._mock_est(6, 30)):
            assert get_run_number() == 2

    def test_just_outside_window(self):
        # 91 min after 5 EST = 6:31 -> should NOT match run 2, should match run 3 (8 EST)
        with patch("nba.core.axiom_writer.datetime", self._mock_est(6, 31)):
            assert get_run_number() == 3


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
        assert ctx["player_context"]["trend"] == "hot"

    def test_confidence_captured(self):
        pick = {"confidence": "HIGH"}
        ctx = _build_context_snapshot(pick)
        assert ctx["confidence"] == "HIGH"

    def test_best_book_type_captured(self):
        pick = {"best_book": "prizepicks_goblin"}
        ctx = _build_context_snapshot(pick)
        assert ctx["best_book_type"] == "prizepicks_goblin"

    def test_prediction_and_edge(self):
        pick = {"prediction": 28.5, "edge": 4.0}
        ctx = _build_context_snapshot(pick)
        assert ctx["prediction"] == pytest.approx(28.5)
        assert ctx["edge"] == pytest.approx(4.0)

    def test_opponent_team_and_is_home(self):
        pick = {"opponent_team": "LAL", "is_home": False}
        ctx = _build_context_snapshot(pick)
        assert ctx["opponent_team"] == "LAL"
        assert ctx["is_home"] is False

    def test_full_pick_all_fields(self):
        pick = _sample_pick()
        ctx = _build_context_snapshot(pick)
        assert ctx["models_agreeing"] == ["xl"]
        assert ctx["risk_level"] == "LOW"
        assert ctx["confidence"] == "HIGH"
        assert ctx["prediction"] == pytest.approx(28.5)
        assert ctx["edge"] == pytest.approx(4.1)
        assert ctx["consensus_line"] == pytest.approx(25.7)
        assert ctx["hit_rate_L5"] == pytest.approx(0.8)
        assert ctx["bp_streak"] == 4


# ─────────────────────────────────────────────────────────────────
# audit_run_start
# ─────────────────────────────────────────────────────────────────


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
        assert "ON CONFLICT" in call_args[0]
        assert call_args[1] == ("2026-03-07", 2, "refresh")

    def test_closes_connection(self):
        mock_conn, _ = _make_mock_conn()
        with patch("nba.core.axiom_writer._connect", return_value=mock_conn):
            audit_run_start("2026-03-07", 1, "full")
        mock_conn.close.assert_called_once()


# ─────────────────────────────────────────────────────────────────
# audit_run_complete
# ─────────────────────────────────────────────────────────────────


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

    def test_passes_all_kwargs_to_sql(self):
        mock_conn, mock_cur = _make_mock_conn()
        with patch("nba.core.axiom_writer._connect", return_value=mock_conn):
            audit_run_complete(
                "2026-03-17",
                2,
                "success",
                props_fetched=500,
                books_available=7,
                injuries_updated=True,
                games_found=12,
                duration_seconds=300,
                picks_generated=15,
                xl_picks=8,
                v3_picks=7,
                error_message=None,
                error_traceback=None,
                anomalies={"low_books": True},
            )
        call_args = mock_cur.execute.call_args[0]
        params = call_args[1]
        # Order: status, props_fetched, books_available, injuries_updated,
        #        games_found, duration_seconds, picks_generated, xl_picks, v3_picks,
        #        error_message, error_traceback, anomalies_json, run_date, run_number
        assert params[0] == "success"
        assert params[1] == 500
        assert params[2] == 7
        assert params[3] is True
        assert params[4] == 12
        assert params[5] == 300
        assert params[6] == 15
        assert params[7] == 8
        assert params[8] == 7
        assert params[9] is None  # error_message
        assert params[10] is None  # error_traceback
        assert json.loads(params[11]) == {"low_books": True}
        assert params[12] == "2026-03-17"
        assert params[13] == 2

    def test_anomalies_none_passed_as_none(self):
        mock_conn, mock_cur = _make_mock_conn()
        with patch("nba.core.axiom_writer._connect", return_value=mock_conn):
            audit_run_complete("2026-03-17", 1, "success", anomalies=None)
        call_args = mock_cur.execute.call_args[0]
        params = call_args[1]
        # anomalies is at index 11
        assert params[11] is None

    def test_error_fields_forwarded(self):
        mock_conn, mock_cur = _make_mock_conn()
        with patch("nba.core.axiom_writer._connect", return_value=mock_conn):
            audit_run_complete(
                "2026-03-17",
                1,
                "failed",
                error_message="Something broke",
                error_traceback="traceback text",
            )
        params = mock_cur.execute.call_args[0][1]
        assert params[0] == "failed"
        assert params[9] == "Something broke"
        assert params[10] == "traceback text"

    def test_closes_connection(self):
        mock_conn, _ = _make_mock_conn()
        with patch("nba.core.axiom_writer._connect", return_value=mock_conn):
            audit_run_complete("2026-03-07", 1, "success")
        mock_conn.close.assert_called_once()


# ─────────────────────────────────────────────────────────────────
# write_picks
# ─────────────────────────────────────────────────────────────────


class TestWritePicks:
    def test_empty_list_returns_zero(self):
        """write_picks should return 0 immediately for empty picks list."""
        result = write_picks("2026-03-17", 1, "2026-03-17T09:00:00", [])
        assert result == 0

    def test_inserts_single_pick(self):
        mock_conn, mock_cur = _make_mock_conn()
        mock_cur.rowcount = 1

        pick = _sample_pick()

        with (
            patch("nba.core.axiom_writer._connect", return_value=mock_conn),
            patch("psycopg2.extras.execute_values") as mock_ev,
        ):
            result = write_picks("2026-03-17", 1, "2026-03-17T09:00:00", [pick])

        assert result == 1
        mock_ev.assert_called_once()
        call_args = mock_ev.call_args
        sql = call_args[0][1]
        rows = call_args[0][2]
        assert "INSERT INTO nba_prediction_history" in sql
        assert "ON CONFLICT DO NOTHING" in sql
        assert len(rows) == 1

    def test_inserts_multiple_picks(self):
        mock_conn, mock_cur = _make_mock_conn()
        mock_cur.rowcount = 3

        picks = [
            _sample_pick(player_name="LeBron James", stat_type="POINTS"),
            _sample_pick(player_name="Stephen Curry", stat_type="POINTS", model_version="v3"),
            _sample_pick(player_name="Nikola Jokic", stat_type="REBOUNDS"),
        ]

        with (
            patch("nba.core.axiom_writer._connect", return_value=mock_conn),
            patch("psycopg2.extras.execute_values") as mock_ev,
        ):
            result = write_picks("2026-03-17", 2, "2026-03-17T11:00:00", picks)

        assert result == 3
        rows = mock_ev.call_args[0][2]
        assert len(rows) == 3
        # Verify player names in the rows
        assert rows[0][3] == "LeBron James"
        assert rows[1][3] == "Stephen Curry"
        assert rows[2][3] == "Nikola Jokic"

    def test_row_tuple_structure(self):
        """Verify the exact shape of each row tuple passed to execute_values."""
        mock_conn, mock_cur = _make_mock_conn()
        mock_cur.rowcount = 1

        pick = _sample_pick(
            player_name="Jayson Tatum",
            stat_type="REBOUNDS",
            model_version="v3",
            filter_tier="META",
            best_line=7.5,
            p_over=0.74,
            edge=3.5,
            line_spread=1.5,
            best_book="fanduel",
            opponent_team="MIA",
            is_home=False,
        )

        with (
            patch("nba.core.axiom_writer._connect", return_value=mock_conn),
            patch("psycopg2.extras.execute_values") as mock_ev,
        ):
            write_picks("2026-03-17", 3, "2026-03-17T14:00:00", [pick])

        row = mock_ev.call_args[0][2][0]
        assert row[0] == "2026-03-17"  # run_date
        assert row[1] == 3  # run_number
        assert row[2] == "2026-03-17T14:00:00"  # run_timestamp
        assert row[3] == "Jayson Tatum"  # player_name
        assert row[4] == "REBOUNDS"  # stat_type
        assert row[5] == "v3"  # model_version
        assert row[6] == "META"  # filter_tier
        assert row[7] == 7.5  # best_line
        assert row[8] == pytest.approx(0.74)  # p_over
        assert row[9] == pytest.approx(3.5)  # edge
        assert row[10] == pytest.approx(1.5)  # line_spread
        assert row[11] == "fanduel"  # best_book
        assert row[12] is None  # game_time (always None)
        assert row[13] == "MIA"  # opponent_team
        assert row[14] is False  # is_home
        # row[15] = context_snapshot JSON string
        ctx = json.loads(row[15])
        assert ctx["confidence"] == "HIGH"

    def test_context_snapshot_serialized_as_json(self):
        mock_conn, mock_cur = _make_mock_conn()
        mock_cur.rowcount = 1

        pick = _sample_pick(prediction=30.0, edge=5.0, confidence="HIGH")

        with (
            patch("nba.core.axiom_writer._connect", return_value=mock_conn),
            patch("psycopg2.extras.execute_values") as mock_ev,
        ):
            write_picks("2026-03-17", 1, "2026-03-17T09:00:00", [pick])

        row = mock_ev.call_args[0][2][0]
        ctx = json.loads(row[15])
        assert ctx["prediction"] == pytest.approx(30.0)
        assert ctx["edge"] == pytest.approx(5.0)
        assert ctx["confidence"] == "HIGH"

    def test_model_version_defaults_to_xl(self):
        """If pick has no model_version, should default to 'xl'."""
        mock_conn, mock_cur = _make_mock_conn()
        mock_cur.rowcount = 1

        pick = _sample_pick()
        del pick["model_version"]

        with (
            patch("nba.core.axiom_writer._connect", return_value=mock_conn),
            patch("psycopg2.extras.execute_values") as mock_ev,
        ):
            write_picks("2026-03-17", 1, "2026-03-17T09:00:00", [pick])

        row = mock_ev.call_args[0][2][0]
        assert row[5] == "xl"

    def test_db_error_returns_zero(self):
        with patch("nba.core.axiom_writer._connect", side_effect=Exception("db refused")):
            result = write_picks("2026-03-17", 1, "2026-03-17T09:00:00", [_sample_pick()])
        assert result == 0

    def test_execute_values_error_returns_zero(self):
        mock_conn, mock_cur = _make_mock_conn()

        with (
            patch("nba.core.axiom_writer._connect", return_value=mock_conn),
            patch(
                "psycopg2.extras.execute_values",
                side_effect=Exception("bulk insert failed"),
            ),
        ):
            result = write_picks("2026-03-17", 1, "2026-03-17T09:00:00", [_sample_pick()])

        assert result == 0

    def test_closes_connection_on_success(self):
        mock_conn, mock_cur = _make_mock_conn()
        mock_cur.rowcount = 1

        with (
            patch("nba.core.axiom_writer._connect", return_value=mock_conn),
            patch("psycopg2.extras.execute_values"),
        ):
            write_picks("2026-03-17", 1, "2026-03-17T09:00:00", [_sample_pick()])

        mock_conn.close.assert_called_once()

    def test_pick_with_minimal_fields(self):
        """A pick with only a player_name should still be insertable (fields default to None)."""
        mock_conn, mock_cur = _make_mock_conn()
        mock_cur.rowcount = 1

        pick = {"player_name": "Unknown Player"}

        with (
            patch("nba.core.axiom_writer._connect", return_value=mock_conn),
            patch("psycopg2.extras.execute_values") as mock_ev,
        ):
            result = write_picks("2026-03-17", 1, "2026-03-17T09:00:00", [pick])

        assert result == 1
        row = mock_ev.call_args[0][2][0]
        assert row[3] == "Unknown Player"
        assert row[4] is None  # stat_type
        assert row[5] == "xl"  # model_version default

    def test_duplicate_handling_via_rowcount(self):
        """Duplicates are silently ignored, rowcount reflects actual inserts."""
        mock_conn, mock_cur = _make_mock_conn()
        mock_cur.rowcount = 1  # Only 1 of 2 was inserted (other was duplicate)

        picks = [_sample_pick(), _sample_pick()]

        with (
            patch("nba.core.axiom_writer._connect", return_value=mock_conn),
            patch("psycopg2.extras.execute_values"),
        ):
            result = write_picks("2026-03-17", 1, "2026-03-17T09:00:00", picks)

        assert result == 1


# ─────────────────────────────────────────────────────────────────
# write_actuals
# ─────────────────────────────────────────────────────────────────


class TestWriteActuals:
    def _make_axiom_conn(self, picks_to_grade=None, update_rowcount=1):
        """Create a mock axiom connection with cursor behaviors for write_actuals."""
        conn = MagicMock()
        conn.__enter__ = lambda s: conn
        conn.__exit__ = MagicMock(return_value=False)

        # We need multiple cursor calls — first for SELECT, then for UPDATE loop.
        # write_actuals opens the connection twice (once for SELECT, once for UPDATE).
        # Each time it uses conn.cursor() as a context manager.
        cur = MagicMock()
        cur.__enter__ = lambda s: cur
        cur.__exit__ = MagicMock(return_value=False)
        cur.fetchall.return_value = picks_to_grade or []
        cur.rowcount = update_rowcount
        conn.cursor.return_value = cur
        return conn, cur

    def test_no_ungraded_picks_returns_zero(self):
        """If there are no ungraded picks, return 0."""
        axiom_conn, axiom_cur = self._make_axiom_conn(picks_to_grade=[])

        with patch("nba.core.axiom_writer._connect", return_value=axiom_conn):
            result = write_actuals("2026-03-16")

        assert result == 0

    def test_no_game_logs_returns_zero(self):
        """If ungraded picks exist but no game logs are found, return 0."""
        # Axiom returns picks to grade
        axiom_conn1, axiom_cur1 = self._make_axiom_conn(
            picks_to_grade=[("LeBron James", "POINTS", 24.5)]
        )

        # Players DB returns empty game logs
        players_conn = MagicMock()
        players_conn.__enter__ = lambda s: players_conn
        players_conn.__exit__ = MagicMock(return_value=False)
        players_cur = MagicMock()
        players_cur.__enter__ = lambda s: players_cur
        players_cur.__exit__ = MagicMock(return_value=False)
        players_cur.fetchall.return_value = []
        players_conn.cursor.return_value = players_cur

        with (
            patch("nba.core.axiom_writer._connect", return_value=axiom_conn1),
            patch("psycopg2.connect", return_value=players_conn),
            patch(
                "nba.core.axiom_writer.get_players_db_config",
                return_value={"host": "localhost"},
                create=True,
            ),
            patch(
                "nba.config.database.get_players_db_config",
                return_value={"host": "localhost"},
            ),
        ):
            result = write_actuals("2026-03-16")

        assert result == 0

    def test_successful_grading(self):
        """Match picks against actuals and update is_hit."""
        # First call to _connect: fetch ungraded picks
        axiom_conn_select, axiom_cur_select = self._make_axiom_conn(
            picks_to_grade=[
                ("LeBron James", "POINTS", 24.5),
                ("Stephen Curry", "REBOUNDS", 6.5),
            ]
        )

        # Second call to _connect: update rows
        axiom_conn_update, axiom_cur_update = self._make_axiom_conn()
        axiom_cur_update.rowcount = 1

        # Track _connect calls — first returns select conn, second returns update conn
        connect_calls = [axiom_conn_select, axiom_conn_update]

        # Players DB returns game logs
        players_conn = MagicMock()
        players_conn.__enter__ = lambda s: players_conn
        players_conn.__exit__ = MagicMock(return_value=False)
        players_cur = MagicMock()
        players_cur.__enter__ = lambda s: players_cur
        players_cur.__exit__ = MagicMock(return_value=False)
        players_cur.fetchall.return_value = [
            ("LeBron James", 28, 7, 9, 3),  # 28 pts > 24.5 line -> HIT
            ("Stephen Curry", 25, 5, 6, 4),  # 5 reb < 6.5 line -> MISS
        ]
        players_conn.cursor.return_value = players_cur

        with (
            patch(
                "nba.core.axiom_writer._connect",
                side_effect=connect_calls,
            ),
            patch("psycopg2.connect", return_value=players_conn),
            patch(
                "nba.config.database.get_players_db_config",
                return_value={"host": "localhost"},
            ),
        ):
            result = write_actuals("2026-03-16")

        # Two picks graded, each updates rowcount=1
        assert result == 2
        # Verify the UPDATE calls
        update_calls = axiom_cur_update.execute.call_args_list
        assert len(update_calls) == 2

        # First UPDATE: LeBron POINTS, actual=28, is_hit=True
        args1 = update_calls[0][0][1]
        assert args1[0] == 28  # actual_result
        assert args1[1] is True  # is_hit (28 > 24.5)
        assert args1[3] == "LeBron James"
        assert args1[4] == "POINTS"

        # Second UPDATE: Curry REBOUNDS, actual=5, is_hit=False
        args2 = update_calls[1][0][1]
        assert args2[0] == 5  # actual_result
        assert args2[1] is False  # is_hit (5 < 6.5)
        assert args2[3] == "Stephen Curry"
        assert args2[4] == "REBOUNDS"

    def test_player_not_found_in_actuals_skipped(self):
        """If a player in picks has no matching game log, skip them."""
        axiom_conn_select, axiom_cur_select = self._make_axiom_conn(
            picks_to_grade=[("Unknown Player", "POINTS", 20.5)]
        )
        axiom_conn_update, axiom_cur_update = self._make_axiom_conn()

        players_conn = MagicMock()
        players_conn.__enter__ = lambda s: players_conn
        players_conn.__exit__ = MagicMock(return_value=False)
        players_cur = MagicMock()
        players_cur.__enter__ = lambda s: players_cur
        players_cur.__exit__ = MagicMock(return_value=False)
        players_cur.fetchall.return_value = [
            ("LeBron James", 28, 7, 9, 3),
        ]
        players_conn.cursor.return_value = players_cur

        with (
            patch(
                "nba.core.axiom_writer._connect",
                side_effect=[axiom_conn_select, axiom_conn_update],
            ),
            patch("psycopg2.connect", return_value=players_conn),
            patch(
                "nba.config.database.get_players_db_config",
                return_value={"host": "localhost"},
            ),
        ):
            result = write_actuals("2026-03-16")

        assert result == 0
        axiom_cur_update.execute.assert_not_called()

    def test_stat_type_not_in_actuals_skipped(self):
        """If a pick's stat_type has no matching column value, skip it."""
        axiom_conn_select, axiom_cur_select = self._make_axiom_conn(
            picks_to_grade=[("LeBron James", "STEALS", 1.5)]
        )
        axiom_conn_update, axiom_cur_update = self._make_axiom_conn()

        players_conn = MagicMock()
        players_conn.__enter__ = lambda s: players_conn
        players_conn.__exit__ = MagicMock(return_value=False)
        players_cur = MagicMock()
        players_cur.__enter__ = lambda s: players_cur
        players_cur.__exit__ = MagicMock(return_value=False)
        players_cur.fetchall.return_value = [
            ("LeBron James", 28, 7, 9, 3),
        ]
        players_conn.cursor.return_value = players_cur

        with (
            patch(
                "nba.core.axiom_writer._connect",
                side_effect=[axiom_conn_select, axiom_conn_update],
            ),
            patch("psycopg2.connect", return_value=players_conn),
            patch(
                "nba.config.database.get_players_db_config",
                return_value={"host": "localhost"},
            ),
        ):
            result = write_actuals("2026-03-16")

        # STEALS is not in the actuals dict keys (POINTS/REBOUNDS/ASSISTS/THREES)
        assert result == 0

    def test_exception_returns_zero(self):
        """Any exception in write_actuals returns 0 (fire-and-forget)."""
        with patch("nba.core.axiom_writer._connect", side_effect=Exception("db down")):
            result = write_actuals("2026-03-16")
        assert result == 0

    def test_case_insensitive_player_lookup(self):
        """Player name matching uses lower() for case insensitivity."""
        axiom_conn_select, axiom_cur_select = self._make_axiom_conn(
            picks_to_grade=[("LEBRON JAMES", "POINTS", 24.5)]
        )
        axiom_conn_update, axiom_cur_update = self._make_axiom_conn()
        axiom_cur_update.rowcount = 1

        players_conn = MagicMock()
        players_conn.__enter__ = lambda s: players_conn
        players_conn.__exit__ = MagicMock(return_value=False)
        players_cur = MagicMock()
        players_cur.__enter__ = lambda s: players_cur
        players_cur.__exit__ = MagicMock(return_value=False)
        players_cur.fetchall.return_value = [
            ("LeBron James", 28, 7, 9, 3),
        ]
        players_conn.cursor.return_value = players_cur

        with (
            patch(
                "nba.core.axiom_writer._connect",
                side_effect=[axiom_conn_select, axiom_conn_update],
            ),
            patch("psycopg2.connect", return_value=players_conn),
            patch(
                "nba.config.database.get_players_db_config",
                return_value={"host": "localhost"},
            ),
        ):
            result = write_actuals("2026-03-16")

        assert result == 1

    def test_null_stat_values_treated_as_zero(self):
        """If game log has None for a stat, float(None or 0) should be 0.0."""
        axiom_conn_select, axiom_cur_select = self._make_axiom_conn(
            picks_to_grade=[("LeBron James", "THREES", 2.5)]
        )
        axiom_conn_update, axiom_cur_update = self._make_axiom_conn()
        axiom_cur_update.rowcount = 1

        players_conn = MagicMock()
        players_conn.__enter__ = lambda s: players_conn
        players_conn.__exit__ = MagicMock(return_value=False)
        players_cur = MagicMock()
        players_cur.__enter__ = lambda s: players_cur
        players_cur.__exit__ = MagicMock(return_value=False)
        # three_pointers_made is None
        players_cur.fetchall.return_value = [
            ("LeBron James", 28, 7, 9, None),
        ]
        players_conn.cursor.return_value = players_cur

        with (
            patch(
                "nba.core.axiom_writer._connect",
                side_effect=[axiom_conn_select, axiom_conn_update],
            ),
            patch("psycopg2.connect", return_value=players_conn),
            patch(
                "nba.config.database.get_players_db_config",
                return_value={"host": "localhost"},
            ),
        ):
            result = write_actuals("2026-03-16")

        assert result == 1
        # actual is 0.0 (from None), is_hit = 0.0 > 2.5 = False
        update_args = axiom_cur_update.execute.call_args[0][1]
        assert update_args[0] == pytest.approx(0.0)
        assert update_args[1] is False

    def test_closes_all_connections(self):
        """All connections (axiom select, players, axiom update) should be closed."""
        axiom_conn_select, _ = self._make_axiom_conn(
            picks_to_grade=[("LeBron James", "POINTS", 24.5)]
        )
        axiom_conn_update, axiom_cur_update = self._make_axiom_conn()
        axiom_cur_update.rowcount = 1

        players_conn = MagicMock()
        players_conn.__enter__ = lambda s: players_conn
        players_conn.__exit__ = MagicMock(return_value=False)
        players_cur = MagicMock()
        players_cur.__enter__ = lambda s: players_cur
        players_cur.__exit__ = MagicMock(return_value=False)
        players_cur.fetchall.return_value = [
            ("LeBron James", 28, 7, 9, 3),
        ]
        players_conn.cursor.return_value = players_cur

        with (
            patch(
                "nba.core.axiom_writer._connect",
                side_effect=[axiom_conn_select, axiom_conn_update],
            ),
            patch("psycopg2.connect", return_value=players_conn),
            patch(
                "nba.config.database.get_players_db_config",
                return_value={"host": "localhost"},
            ),
        ):
            write_actuals("2026-03-16")

        axiom_conn_select.close.assert_called_once()
        axiom_conn_update.close.assert_called_once()
        players_conn.close.assert_called_once()

    def test_all_four_stat_types(self):
        """Verify POINTS, REBOUNDS, ASSISTS, THREES all map correctly."""
        picks = [
            ("Player A", "POINTS", 20.5),
            ("Player A", "REBOUNDS", 6.5),
            ("Player A", "ASSISTS", 5.5),
            ("Player A", "THREES", 2.5),
        ]
        axiom_conn_select, _ = self._make_axiom_conn(picks_to_grade=picks)
        axiom_conn_update, axiom_cur_update = self._make_axiom_conn()
        axiom_cur_update.rowcount = 1

        players_conn = MagicMock()
        players_conn.__enter__ = lambda s: players_conn
        players_conn.__exit__ = MagicMock(return_value=False)
        players_cur = MagicMock()
        players_cur.__enter__ = lambda s: players_cur
        players_cur.__exit__ = MagicMock(return_value=False)
        # pts=25, reb=8, ast=6, threes=3
        players_cur.fetchall.return_value = [
            ("Player A", 25, 8, 6, 3),
        ]
        players_conn.cursor.return_value = players_cur

        with (
            patch(
                "nba.core.axiom_writer._connect",
                side_effect=[axiom_conn_select, axiom_conn_update],
            ),
            patch("psycopg2.connect", return_value=players_conn),
            patch(
                "nba.config.database.get_players_db_config",
                return_value={"host": "localhost"},
            ),
        ):
            result = write_actuals("2026-03-16")

        assert result == 4
        calls = axiom_cur_update.execute.call_args_list
        # POINTS: 25 > 20.5 = True
        assert calls[0][0][1][0] == pytest.approx(25.0)
        assert calls[0][0][1][1] is True
        # REBOUNDS: 8 > 6.5 = True
        assert calls[1][0][1][0] == pytest.approx(8.0)
        assert calls[1][0][1][1] is True
        # ASSISTS: 6 > 5.5 = True
        assert calls[2][0][1][0] == pytest.approx(6.0)
        assert calls[2][0][1][1] is True
        # THREES: 3 > 2.5 = True
        assert calls[3][0][1][0] == pytest.approx(3.0)
        assert calls[3][0][1][1] is True

    def test_is_hit_false_when_actual_equals_line(self):
        """is_hit uses strict > (not >=), so actual == line should be False."""
        axiom_conn_select, _ = self._make_axiom_conn(
            picks_to_grade=[("LeBron James", "POINTS", 28.0)]
        )
        axiom_conn_update, axiom_cur_update = self._make_axiom_conn()
        axiom_cur_update.rowcount = 1

        players_conn = MagicMock()
        players_conn.__enter__ = lambda s: players_conn
        players_conn.__exit__ = MagicMock(return_value=False)
        players_cur = MagicMock()
        players_cur.__enter__ = lambda s: players_cur
        players_cur.__exit__ = MagicMock(return_value=False)
        players_cur.fetchall.return_value = [
            ("LeBron James", 28, 7, 9, 3),  # 28 == 28.0 line -> NOT a hit
        ]
        players_conn.cursor.return_value = players_cur

        with (
            patch(
                "nba.core.axiom_writer._connect",
                side_effect=[axiom_conn_select, axiom_conn_update],
            ),
            patch("psycopg2.connect", return_value=players_conn),
            patch(
                "nba.config.database.get_players_db_config",
                return_value={"host": "localhost"},
            ),
        ):
            result = write_actuals("2026-03-16")

        assert result == 1
        update_args = axiom_cur_update.execute.call_args[0][1]
        assert update_args[1] is False  # 28 > 28.0 is False

    def test_players_db_connect_failure(self):
        """If players DB connection fails, return 0."""
        axiom_conn_select, _ = self._make_axiom_conn(
            picks_to_grade=[("LeBron James", "POINTS", 24.5)]
        )

        with (
            patch("nba.core.axiom_writer._connect", return_value=axiom_conn_select),
            patch("psycopg2.connect", side_effect=Exception("players DB down")),
            patch(
                "nba.config.database.get_players_db_config",
                return_value={"host": "localhost"},
            ),
        ):
            result = write_actuals("2026-03-16")

        assert result == 0


# ─────────────────────────────────────────────────────────────────
# count_todays_props
# ─────────────────────────────────────────────────────────────────


class TestCountTodaysProps:
    def test_returns_counts_from_db(self):
        mock_conn, mock_cur = _make_mock_conn()
        mock_cur.fetchone.return_value = (627, 7)

        with (
            patch("psycopg2.connect", return_value=mock_conn),
            patch(
                "nba.config.database.get_intelligence_db_config",
                return_value={"host": "localhost", "port": 5539},
            ),
        ):
            result = count_todays_props("2026-03-17")

        assert result == (627, 7)

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

    def test_none_row_returns_zeros(self):
        mock_conn, mock_cur = _make_mock_conn()
        mock_cur.fetchone.return_value = None

        with (
            patch("psycopg2.connect", return_value=mock_conn),
            patch(
                "nba.config.database.get_intelligence_db_config",
                return_value={"host": "localhost"},
            ),
        ):
            result = count_todays_props("2026-03-17")

        assert result == (0, 0)

    def test_null_counts_treated_as_zero(self):
        mock_conn, mock_cur = _make_mock_conn()
        mock_cur.fetchone.return_value = (None, None)

        with (
            patch("psycopg2.connect", return_value=mock_conn),
            patch(
                "nba.config.database.get_intelligence_db_config",
                return_value={"host": "localhost"},
            ),
        ):
            result = count_todays_props("2026-03-17")

        assert result == (0, 0)

    def test_executes_correct_query(self):
        mock_conn, mock_cur = _make_mock_conn()
        mock_cur.fetchone.return_value = (100, 5)

        with (
            patch("psycopg2.connect", return_value=mock_conn),
            patch(
                "nba.config.database.get_intelligence_db_config",
                return_value={"host": "localhost"},
            ),
        ):
            count_todays_props("2026-03-17")

        mock_cur.execute.assert_called_once()
        sql = mock_cur.execute.call_args[0][0]
        assert "nba_props_xl" in sql
        assert "game_date" in sql
        params = mock_cur.execute.call_args[0][1]
        assert params == ("2026-03-17",)

    def test_closes_connection(self):
        mock_conn, mock_cur = _make_mock_conn()
        mock_cur.fetchone.return_value = (100, 5)

        with (
            patch("psycopg2.connect", return_value=mock_conn),
            patch(
                "nba.config.database.get_intelligence_db_config",
                return_value={"host": "localhost"},
            ),
        ):
            count_todays_props("2026-03-17")

        mock_conn.close.assert_called_once()
