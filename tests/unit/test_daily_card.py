"""
Unit tests for nba.core.daily_card — formatting functions and orchestration.

DB and Discord API calls are mocked throughout.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from nba.core.daily_card import (
    _already_sent_today,
    _bp_stars,
    _fmt_line,
    _fmt_p_over,
    _get_bot_token,
    build_embed,
    send_daily_card,
)

# ─────────────────────────────────────────────────────────────────
# _fmt_line
# ─────────────────────────────────────────────────────────────────


class TestFmtLine:
    def test_stable(self):
        result = _fmt_line(25.5, 25.5, "stable")
        assert "holding" in result
        assert "25.5" in result

    def test_falling(self):
        result = _fmt_line(26.5, 25.0, "falling")
        assert "dropped" in result
        assert "26.5" in result
        assert "25.0" in result

    def test_rising(self):
        result = _fmt_line(24.5, 26.0, "rising")
        assert "rose" in result
        assert "24.5" in result
        assert "26.0" in result

    def test_unknown_direction_treated_as_stable(self):
        result = _fmt_line(25.5, 25.5, "unknown")
        assert "holding" in result


# ─────────────────────────────────────────────────────────────────
# _fmt_p_over
# ─────────────────────────────────────────────────────────────────


class TestFmtPOver:
    def test_stable(self):
        result = _fmt_p_over(0.720, 0.721, 0.001)
        assert "steady" in result

    def test_trending_up(self):
        result = _fmt_p_over(0.65, 0.80, 0.15)
        assert "rising" in result
        assert "65%" in result
        assert "80%" in result

    def test_trending_down(self):
        result = _fmt_p_over(0.80, 0.65, -0.15)
        assert "fading" in result

    def test_none_trend_treated_as_zero(self):
        result = _fmt_p_over(0.720, 0.720, None)
        assert "steady" in result

    def test_boundary_not_stable(self):
        # trend of exactly 0.005 is at the boundary — not stable
        result = _fmt_p_over(0.70, 0.705, 0.005)
        assert "steady" not in result


# ─────────────────────────────────────────────────────────────────
# _bp_stars
# ─────────────────────────────────────────────────────────────────


class TestBpStars:
    def test_no_rating_returns_empty(self):
        assert _bp_stars({}) == ""
        assert _bp_stars({"bp_bet_rating": None}) == ""
        assert _bp_stars({"bp_bet_rating": 0}) == ""

    def test_rating_produces_stars(self):
        result = _bp_stars({"bp_bet_rating": 3})
        assert "★" in result or "\u2b50" in result
        assert "BP" in result

    def test_rating_with_side(self):
        result = _bp_stars({"bp_bet_rating": 4, "bp_recommended_side": "over"})
        assert "over" in result

    def test_rating_without_side(self):
        result = _bp_stars({"bp_bet_rating": 5})
        assert "BP" in result
        # No side annotation
        assert "()" not in result

    def test_five_stars(self):
        result = _bp_stars({"bp_bet_rating": 5})
        star_char = "\u2b50"
        assert result.count(star_char) == 5


# ─────────────────────────────────────────────────────────────────
# build_embed
# ─────────────────────────────────────────────────────────────────


class TestBuildEmbed:
    def _make_pick(self, **overrides):
        base = {
            "player_name": "Anthony Edwards",
            "stat_type": "POINTS",
            "conviction": 0.85,
            "conviction_label": "LOCKED",
            "appearances": 5,
            "total_runs": 6,
            "line_at_entry": 24.5,
            "line_latest": 23.5,
            "line_direction": "falling",
            "p_over_at_entry": 0.72,
            "p_over_latest": 0.78,
            "p_over_trend": 0.06,
            "book_latest": "draftkings",
            "context_snapshot": {},
        }
        base.update(overrides)
        return base

    def test_empty_picks_returns_no_picks_embed(self):
        embed = build_embed("2026-03-07", "7:30 PM ET", [], 3)
        assert embed["color"] == 0x36393F
        assert "Nothing meets conviction" in embed["description"]
        assert "0 picks" in embed["footer"]["text"]

    def test_embed_with_picks_has_color(self):
        picks = [self._make_pick()]
        embed = build_embed("2026-03-07", "7:30 PM ET", picks, 4)
        # Active picks use green (0x2ECC71), not the empty gray
        assert embed["color"] != 0x36393F

    def test_embed_has_fields(self):
        picks = [self._make_pick(), self._make_pick(player_name="Jayson Tatum")]
        embed = build_embed("2026-03-07", "7:30 PM ET", picks, 4)
        # Fields are grouped by stat type, not 1:1 with picks
        assert len(embed["fields"]) >= 1

    def test_field_name_contains_stat(self):
        picks = [self._make_pick()]
        embed = build_embed("2026-03-07", "7:00 PM ET", picks, 2)
        name = embed["fields"][0]["name"]
        assert "PTS" in name

    def test_footer_contains_pick_count(self):
        picks = [self._make_pick()]
        embed = build_embed("2026-03-07", "7:00 PM ET", picks, 3)
        assert "1 pick" in embed["footer"]["text"]

    def test_footer_contains_run_number(self):
        picks = [self._make_pick()]
        embed = build_embed("2026-03-07", "7:00 PM ET", picks, 5)
        assert "Run 5 of 7" in embed["footer"]["text"]

    def test_description_contains_tip_time(self):
        picks = [self._make_pick()]
        embed = build_embed("2026-03-07", "8:00 PM ET", picks, 3)
        assert "8:00 PM ET" in embed["description"]

    def test_description_contains_locked_count(self):
        picks = [self._make_pick(conviction_label="LOCKED")]
        embed = build_embed("2026-03-07", "7:00 PM ET", picks, 3)
        assert "locked" in embed["description"].lower()

    def test_stat_abbreviation_rebounds(self):
        picks = [self._make_pick(stat_type="REBOUNDS")]
        embed = build_embed("2026-03-07", "7:00 PM ET", picks, 3)
        # REB should appear in either fields or description
        full_text = str(embed)
        assert "REB" in full_text

    def test_date_formatting(self):
        embed = build_embed("2026-03-07", "7:00 PM ET", [], 1)
        assert "Mar 7, 2026" in embed["description"]

    def test_empty_picks_footer_has_run(self):
        embed = build_embed("2026-03-07", "7:00 PM ET", [], 2)
        assert "Run 2 of 7" in embed["footer"]["text"]


# ─────────────────────────────────────────────────────────────────
# _get_bot_token
# ─────────────────────────────────────────────────────────────────


class TestGetBotToken:
    def test_returns_env_var_when_set(self):
        with patch.dict(os.environ, {"AXIOM_BOT_TOKEN": "test-token-123"}):
            assert _get_bot_token() == "test-token-123"

    def test_returns_none_when_no_env_and_no_file(self):
        env = {k: v for k, v in os.environ.items() if k != "AXIOM_BOT_TOKEN"}
        with patch.dict(os.environ, env, clear=True):
            with patch("builtins.open", side_effect=OSError("not found")):
                result = _get_bot_token()
                assert result is None

    def test_reads_token_from_env_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write('AXIOM_BOT_TOKEN="file-token-456"\n')
            f.write("OTHER_VAR=something\n")
            tmp_path = f.name

        env = {k: v for k, v in os.environ.items() if k != "AXIOM_BOT_TOKEN"}
        with patch.dict(os.environ, env, clear=True):
            with patch("nba.core.daily_card.os.path.join", return_value=tmp_path):
                result = _get_bot_token()
                assert result == "file-token-456"

        os.unlink(tmp_path)

    def test_env_var_takes_priority_over_file(self):
        with patch.dict(os.environ, {"AXIOM_BOT_TOKEN": "env-wins"}):
            # File should never be opened if env var is set
            with patch("builtins.open") as mock_open:
                result = _get_bot_token()
                assert result == "env-wins"
                mock_open.assert_not_called()


# ─────────────────────────────────────────────────────────────────
# _already_sent_today
# ─────────────────────────────────────────────────────────────────


class TestAlreadySentToday:
    def test_returns_true_when_post_exists(self):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = lambda s: mock_cursor
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchone.return_value = (1,)
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.close = MagicMock()

        with patch("nba.core.daily_card._connect", return_value=mock_conn):
            assert _already_sent_today("2026-03-07") is True

    def test_returns_false_when_no_post(self):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = lambda s: mock_cursor
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchone.return_value = (0,)
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.close = MagicMock()

        with patch("nba.core.daily_card._connect", return_value=mock_conn):
            assert _already_sent_today("2026-03-07") is False

    def test_returns_false_on_db_error(self):
        with patch("nba.core.daily_card._connect", side_effect=Exception("conn failed")):
            assert _already_sent_today("2026-03-07") is False


# ─────────────────────────────────────────────────────────────────
# send_daily_card
# ─────────────────────────────────────────────────────────────────


class TestSendDailyCard:
    def _mock_pick(self):
        return {
            "player_name": "Giannis Antetokounmpo",
            "stat_type": "POINTS",
            "conviction": 0.88,
            "conviction_label": "LOCKED",
            "appearances": 5,
            "total_runs": 6,
            "line_at_entry": 30.5,
            "line_latest": 29.5,
            "line_direction": "falling",
            "p_over_at_entry": 0.75,
            "p_over_latest": 0.80,
            "p_over_trend": 0.05,
            "book_latest": "draftkings",
            "context_snapshot": {},
        }

    def test_already_sent_returns_early(self):
        with patch("nba.core.daily_card._already_sent_today", return_value=True):
            result = send_daily_card("2026-03-07")
        assert result["sent"] is False
        assert result["reason"] == "already_sent"

    def test_force_bypasses_dedup(self):
        with (
            patch("nba.core.daily_card._already_sent_today", return_value=True),
            patch("nba.core.daily_card._load_conviction_picks", return_value=([], 3)),
            patch("nba.core.daily_card.get_first_tip_time", return_value="7:00 PM ET"),
            patch("nba.core.daily_card._post_dm_embed", return_value="msg-123"),
            patch("nba.core.daily_card._record_post"),
        ):
            result = send_daily_card("2026-03-07", force=True)
        assert result["sent"] is True

    def test_discord_error_returns_not_sent(self):
        with (
            patch("nba.core.daily_card._already_sent_today", return_value=False),
            patch(
                "nba.core.daily_card._load_conviction_picks",
                return_value=([self._mock_pick()], 4),
            ),
            patch("nba.core.daily_card.get_first_tip_time", return_value="7:30 PM ET"),
            patch("nba.core.daily_card._post_dm_embed", return_value=None),
        ):
            result = send_daily_card("2026-03-07")
        assert result["sent"] is False
        assert result["reason"] == "discord_error"
        assert result["picks_count"] == 1

    def test_success_returns_message_id(self):
        with (
            patch("nba.core.daily_card._already_sent_today", return_value=False),
            patch(
                "nba.core.daily_card._load_conviction_picks",
                return_value=([self._mock_pick()], 5),
            ),
            patch("nba.core.daily_card.get_first_tip_time", return_value="7:00 PM ET"),
            patch("nba.core.daily_card._post_dm_embed", return_value="msg-999"),
            patch("nba.core.daily_card._record_post"),
        ):
            result = send_daily_card("2026-03-07", trigger="airflow")
        assert result["sent"] is True
        assert result["message_id"] == "msg-999"
        assert result["picks_count"] == 1
        assert result["reason"] == "ok"

    def test_db_error_on_load_returns_not_sent(self):
        with (
            patch("nba.core.daily_card._already_sent_today", return_value=False),
            patch(
                "nba.core.daily_card._load_conviction_picks",
                side_effect=Exception("db down"),
            ),
        ):
            result = send_daily_card("2026-03-07")
        assert result["sent"] is False
        assert "db_error" in result["reason"]

    def test_empty_picks_still_sends(self):
        with (
            patch("nba.core.daily_card._already_sent_today", return_value=False),
            patch("nba.core.daily_card._load_conviction_picks", return_value=([], 2)),
            patch("nba.core.daily_card.get_first_tip_time", return_value="7:00 PM ET"),
            patch("nba.core.daily_card._post_dm_embed", return_value="msg-000"),
            patch("nba.core.daily_card._record_post"),
        ):
            result = send_daily_card("2026-03-07")
        assert result["sent"] is True
        assert result["picks_count"] == 0
