"""
Unit tests for nba.core.daily_card — formatting functions and orchestration.

DB and Discord API calls are mocked throughout.
"""

import json
import os
import tempfile
import urllib.error
from io import BytesIO
from unittest.mock import MagicMock, call, patch

import pytest

from nba.core.daily_card import (
    _already_sent_today,
    _axiom_note,
    _bp_signal,
    _bp_stars,
    _conviction_color,
    _fmt_line,
    _fmt_p_over,
    _get_bot_token,
    _load_conviction_picks,
    _post_dm_embed,
    _record_post,
    build_embed,
    get_first_tip_time,
    send_daily_card,
)

# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────


def _make_pick(**overrides):
    """Create a realistic pick dict with sensible defaults."""
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

    def test_string_inputs(self):
        """_fmt_line casts to float, so string inputs should work."""
        result = _fmt_line("24.5", "23.0", "falling")
        assert "dropped" in result
        assert "24.5" in result
        assert "23.0" in result


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

    def test_negative_boundary(self):
        result = _fmt_p_over(0.705, 0.70, -0.005)
        assert "steady" not in result
        assert "fading" in result

    def test_zero_trend_stable(self):
        result = _fmt_p_over(0.80, 0.80, 0.0)
        assert "steady" in result


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
        assert "BP" in result

    def test_rating_with_side(self):
        result = _bp_stars({"bp_bet_rating": 4, "bp_recommended_side": "over"})
        assert "over" in result

    def test_rating_without_side(self):
        result = _bp_stars({"bp_bet_rating": 5})
        assert "BP" in result
        assert "()" not in result

    def test_five_stars(self):
        result = _bp_stars({"bp_bet_rating": 5})
        star_char = "\u2b50"
        assert result.count(star_char) == 5


# ─────────────────────────────────────────────────────────────────
# _bp_signal
# ─────────────────────────────────────────────────────────────────


class TestBpSignal:
    def test_no_rating_returns_empty(self):
        assert _bp_signal({}) == ""
        assert _bp_signal({"bp_bet_rating": None}) == ""

    def test_five_star_over_sharp_alignment(self):
        result = _bp_signal({"bp_bet_rating": 5, "bp_recommended_side": "over"})
        assert "sharp alignment" in result
        assert "5/5" in result

    def test_four_star_over_confirms(self):
        result = _bp_signal({"bp_bet_rating": 4, "bp_recommended_side": "over"})
        assert "confirms" in result
        assert "4/5" in result

    def test_three_star_over_confirms(self):
        result = _bp_signal({"bp_bet_rating": 3, "bp_recommended_side": "over"})
        assert "confirms" in result
        assert "3/5" in result

    def test_under_side_contrarian(self):
        result = _bp_signal({"bp_bet_rating": 4, "bp_recommended_side": "under"})
        assert "contrarian" in result
        assert "4/5" in result

    def test_under_side_case_insensitive(self):
        result = _bp_signal({"bp_bet_rating": 3, "bp_recommended_side": "UNDER"})
        assert "contrarian" in result.lower() or "leans other way" in result

    def test_no_side_just_rating(self):
        result = _bp_signal({"bp_bet_rating": 3, "bp_recommended_side": ""})
        assert "BP 3/5" in result
        assert "confirms" not in result
        assert "contrarian" not in result

    def test_none_side_just_rating(self):
        result = _bp_signal({"bp_bet_rating": 4})
        assert "BP 4/5" in result

    def test_zero_rating_returns_empty(self):
        # int(0) is falsy
        assert _bp_signal({"bp_bet_rating": 0}) == ""


# ─────────────────────────────────────────────────────────────────
# _conviction_color
# ─────────────────────────────────────────────────────────────────


class TestConvictionColor:
    def test_empty_picks_dark_gray(self):
        assert _conviction_color([]) == 0x36393F

    def test_elite_avg_gold(self):
        picks = [{"conviction": 0.95}, {"conviction": 0.92}]
        # avg = 0.935 >= 0.90
        assert _conviction_color(picks) == 0xFFD700

    def test_strong_avg_green(self):
        picks = [{"conviction": 0.87}, {"conviction": 0.86}]
        # avg = 0.865 >= 0.85
        assert _conviction_color(picks) == 0x2ECC71

    def test_standard_avg_blue(self):
        picks = [{"conviction": 0.80}, {"conviction": 0.82}]
        # avg = 0.81 < 0.85
        assert _conviction_color(picks) == 0x3498DB

    def test_boundary_at_090(self):
        picks = [{"conviction": 0.90}]
        assert _conviction_color(picks) == 0xFFD700

    def test_boundary_at_085(self):
        picks = [{"conviction": 0.85}]
        assert _conviction_color(picks) == 0x2ECC71

    def test_just_below_085(self):
        picks = [{"conviction": 0.849}]
        assert _conviction_color(picks) == 0x3498DB


# ─────────────────────────────────────────────────────────────────
# _axiom_note
# ─────────────────────────────────────────────────────────────────


class TestAxiomNote:
    def test_top_play_of_slate(self):
        pick = _make_pick(conviction=0.95, p_over_trend=0.0)
        result = _axiom_note(pick, rank=1, total=5)
        assert "Top play of the slate" in result

    def test_elite_conviction_not_rank_1(self):
        pick = _make_pick(conviction=0.96, p_over_trend=0.0)
        result = _axiom_note(pick, rank=2, total=5)
        assert "Elite conviction" in result

    def test_both_models_agree(self):
        pick = _make_pick(
            context_snapshot={"models_agreeing": ["xl", "v3"]},
            p_over_trend=0.0,
        )
        result = _axiom_note(pick, rank=3, total=5)
        assert "both models agree" in result

    def test_picked_every_run(self):
        pick = _make_pick(appearances=4, total_runs=4, p_over_trend=0.0)
        result = _axiom_note(pick, rank=3, total=5)
        assert "picked every run" in result
        assert "4/4" in result

    def test_picked_every_run_requires_at_least_3(self):
        pick = _make_pick(appearances=2, total_runs=2, p_over_trend=0.0)
        result = _axiom_note(pick, rank=3, total=5)
        # 2 runs is not >= 3, so no "picked every run"
        assert "picked every run" not in result

    def test_multiple_appearances(self):
        pick = _make_pick(appearances=4, total_runs=6, p_over_trend=0.0)
        result = _axiom_note(pick, rank=3, total=5)
        assert "4/6 runs" in result

    def test_bp_sharp_money_confirms(self):
        pick = _make_pick(
            context_snapshot={"bp_bet_rating": 5, "bp_recommended_side": "over"},
            p_over_trend=0.0,
        )
        result = _axiom_note(pick, rank=3, total=5)
        assert "BP sharp money confirms" in result

    def test_bp_aligns(self):
        pick = _make_pick(
            context_snapshot={"bp_bet_rating": 4, "bp_recommended_side": "over"},
            p_over_trend=0.0,
        )
        result = _axiom_note(pick, rank=3, total=5)
        assert "BP aligns" in result

    def test_bp_contrarian(self):
        pick = _make_pick(
            context_snapshot={"bp_bet_rating": 3, "bp_recommended_side": "under"},
            p_over_trend=0.0,
        )
        result = _axiom_note(pick, rank=3, total=5)
        assert "contrarian vs BP" in result

    def test_probability_rising(self):
        pick = _make_pick(p_over_trend=0.05)
        result = _axiom_note(pick, rank=3, total=5)
        assert "probability rising" in result

    def test_probability_fading(self):
        pick = _make_pick(p_over_trend=-0.05)
        result = _axiom_note(pick, rank=3, total=5)
        assert "slight fade across runs" in result

    def test_line_falling(self):
        pick = _make_pick(line_direction="falling", p_over_trend=0.0)
        result = _axiom_note(pick, rank=3, total=5)
        assert "line moved our way" in result

    def test_line_rising(self):
        pick = _make_pick(line_direction="rising", p_over_trend=0.0)
        result = _axiom_note(pick, rank=3, total=5)
        assert "line moved against" in result

    def test_empty_notes_fallback(self):
        """When no signals trigger, falls back to appearances/total."""
        pick = _make_pick(
            conviction=0.80,
            appearances=2,
            total_runs=6,
            p_over_trend=0.0,
            line_direction="stable",
            context_snapshot={},
        )
        result = _axiom_note(pick, rank=3, total=5)
        assert "2/6 runs" in result

    def test_max_three_notes(self):
        """Notes are truncated to 3 items joined by ' - '."""
        pick = _make_pick(
            conviction=0.96,
            appearances=5,
            total_runs=5,
            p_over_trend=0.05,
            line_direction="falling",
            context_snapshot={
                "models_agreeing": ["xl", "v3"],
                "bp_bet_rating": 5,
                "bp_recommended_side": "over",
            },
        )
        result = _axiom_note(pick, rank=1, total=5)
        # Should have at most 3 segments separated by " - "
        assert result.count(" \u2014 ") <= 2  # em-dash from join

    def test_none_p_over_trend(self):
        pick = _make_pick(p_over_trend=None, line_direction="stable")
        result = _axiom_note(pick, rank=3, total=5)
        # Should not crash; trend treated as 0.0
        assert isinstance(result, str)

    def test_missing_context_snapshot(self):
        pick = _make_pick(context_snapshot=None, p_over_trend=0.0)
        result = _axiom_note(pick, rank=3, total=5)
        assert isinstance(result, str)


# ─────────────────────────────────────────────────────────────────
# get_first_tip_time
# ─────────────────────────────────────────────────────────────────


class TestGetFirstTipTime:
    def test_returns_tip_time_from_espn(self):
        mock_data = {
            "events": [
                {"date": "2026-03-17T23:30:00Z"},
                {"date": "2026-03-18T00:00:00Z"},
            ]
        }
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(mock_data).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("nba.core.daily_card.urllib.request.urlopen", return_value=mock_response):
            result = get_first_tip_time("2026-03-17")
        assert "PM ET" in result or "AM ET" in result

    def test_returns_fallback_on_no_events(self):
        mock_data = {"events": []}
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(mock_data).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("nba.core.daily_card.urllib.request.urlopen", return_value=mock_response):
            result = get_first_tip_time("2026-03-17")
        assert result == "7:00 PM ET"

    def test_returns_fallback_on_exception(self):
        with patch(
            "nba.core.daily_card.urllib.request.urlopen",
            side_effect=urllib.error.URLError("timeout"),
        ):
            result = get_first_tip_time("2026-03-17")
        assert result == "7:00 PM ET"

    def test_returns_fallback_on_missing_date_field(self):
        mock_data = {"events": [{"date": ""}]}
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(mock_data).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("nba.core.daily_card.urllib.request.urlopen", return_value=mock_response):
            result = get_first_tip_time("2026-03-17")
        # Empty date string is falsy, so times list is empty -> fallback
        assert result == "7:00 PM ET"

    def test_picks_earliest_game(self):
        mock_data = {
            "events": [
                {"date": "2026-03-18T01:00:00Z"},  # later
                {"date": "2026-03-17T23:00:00Z"},  # earlier
            ]
        }
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(mock_data).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("nba.core.daily_card.urllib.request.urlopen", return_value=mock_response):
            result = get_first_tip_time("2026-03-17")
        # 23:00 UTC = 7:00 PM ET (or 6:00 PM depending on DST)
        assert "PM ET" in result


# ─────────────────────────────────────────────────────────────────
# _load_conviction_picks
# ─────────────────────────────────────────────────────────────────


class TestLoadConvictionPicks:
    def test_returns_picks_and_max_run(self):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = lambda s: mock_cursor
        mock_cursor.__exit__ = MagicMock(return_value=False)

        # First query: conviction picks
        # Second query: max run number
        mock_cursor.fetchall.return_value = [
            (
                "LeBron James",
                "POINTS",
                0.90,
                "LOCKED",
                5,
                6,
                25.5,
                24.5,
                "falling",
                0.72,
                0.78,
                0.06,
                "draftkings",
                None,
            )
        ]
        mock_cursor.description = [
            ("player_name",),
            ("stat_type",),
            ("conviction",),
            ("conviction_label",),
            ("appearances",),
            ("total_runs",),
            ("line_at_entry",),
            ("line_latest",),
            ("line_direction",),
            ("p_over_at_entry",),
            ("p_over_latest",),
            ("p_over_trend",),
            ("book_latest",),
            ("context_snapshot",),
        ]
        mock_cursor.fetchone.return_value = (4,)
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.close = MagicMock()

        with patch("nba.core.daily_card._connect", return_value=mock_conn):
            picks, max_run = _load_conviction_picks("2026-03-17")

        assert len(picks) == 1
        assert picks[0]["player_name"] == "LeBron James"
        assert picks[0]["conviction"] == 0.90
        assert max_run == 4
        mock_conn.close.assert_called_once()

    def test_returns_empty_when_no_picks(self):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = lambda s: mock_cursor
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchall.return_value = []
        mock_cursor.description = [
            ("player_name",),
            ("stat_type",),
            ("conviction",),
            ("conviction_label",),
            ("appearances",),
            ("total_runs",),
            ("line_at_entry",),
            ("line_latest",),
            ("line_direction",),
            ("p_over_at_entry",),
            ("p_over_latest",),
            ("p_over_trend",),
            ("book_latest",),
            ("context_snapshot",),
        ]
        mock_cursor.fetchone.return_value = (1,)
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.close = MagicMock()

        with patch("nba.core.daily_card._connect", return_value=mock_conn):
            picks, max_run = _load_conviction_picks("2026-03-17")

        assert picks == []
        assert max_run == 1

    def test_connection_closed_on_error(self):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = lambda s: mock_cursor
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchall.side_effect = Exception("query failed")
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.close = MagicMock()

        with patch("nba.core.daily_card._connect", return_value=mock_conn):
            with pytest.raises(Exception, match="query failed"):
                _load_conviction_picks("2026-03-17")

        mock_conn.close.assert_called_once()


# ─────────────────────────────────────────────────────────────────
# _post_dm_embed
# ─────────────────────────────────────────────────────────────────


class TestPostDmEmbed:
    def test_returns_none_when_no_token(self):
        with patch("nba.core.daily_card._get_bot_token", return_value=None):
            result = _post_dm_embed({"title": "test"})
        assert result is None

    def test_successful_dm_returns_message_id(self):
        # Mock the two urlopen calls: create DM channel + send message
        dm_response = MagicMock()
        dm_response.read.return_value = json.dumps({"id": "channel-123"}).encode()
        dm_response.__enter__ = lambda s: s
        dm_response.__exit__ = MagicMock(return_value=False)

        msg_response = MagicMock()
        msg_response.read.return_value = json.dumps({"id": "msg-456"}).encode()
        msg_response.__enter__ = lambda s: s
        msg_response.__exit__ = MagicMock(return_value=False)

        with patch("nba.core.daily_card._get_bot_token", return_value="bot-token"):
            with patch.dict(os.environ, {"DISCORD_OWNER_ID": "12345"}):
                with patch(
                    "nba.core.daily_card.urllib.request.urlopen",
                    side_effect=[dm_response, msg_response],
                ):
                    result = _post_dm_embed({"title": "test embed"})

        assert result == "msg-456"

    def test_http_error_returns_none(self):
        dm_response = MagicMock()
        dm_response.read.return_value = json.dumps({"id": "channel-123"}).encode()
        dm_response.__enter__ = lambda s: s
        dm_response.__exit__ = MagicMock(return_value=False)

        http_err = urllib.error.HTTPError(
            url="http://discord.com", code=403, msg="Forbidden", hdrs={}, fp=BytesIO(b"forbidden")
        )

        with patch("nba.core.daily_card._get_bot_token", return_value="bot-token"):
            with patch.dict(os.environ, {"DISCORD_OWNER_ID": "12345"}):
                with patch(
                    "nba.core.daily_card.urllib.request.urlopen",
                    side_effect=[dm_response, http_err],
                ):
                    result = _post_dm_embed({"title": "test"})

        assert result is None

    def test_generic_exception_returns_none(self):
        with patch("nba.core.daily_card._get_bot_token", return_value="bot-token"):
            with patch.dict(os.environ, {"DISCORD_OWNER_ID": "12345"}):
                with patch(
                    "nba.core.daily_card.urllib.request.urlopen",
                    side_effect=ConnectionError("no connection"),
                ):
                    result = _post_dm_embed({"title": "test"})

        assert result is None

    def test_http_error_on_dm_channel_creation(self):
        http_err = urllib.error.HTTPError(
            url="http://discord.com", code=401, msg="Unauthorized", hdrs={}, fp=BytesIO(b"unauth")
        )

        with patch("nba.core.daily_card._get_bot_token", return_value="bot-token"):
            with patch.dict(os.environ, {"DISCORD_OWNER_ID": "12345"}):
                with patch(
                    "nba.core.daily_card.urllib.request.urlopen",
                    side_effect=http_err,
                ):
                    result = _post_dm_embed({"title": "test"})

        assert result is None


# ─────────────────────────────────────────────────────────────────
# _record_post
# ─────────────────────────────────────────────────────────────────


class TestRecordPost:
    def test_successful_recording(self):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = lambda s: mock_cursor
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = lambda s: s
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.close = MagicMock()

        picks = [
            {
                "player_name": "LeBron",
                "stat_type": "POINTS",
                "conviction": 0.90,
                "conviction_label": "LOCKED",
            },
        ]

        with patch("nba.core.daily_card._connect", return_value=mock_conn):
            _record_post("2026-03-17", picks, "msg-123", "airflow")

        mock_cursor.execute.assert_called_once()
        # The first arg to execute call
        call_args = mock_cursor.execute.call_args
        assert "INSERT INTO axiom_posts" in call_args[0][0]
        mock_conn.close.assert_called_once()

    def test_db_error_does_not_raise(self):
        """_record_post swallows exceptions (non-critical)."""
        with patch("nba.core.daily_card._connect", side_effect=Exception("db down")):
            # Should not raise
            _record_post("2026-03-17", [], None, "manual")

    def test_records_picks_as_json(self):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = lambda s: mock_cursor
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = lambda s: s
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.close = MagicMock()

        picks = [
            {
                "player_name": "Giannis",
                "stat_type": "REBOUNDS",
                "conviction": 0.88,
                "conviction_label": "STRONG",
            },
        ]

        with patch("nba.core.daily_card._connect", return_value=mock_conn):
            _record_post("2026-03-17", picks, "msg-999", "discord")

        call_args = mock_cursor.execute.call_args
        params = call_args[0][1]
        # params is (run_date, trigger, picks_json, message_id)
        assert params[0] == "2026-03-17"
        assert params[1] == "discord"
        picks_json = json.loads(params[2])
        assert picks_json[0]["player_name"] == "Giannis"
        assert params[3] == "msg-999"


# ─────────────────────────────────────────────────────────────────
# build_embed — rich fields and context coverage
# ─────────────────────────────────────────────────────────────────


class TestBuildEmbed:
    def test_empty_picks_returns_no_picks_embed(self):
        embed = build_embed("2026-03-07", "7:30 PM ET", [], 3)
        assert embed["color"] == 0x36393F
        assert "Nothing meets conviction" in embed["description"]
        assert "0 picks" in embed["footer"]["text"]

    def test_embed_with_picks_has_color(self):
        picks = [_make_pick()]
        embed = build_embed("2026-03-07", "7:30 PM ET", picks, 4)
        assert embed["color"] != 0x36393F

    def test_embed_has_fields(self):
        picks = [_make_pick(), _make_pick(player_name="Jayson Tatum")]
        embed = build_embed("2026-03-07", "7:30 PM ET", picks, 4)
        assert len(embed["fields"]) >= 1

    def test_field_name_contains_stat(self):
        picks = [_make_pick()]
        embed = build_embed("2026-03-07", "7:00 PM ET", picks, 2)
        # First field is market separator
        name = embed["fields"][0]["name"]
        assert "PTS" in name

    def test_footer_contains_pick_count(self):
        picks = [_make_pick()]
        embed = build_embed("2026-03-07", "7:00 PM ET", picks, 3)
        assert "1 pick" in embed["footer"]["text"]

    def test_footer_contains_run_number(self):
        picks = [_make_pick()]
        embed = build_embed("2026-03-07", "7:00 PM ET", picks, 5)
        assert "Run 5 of 7" in embed["footer"]["text"]

    def test_description_contains_tip_time(self):
        picks = [_make_pick()]
        embed = build_embed("2026-03-07", "8:00 PM ET", picks, 3)
        assert "8:00 PM ET" in embed["description"]

    def test_description_contains_locked_count(self):
        picks = [_make_pick(conviction_label="LOCKED")]
        embed = build_embed("2026-03-07", "7:00 PM ET", picks, 3)
        assert "locked" in embed["description"].lower()

    def test_description_contains_strong_count(self):
        picks = [_make_pick(conviction_label="STRONG")]
        embed = build_embed("2026-03-07", "7:00 PM ET", picks, 3)
        assert "strong" in embed["description"].lower()

    def test_description_both_locked_and_strong(self):
        picks = [
            _make_pick(conviction_label="LOCKED"),
            _make_pick(conviction_label="STRONG", player_name="Jayson Tatum"),
        ]
        embed = build_embed("2026-03-07", "7:00 PM ET", picks, 3)
        assert "locked" in embed["description"].lower()
        assert "strong" in embed["description"].lower()

    def test_stat_abbreviation_rebounds(self):
        picks = [_make_pick(stat_type="REBOUNDS")]
        embed = build_embed("2026-03-07", "7:00 PM ET", picks, 3)
        full_text = str(embed)
        assert "REB" in full_text

    def test_stat_abbreviation_unknown_market(self):
        """Unknown markets use first 3 chars of the market name."""
        picks = [_make_pick(stat_type="STEALS")]
        embed = build_embed("2026-03-07", "7:00 PM ET", picks, 3)
        full_text = str(embed)
        assert "STL" in full_text

    def test_date_formatting(self):
        embed = build_embed("2026-03-07", "7:00 PM ET", [], 1)
        assert "Mar 7, 2026" in embed["description"]

    def test_empty_picks_footer_has_run(self):
        embed = build_embed("2026-03-07", "7:00 PM ET", [], 2)
        assert "Run 2 of 7" in embed["footer"]["text"]

    def test_models_agreeing_in_description(self):
        picks = [
            _make_pick(context_snapshot={"models_agreeing": ["xl", "v3"]}),
        ]
        embed = build_embed("2026-03-07", "7:00 PM ET", picks, 3)
        assert "V3" in embed["description"] or "XL" in embed["description"]

    def test_models_default_to_xl(self):
        picks = [_make_pick(context_snapshot={})]
        embed = build_embed("2026-03-07", "7:00 PM ET", picks, 3)
        assert "XL" in embed["description"]

    def test_book_display_name(self):
        picks = [_make_pick(book_latest="fanduel")]
        embed = build_embed("2026-03-07", "7:00 PM ET", picks, 3)
        full_text = str(embed)
        assert "FanDuel" in full_text

    def test_unknown_book_uses_raw_name(self):
        picks = [_make_pick(book_latest="some_new_book")]
        embed = build_embed("2026-03-07", "7:00 PM ET", picks, 3)
        full_text = str(embed)
        assert "some_new_book" in full_text

    def test_multiple_markets_ordering(self):
        """Markets should be ordered: POINTS, REBOUNDS, etc."""
        picks = [
            _make_pick(stat_type="REBOUNDS", player_name="Giannis"),
            _make_pick(stat_type="POINTS", player_name="LeBron"),
        ]
        embed = build_embed("2026-03-07", "7:00 PM ET", picks, 3)
        fields_text = " ".join(f["name"] for f in embed["fields"])
        pts_idx = fields_text.index("PTS")
        reb_idx = fields_text.index("REB")
        assert pts_idx < reb_idx

    def test_player_context_avg_l5_l10(self):
        picks = [
            _make_pick(
                context_snapshot={
                    "player_context": {"avg_L5": 28.2, "avg_L10": 26.5},
                }
            )
        ]
        embed = build_embed("2026-03-07", "7:00 PM ET", picks, 3)
        full_text = str(embed)
        assert "28.2" in full_text
        assert "26.5" in full_text

    def test_player_context_h2h(self):
        picks = [
            _make_pick(
                context_snapshot={
                    "opponent_team": "LAL",
                    "player_context": {"h2h_avg": 31.5, "h2h_games": 5},
                }
            )
        ]
        embed = build_embed("2026-03-07", "7:00 PM ET", picks, 3)
        full_text = str(embed)
        assert "vs LAL" in full_text
        assert "31.5" in full_text
        assert "5g" in full_text

    def test_player_context_h2h_too_few_games(self):
        """h2h shown only when >= 2 games. With fewer, show vs/@ opponent."""
        picks = [
            _make_pick(
                context_snapshot={
                    "opponent_team": "GSW",
                    "is_home": True,
                    "player_context": {"h2h_avg": 30.0, "h2h_games": 1},
                }
            )
        ]
        embed = build_embed("2026-03-07", "7:00 PM ET", picks, 3)
        full_text = str(embed)
        # Should show "vs GSW" not h2h
        assert "vs GSW" in full_text
        assert "1g" not in full_text

    def test_player_context_away_opponent(self):
        picks = [
            _make_pick(
                context_snapshot={
                    "opponent_team": "BOS",
                    "is_home": False,
                    "player_context": {},
                }
            )
        ]
        embed = build_embed("2026-03-07", "7:00 PM ET", picks, 3)
        full_text = str(embed)
        assert "@ BOS" in full_text

    def test_player_context_is_home_none(self):
        """When is_home is None, just show 'vs OPP'."""
        picks = [
            _make_pick(
                context_snapshot={
                    "opponent_team": "MIA",
                    "is_home": None,
                    "player_context": {},
                }
            )
        ]
        embed = build_embed("2026-03-07", "7:00 PM ET", picks, 3)
        full_text = str(embed)
        assert "vs MIA" in full_text

    def test_player_context_trend_label(self):
        picks = [
            _make_pick(
                context_snapshot={
                    "player_context": {"trend": "HOT"},
                }
            )
        ]
        embed = build_embed("2026-03-07", "7:00 PM ET", picks, 3)
        full_text = str(embed)
        assert "HOT" in full_text

    def test_player_context_stable_trend_not_shown(self):
        picks = [
            _make_pick(
                context_snapshot={
                    "player_context": {"trend": "STABLE"},
                }
            )
        ]
        embed = build_embed("2026-03-07", "7:00 PM ET", picks, 3)
        # STABLE should be filtered out
        full_text = str(embed)
        assert "STABLE" not in full_text

    def test_projection_and_edge_shown(self):
        picks = [
            _make_pick(
                context_snapshot={
                    "prediction": 28.5,
                    "edge": 4.0,
                }
            )
        ]
        embed = build_embed("2026-03-07", "7:00 PM ET", picks, 3)
        full_text = str(embed)
        assert "28.5" in full_text
        assert "edge" in full_text.lower()

    def test_no_projection_shows_probability_line(self):
        picks = [
            _make_pick(
                context_snapshot={},
                p_over_at_entry=0.72,
                p_over_latest=0.78,
                p_over_trend=0.06,
                line_direction="falling",
            )
        ]
        embed = build_embed("2026-03-07", "7:00 PM ET", picks, 3)
        full_text = str(embed)
        # Should contain probability info from _fmt_p_over and line info from _fmt_line
        assert "rising" in full_text or "Probability" in full_text

    def test_hit_rates_displayed(self):
        picks = [
            _make_pick(
                context_snapshot={
                    "hit_rate_L5": 0.80,
                    "hit_rate_season": 0.65,
                }
            )
        ]
        embed = build_embed("2026-03-07", "7:00 PM ET", picks, 3)
        full_text = str(embed)
        assert "80%" in full_text
        assert "65%" in full_text

    def test_opp_defense_rank_ordinal_suffix(self):
        picks = [
            _make_pick(
                context_snapshot={
                    "opposition_rank": 3,
                }
            )
        ]
        embed = build_embed("2026-03-07", "7:00 PM ET", picks, 3)
        full_text = str(embed)
        assert "3rd" in full_text

    def test_opp_defense_rank_11th_12th_13th(self):
        """11th, 12th, 13th use 'th' not 'st'/'nd'/'rd'."""
        for rank, expected in [(11, "11th"), (12, "12th"), (13, "13th")]:
            picks = [_make_pick(context_snapshot={"opposition_rank": rank})]
            embed = build_embed("2026-03-07", "7:00 PM ET", picks, 3)
            full_text = str(embed)
            assert expected in full_text, f"Expected {expected} for rank {rank}"

    def test_opp_defense_rank_1st_2nd(self):
        picks = [_make_pick(context_snapshot={"opposition_rank": 1})]
        embed = build_embed("2026-03-07", "7:00 PM ET", picks, 3)
        assert "1st" in str(embed)

        picks = [_make_pick(context_snapshot={"opposition_rank": 2})]
        embed = build_embed("2026-03-07", "7:00 PM ET", picks, 3)
        assert "2nd" in str(embed)

    def test_opp_defense_rank_21st(self):
        picks = [_make_pick(context_snapshot={"opposition_rank": 21})]
        embed = build_embed("2026-03-07", "7:00 PM ET", picks, 3)
        assert "21st" in str(embed)

    def test_opp_rank_fallback_key(self):
        """Tests opp_rank as fallback when opposition_rank is missing."""
        picks = [_make_pick(context_snapshot={"opp_rank": 5})]
        embed = build_embed("2026-03-07", "7:00 PM ET", picks, 3)
        assert "5th" in str(embed)

    def test_bp_signal_in_field(self):
        picks = [
            _make_pick(
                context_snapshot={
                    "bp_bet_rating": 5,
                    "bp_recommended_side": "over",
                }
            )
        ]
        embed = build_embed("2026-03-07", "7:00 PM ET", picks, 3)
        full_text = str(embed)
        assert "sharp alignment" in full_text

    def test_axiom_analysis_in_field(self):
        picks = [_make_pick(conviction=0.95)]
        embed = build_embed("2026-03-07", "7:00 PM ET", picks, 1)
        full_text = str(embed)
        # Axiom note line starts with triangular bullet
        assert "\u25b8" in full_text

    def test_strong_label_tag(self):
        picks = [_make_pick(conviction_label="STRONG")]
        embed = build_embed("2026-03-07", "7:00 PM ET", picks, 3)
        full_text = str(embed)
        assert "[STRONG]" in full_text

    def test_locked_label_tag(self):
        picks = [_make_pick(conviction_label="LOCKED")]
        embed = build_embed("2026-03-07", "7:00 PM ET", picks, 3)
        full_text = str(embed)
        assert "[LOCKED]" in full_text

    def test_market_separator_fields(self):
        """Each market gets a separator field with count."""
        picks = [
            _make_pick(stat_type="POINTS"),
            _make_pick(stat_type="REBOUNDS", player_name="Giannis"),
        ]
        embed = build_embed("2026-03-07", "7:00 PM ET", picks, 3)
        sep_fields = [f for f in embed["fields"] if "\u2500" in f["name"]]
        assert len(sep_fields) == 2

    def test_no_picks_for_a_market_skipped(self):
        """Markets with no picks should not produce separator fields."""
        picks = [_make_pick(stat_type="POINTS")]
        embed = build_embed("2026-03-07", "7:00 PM ET", picks, 3)
        sep_fields = [f for f in embed["fields"] if "\u2500" in f["name"]]
        assert len(sep_fields) == 1
        assert "PTS" in sep_fields[0]["name"]

    def test_embed_timestamp_present(self):
        embed = build_embed("2026-03-07", "7:00 PM ET", [], 1)
        assert "timestamp" in embed


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
            with patch("builtins.open") as mock_open:
                result = _get_bot_token()
                assert result == "env-wins"
                mock_open.assert_not_called()

    def test_reads_unquoted_token(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("AXIOM_BOT_TOKEN=plain-token\n")
            tmp_path = f.name

        env = {k: v for k, v in os.environ.items() if k != "AXIOM_BOT_TOKEN"}
        with patch.dict(os.environ, env, clear=True):
            with patch("nba.core.daily_card.os.path.join", return_value=tmp_path):
                result = _get_bot_token()
                assert result == "plain-token"

        os.unlink(tmp_path)

    def test_reads_single_quoted_token(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("AXIOM_BOT_TOKEN='single-quoted'\n")
            tmp_path = f.name

        env = {k: v for k, v in os.environ.items() if k != "AXIOM_BOT_TOKEN"}
        with patch.dict(os.environ, env, clear=True):
            with patch("nba.core.daily_card.os.path.join", return_value=tmp_path):
                result = _get_bot_token()
                assert result == "single-quoted"

        os.unlink(tmp_path)

    def test_file_without_token_returns_none(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("SOME_OTHER_VAR=value\n")
            tmp_path = f.name

        env = {k: v for k, v in os.environ.items() if k != "AXIOM_BOT_TOKEN"}
        with patch.dict(os.environ, env, clear=True):
            with patch("nba.core.daily_card.os.path.join", return_value=tmp_path):
                result = _get_bot_token()
                assert result is None

        os.unlink(tmp_path)


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

    def test_record_post_called_on_success(self):
        with (
            patch("nba.core.daily_card._already_sent_today", return_value=False),
            patch(
                "nba.core.daily_card._load_conviction_picks",
                return_value=([self._mock_pick()], 3),
            ),
            patch("nba.core.daily_card.get_first_tip_time", return_value="7:00 PM ET"),
            patch("nba.core.daily_card._post_dm_embed", return_value="msg-abc"),
            patch("nba.core.daily_card._record_post") as mock_record,
        ):
            send_daily_card("2026-03-07", trigger="discord")
        mock_record.assert_called_once_with(
            "2026-03-07",
            [self._mock_pick()],
            "msg-abc",
            "discord",
        )

    def test_record_post_not_called_on_discord_failure(self):
        with (
            patch("nba.core.daily_card._already_sent_today", return_value=False),
            patch(
                "nba.core.daily_card._load_conviction_picks",
                return_value=([self._mock_pick()], 3),
            ),
            patch("nba.core.daily_card.get_first_tip_time", return_value="7:00 PM ET"),
            patch("nba.core.daily_card._post_dm_embed", return_value=None),
            patch("nba.core.daily_card._record_post") as mock_record,
        ):
            send_daily_card("2026-03-07")
        mock_record.assert_not_called()


# ─────────────────────────────────────────────────────────────────
# _connect (basic coverage)
# ─────────────────────────────────────────────────────────────────


class TestConnect:
    def test_connect_calls_psycopg2(self):
        mock_psycopg2 = MagicMock()
        mock_config = {
            "host": "testhost",
            "port": 5500,
            "dbname": "sportsuite",
            "user": "testuser",
            "password": "testpw",
        }
        with patch.dict("sys.modules", {"psycopg2": mock_psycopg2}):
            with patch("nba.core.daily_card.get_axiom_db_config", return_value=mock_config.copy()):
                from nba.core.daily_card import _connect

                _connect()
                call_kwargs = mock_psycopg2.connect.call_args[1]
                assert call_kwargs["host"] == "testhost"
                assert call_kwargs["connect_timeout"] == 5
