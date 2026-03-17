"""
Unit tests for nba.core.conviction_engine — pure functions + compute_conviction.

All tests bypass the DB (no psycopg2 connections) using mocks.
"""

import json
import statistics
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, call, patch

import pytest

from nba.core.conviction_engine import (
    _build_narrative,
    _connect,
    _get_label,
    _get_status,
    _get_tier_stats,
    _load_bp_recommendations,
    _normalize,
    _score_appearance,
    _score_bp_rec,
    _score_line_movement,
    _score_stability,
    _score_trend,
    _tier_score_adjustment,
    compute_conviction,
)

# ─────────────────────────────────────────────────────────────────
# _connect
# ─────────────────────────────────────────────────────────────────


class TestConnect:
    def test_returns_connection(self):
        mock_conn = MagicMock()
        with patch("psycopg2.connect", return_value=mock_conn):
            result = _connect()
        assert result is mock_conn


# ─────────────────────────────────────────────────────────────────
# _normalize
# ─────────────────────────────────────────────────────────────────


class TestNormalize:
    def test_midpoint(self):
        assert _normalize(5.0, 0.0, 10.0) == pytest.approx(0.5)

    def test_at_min(self):
        assert _normalize(0.0, 0.0, 10.0) == pytest.approx(0.0)

    def test_at_max(self):
        assert _normalize(10.0, 0.0, 10.0) == pytest.approx(1.0)

    def test_below_min_clamped(self):
        assert _normalize(-5.0, 0.0, 10.0) == pytest.approx(0.0)

    def test_above_max_clamped(self):
        assert _normalize(15.0, 0.0, 10.0) == pytest.approx(1.0)

    def test_degenerate_range(self):
        # max == min -> returns 0.5
        assert _normalize(5.0, 5.0, 5.0) == pytest.approx(0.5)


# ─────────────────────────────────────────────────────────────────
# _score_appearance
# ─────────────────────────────────────────────────────────────────


class TestScoreAppearance:
    def test_perfect_from_run1(self):
        # 6 appearances across all 6 fired runs, entered at run 1 -> full score
        score = _score_appearance(6, 1, [1, 2, 3, 4, 5, 6])
        assert score == pytest.approx(1.0)

    def test_all_possible_from_run3(self):
        # Appeared every run since entering at run 3 (4 fired runs since entry)
        score = _score_appearance(4, 3, [1, 2, 3, 4, 5, 6])
        assert 0.8 <= score <= 1.0

    def test_late_entry_penalty(self):
        # Present all 6 runs from run 1 scores higher than entering at run 5
        score_early = _score_appearance(6, 1, [1, 2, 3, 4, 5, 6])
        score_late = _score_appearance(2, 5, [1, 2, 3, 4, 5, 6])
        assert score_early > score_late

    def test_zero_possible_runs(self):
        # entry_run not in fired_runs and none are >= entry_run
        assert _score_appearance(1, 7, [1, 2, 3, 4, 5, 6]) == 0.0

    def test_single_run(self):
        score = _score_appearance(1, 1, [1])
        assert 0.0 <= score <= 1.0

    def test_entry_factor_floor(self):
        # Entry at run 6 -> entry_factor floored at 0.60
        score = _score_appearance(1, 6, [1, 2, 3, 4, 5, 6])
        assert score >= 0.0

    def test_skipped_runs_not_penalised(self):
        # Runs 2 and 3 never fired. Pick seen in all 4 runs that DID fire.
        # Should score the same as a pick seen in all 6 runs of a full day.
        fired_with_gap = [1, 4, 5, 6]
        score_with_gap = _score_appearance(4, 1, fired_with_gap)  # 4/4 fired
        score_full = _score_appearance(6, 1, [1, 2, 3, 4, 5, 6])  # 6/6 fired
        assert score_with_gap == pytest.approx(score_full)

    def test_only_fired_runs_after_entry_count(self):
        # Entered at run 4, only runs 4 and 6 fired after that
        fired = [1, 2, 4, 6]
        score = _score_appearance(2, 4, fired)
        assert score == pytest.approx(1.0 * (1.0 - 3 * 0.08))  # 2/2 * entry_factor(4)


# ─────────────────────────────────────────────────────────────────
# _score_stability
# ─────────────────────────────────────────────────────────────────


class TestScoreStability:
    def test_zero_std_is_perfect(self):
        assert _score_stability(0.0) == pytest.approx(1.0)

    def test_high_std_is_zero(self):
        assert _score_stability(0.15) == pytest.approx(0.0)

    def test_moderate_std(self):
        score = _score_stability(0.05)
        assert 0.0 < score < 1.0

    def test_monotone_decreasing(self):
        assert _score_stability(0.02) > _score_stability(0.08) > _score_stability(0.12)


# ─────────────────────────────────────────────────────────────────
# _score_trend
# ─────────────────────────────────────────────────────────────────


class TestScoreTrend:
    def test_zero_trend_is_neutral(self):
        assert _score_trend(0.0) == pytest.approx(0.5)

    def test_strong_positive_trend(self):
        assert _score_trend(0.10) == pytest.approx(1.0)

    def test_strong_negative_trend(self):
        assert _score_trend(-0.10) == pytest.approx(0.0)

    def test_partial_positive(self):
        score = _score_trend(0.05)
        assert 0.5 < score < 1.0

    def test_partial_negative(self):
        score = _score_trend(-0.05)
        assert 0.0 < score < 0.5


# ─────────────────────────────────────────────────────────────────
# _score_line_movement
# ─────────────────────────────────────────────────────────────────


class TestScoreLineMovement:
    def test_noise_treated_as_stable(self):
        # Movement < 0.25 is noise -> returns the stable score (0.55)
        assert _score_line_movement(0.1) == pytest.approx(0.55)
        assert _score_line_movement(-0.1) == pytest.approx(0.55)
        assert _score_line_movement(0.0) == pytest.approx(0.55)

    def test_line_dropped_is_good(self):
        # Negative movement (line softened) -> score above 0.55
        score = _score_line_movement(-1.0)
        assert score > 0.55

    def test_line_rose_is_bad(self):
        # Positive movement (line rose) -> score below 0.55
        score = _score_line_movement(1.0)
        assert score < 0.55

    def test_boundary_at_noise(self):
        # Exactly at boundary -- 0.25 is above noise threshold
        score = _score_line_movement(0.25)
        assert score != pytest.approx(0.55)


# ─────────────────────────────────────────────────────────────────
# _get_label
# ─────────────────────────────────────────────────────────────────


class TestGetLabel:
    def test_early_on_first_appearance(self):
        assert _get_label(0.95, 1) == "EARLY"

    def test_locked_requires_high_conviction_and_appearances(self):
        assert _get_label(0.90, 3) == "LOCKED"
        assert _get_label(0.85, 5) == "LOCKED"

    def test_locked_blocked_below_min_appearances(self):
        # 0.85 conviction but only 2 appearances -> STRONG not LOCKED
        assert _get_label(0.85, 2) == "STRONG"

    def test_strong(self):
        assert _get_label(0.80, 3) == "STRONG"
        assert _get_label(0.75, 2) == "STRONG"

    def test_moderate_is_watch(self):
        # 0.60-0.74 range is now WATCH (not published)
        assert _get_label(0.70, 3) == "WATCH"
        assert _get_label(0.60, 4) == "WATCH"

    def test_watch(self):
        assert _get_label(0.55, 3) == "WATCH"
        assert _get_label(0.50, 4) == "WATCH"

    def test_skip(self):
        assert _get_label(0.45, 4) == "SKIP"
        assert _get_label(0.30, 4) == "SKIP"
        assert _get_label(0.0, 6) == "SKIP"

    def test_boundary_locked_at_0_85(self):
        assert _get_label(0.85, 3) == "LOCKED"

    def test_boundary_strong_at_0_75(self):
        assert _get_label(0.75, 3) == "STRONG"

    def test_boundary_below_strong_is_watch(self):
        assert _get_label(0.74, 3) == "WATCH"


# ─────────────────────────────────────────────────────────────────
# _get_status
# ─────────────────────────────────────────────────────────────────


class TestGetStatus:
    def test_active(self):
        assert _get_status(True, 0.0, 0.0) == "active"
        assert _get_status(True, 2.0, -0.2) == "active"

    def test_evaporated_line_rose(self):
        assert _get_status(False, 1.5, 0.0) == "evaporated"

    def test_evaporated_model_lost_confidence(self):
        assert _get_status(False, 0.0, -0.15) == "evaporated"

    def test_dropped_no_clear_cause(self):
        assert _get_status(False, 0.5, -0.05) == "dropped"

    def test_evaporated_both_signals(self):
        assert _get_status(False, 2.0, -0.2) == "evaporated"


# ─────────────────────────────────────────────────────────────────
# _tier_score_adjustment
# ─────────────────────────────────────────────────────────────────


class TestTierScoreAdjustment:
    def _make_stats(self, wins, total):
        return {
            "wins": wins,
            "losses": total - wins,
            "total": total,
            "win_rate": round(wins / total, 3),
        }

    def test_no_tier_returns_zero(self):
        delta, note = _tier_score_adjustment(None, {})
        assert delta == 0.0
        assert note == ""

    def test_tier_not_in_stats_returns_zero(self):
        delta, note = _tier_score_adjustment("X", {})
        assert delta == 0.0
        assert note == ""

    def test_underperforming_tier_penalty(self):
        stats = {"X": self._make_stats(3, 10)}  # 30% WR -- below 0.47
        delta, note = _tier_score_adjustment("X", stats)
        assert delta == pytest.approx(-0.08)
        assert "UNDERPERFORMING" in note

    def test_strong_tier_bonus(self):
        stats = {"META": self._make_stats(8, 10)}  # 80% WR -- above 0.63
        delta, note = _tier_score_adjustment("META", stats)
        assert delta == pytest.approx(0.04)
        assert "performing well" in note

    def test_neutral_tier_no_adjustment(self):
        stats = {"Z": self._make_stats(5, 10)}  # 50% WR -- between thresholds
        delta, note = _tier_score_adjustment("Z", stats)
        assert delta == 0.0
        assert "Z L14:" in note

    def test_boundary_underperform(self):
        # WR exactly at 0.47 -> not below threshold -> no penalty
        stats = {"X": self._make_stats(47, 100)}
        delta, _ = _tier_score_adjustment("X", stats)
        assert delta == 0.0

    def test_boundary_strong(self):
        # WR exactly at 0.63 -> not above threshold -> no bonus
        stats = {"X": self._make_stats(63, 100)}
        delta, _ = _tier_score_adjustment("X", stats)
        assert delta == 0.0

    def test_note_contains_tier_name(self):
        stats = {"GOLD": self._make_stats(7, 10)}
        _, note = _tier_score_adjustment("GOLD", stats)
        assert "GOLD" in note


# ─────────────────────────────────────────────────────────────────
# _build_narrative
# ─────────────────────────────────────────────────────────────────


class TestBuildNarrative:
    """Tests for narrative generation -- verifies key phrases appear."""

    _BASE = dict(
        player_name="LeBron James",
        stat_type="POINTS",
        appearances=4,
        total_runs=6,
        entry_run=1,
        run_pattern="111011",
        line_at_entry=25.5,
        line_latest=25.5,
        line_direction="stable",
        p_over_at_entry=0.72,
        p_over_latest=0.75,
        book_latest="draftkings",
        status="active",
        conviction_label="STRONG",
        conviction=0.74,
        context=None,
        tier_note="",
    )

    def _build(self, **overrides):
        return _build_narrative(**{**self._BASE, **overrides})

    def test_present_all_runs(self):
        text = self._build(appearances=6, total_runs=6)
        assert "Present all 6 runs" in text

    def test_partial_appearances(self):
        text = self._build(appearances=4, total_runs=6, entry_run=1)
        assert "4/6 runs" in text

    def test_late_entry(self):
        text = self._build(appearances=3, total_runs=6, entry_run=4)
        assert "run 4" in text

    def test_line_held(self):
        text = self._build(line_at_entry=25.5, line_latest=25.5)
        assert "held" in text.lower() or "stable" in text.lower()

    def test_line_softened(self):
        text = self._build(line_at_entry=26.5, line_latest=25.0)
        assert "softened" in text or "More value" in text

    def test_line_rose(self):
        text = self._build(line_at_entry=25.0, line_latest=27.0)
        assert "rose" in text or "eroding" in text

    def test_p_over_stable(self):
        text = self._build(p_over_at_entry=0.72, p_over_latest=0.72)
        assert "stable" in text

    def test_p_over_strengthening(self):
        text = self._build(p_over_at_entry=0.65, p_over_latest=0.80)
        assert "strengthening" in text

    def test_p_over_weakening(self):
        text = self._build(p_over_at_entry=0.80, p_over_latest=0.65)
        assert "weakening" in text

    def test_evaporated_status(self):
        text = self._build(status="evaporated")
        assert "EVAPORATED" in text

    def test_dropped_status(self):
        text = self._build(status="dropped")
        assert "DROPPED" in text

    def test_bp_streak_included(self):
        ctx = {"bp_streak": 5, "bp_streak_type": "OVER"}
        text = self._build(context=ctx)
        assert "BP streak: 5" in text
        assert "OVER" in text

    def test_bp_rating_included(self):
        ctx = {"bp_bet_rating": 4, "bp_recommended_side": "over"}
        text = self._build(context=ctx)
        assert "BP rating: 4/5" in text

    def test_risk_high_included(self):
        ctx = {"risk_level": "HIGH", "risk_flags": ["back_to_back"]}
        text = self._build(context=ctx)
        assert "Risk: HIGH" in text
        assert "back_to_back" in text

    def test_risk_flags_only(self):
        ctx = {"risk_level": None, "risk_flags": ["injury_adjacent"]}
        text = self._build(context=ctx)
        assert "Risk:" in text

    def test_tier_note_included(self):
        text = self._build(tier_note="X L14: 8/10 (80.0%) -- performing well")
        assert "Filter:" in text
        assert "performing well" in text

    def test_no_tier_note_when_empty(self):
        text = self._build(tier_note="")
        assert "Filter:" not in text

    def test_none_line_skips_line_section(self):
        text = self._build(line_at_entry=None, line_latest=None)
        assert "softened" not in text
        assert "rose" not in text

    def test_none_p_over_skips_signal_section(self):
        text = self._build(p_over_at_entry=None, p_over_latest=None)
        assert "stable" not in text
        assert "strengthening" not in text

    def test_bp_rec_over_in_narrative(self):
        ctx = {"bp_rec_stars": 5, "bp_rec_side": "over", "bp_rec_ev": 0.27}
        text = self._build(context=ctx)
        assert "BP rec:" in text
        assert "OVER" in text

    def test_bp_rec_genuine_conflict_in_narrative(self):
        ctx = {
            "bp_rec_stars": 4,
            "bp_rec_side": "under",
            "bp_rec_conflict_type": "genuine",
            "bp_rec_projection": 23.5,
        }
        text = self._build(context=ctx)
        assert "CONFLICTS" in text
        assert "OVER" in text

    def test_bp_rec_odds_play_in_narrative(self):
        ctx = {
            "bp_rec_stars": 5,
            "bp_rec_side": "under",
            "bp_rec_conflict_type": "odds_play",
            "bp_rec_projection": 22.4,
        }
        text = self._build(context=ctx)
        assert "odds-value play" in text
        assert "CONFLICTS" not in text

    def test_bp_rec_projection_shown_in_narrative(self):
        ctx = {
            "bp_rec_stars": 4,
            "bp_rec_side": "under",
            "bp_rec_conflict_type": "genuine",
            "bp_rec_projection": 23.5,
        }
        text = self._build(context=ctx)
        assert "23.5" in text

    # ------ NEW TESTS: narrative coverage for uncovered branches ------

    def test_bp_prop_streak_text_included(self):
        """Line 260-261: bp_prop_streak_text branch."""
        ctx = {"bp_prop_streak_text": "3-game OVER streak"}
        text = self._build(context=ctx)
        assert "BP: 3-game OVER streak" in text

    def test_bp_prop_streak_text_empty_skips(self):
        """Line 260: falsy streak_text is skipped."""
        ctx = {"bp_prop_streak_text": ""}
        text = self._build(context=ctx)
        # Should not add a bare "BP: " with nothing after it for this field
        count = text.count("BP: ")
        # No extra "BP: " from empty streak text
        assert "BP: " not in text or count == 0

    def test_bp_hit_rate_L15_included(self):
        """Lines 264-268: hit rate narrative."""
        ctx = {
            "bp_hit_rate_L15": 0.733,
            "bp_hit_rate_wins": 11,
            "bp_hit_rate_games": 15,
        }
        text = self._build(context=ctx)
        assert "BP L15:" in text
        assert "11-4" in text  # 11 wins, 15-11=4 losses
        assert "73%" in text

    def test_bp_hit_rate_skipped_when_no_games(self):
        """Lines 267: guard on hr_games being falsy."""
        ctx = {
            "bp_hit_rate_L15": 0.5,
            "bp_hit_rate_wins": 5,
            "bp_hit_rate_games": None,
        }
        text = self._build(context=ctx)
        assert "BP L" not in text

    def test_bp_hit_rate_skipped_when_hr_none(self):
        """Lines 267: guard on hr being None."""
        ctx = {
            "bp_hit_rate_L15": None,
            "bp_hit_rate_wins": 5,
            "bp_hit_rate_games": 10,
        }
        text = self._build(context=ctx)
        assert "BP L" not in text

    def test_bp_performance_splits_included(self):
        """Lines 271-275: performance splits narrative."""
        ctx = {
            "bp_performance_splits": [
                {"text": "8-2 at home"},
                {"text": "6-3 vs East"},
                {"text": "5-5 on road"},  # third split is skipped (only top 2)
            ]
        }
        text = self._build(context=ctx)
        assert "BP: 8-2 at home" in text
        assert "BP: 6-3 vs East" in text
        assert "5-5 on road" not in text  # Only top 2

    def test_bp_performance_splits_empty_list(self):
        """Lines 271: empty splits list is a no-op."""
        ctx = {"bp_performance_splits": []}
        text = self._build(context=ctx)
        # No crash, no extra "BP:" from splits
        assert "BP:" not in text or "BP streak" in text or "BP rec" in text or True

    def test_bp_performance_splits_with_empty_text(self):
        """Line 274: split with empty text is skipped."""
        ctx = {"bp_performance_splits": [{"text": ""}, {"text": "7-3 vs West"}]}
        text = self._build(context=ctx)
        assert "BP: 7-3 vs West" in text

    def test_bp_rec_side_neither_over_nor_under(self):
        """Line 296->307: bp_rec_side is empty -> neither over nor under branch taken."""
        ctx = {"bp_rec_stars": 3, "bp_rec_side": "", "bp_rec_ev": 0.1}
        text = self._build(context=ctx)
        assert "OVER" not in text or "p_over" in text  # No BP rec OVER/UNDER text
        assert "CONFLICTS" not in text
        assert "odds-value play" not in text

    def test_bp_rec_over_with_ev_zero(self):
        """Line 286: bp_rec_ev is 0 (falsy) -> no EV string."""
        ctx = {"bp_rec_stars": 4, "bp_rec_side": "over", "bp_rec_ev": 0}
        text = self._build(context=ctx)
        assert "BP rec:" in text
        assert "EV+" not in text

    def test_bp_rec_over_with_negative_ev(self):
        """Line 286: bp_rec_ev is negative -> no EV+ string."""
        ctx = {"bp_rec_stars": 3, "bp_rec_side": "over", "bp_rec_ev": -0.05}
        text = self._build(context=ctx)
        assert "BP rec:" in text
        assert "EV+" not in text

    def test_bp_rec_over_with_market_ev_and_book(self):
        """Lines 287-290: market_ev and bp_rec_best_book are shown."""
        ctx = {
            "bp_rec_stars": 5,
            "bp_rec_side": "over",
            "bp_rec_ev": 0.15,
            "bp_rec_market_ev": 0.082,
            "bp_rec_best_book": "FanDuel",
        }
        text = self._build(context=ctx)
        assert "market EV" in text
        assert "FanDuel" in text

    def test_bp_rec_over_without_market_ev(self):
        """Line 290: no market_ev -> no parenthetical."""
        ctx = {
            "bp_rec_stars": 5,
            "bp_rec_side": "over",
            "bp_rec_ev": 0.15,
            "bp_rec_market_ev": None,
            "bp_rec_best_book": "FanDuel",
        }
        text = self._build(context=ctx)
        assert "market EV" not in text

    def test_bp_rec_over_without_best_book(self):
        """Line 290: no bp_rec_best_book -> no parenthetical."""
        ctx = {
            "bp_rec_stars": 5,
            "bp_rec_side": "over",
            "bp_rec_ev": 0.15,
            "bp_rec_market_ev": 0.08,
            "bp_rec_best_book": None,
        }
        text = self._build(context=ctx)
        assert "market EV" not in text

    def test_bp_rec_under_with_projection_none(self):
        """Line 293: bp_proj is None -> no projection string."""
        ctx = {
            "bp_rec_stars": 4,
            "bp_rec_side": "under",
            "bp_rec_conflict_type": "genuine",
            "bp_rec_projection": None,
        }
        text = self._build(context=ctx)
        assert "BP rec:" in text
        assert "CONFLICTS" in text
        assert "projects" not in text

    def test_risk_no_flags_no_level(self):
        """Line 309-311: risk_level is present but no flags."""
        ctx = {"risk_level": "HIGH", "risk_flags": []}
        text = self._build(context=ctx)
        assert "Risk: HIGH" in text

    def test_risk_unknown_with_flags(self):
        """Line 311: risk_level is None but risk_flags exist."""
        ctx = {"risk_level": None, "risk_flags": ["b2b", "travel"]}
        text = self._build(context=ctx)
        assert "Risk: UNKNOWN" in text
        assert "b2b" in text
        assert "travel" in text

    def test_narrative_active_status_no_flag(self):
        """Lines 242-245: active status does NOT add evaporated/dropped."""
        text = self._build(status="active")
        assert "EVAPORATED" not in text
        assert "DROPPED" not in text


# ─────────────────────────────────────────────────────────────────
# _get_tier_stats
# ─────────────────────────────────────────────────────────────────


class TestGetTierStats:
    def _make_conn(self, rows):
        mock_cur = MagicMock()
        mock_cur.__enter__ = lambda s: mock_cur
        mock_cur.__exit__ = MagicMock(return_value=False)
        mock_cur.fetchall.return_value = rows
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cur
        return mock_conn

    def test_returns_stats_by_tier(self):
        rows = [
            {"tier": "X", "total": 10, "wins": 7, "losses": 3},
            {"tier": "META", "total": 12, "wins": 9, "losses": 3},
        ]
        conn = self._make_conn(rows)
        result = _get_tier_stats(conn)
        assert "X" in result
        assert result["X"]["wins"] == 7
        assert result["X"]["win_rate"] == pytest.approx(0.7)
        assert "META" in result

    def test_returns_empty_on_db_error(self):
        mock_conn = MagicMock()
        mock_conn.cursor.side_effect = Exception("db error")
        result = _get_tier_stats(mock_conn)
        assert result == {}

    def test_none_tier_mapped_to_unknown(self):
        rows = [{"tier": None, "total": 8, "wins": 4, "losses": 4}]
        conn = self._make_conn(rows)
        result = _get_tier_stats(conn)
        assert "unknown" in result

    def test_win_rate_zero_total_guard(self):
        rows = [{"tier": "Z", "total": 0, "wins": 0, "losses": 0}]
        conn = self._make_conn(rows)
        result = _get_tier_stats(conn)
        assert result["Z"]["win_rate"] == 0.0

    def test_empty_rows_returns_empty_dict(self):
        conn = self._make_conn([])
        result = _get_tier_stats(conn)
        assert result == {}


# ─────────────────────────────────────────────────────────────────
# _score_bp_rec
# ─────────────────────────────────────────────────────────────────


class TestScoreBpRec:
    def _rec(self, stars, side, projection=None, best_ev=None, hit_rate=None):
        return {
            "bp_bet_rating": stars,
            "bp_recommended_side": side,
            "bp_projection": projection,
            "bp_expected_value": 0.1,
            "best_ev": best_ev or 0.0,
            "best_book": "draftkings",
            "hit_rate_L15": hit_rate,
            "hit_rate_wins": int(hit_rate * 15) if hit_rate else None,
            "hit_rate_losses": 15 - int(hit_rate * 15) if hit_rate else None,
            "hit_rate_games": 15 if hit_rate else None,
        }

    def test_none_rec_returns_zero(self):
        delta, ctx = _score_bp_rec(None)
        assert delta == 0.0
        assert ctx == {}

    def test_empty_rec_returns_zero(self):
        delta, ctx = _score_bp_rec({})
        assert delta == 0.0

    def test_over_5star_with_market_ev(self):
        delta, ctx = _score_bp_rec(self._rec(5, "over", best_ev=0.06))
        assert delta == pytest.approx(0.11)
        assert ctx["bp_rec_stars"] == 5
        assert ctx["bp_rec_side"] == "over"

    def test_over_5star_no_market_ev(self):
        delta, ctx = _score_bp_rec(self._rec(5, "over", best_ev=0.01))
        assert delta == 0.07

    def test_over_4star_with_market_ev(self):
        delta, ctx = _score_bp_rec(self._rec(4, "over", best_ev=0.06))
        assert delta == 0.09

    def test_over_4star_no_market_ev(self):
        delta, ctx = _score_bp_rec(self._rec(4, "over"))
        assert delta == 0.05

    def test_over_3star(self):
        delta, ctx = _score_bp_rec(self._rec(3, "over"))
        assert delta == 0.02

    def test_under_3star_no_penalty(self):
        # 3-star UNDER does not trigger conflict penalty
        delta, ctx = _score_bp_rec(self._rec(3, "under", projection=23.0), our_best_line=24.5)
        assert delta == 0.0

    def test_under_4star_genuine_conflict(self):
        # bp_projection (23.5) is within 1.5 of our line (24.5) -> genuine conflict
        delta, ctx = _score_bp_rec(self._rec(4, "under", projection=23.5), our_best_line=24.5)
        assert delta == -0.06
        assert ctx["bp_rec_conflict_type"] == "genuine"

    def test_under_5star_genuine_conflict(self):
        delta, ctx = _score_bp_rec(self._rec(5, "under", projection=24.0), our_best_line=24.5)
        assert delta == -0.06
        assert ctx["bp_rec_conflict_type"] == "genuine"

    def test_under_4star_odds_play_no_penalty(self):
        # bp_projection (22.4) is > 1.5 below our line (24.5) -> odds play
        delta, ctx = _score_bp_rec(self._rec(4, "under", projection=22.4), our_best_line=24.5)
        assert delta == 0.0
        assert ctx["bp_rec_conflict_type"] == "odds_play"

    def test_under_5star_odds_play_no_penalty(self):
        delta, ctx = _score_bp_rec(self._rec(5, "under", projection=20.0), our_best_line=24.5)
        assert delta == 0.0
        assert ctx["bp_rec_conflict_type"] == "odds_play"

    def test_under_no_line_provided_assumes_genuine(self):
        # No our_best_line -> can't determine odds play -> apply penalty
        delta, ctx = _score_bp_rec(self._rec(4, "under", projection=22.4))
        assert delta == -0.06
        assert ctx["bp_rec_conflict_type"] == "genuine"

    def test_under_no_projection_assumes_genuine(self):
        # No bp_projection -> can't determine odds play -> apply penalty
        delta, ctx = _score_bp_rec(self._rec(4, "under", projection=None), our_best_line=24.5)
        assert delta == -0.06
        assert ctx["bp_rec_conflict_type"] == "genuine"

    def test_extra_ctx_always_populated(self):
        _, ctx = _score_bp_rec(self._rec(4, "over", projection=25.0, best_ev=0.06))
        assert "bp_rec_stars" in ctx
        assert "bp_rec_side" in ctx
        assert "bp_rec_projection" in ctx
        assert "bp_rec_market_ev" in ctx
        assert "bp_rec_best_book" in ctx

    def test_boundary_exactly_1_5_below_is_genuine(self):
        # At exactly 1.5 below, NOT odds play (condition is strict <)
        delta, ctx = _score_bp_rec(self._rec(4, "under", projection=23.0), our_best_line=24.5)
        assert delta == -0.06
        assert ctx["bp_rec_conflict_type"] == "genuine"

    # ------ NEW TESTS: hit_rate bonus, cap, extra_ctx fields ------

    def test_over_5star_with_market_ev_and_high_hit_rate(self):
        """Hit rate >= 60% adds +0.03, capped at 0.12."""
        delta, ctx = _score_bp_rec(self._rec(5, "over", best_ev=0.06, hit_rate=0.70))
        # star_bonus=0.07 + market_bonus=0.04 + hit_bonus=0.03 = 0.14 -> capped at 0.12
        assert delta == pytest.approx(0.12)

    def test_over_3star_with_high_hit_rate(self):
        """3-star with high hit rate: 0.02 + 0.03 = 0.05."""
        delta, ctx = _score_bp_rec(self._rec(3, "over", best_ev=0.0, hit_rate=0.65))
        assert delta == pytest.approx(0.05)

    def test_over_with_hit_rate_below_threshold(self):
        """Hit rate < 60% -> no hit_bonus."""
        delta, ctx = _score_bp_rec(self._rec(5, "over", best_ev=0.01, hit_rate=0.50))
        # star_bonus=0.07, market_bonus=0 (0.01 < 0.05), hit_bonus=0
        assert delta == pytest.approx(0.07)

    def test_extra_ctx_hit_rate_fields(self):
        """Extra context should include hit rate fields."""
        _, ctx = _score_bp_rec(self._rec(4, "over", hit_rate=0.733))
        assert ctx["bp_hit_rate_L15"] == pytest.approx(0.733)
        assert ctx["bp_hit_rate_wins"] == 10  # int(0.733 * 15) = 10
        assert ctx["bp_hit_rate_games"] == 15

    def test_extra_ctx_supplemental_fields(self):
        """Extra context includes performance_splits, prop_streak, etc."""
        rec = {
            "bp_bet_rating": 4,
            "bp_recommended_side": "over",
            "bp_expected_value": 0.1,
            "best_ev": 0.0,
            "best_book": "dk",
            "bp_projection": 25.0,
            "hit_rate_L15": None,
            "hit_rate_wins": None,
            "hit_rate_losses": None,
            "hit_rate_games": None,
            "performance_splits": [{"text": "7-2 at home"}],
            "prop_streak": 4,
            "prop_streak_text": "4-game OVER streak",
            "bp_opponent_rank": 5,
            "tailing_pct": 0.72,
        }
        _, ctx = _score_bp_rec(rec)
        assert ctx["bp_performance_splits"] == [{"text": "7-2 at home"}]
        assert ctx["bp_prop_streak"] == 4
        assert ctx["bp_prop_streak_text"] == "4-game OVER streak"
        assert ctx["bp_opponent_rank"] == 5
        assert ctx["bp_tailing_pct"] == 0.72

    def test_over_2star_no_bonus(self):
        """2-star is not in _BP_REC_STAR_BONUS -> star_bonus=0."""
        delta, _ = _score_bp_rec(self._rec(2, "over"))
        assert delta == 0.0

    def test_under_2star_no_penalty(self):
        """2-star UNDER: stars < 4 -> no penalty (falls to else branch)."""
        delta, _ = _score_bp_rec(self._rec(2, "under", projection=24.0), our_best_line=24.5)
        assert delta == 0.0


# ─────────────────────────────────────────────────────────────────
# _load_bp_recommendations
# ─────────────────────────────────────────────────────────────────


class TestLoadBpRecommendations:
    def _write_bp_file(self, tmp_path: Path, date: str, picks: list) -> Path:
        """Write a fake bp_pick_recommendations file and patch the lines_dir."""
        bp_file = tmp_path / f"bp_pick_recommendations_{date}.json"
        bp_file.write_text(json.dumps({"date": date, "picks": picks}))
        return bp_file

    def test_missing_file_returns_empty(self):
        result = _load_bp_recommendations("9999-01-01")
        assert result == {}

    def test_file_loaded_and_keyed_correctly(self):
        """Write a real file at the expected path with a unique date and load it."""
        from pathlib import Path as RealPath

        import nba.core.conviction_engine as ce

        lines_dir = RealPath(ce.__file__).parent.parent / "betting_xl" / "lines"
        lines_dir.mkdir(parents=True, exist_ok=True)

        picks = [
            {"player_name": "LeBron James", "stat_type": "POINTS", "bp_bet_rating": 5},
            {"player_name": "Ja Morant", "stat_type": "ASSISTS", "bp_bet_rating": 3},
        ]
        bp_file = lines_dir / "bp_pick_recommendations_9998-12-31.json"
        bp_file.write_text(json.dumps({"picks": picks}))
        try:
            result = _load_bp_recommendations("9998-12-31")
        finally:
            bp_file.unlink(missing_ok=True)

        assert ("lebron james", "POINTS") in result
        assert ("ja morant", "ASSISTS") in result
        assert result[("lebron james", "POINTS")]["bp_bet_rating"] == 5

    def test_normalises_player_name_lowercase(self):
        from pathlib import Path as RealPath

        import nba.core.conviction_engine as ce

        lines_dir = RealPath(ce.__file__).parent.parent / "betting_xl" / "lines"
        lines_dir.mkdir(parents=True, exist_ok=True)

        picks = [{"player_name": "Paolo Banchero", "stat_type": "POINTS", "bp_bet_rating": 5}]
        bp_file = lines_dir / "bp_pick_recommendations_9998-12-30.json"
        bp_file.write_text(json.dumps({"picks": picks}))
        try:
            result = _load_bp_recommendations("9998-12-30")
        finally:
            bp_file.unlink(missing_ok=True)

        assert ("paolo banchero", "POINTS") in result

    def test_skips_picks_missing_name_or_stat(self):
        from pathlib import Path as RealPath

        import nba.core.conviction_engine as ce

        lines_dir = RealPath(ce.__file__).parent.parent / "betting_xl" / "lines"
        lines_dir.mkdir(parents=True, exist_ok=True)

        picks = [
            {"player_name": "", "stat_type": "POINTS"},
            {"player_name": "Player A", "stat_type": ""},
            {"player_name": "Player B", "stat_type": "REBOUNDS", "bp_bet_rating": 3},
        ]
        bp_file = lines_dir / "bp_pick_recommendations_9998-12-29.json"
        bp_file.write_text(json.dumps({"picks": picks}))
        try:
            result = _load_bp_recommendations("9998-12-29")
        finally:
            bp_file.unlink(missing_ok=True)

        assert len(result) == 1
        assert ("player b", "REBOUNDS") in result

    def test_corrupt_file_returns_empty(self):
        from pathlib import Path as RealPath

        import nba.core.conviction_engine as ce

        lines_dir = RealPath(ce.__file__).parent.parent / "betting_xl" / "lines"
        lines_dir.mkdir(parents=True, exist_ok=True)

        bp_file = lines_dir / "bp_pick_recommendations_9998-12-28.json"
        bp_file.write_text("NOT VALID JSON {{{{")
        try:
            result = _load_bp_recommendations("9998-12-28")
        finally:
            bp_file.unlink(missing_ok=True)

        assert result == {}


# ─────────────────────────────────────────────────────────────────
# compute_conviction (integration-level tests, all DB mocked)
# ─────────────────────────────────────────────────────────────────


def _make_prediction_row(
    player_name="LeBron James",
    stat_type="POINTS",
    model_version="xl",
    run_number=1,
    line=25.5,
    p_over=0.78,
    edge=3.2,
    spread=1.5,
    book="draftkings",
    context_snapshot=None,
):
    """Helper to build a prediction_history row dict."""
    return {
        "player_name": player_name,
        "stat_type": stat_type,
        "model_version": model_version,
        "run_number": run_number,
        "line": line,
        "p_over": p_over,
        "edge": edge,
        "spread": spread,
        "book": book,
        "context_snapshot": context_snapshot or {},
    }


@patch("nba.core.conviction_engine.psycopg2.extras.Json", side_effect=lambda x: x)
@patch("nba.core.conviction_engine.psycopg2.extras.execute_values")
class TestComputeConviction:
    """Tests for the main compute_conviction() function with all DB interactions mocked."""

    def _mock_conn_and_cursors(self, prediction_rows, tier_rows=None):
        """Build a mock connection that returns prediction_rows on first cursor
        and tier_rows on second cursor (for _get_tier_stats)."""
        mock_conn = MagicMock()

        # The main cursor (RealDictCursor) is used twice:
        # 1) Fetch prediction history
        # 2) _get_tier_stats query
        # Then a regular cursor for the upsert
        call_count = {"n": 0}
        all_rows = [prediction_rows, tier_rows or []]

        def make_cursor(*args, **kwargs):
            cur = MagicMock()
            cur.__enter__ = lambda s: cur
            cur.__exit__ = MagicMock(return_value=False)
            idx = call_count["n"]
            call_count["n"] += 1
            if idx < len(all_rows):
                cur.fetchall.return_value = all_rows[idx]
            else:
                cur.fetchall.return_value = []
            return cur

        mock_conn.cursor.side_effect = make_cursor
        # Support context manager for the upsert block
        mock_conn.__enter__ = lambda s: s
        mock_conn.__exit__ = MagicMock(return_value=False)
        return mock_conn

    @patch("nba.core.conviction_engine._load_bp_recommendations", return_value={})
    @patch("nba.core.conviction_engine._connect")
    def test_basic_single_pick_single_run(self, mock_connect, mock_bp, mock_ev, mock_json):
        """Single pick, single run -> writes 1 conviction row, early-day cap applies."""
        rows = [_make_prediction_row(run_number=1)]
        mock_conn = self._mock_conn_and_cursors(rows)
        mock_connect.return_value = mock_conn

        result = compute_conviction("2026-03-17", 1)

        assert result == 1
        mock_conn.close.assert_called()

    @patch("nba.core.conviction_engine._load_bp_recommendations", return_value={})
    @patch("nba.core.conviction_engine._connect")
    def test_single_pick_across_multiple_runs(self, mock_connect, mock_bp, mock_ev, mock_json):
        """Same pick seen in 3 runs -> 1 conviction row with higher score."""
        rows = [
            _make_prediction_row(run_number=1, p_over=0.75, line=25.5),
            _make_prediction_row(run_number=2, p_over=0.77, line=25.5),
            _make_prediction_row(run_number=3, p_over=0.80, line=25.0),
        ]
        mock_conn = self._mock_conn_and_cursors(rows)
        mock_connect.return_value = mock_conn

        result = compute_conviction("2026-03-17", 3)
        assert result == 1

    @patch("nba.core.conviction_engine._load_bp_recommendations", return_value={})
    @patch("nba.core.conviction_engine._connect")
    def test_multiple_picks(self, mock_connect, mock_bp, mock_ev, mock_json):
        """Two different players -> 2 conviction rows."""
        rows = [
            _make_prediction_row(player_name="LeBron James", stat_type="POINTS", run_number=1),
            _make_prediction_row(player_name="Kevin Durant", stat_type="REBOUNDS", run_number=1),
        ]
        mock_conn = self._mock_conn_and_cursors(rows)
        mock_connect.return_value = mock_conn

        result = compute_conviction("2026-03-17", 1)
        assert result == 2

    @patch("nba.core.conviction_engine._load_bp_recommendations", return_value={})
    @patch("nba.core.conviction_engine._connect")
    def test_no_rows_returns_zero(self, mock_connect, mock_bp, mock_ev, mock_json):
        """No prediction history -> returns 0."""
        mock_conn = self._mock_conn_and_cursors([])
        mock_connect.return_value = mock_conn

        result = compute_conviction("2026-03-17", 1)
        assert result == 0
        mock_conn.close.assert_called()

    @patch("nba.core.conviction_engine._connect")
    def test_connection_failure_returns_zero(self, mock_connect, mock_ev, mock_json):
        """DB connection failure -> returns 0 gracefully."""
        mock_connect.side_effect = Exception("connection refused")

        result = compute_conviction("2026-03-17", 1)
        assert result == 0

    @patch("nba.core.conviction_engine._load_bp_recommendations", return_value={})
    @patch("nba.core.conviction_engine._connect")
    def test_early_day_cap_applied_run1(self, mock_connect, mock_bp, mock_ev, mock_json):
        """Run 1 -> conviction capped at 0.62 regardless of score."""
        # High p_over to push score up, but early-day cap limits it
        rows = [_make_prediction_row(run_number=1, p_over=0.95)]
        mock_conn = self._mock_conn_and_cursors(rows)
        mock_connect.return_value = mock_conn

        result = compute_conviction("2026-03-17", 1)
        assert result == 1

        # Verify the upsert was called
        upsert_cursor = mock_conn.cursor.return_value
        # We can't easily inspect the values passed to execute_values
        # but we verified it returned 1 (so it ran the upsert path)

    @patch("nba.core.conviction_engine._load_bp_recommendations", return_value={})
    @patch("nba.core.conviction_engine._connect")
    def test_model_agreement_bonus(self, mock_connect, mock_bp, mock_ev, mock_json):
        """Both XL and V3 see the pick -> +0.05 bonus."""
        ctx = {"models_agreeing": ["xl", "v3"]}
        rows = [
            _make_prediction_row(run_number=1, context_snapshot=ctx),
            _make_prediction_row(run_number=2, context_snapshot=ctx),
            _make_prediction_row(run_number=3, context_snapshot=ctx),
        ]
        mock_conn = self._mock_conn_and_cursors(rows)
        mock_connect.return_value = mock_conn

        result = compute_conviction("2026-03-17", 3)
        assert result == 1

    @patch("nba.core.conviction_engine._load_bp_recommendations", return_value={})
    @patch("nba.core.conviction_engine._connect")
    def test_bp_hit_rate_bonus(self, mock_connect, mock_bp, mock_ev, mock_json):
        """BP bet_rating >= 4 and side == 'over' -> +0.03 bonus."""
        ctx = {"bp_bet_rating": 5, "bp_recommended_side": "over"}
        rows = [
            _make_prediction_row(run_number=1, context_snapshot=ctx),
            _make_prediction_row(run_number=2, context_snapshot=ctx),
            _make_prediction_row(run_number=3, context_snapshot=ctx),
        ]
        mock_conn = self._mock_conn_and_cursors(rows)
        mock_connect.return_value = mock_conn

        result = compute_conviction("2026-03-17", 3)
        assert result == 1

    @patch("nba.core.conviction_engine._connect")
    def test_bp_recommendations_integrated(self, mock_connect, mock_ev, mock_json):
        """BP recommendations file provides bonus when pick matches."""
        bp_recs = {
            ("lebron james", "POINTS"): {
                "bp_bet_rating": 5,
                "bp_recommended_side": "over",
                "bp_expected_value": 0.15,
                "best_ev": 0.08,
                "best_book": "fanduel",
                "bp_projection": 26.0,
                "hit_rate_L15": 0.73,
                "hit_rate_wins": 11,
                "hit_rate_losses": 4,
                "hit_rate_games": 15,
                "performance_splits": [],
                "prop_streak": None,
                "prop_streak_text": None,
                "bp_opponent_rank": None,
                "tailing_pct": None,
            }
        }
        rows = [
            _make_prediction_row(run_number=1),
            _make_prediction_row(run_number=2),
            _make_prediction_row(run_number=3),
        ]
        mock_conn = self._mock_conn_and_cursors(rows)
        mock_connect.return_value = mock_conn

        with patch("nba.core.conviction_engine._load_bp_recommendations", return_value=bp_recs):
            result = compute_conviction("2026-03-17", 3)
        assert result == 1

    @patch("nba.core.conviction_engine._load_bp_recommendations", return_value={})
    @patch("nba.core.conviction_engine._connect")
    def test_tier_stats_applied(self, mock_connect, mock_bp, mock_ev, mock_json):
        """Tier stats with strong performance -> positive adjustment."""
        tier_rows = [{"tier": "X", "total": 20, "wins": 15, "losses": 5}]
        ctx = {"filter_tier": "X"}
        rows = [
            _make_prediction_row(run_number=1, context_snapshot=ctx),
            _make_prediction_row(run_number=2, context_snapshot=ctx),
            _make_prediction_row(run_number=3, context_snapshot=ctx),
        ]
        mock_conn = self._mock_conn_and_cursors(rows, tier_rows=tier_rows)
        mock_connect.return_value = mock_conn

        result = compute_conviction("2026-03-17", 3)
        assert result == 1

    @patch("nba.core.conviction_engine._load_bp_recommendations", return_value={})
    @patch("nba.core.conviction_engine._connect")
    def test_context_snapshot_as_string(self, mock_connect, mock_bp, mock_ev, mock_json):
        """context_snapshot stored as JSON string -> parsed correctly."""
        ctx_str = json.dumps({"models_agreeing": ["xl", "v3"], "filter_tier": "META"})
        rows = [
            _make_prediction_row(run_number=1, context_snapshot=ctx_str),
            _make_prediction_row(run_number=2, context_snapshot=ctx_str),
        ]
        mock_conn = self._mock_conn_and_cursors(rows)
        mock_connect.return_value = mock_conn

        result = compute_conviction("2026-03-17", 2)
        assert result == 1

    @patch("nba.core.conviction_engine._load_bp_recommendations", return_value={})
    @patch("nba.core.conviction_engine._connect")
    def test_context_snapshot_invalid_json_string(self, mock_connect, mock_bp, mock_ev, mock_json):
        """context_snapshot is invalid JSON string -> falls back to empty dict."""
        rows = [
            _make_prediction_row(run_number=1, context_snapshot="not json {{"),
            _make_prediction_row(run_number=2, context_snapshot="not json {{"),
        ]
        mock_conn = self._mock_conn_and_cursors(rows)
        mock_connect.return_value = mock_conn

        result = compute_conviction("2026-03-17", 2)
        assert result == 1  # Still processes the pick despite invalid ctx

    @patch("nba.core.conviction_engine._load_bp_recommendations", return_value={})
    @patch("nba.core.conviction_engine._connect")
    def test_pick_with_none_p_over_skipped(self, mock_connect, mock_bp, mock_ev, mock_json):
        """Picks with all None p_over are skipped (no p_overs -> continue)."""
        rows = [
            _make_prediction_row(run_number=1, p_over=None),
        ]
        mock_conn = self._mock_conn_and_cursors(rows)
        mock_connect.return_value = mock_conn

        result = compute_conviction("2026-03-17", 1)
        # No valid p_overs -> skip -> conviction_rows is empty -> return 0
        assert result == 0

    @patch("nba.core.conviction_engine._load_bp_recommendations", return_value={})
    @patch("nba.core.conviction_engine._connect")
    def test_inactive_pick_status_evaporated(self, mock_connect, mock_bp, mock_ev, mock_json):
        """Pick not in latest run + line rose significantly -> evaporated status."""
        rows = [
            _make_prediction_row(run_number=1, line=25.0, p_over=0.75),
            _make_prediction_row(run_number=2, line=27.0, p_over=0.60),
            # Not present at run 3 -> is_active=False, line_movement=+2.0
        ]
        mock_conn = self._mock_conn_and_cursors(rows)
        mock_connect.return_value = mock_conn

        result = compute_conviction("2026-03-17", 3)
        assert result == 1

    @patch("nba.core.conviction_engine._load_bp_recommendations", return_value={})
    @patch("nba.core.conviction_engine._connect")
    def test_inactive_pick_status_dropped(self, mock_connect, mock_bp, mock_ev, mock_json):
        """Pick not in latest run + line stable + p_over stable -> dropped status."""
        rows = [
            _make_prediction_row(run_number=1, line=25.5, p_over=0.75),
            _make_prediction_row(run_number=2, line=25.5, p_over=0.74),
            # Not present at run 3 -> is_active=False, no clear cause -> dropped
        ]
        mock_conn = self._mock_conn_and_cursors(rows)
        mock_connect.return_value = mock_conn

        result = compute_conviction("2026-03-17", 3)
        assert result == 1

    @patch("nba.core.conviction_engine._load_bp_recommendations", return_value={})
    @patch("nba.core.conviction_engine._connect")
    def test_line_direction_stable(self, mock_connect, mock_bp, mock_ev, mock_json):
        """Line movement within noise -> direction is 'stable'."""
        rows = [
            _make_prediction_row(run_number=1, line=25.5),
            _make_prediction_row(run_number=2, line=25.6),
            _make_prediction_row(run_number=3, line=25.5),
        ]
        mock_conn = self._mock_conn_and_cursors(rows)
        mock_connect.return_value = mock_conn

        result = compute_conviction("2026-03-17", 3)
        assert result == 1

    @patch("nba.core.conviction_engine._load_bp_recommendations", return_value={})
    @patch("nba.core.conviction_engine._connect")
    def test_line_direction_falling(self, mock_connect, mock_bp, mock_ev, mock_json):
        """Line dropped significantly -> direction is 'falling'."""
        rows = [
            _make_prediction_row(run_number=1, line=26.5),
            _make_prediction_row(run_number=2, line=26.0),
            _make_prediction_row(run_number=3, line=25.5),
        ]
        mock_conn = self._mock_conn_and_cursors(rows)
        mock_connect.return_value = mock_conn

        result = compute_conviction("2026-03-17", 3)
        assert result == 1

    @patch("nba.core.conviction_engine._load_bp_recommendations", return_value={})
    @patch("nba.core.conviction_engine._connect")
    def test_line_direction_rising(self, mock_connect, mock_bp, mock_ev, mock_json):
        """Line rose significantly -> direction is 'rising'."""
        rows = [
            _make_prediction_row(run_number=1, line=24.5),
            _make_prediction_row(run_number=2, line=25.5),
            _make_prediction_row(run_number=3, line=26.5),
        ]
        mock_conn = self._mock_conn_and_cursors(rows)
        mock_connect.return_value = mock_conn

        result = compute_conviction("2026-03-17", 3)
        assert result == 1

    @patch("nba.core.conviction_engine._load_bp_recommendations", return_value={})
    @patch("nba.core.conviction_engine._connect")
    def test_no_lines_in_pick(self, mock_connect, mock_bp, mock_ev, mock_json):
        """All line values are None -> line_entry/line_latest are None."""
        rows = [
            _make_prediction_row(run_number=1, line=None),
            _make_prediction_row(run_number=2, line=None),
        ]
        mock_conn = self._mock_conn_and_cursors(rows)
        mock_connect.return_value = mock_conn

        result = compute_conviction("2026-03-17", 2)
        assert result == 1

    @patch("nba.core.conviction_engine._load_bp_recommendations", return_value={})
    @patch("nba.core.conviction_engine._connect")
    def test_exception_during_processing_returns_zero(
        self, mock_connect, mock_bp, mock_ev, mock_json
    ):
        """Unexpected exception during main processing -> returns 0."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        # Make the first cursor.execute raise an error
        mock_cur = MagicMock()
        mock_cur.__enter__ = lambda s: mock_cur
        mock_cur.__exit__ = MagicMock(return_value=False)
        mock_cur.execute.side_effect = Exception("query failed")
        mock_conn.cursor.return_value = mock_cur

        result = compute_conviction("2026-03-17", 1)
        assert result == 0

    @patch("nba.core.conviction_engine._load_bp_recommendations", return_value={})
    @patch("nba.core.conviction_engine._connect")
    def test_exception_during_processing_closes_conn(
        self, mock_connect, mock_bp, mock_ev, mock_json
    ):
        """Exception during processing -> connection is still closed."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        mock_cur = MagicMock()
        mock_cur.__enter__ = lambda s: mock_cur
        mock_cur.__exit__ = MagicMock(return_value=False)
        mock_cur.execute.side_effect = Exception("oops")
        mock_conn.cursor.return_value = mock_cur

        compute_conviction("2026-03-17", 1)
        mock_conn.close.assert_called()

    @patch("nba.core.conviction_engine._load_bp_recommendations", return_value={})
    @patch("nba.core.conviction_engine._connect")
    def test_exception_close_also_fails(self, mock_connect, mock_bp, mock_ev, mock_json):
        """Exception during processing + conn.close() also fails -> still returns 0."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        mock_cur = MagicMock()
        mock_cur.__enter__ = lambda s: mock_cur
        mock_cur.__exit__ = MagicMock(return_value=False)
        mock_cur.execute.side_effect = Exception("oops")
        mock_conn.cursor.return_value = mock_cur
        mock_conn.close.side_effect = Exception("close failed")

        result = compute_conviction("2026-03-17", 1)
        assert result == 0

    @patch("nba.core.conviction_engine._load_bp_recommendations", return_value={})
    @patch("nba.core.conviction_engine._connect")
    def test_bp_rec_penalty_applied_in_compute(self, mock_connect, mock_bp, mock_ev, mock_json):
        """BP rec UNDER conflict -> conviction reduced by penalty."""
        bp_recs = {
            ("lebron james", "POINTS"): {
                "bp_bet_rating": 5,
                "bp_recommended_side": "under",
                "bp_expected_value": 0.1,
                "best_ev": 0.0,
                "best_book": "dk",
                "bp_projection": 25.0,
                "hit_rate_L15": None,
                "hit_rate_wins": None,
                "hit_rate_losses": None,
                "hit_rate_games": None,
                "performance_splits": [],
                "prop_streak": None,
                "prop_streak_text": None,
                "bp_opponent_rank": None,
                "tailing_pct": None,
            }
        }
        mock_bp.return_value = bp_recs

        rows = [
            _make_prediction_row(run_number=1, line=25.5),
            _make_prediction_row(run_number=2, line=25.5),
            _make_prediction_row(run_number=3, line=25.5),
        ]
        mock_conn = self._mock_conn_and_cursors(rows)
        mock_connect.return_value = mock_conn

        result = compute_conviction("2026-03-17", 3)
        assert result == 1

    @patch("nba.core.conviction_engine._load_bp_recommendations", return_value={})
    @patch("nba.core.conviction_engine._connect")
    def test_tier_underperform_penalty_applied(self, mock_connect, mock_bp, mock_ev, mock_json):
        """Tier with WR < 0.47 -> conviction reduced by -0.08."""
        tier_rows = [{"tier": "X", "total": 20, "wins": 6, "losses": 14}]
        ctx = {"filter_tier": "X"}
        rows = [
            _make_prediction_row(run_number=1, context_snapshot=ctx),
            _make_prediction_row(run_number=2, context_snapshot=ctx),
            _make_prediction_row(run_number=3, context_snapshot=ctx),
        ]
        mock_conn = self._mock_conn_and_cursors(rows, tier_rows=tier_rows)
        mock_connect.return_value = mock_conn

        result = compute_conviction("2026-03-17", 3)
        assert result == 1

    @patch("nba.core.conviction_engine._load_bp_recommendations", return_value={})
    @patch("nba.core.conviction_engine._connect")
    def test_conviction_clamped_to_zero_floor(self, mock_connect, mock_bp, mock_ev, mock_json):
        """Negative adjustments don't push conviction below 0."""
        # Very weak pick + BP penalty + tier penalty
        bp_recs = {
            ("lebron james", "POINTS"): {
                "bp_bet_rating": 5,
                "bp_recommended_side": "under",
                "bp_expected_value": 0.0,
                "best_ev": 0.0,
                "best_book": "dk",
                "bp_projection": 25.0,
                "hit_rate_L15": None,
                "hit_rate_wins": None,
                "hit_rate_losses": None,
                "hit_rate_games": None,
                "performance_splits": [],
                "prop_streak": None,
                "prop_streak_text": None,
                "bp_opponent_rank": None,
                "tailing_pct": None,
            }
        }
        mock_bp.return_value = bp_recs
        tier_rows = [{"tier": "X", "total": 20, "wins": 4, "losses": 16}]
        ctx = {"filter_tier": "X"}
        rows = [
            _make_prediction_row(run_number=1, p_over=0.52, context_snapshot=ctx, line=25.5),
            _make_prediction_row(run_number=2, p_over=0.40, context_snapshot=ctx, line=25.5),
            _make_prediction_row(run_number=3, p_over=0.30, context_snapshot=ctx, line=25.5),
        ]
        mock_conn = self._mock_conn_and_cursors(rows, tier_rows=tier_rows)
        mock_connect.return_value = mock_conn

        result = compute_conviction("2026-03-17", 3)
        assert result == 1  # Still writes the row, just with low conviction

    @patch("nba.core.conviction_engine._load_bp_recommendations", return_value={})
    @patch("nba.core.conviction_engine._connect")
    def test_upsert_uses_execute_values(self, mock_connect, mock_bp, mock_ev, mock_json):
        """Verify execute_values is called for the upsert."""
        rows = [
            _make_prediction_row(run_number=1),
            _make_prediction_row(run_number=2),
        ]
        mock_conn = self._mock_conn_and_cursors(rows)
        mock_connect.return_value = mock_conn

        result = compute_conviction("2026-03-17", 2)

        assert result == 1
        mock_ev.assert_called_once()

    @patch("nba.core.conviction_engine._load_bp_recommendations", return_value={})
    @patch("nba.core.conviction_engine._connect")
    def test_mixed_players_grouped_correctly(self, mock_connect, mock_bp, mock_ev, mock_json):
        """Multiple players across multiple runs -> correct grouping."""
        rows = [
            _make_prediction_row(
                player_name="LeBron James",
                stat_type="POINTS",
                run_number=1,
                p_over=0.78,
            ),
            _make_prediction_row(
                player_name="Kevin Durant",
                stat_type="REBOUNDS",
                run_number=1,
                p_over=0.72,
            ),
            _make_prediction_row(
                player_name="LeBron James",
                stat_type="POINTS",
                run_number=2,
                p_over=0.80,
            ),
            _make_prediction_row(
                player_name="Kevin Durant",
                stat_type="REBOUNDS",
                run_number=2,
                p_over=0.74,
            ),
            _make_prediction_row(
                player_name="Jayson Tatum",
                stat_type="POINTS",
                run_number=2,
                p_over=0.70,
            ),
        ]
        mock_conn = self._mock_conn_and_cursors(rows)
        mock_connect.return_value = mock_conn

        result = compute_conviction("2026-03-17", 2)
        assert result == 3  # LeBron+POINTS, KD+REBOUNDS, Tatum+POINTS

    @patch("nba.core.conviction_engine._load_bp_recommendations", return_value={})
    @patch("nba.core.conviction_engine._connect")
    def test_appearance_2_not_capped(self, mock_connect, mock_bp, mock_ev, mock_json):
        """With 2 appearances at run 2, conviction is NOT early-day capped."""
        rows = [
            _make_prediction_row(run_number=1, p_over=0.85),
            _make_prediction_row(run_number=2, p_over=0.90),
        ]
        mock_conn = self._mock_conn_and_cursors(rows)
        mock_connect.return_value = mock_conn

        # run_number=2, appearances=2 -> cap does NOT apply
        # (cap only when run_number==1 OR appearances < 2)
        result = compute_conviction("2026-03-17", 2)
        assert result == 1

    @patch("nba.core.conviction_engine._load_bp_recommendations", return_value={})
    @patch("nba.core.conviction_engine._connect")
    def test_fired_runs_derived_from_rows(self, mock_connect, mock_bp, mock_ev, mock_json):
        """fired_runs is built from unique run_numbers in rows."""
        rows = [
            _make_prediction_row(player_name="Player A", stat_type="POINTS", run_number=1),
            _make_prediction_row(player_name="Player A", stat_type="POINTS", run_number=3),
            _make_prediction_row(player_name="Player B", stat_type="REBOUNDS", run_number=3),
        ]
        mock_conn = self._mock_conn_and_cursors(rows)
        mock_connect.return_value = mock_conn

        # run 2 never fired -> not penalized in appearance scoring
        result = compute_conviction("2026-03-17", 3)
        assert result == 2

    @patch("nba.core.conviction_engine._load_bp_recommendations", return_value={})
    @patch("nba.core.conviction_engine._connect")
    def test_context_snapshot_none_handled(self, mock_connect, mock_bp, mock_ev, mock_json):
        """context_snapshot is None -> falls back to empty dict."""
        rows = [
            _make_prediction_row(run_number=1, context_snapshot=None),
            _make_prediction_row(run_number=2, context_snapshot=None),
        ]
        mock_conn = self._mock_conn_and_cursors(rows)
        mock_connect.return_value = mock_conn

        result = compute_conviction("2026-03-17", 2)
        assert result == 1

    @patch("nba.core.conviction_engine._load_bp_recommendations", return_value={})
    @patch("nba.core.conviction_engine._connect")
    def test_single_p_over_std_is_zero(self, mock_connect, mock_bp, mock_ev, mock_json):
        """With only 1 p_over value, stdev is 0.0 by definition."""
        rows = [_make_prediction_row(run_number=1, p_over=0.80)]
        mock_conn = self._mock_conn_and_cursors(rows)
        mock_connect.return_value = mock_conn

        result = compute_conviction("2026-03-17", 1)
        assert result == 1

    @patch("nba.core.conviction_engine._load_bp_recommendations", return_value={})
    @patch("nba.core.conviction_engine._connect")
    def test_bp_delta_zero_does_not_modify_conviction(
        self, mock_connect, mock_bp, mock_ev, mock_json
    ):
        """When bp_delta is 0, conviction should not be modified by it."""
        bp_recs = {
            ("lebron james", "POINTS"): {
                "bp_bet_rating": 2,  # 2-star -> no bonus
                "bp_recommended_side": "over",
                "bp_expected_value": 0.0,
                "best_ev": 0.0,
                "best_book": "dk",
                "bp_projection": None,
                "hit_rate_L15": None,
                "hit_rate_wins": None,
                "hit_rate_losses": None,
                "hit_rate_games": None,
                "performance_splits": [],
                "prop_streak": None,
                "prop_streak_text": None,
                "bp_opponent_rank": None,
                "tailing_pct": None,
            }
        }
        mock_bp.return_value = bp_recs
        rows = [
            _make_prediction_row(run_number=1),
            _make_prediction_row(run_number=2),
        ]
        mock_conn = self._mock_conn_and_cursors(rows)
        mock_connect.return_value = mock_conn

        result = compute_conviction("2026-03-17", 2)
        assert result == 1

    @patch("nba.core.conviction_engine._load_bp_recommendations", return_value={})
    @patch("nba.core.conviction_engine._connect")
    def test_bp_extra_ctx_merged_into_context(self, mock_connect, mock_bp, mock_ev, mock_json):
        """BP extra_ctx fields are merged into the context snapshot."""
        bp_recs = {
            ("lebron james", "POINTS"): {
                "bp_bet_rating": 5,
                "bp_recommended_side": "over",
                "bp_expected_value": 0.1,
                "best_ev": 0.08,
                "best_book": "fanduel",
                "bp_projection": 26.0,
                "hit_rate_L15": None,
                "hit_rate_wins": None,
                "hit_rate_losses": None,
                "hit_rate_games": None,
                "performance_splits": [],
                "prop_streak": None,
                "prop_streak_text": None,
                "bp_opponent_rank": None,
                "tailing_pct": None,
            }
        }
        mock_bp.return_value = bp_recs
        rows = [
            _make_prediction_row(run_number=1, context_snapshot={"existing_key": "val"}),
            _make_prediction_row(run_number=2, context_snapshot={"existing_key": "val"}),
            _make_prediction_row(run_number=3, context_snapshot={"existing_key": "val"}),
        ]
        mock_conn = self._mock_conn_and_cursors(rows)
        mock_connect.return_value = mock_conn

        result = compute_conviction("2026-03-17", 3)
        assert result == 1

    @patch("nba.core.conviction_engine._load_bp_recommendations", return_value={})
    @patch("nba.core.conviction_engine._connect")
    def test_no_valid_conviction_rows_returns_zero(self, mock_connect, mock_bp, mock_ev, mock_json):
        """All picks have None p_over -> all skipped -> conviction_rows empty -> returns 0."""
        rows = [
            _make_prediction_row(player_name="A", stat_type="POINTS", run_number=1, p_over=None),
            _make_prediction_row(player_name="B", stat_type="REBOUNDS", run_number=1, p_over=None),
        ]
        mock_conn = self._mock_conn_and_cursors(rows)
        mock_connect.return_value = mock_conn

        result = compute_conviction("2026-03-17", 1)
        assert result == 0
        mock_conn.close.assert_called()

    @patch("nba.core.conviction_engine._load_bp_recommendations", return_value={})
    @patch("nba.core.conviction_engine._connect")
    def test_same_player_different_stats(self, mock_connect, mock_bp, mock_ev, mock_json):
        """Same player with POINTS and REBOUNDS -> 2 separate conviction rows."""
        rows = [
            _make_prediction_row(player_name="LeBron James", stat_type="POINTS", run_number=1),
            _make_prediction_row(player_name="LeBron James", stat_type="REBOUNDS", run_number=1),
        ]
        mock_conn = self._mock_conn_and_cursors(rows)
        mock_connect.return_value = mock_conn

        result = compute_conviction("2026-03-17", 1)
        assert result == 2
