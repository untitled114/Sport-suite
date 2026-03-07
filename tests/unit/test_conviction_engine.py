"""
Unit tests for nba.core.conviction_engine — pure functions only.

All tests bypass the DB (no psycopg2 connections).
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

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
)

# ─────────────────────────────────────────────────────────────────
# _normalize
# ─────────────────────────────────────────────────────────────────


class TestConnect:
    def test_returns_connection(self):
        mock_conn = MagicMock()
        with patch("psycopg2.connect", return_value=mock_conn):
            result = _connect()
        assert result is mock_conn


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
        # max == min → returns 0.5
        assert _normalize(5.0, 5.0, 5.0) == pytest.approx(0.5)


# ─────────────────────────────────────────────────────────────────
# _score_appearance
# ─────────────────────────────────────────────────────────────────


class TestScoreAppearance:
    def test_perfect_from_run1(self):
        # 6 appearances across all 6 fired runs, entered at run 1 → full score
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
        # Entry at run 6 → entry_factor floored at 0.60
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
        # Movement < 0.25 is noise → returns the stable score (0.55)
        assert _score_line_movement(0.1) == pytest.approx(0.55)
        assert _score_line_movement(-0.1) == pytest.approx(0.55)
        assert _score_line_movement(0.0) == pytest.approx(0.55)

    def test_line_dropped_is_good(self):
        # Negative movement (line softened) → score above 0.55
        score = _score_line_movement(-1.0)
        assert score > 0.55

    def test_line_rose_is_bad(self):
        # Positive movement (line rose) → score below 0.55
        score = _score_line_movement(1.0)
        assert score < 0.55

    def test_boundary_at_noise(self):
        # Exactly at boundary — 0.25 is above noise threshold
        score = _score_line_movement(0.25)
        assert score != pytest.approx(0.55)


# ─────────────────────────────────────────────────────────────────
# _get_label
# ─────────────────────────────────────────────────────────────────


class TestGetLabel:
    def test_early_on_first_appearance(self):
        assert _get_label(0.95, 1) == "EARLY"

    def test_locked_requires_high_conviction_and_appearances(self):
        assert _get_label(0.85, 3) == "LOCKED"
        assert _get_label(0.85, 5) == "LOCKED"

    def test_locked_blocked_below_min_appearances(self):
        # 0.85 conviction but only 2 appearances → STRONG not LOCKED
        assert _get_label(0.85, 2) == "STRONG"

    def test_strong(self):
        assert _get_label(0.70, 3) == "STRONG"
        assert _get_label(0.60, 2) == "STRONG"

    def test_watch(self):
        assert _get_label(0.50, 3) == "WATCH"
        assert _get_label(0.40, 4) == "WATCH"

    def test_skip(self):
        assert _get_label(0.30, 4) == "SKIP"
        assert _get_label(0.0, 6) == "SKIP"

    def test_boundary_locked_at_0_80(self):
        assert _get_label(0.80, 3) == "LOCKED"

    def test_boundary_strong_at_0_60(self):
        assert _get_label(0.60, 3) == "STRONG"


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
        stats = {"X": self._make_stats(3, 10)}  # 30% WR — below 0.47
        delta, note = _tier_score_adjustment("X", stats)
        assert delta == pytest.approx(-0.08)
        assert "UNDERPERFORMING" in note

    def test_strong_tier_bonus(self):
        stats = {"META": self._make_stats(8, 10)}  # 80% WR — above 0.63
        delta, note = _tier_score_adjustment("META", stats)
        assert delta == pytest.approx(0.04)
        assert "performing well" in note

    def test_neutral_tier_no_adjustment(self):
        stats = {"Z": self._make_stats(5, 10)}  # 50% WR — between thresholds
        delta, note = _tier_score_adjustment("Z", stats)
        assert delta == 0.0
        assert "Z L14:" in note

    def test_boundary_underperform(self):
        # WR exactly at 0.47 → not below threshold → no penalty
        stats = {"X": self._make_stats(47, 100)}
        delta, _ = _tier_score_adjustment("X", stats)
        assert delta == 0.0

    def test_boundary_strong(self):
        # WR exactly at 0.63 → not above threshold → no bonus
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
    """Tests for narrative generation — verifies key phrases appear."""

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
        text = self._build(tier_note="X L14: 8/10 (80.0%) — performing well")
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
    def _rec(self, stars, side, projection=None, best_ev=None):
        return {
            "bp_bet_rating": stars,
            "bp_recommended_side": side,
            "bp_projection": projection,
            "bp_expected_value": 0.1,
            "best_ev": best_ev or 0.0,
            "best_book": "draftkings",
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
        assert delta == 0.11
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
        # bp_projection (23.5) is within 1.5 of our line (24.5) → genuine conflict
        delta, ctx = _score_bp_rec(self._rec(4, "under", projection=23.5), our_best_line=24.5)
        assert delta == -0.06
        assert ctx["bp_rec_conflict_type"] == "genuine"

    def test_under_5star_genuine_conflict(self):
        delta, ctx = _score_bp_rec(self._rec(5, "under", projection=24.0), our_best_line=24.5)
        assert delta == -0.06
        assert ctx["bp_rec_conflict_type"] == "genuine"

    def test_under_4star_odds_play_no_penalty(self):
        # bp_projection (22.4) is > 1.5 below our line (24.5) → odds play
        delta, ctx = _score_bp_rec(self._rec(4, "under", projection=22.4), our_best_line=24.5)
        assert delta == 0.0
        assert ctx["bp_rec_conflict_type"] == "odds_play"

    def test_under_5star_odds_play_no_penalty(self):
        delta, ctx = _score_bp_rec(self._rec(5, "under", projection=20.0), our_best_line=24.5)
        assert delta == 0.0
        assert ctx["bp_rec_conflict_type"] == "odds_play"

    def test_under_no_line_provided_assumes_genuine(self):
        # No our_best_line → can't determine odds play → apply penalty
        delta, ctx = _score_bp_rec(self._rec(4, "under", projection=22.4))
        assert delta == -0.06
        assert ctx["bp_rec_conflict_type"] == "genuine"

    def test_under_no_projection_assumes_genuine(self):
        # No bp_projection → can't determine odds play → apply penalty
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
