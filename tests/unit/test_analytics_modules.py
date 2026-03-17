"""
Unit tests for analytics and utility modules:
- nba.betting_xl.analyze_tv_impact (compute_stats, chi_squared_test, tag_national_tv, load_all_picks)
- nba.betting_xl.analyze_cheatsheet_filters (tested via main patterns)
- nba.betting_xl.quick_refresh (run_step)

All DB/API calls are mocked. Focus on pure functions and logic branches.
"""

import json
import math
import os
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nba.betting_xl.analyze_tv_impact import (
    ESPN_TEAM_MAP,
    STAT_COLUMN,
    chi_squared_test,
    compute_stats,
    fetch_broadcasts_for_date,
    load_or_fetch_tv_cache,
    print_comparison,
    print_section,
    tag_national_tv,
)

# ═════════════════════════════════════════════════════════════════
# nba.betting_xl.analyze_tv_impact
# ═════════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────────
# compute_stats
# ─────────────────────────────────────────────────────────────────


class TestComputeStats:
    def test_empty_picks(self):
        result = compute_stats([], "empty")
        assert result["n"] == 0
        assert result["wins"] == 0
        assert result["wr"] == 0
        assert result["roi"] == 0
        assert result["label"] == "empty"

    def test_all_wins(self):
        picks = [{"won": True}] * 10
        result = compute_stats(picks, "winners")
        assert result["n"] == 10
        assert result["wins"] == 10
        assert result["losses"] == 0
        assert result["wr"] == 100.0
        # ROI = (10*0.909 - 0) / 10 * 100 = 90.9%
        assert result["roi"] == pytest.approx(90.9, abs=0.1)

    def test_all_losses(self):
        picks = [{"won": False}] * 10
        result = compute_stats(picks, "losers")
        assert result["n"] == 10
        assert result["wins"] == 0
        assert result["losses"] == 10
        assert result["wr"] == 0.0
        # ROI = (0 - 10) / 10 * 100 = -100%
        assert result["roi"] == -100.0

    def test_mixed_results(self):
        picks = [{"won": True}] * 6 + [{"won": False}] * 4
        result = compute_stats(picks, "mixed")
        assert result["n"] == 10
        assert result["wins"] == 6
        assert result["losses"] == 4
        assert result["wr"] == 60.0
        profit = 6 * 0.909 - 4 * 1.0
        expected_roi = profit / 10 * 100
        assert result["roi"] == pytest.approx(expected_roi, abs=0.1)
        assert result["profit"] == pytest.approx(profit, abs=0.01)

    def test_single_win(self):
        result = compute_stats([{"won": True}], "single")
        assert result["n"] == 1
        assert result["wr"] == 100.0


# ─────────────────────────────────────────────────────────────────
# chi_squared_test
# ─────────────────────────────────────────────────────────────────


class TestChiSquaredTest:
    def test_empty_groups(self):
        result = chi_squared_test([], [])
        assert result["chi2"] == 0
        assert result["p_value"] == 1.0
        assert result["significant"] is False

    def test_one_empty_group(self):
        result = chi_squared_test([{"won": True}], [])
        assert result["chi2"] == 0
        assert result["significant"] is False

    def test_identical_groups(self):
        a = [{"won": True}, {"won": False}]
        b = [{"won": True}, {"won": False}]
        result = chi_squared_test(a, b)
        assert result["chi2"] == pytest.approx(0.0, abs=0.01)
        assert result["significant"] is False

    def test_extreme_difference(self):
        """One group all wins, other all losses → high chi2."""
        a = [{"won": True}] * 50
        b = [{"won": False}] * 50
        result = chi_squared_test(a, b)
        assert result["chi2"] > 10
        assert result["p_value"] < 0.05
        assert result["significant"] is True

    def test_moderate_difference(self):
        """Test that results are reasonable for moderate difference."""
        a = [{"won": True}] * 30 + [{"won": False}] * 20
        b = [{"won": True}] * 20 + [{"won": False}] * 30
        result = chi_squared_test(a, b)
        assert result["chi2"] > 0
        assert 0 <= result["p_value"] <= 1

    def test_p_value_uses_erfc(self):
        """Verify p_value is computed using erfc."""
        a = [{"won": True}] * 10 + [{"won": False}] * 10
        b = [{"won": True}] * 15 + [{"won": False}] * 5
        result = chi_squared_test(a, b)
        # Re-derive expected p_value
        expected_p = math.erfc(math.sqrt(result["chi2"] / 2))
        assert result["p_value"] == pytest.approx(expected_p, abs=0.0001)


# ─────────────────────────────────────────────────────────────────
# tag_national_tv
# ─────────────────────────────────────────────────────────────────


class TestTagNationalTv:
    def test_tags_national_game(self):
        picks = [
            {"game_date": "2026-03-16", "team_abbrev": "LAL", "opponent_team": "GSW"},
        ]
        tv_cache = {
            "2026-03-16": {
                "national": {"LAL_GSW": ["ESPN", "TNT"]},
                "all": ["LAL_GSW"],
            },
        }
        result = tag_national_tv(picks, tv_cache)
        assert result[0]["is_national_tv"] is True
        assert result[0]["tv_networks"] == ["ESPN", "TNT"]

    def test_tags_local_game(self):
        picks = [
            {"game_date": "2026-03-16", "team_abbrev": "ORL", "opponent_team": "CHA"},
        ]
        tv_cache = {
            "2026-03-16": {
                "national": {"LAL_GSW": ["ESPN"]},
                "all": ["LAL_GSW", "ORL_CHA"],
            },
        }
        result = tag_national_tv(picks, tv_cache)
        assert result[0]["is_national_tv"] is False
        assert result[0]["tv_networks"] == []

    def test_missing_date_in_cache(self):
        picks = [
            {"game_date": "2026-03-20", "team_abbrev": "BOS", "opponent_team": "MIL"},
        ]
        tv_cache = {}
        result = tag_national_tv(picks, tv_cache)
        assert result[0]["is_national_tv"] is False

    def test_multiple_picks(self):
        picks = [
            {"game_date": "2026-03-16", "team_abbrev": "LAL", "opponent_team": "GSW"},
            {"game_date": "2026-03-16", "team_abbrev": "ORL", "opponent_team": "CHA"},
        ]
        tv_cache = {
            "2026-03-16": {
                "national": {"LAL_GSW": ["ABC"]},
                "all": ["LAL_GSW", "ORL_CHA"],
            },
        }
        result = tag_national_tv(picks, tv_cache)
        assert result[0]["is_national_tv"] is True
        assert result[1]["is_national_tv"] is False


# ─────────────────────────────────────────────────────────────────
# ESPN_TEAM_MAP and STAT_COLUMN
# ─────────────────────────────────────────────────────────────────


class TestTvImpactConstants:
    def test_espn_team_map(self):
        assert ESPN_TEAM_MAP["WSH"] == "WAS"
        assert ESPN_TEAM_MAP["GS"] == "GSW"
        assert ESPN_TEAM_MAP["NO"] == "NOP"
        assert ESPN_TEAM_MAP["SA"] == "SAS"

    def test_stat_column(self):
        assert STAT_COLUMN["POINTS"] == "points"
        assert STAT_COLUMN["REBOUNDS"] == "rebounds"
        assert STAT_COLUMN["ASSISTS"] == "assists"
        assert STAT_COLUMN["THREES"] == "three_pointers_made"


# ─────────────────────────────────────────────────────────────────
# fetch_broadcasts_for_date
# ─────────────────────────────────────────────────────────────────


class TestFetchBroadcasts:
    @patch("nba.betting_xl.analyze_tv_impact.requests.get")
    def test_parses_espn_response(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "events": [
                {
                    "shortName": "OKC @ NYK",
                    "competitions": [
                        {
                            "geoBroadcasts": [
                                {
                                    "market": {"type": "National"},
                                    "type": {"shortName": "TV"},
                                    "media": {"shortName": "ESPN"},
                                },
                            ],
                        },
                    ],
                },
                {
                    "shortName": "ORL @ CHA",
                    "competitions": [
                        {
                            "geoBroadcasts": [
                                {
                                    "market": {"type": "Local"},
                                    "type": {"shortName": "TV"},
                                    "media": {"shortName": "BSFL"},
                                },
                            ],
                        },
                    ],
                },
            ],
        }
        mock_get.return_value = mock_resp

        result = fetch_broadcasts_for_date("2026-03-16")
        assert "national" in result
        assert "all" in result
        # OKC @ NYK → NYK_OKC (NY maps to NYK)
        assert "NYK_OKC" in result["national"]
        assert result["national"]["NYK_OKC"] == ["ESPN"]
        # ORL vs CHA is not national
        assert "CHA_ORL" not in result["national"]

    @patch("nba.betting_xl.analyze_tv_impact.requests.get")
    def test_empty_events(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"events": []}
        mock_get.return_value = mock_resp

        result = fetch_broadcasts_for_date("2026-03-16")
        assert result["national"] == {}
        assert result["all"] == []

    @patch("nba.betting_xl.analyze_tv_impact.requests.get")
    def test_malformed_shortname_skipped(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "events": [{"shortName": "INVALID", "competitions": []}],
        }
        mock_get.return_value = mock_resp
        result = fetch_broadcasts_for_date("2026-03-16")
        assert result["national"] == {}


# ─────────────────────────────────────────────────────────────────
# load_or_fetch_tv_cache
# ─────────────────────────────────────────────────────────────────


class TestLoadOrFetchTvCache:
    def test_loads_from_cache_file(self, tmp_path):
        cache_file = tmp_path / "tv_cache.json"
        cache_data = {"2026-03-16": {"national": {}, "all": []}}
        cache_file.write_text(json.dumps(cache_data))

        with patch("nba.betting_xl.analyze_tv_impact.TV_CACHE_FILE", cache_file):
            result = load_or_fetch_tv_cache(["2026-03-16"])
        assert "2026-03-16" in result

    @patch("nba.betting_xl.analyze_tv_impact.fetch_broadcasts_for_date")
    def test_fetches_missing_dates(self, mock_fetch, tmp_path):
        cache_file = tmp_path / "tv_cache.json"
        # No existing cache
        mock_fetch.return_value = {"national": {"LAL_GSW": ["ESPN"]}, "all": ["LAL_GSW"]}

        with (
            patch("nba.betting_xl.analyze_tv_impact.TV_CACHE_FILE", cache_file),
            patch("nba.betting_xl.analyze_tv_impact.time.sleep"),
        ):
            result = load_or_fetch_tv_cache(["2026-03-16"])

        assert "2026-03-16" in result
        assert result["2026-03-16"]["national"]["LAL_GSW"] == ["ESPN"]
        # Should have written cache file
        assert cache_file.exists()

    @patch("nba.betting_xl.analyze_tv_impact.fetch_broadcasts_for_date")
    def test_handles_fetch_error(self, mock_fetch, tmp_path):
        cache_file = tmp_path / "tv_cache.json"
        mock_fetch.side_effect = Exception("network error")

        with (
            patch("nba.betting_xl.analyze_tv_impact.TV_CACHE_FILE", cache_file),
            patch("nba.betting_xl.analyze_tv_impact.time.sleep"),
        ):
            result = load_or_fetch_tv_cache(["2026-03-16"])

        assert "2026-03-16" in result
        assert result["2026-03-16"]["national"] == {}


# ─────────────────────────────────────────────────────────────────
# load_all_picks
# ─────────────────────────────────────────────────────────────────


class TestLoadAllPicks:
    def test_loads_xl_picks(self, tmp_path):
        from nba.betting_xl.analyze_tv_impact import load_all_picks

        picks_data = {
            "date": "2026-03-16",
            "picks": [
                {
                    "player_name": "LeBron James",
                    "stat_type": "POINTS",
                    "side": "OVER",
                    "best_line": 25.5,
                    "opponent_team": "GSW",
                    "is_home": True,
                    "model_version": "xl",
                    "filter_tier": "X",
                    "confidence": "HIGH",
                },
            ],
        }
        pred_dir = tmp_path / "predictions"
        pred_dir.mkdir()
        (pred_dir / "xl_picks_2026-03-16.json").write_text(json.dumps(picks_data))

        with patch("nba.betting_xl.analyze_tv_impact.PREDICTIONS_DIR", pred_dir):
            picks = load_all_picks()

        assert len(picks) == 1
        assert picks[0]["player_name"] == "LeBron James"
        assert picks[0]["system"] == "XL"
        assert picks[0]["line"] == 25.5
        assert picks[0]["game_date"] == "2026-03-16"

    def test_skips_invalid_stat_types(self, tmp_path):
        from nba.betting_xl.analyze_tv_impact import load_all_picks

        picks_data = {
            "date": "2026-03-16",
            "picks": [
                {"player_name": "A", "stat_type": "INVALID_STAT", "best_line": 10},
                {"player_name": "B", "stat_type": "POINTS", "best_line": 20},
            ],
        }
        pred_dir = tmp_path / "predictions"
        pred_dir.mkdir()
        (pred_dir / "xl_picks_2026-03-16.json").write_text(json.dumps(picks_data))

        with patch("nba.betting_xl.analyze_tv_impact.PREDICTIONS_DIR", pred_dir):
            picks = load_all_picks()

        assert len(picks) == 1
        assert picks[0]["player_name"] == "B"

    def test_skips_bad_json(self, tmp_path):
        from nba.betting_xl.analyze_tv_impact import load_all_picks

        pred_dir = tmp_path / "predictions"
        pred_dir.mkdir()
        (pred_dir / "xl_picks_2026-03-16.json").write_text("NOT JSON")

        with patch("nba.betting_xl.analyze_tv_impact.PREDICTIONS_DIR", pred_dir):
            picks = load_all_picks()

        assert picks == []

    def test_date_normalization(self, tmp_path):
        from nba.betting_xl.analyze_tv_impact import load_all_picks

        picks_data = {
            "date": "20260316",  # Compact format
            "picks": [
                {"player_name": "A", "stat_type": "POINTS", "best_line": 20},
            ],
        }
        pred_dir = tmp_path / "predictions"
        pred_dir.mkdir()
        (pred_dir / "xl_picks_20260316.json").write_text(json.dumps(picks_data))

        with patch("nba.betting_xl.analyze_tv_impact.PREDICTIONS_DIR", pred_dir):
            picks = load_all_picks()

        assert len(picks) == 1
        assert picks[0]["game_date"] == "2026-03-16"


# ─────────────────────────────────────────────────────────────────
# grade_picks
# ─────────────────────────────────────────────────────────────────


class TestGradePicks:
    @patch("nba.betting_xl.analyze_tv_impact.psycopg2.connect")
    def test_grades_correctly(self, mock_connect):
        from nba.betting_xl.analyze_tv_impact import grade_picks

        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = [
            ("LeBron James", "2026-03-16", "LAL", "GSW", 30, 8, 7, 3, 36),
        ]
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cur
        mock_connect.return_value = mock_conn

        picks = [
            {
                "player_name": "LeBron James",
                "stat_type": "POINTS",
                "side": "OVER",
                "line": 25.5,
                "game_date": "2026-03-16",
            },
        ]
        graded = grade_picks(picks)
        assert len(graded) == 1
        assert graded[0]["actual"] == 30
        assert graded[0]["won"] is True

    @patch("nba.betting_xl.analyze_tv_impact.psycopg2.connect")
    def test_push_is_loss(self, mock_connect):
        from nba.betting_xl.analyze_tv_impact import grade_picks

        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = [
            ("Player A", "2026-03-16", "BOS", "MIL", 25, 10, 5, 2, 34),
        ]
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cur
        mock_connect.return_value = mock_conn

        picks = [
            {
                "player_name": "Player A",
                "stat_type": "POINTS",
                "side": "OVER",
                "line": 25.0,  # actual == line → push → loss
                "game_date": "2026-03-16",
            },
        ]
        graded = grade_picks(picks)
        assert graded[0]["won"] is False

    @patch("nba.betting_xl.analyze_tv_impact.psycopg2.connect")
    def test_under_pick(self, mock_connect):
        from nba.betting_xl.analyze_tv_impact import grade_picks

        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = [
            ("Player B", "2026-03-16", "PHX", "DAL", 20, 5, 4, 1, 30),
        ]
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cur
        mock_connect.return_value = mock_conn

        picks = [
            {
                "player_name": "Player B",
                "stat_type": "POINTS",
                "side": "UNDER",
                "line": 25.0,  # actual=20 < 25 → under wins
                "game_date": "2026-03-16",
            },
        ]
        graded = grade_picks(picks)
        assert graded[0]["won"] is True


# ─────────────────────────────────────────────────────────────────
# print_section and print_comparison (output tests)
# ─────────────────────────────────────────────────────────────────


class TestPrintHelpers:
    def test_print_section(self, capsys):
        print_section("TEST SECTION")
        out = capsys.readouterr().out
        assert "TEST SECTION" in out
        assert "=" in out

    def test_print_comparison(self, capsys):
        ntv = {"n": 10, "wins": 7, "losses": 3, "wr": 70.0, "roi": 36.36}
        local = {"n": 20, "wins": 10, "losses": 10, "wr": 50.0, "roi": -4.55}
        test = {"chi2": 1.234, "p_value": 0.456, "significant": False}
        print_comparison(ntv, local, test)
        out = capsys.readouterr().out
        assert "National TV" in out
        assert "Local Only" in out
        assert "Chi-squared" in out


# ═════════════════════════════════════════════════════════════════
# nba.betting_xl.quick_refresh
# ═════════════════════════════════════════════════════════════════


from nba.betting_xl.quick_refresh import run_step


class TestRunStep:
    def test_successful_command(self):
        name, ok, elapsed, output = run_step("test", "echo hello", timeout=10)
        assert name == "test"
        assert ok is True
        assert elapsed < 10
        assert "hello" in output

    def test_failing_command(self):
        name, ok, elapsed, output = run_step("fail_test", "false", timeout=10)
        assert name == "fail_test"
        assert ok is False

    def test_timeout_command(self):
        name, ok, elapsed, output = run_step("timeout_test", "sleep 60", timeout=1)
        assert name == "timeout_test"
        assert ok is False
        assert "TIMEOUT" in output

    def test_nonexistent_command(self):
        name, ok, elapsed, output = run_step("noexist", "nonexistent_command_xyz_123", timeout=5)
        assert name == "noexist"
        assert ok is False

    def test_output_truncated(self):
        """Verify stdout is truncated to last 500 chars on success."""
        # Generate more than 500 chars of output
        name, ok, elapsed, output = run_step(
            "long_output", "python3 -c \"print('x' * 1000)\"", timeout=10
        )
        assert ok is True
        assert len(output) <= 501  # 500 chars + possible newline


# ═════════════════════════════════════════════════════════════════
# nba.betting_xl.analyze_cheatsheet_filters
# (Module is DB-heavy. Test constants and import.)
# ═════════════════════════════════════════════════════════════════


class TestAnalyzeCheatsheetFilters:
    """Test that the module imports without error and has expected structure."""

    def test_module_imports(self):
        import nba.betting_xl.analyze_cheatsheet_filters as acf

        assert hasattr(acf, "main")
        assert hasattr(acf, "analyze_losses")

    @patch("nba.betting_xl.analyze_cheatsheet_filters.psycopg2.connect")
    def test_analyze_losses_queries_db(self, mock_connect):
        """Test analyze_losses wires up DB queries. We just verify it calls connect."""
        from nba.betting_xl.analyze_cheatsheet_filters import analyze_losses

        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = []
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cur
        mock_connect.return_value = mock_conn

        analyze_losses()
        # Should connect to both intel and players DBs
        assert mock_connect.call_count == 2

    @patch("nba.betting_xl.analyze_cheatsheet_filters.psycopg2.connect")
    def test_main_queries_db(self, mock_connect):
        """Test main function wires up DB and handles empty data."""
        from nba.betting_xl.analyze_cheatsheet_filters import main

        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = []
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cur
        mock_connect.return_value = mock_conn

        main()
        assert mock_connect.call_count == 2

    @patch("nba.betting_xl.analyze_cheatsheet_filters.psycopg2.connect")
    def test_analyze_losses_with_data(self, mock_connect, capsys):
        """Test analyze_losses with sample cheatsheet + actuals data."""
        from nba.betting_xl.analyze_cheatsheet_filters import analyze_losses

        intel_cur = MagicMock()
        intel_cur.fetchall.return_value = [
            # player, date, stat, line, proj, proj_diff, rating, ev_pct, opp_rank, hr_l5, hr_l15, hr_szn
            ("LeBron James", "2026-03-01", "POINTS", 25.5, 28.0, 2.5, 4, 15, 20, 0.80, 0.70, 0.75),
        ]
        players_cur = MagicMock()
        players_cur.fetchall.return_value = [
            # name, date, pts, reb, ast
            ("LeBron James", "2026-03-01", 30, 8, 7),
        ]

        mock_intel_conn = MagicMock()
        mock_intel_conn.cursor.return_value = intel_cur
        mock_players_conn = MagicMock()
        mock_players_conn.cursor.return_value = players_cur

        mock_connect.side_effect = [mock_intel_conn, mock_players_conn]

        analyze_losses()
        out = capsys.readouterr().out
        # Should produce output (even if minimal due to filter criteria)
        assert "POINTS" in out or "Additional filter" in out

    @patch("nba.betting_xl.analyze_cheatsheet_filters.psycopg2.connect")
    def test_main_with_data(self, mock_connect, capsys):
        """Test main with sample data to cover filter loop."""
        from nba.betting_xl.analyze_cheatsheet_filters import main

        intel_cur = MagicMock()
        intel_cur.fetchall.return_value = [
            ("Player A", "2026-03-01", "POINTS", 20.5, 23.0, 2.5, 4, 20, 15, 0.80, 0.70, 0.75),
            ("Player B", "2026-03-01", "POINTS", 18.0, 20.0, 2.0, 3, 15, 22, 0.60, 0.60, 0.60),
            ("Player C", "2026-03-01", "POINTS", 22.0, 24.0, 2.0, 5, 25, 25, 0.90, 0.80, 0.80),
            ("Player D", "2026-03-01", "REBOUNDS", 8.5, 10.0, 1.5, 4, 18, 11, 0.70, 0.65, 0.70),
        ]
        players_cur = MagicMock()
        players_cur.fetchall.return_value = [
            ("Player A", "2026-03-01", 25, 6, 5, 2),
            ("Player B", "2026-03-01", 15, 4, 3, 1),
            ("Player C", "2026-03-01", 28, 7, 6, 3),
            ("Player D", "2026-03-01", 10, 9, 4, 0),
        ]

        mock_intel_conn = MagicMock()
        mock_intel_conn.cursor.return_value = intel_cur
        mock_players_conn = MagicMock()
        mock_players_conn.cursor.return_value = players_cur

        mock_connect.side_effect = [mock_intel_conn, mock_players_conn]

        main()
        out = capsys.readouterr().out
        assert "POINTS" in out
        assert "Filter" in out
