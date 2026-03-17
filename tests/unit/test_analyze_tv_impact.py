"""
Unit Tests for analyze_tv_impact.py
====================================
Tests for NBA pick performance analysis split by TV broadcast status.
All external calls (ESPN API, database, file I/O) are mocked.
"""

import json
import math
from datetime import datetime
from unittest.mock import MagicMock, mock_open, patch

import pytest


# ---------------------------------------------------------------------------
# fetch_broadcasts_for_date
# ---------------------------------------------------------------------------
class TestFetchBroadcastsForDate:
    """Tests for the ESPN API broadcast fetcher."""

    @patch("nba.betting_xl.analyze_tv_impact.requests.get")
    def test_returns_national_games(self, mock_get):
        """Test parsing national TV games from ESPN response."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "events": [
                {
                    "shortName": "LAL @ NYK",
                    "competitions": [
                        {
                            "geoBroadcasts": [
                                {
                                    "market": {"type": "National"},
                                    "type": {"shortName": "TV"},
                                    "media": {"shortName": "ESPN"},
                                }
                            ]
                        }
                    ],
                }
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        from nba.betting_xl.analyze_tv_impact import fetch_broadcasts_for_date

        result = fetch_broadcasts_for_date("2025-01-15")

        assert "national" in result
        assert "all" in result
        # NY maps to NYK via ESPN_TEAM_MAP
        assert "NYK_LAL" in result["national"]
        assert result["national"]["NYK_LAL"] == ["ESPN"]
        assert "NYK_LAL" in result["all"]

    @patch("nba.betting_xl.analyze_tv_impact.requests.get")
    def test_local_only_game(self, mock_get):
        """Test a game with no national broadcasts."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "events": [
                {
                    "shortName": "MEM @ CLE",
                    "competitions": [
                        {
                            "geoBroadcasts": [
                                {
                                    "market": {"type": "Local"},
                                    "type": {"shortName": "TV"},
                                    "media": {"shortName": "BSOH"},
                                }
                            ]
                        }
                    ],
                }
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        from nba.betting_xl.analyze_tv_impact import fetch_broadcasts_for_date

        result = fetch_broadcasts_for_date("2025-01-15")

        assert result["national"] == {}
        assert "CLE_MEM" in result["all"]

    @patch("nba.betting_xl.analyze_tv_impact.requests.get")
    def test_espn_team_map_applied(self, mock_get):
        """Test that ESPN team abbreviation mappings are applied."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "events": [
                {
                    "shortName": "GS @ WSH",
                    "competitions": [
                        {
                            "geoBroadcasts": [
                                {
                                    "market": {"type": "National"},
                                    "type": {"shortName": "TV"},
                                    "media": {"shortName": "TNT"},
                                }
                            ]
                        }
                    ],
                }
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        from nba.betting_xl.analyze_tv_impact import fetch_broadcasts_for_date

        result = fetch_broadcasts_for_date("2025-01-15")

        # GS -> GSW, WSH -> WAS
        assert "WAS_GSW" in result["national"]

    @patch("nba.betting_xl.analyze_tv_impact.requests.get")
    def test_invalid_short_name_skipped(self, mock_get):
        """Test that events without proper shortName format are skipped."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "events": [
                {
                    "shortName": "All-Star Game",
                    "competitions": [{"geoBroadcasts": []}],
                }
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        from nba.betting_xl.analyze_tv_impact import fetch_broadcasts_for_date

        result = fetch_broadcasts_for_date("2025-02-15")

        assert result["national"] == {}
        assert result["all"] == []

    @patch("nba.betting_xl.analyze_tv_impact.requests.get")
    def test_empty_events(self, mock_get):
        """Test empty events list (no games on date)."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"events": []}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        from nba.betting_xl.analyze_tv_impact import fetch_broadcasts_for_date

        result = fetch_broadcasts_for_date("2025-07-15")

        assert result["national"] == {}
        assert result["all"] == []

    @patch("nba.betting_xl.analyze_tv_impact.requests.get")
    def test_multiple_national_networks(self, mock_get):
        """Test a game broadcast on multiple national networks."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "events": [
                {
                    "shortName": "BOS @ LAL",
                    "competitions": [
                        {
                            "geoBroadcasts": [
                                {
                                    "market": {"type": "National"},
                                    "type": {"shortName": "TV"},
                                    "media": {"shortName": "ABC"},
                                },
                                {
                                    "market": {"type": "National"},
                                    "type": {"shortName": "TV"},
                                    "media": {"shortName": "ESPN"},
                                },
                            ]
                        }
                    ],
                }
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        from nba.betting_xl.analyze_tv_impact import fetch_broadcasts_for_date

        result = fetch_broadcasts_for_date("2025-01-15")

        assert "LAL_BOS" in result["national"]
        assert "ABC" in result["national"]["LAL_BOS"]
        assert "ESPN" in result["national"]["LAL_BOS"]


# ---------------------------------------------------------------------------
# load_or_fetch_tv_cache
# ---------------------------------------------------------------------------
class TestLoadOrFetchTvCache:
    """Tests for TV broadcast cache loading/fetching."""

    @patch("nba.betting_xl.analyze_tv_impact.TV_CACHE_FILE")
    @patch("nba.betting_xl.analyze_tv_impact.fetch_broadcasts_for_date")
    @patch("nba.betting_xl.analyze_tv_impact.time.sleep")
    @patch("builtins.open", new_callable=mock_open)
    def test_fetches_missing_dates(self, mock_file, mock_sleep, mock_fetch, mock_cache_path):
        """Test that missing dates are fetched from ESPN API."""
        mock_cache_path.exists.return_value = False
        mock_fetch.return_value = {"national": {}, "all": []}

        from nba.betting_xl.analyze_tv_impact import load_or_fetch_tv_cache

        result = load_or_fetch_tv_cache(["2025-01-15", "2025-01-16"])

        assert mock_fetch.call_count == 2
        assert "2025-01-15" in result
        assert "2025-01-16" in result

    @patch("nba.betting_xl.analyze_tv_impact.TV_CACHE_FILE")
    @patch("builtins.open", new_callable=mock_open)
    def test_uses_cached_data(self, mock_file, mock_cache_path):
        """Test that cached dates are not re-fetched."""
        cached = {"2025-01-15": {"national": {"LAL_BOS": ["ESPN"]}, "all": ["LAL_BOS"]}}
        mock_cache_path.exists.return_value = True
        mock_file.return_value.read.return_value = json.dumps(cached)

        # When all dates are cached, the read path returns the json.load result
        with patch("json.load", return_value=cached):
            from nba.betting_xl.analyze_tv_impact import load_or_fetch_tv_cache

            result = load_or_fetch_tv_cache(["2025-01-15"])

        assert "2025-01-15" in result

    @patch("nba.betting_xl.analyze_tv_impact.TV_CACHE_FILE")
    @patch("nba.betting_xl.analyze_tv_impact.fetch_broadcasts_for_date")
    @patch("nba.betting_xl.analyze_tv_impact.time.sleep")
    @patch("builtins.open", new_callable=mock_open)
    def test_handles_fetch_errors(self, mock_file, mock_sleep, mock_fetch, mock_cache_path):
        """Test that API errors are caught and cached as empty."""
        mock_cache_path.exists.return_value = False
        mock_fetch.side_effect = Exception("API error")

        from nba.betting_xl.analyze_tv_impact import load_or_fetch_tv_cache

        result = load_or_fetch_tv_cache(["2025-01-15"])

        assert "2025-01-15" in result
        assert result["2025-01-15"]["national"] == {}

    @patch("nba.betting_xl.analyze_tv_impact.TV_CACHE_FILE")
    @patch("nba.betting_xl.analyze_tv_impact.fetch_broadcasts_for_date")
    @patch("nba.betting_xl.analyze_tv_impact.time.sleep")
    @patch("builtins.open", new_callable=mock_open)
    def test_progress_print_at_20_dates(
        self, mock_file, mock_sleep, mock_fetch, mock_cache_path, capsys
    ):
        """Test progress print is shown every 20 dates."""
        mock_cache_path.exists.return_value = False
        mock_fetch.return_value = {"national": {}, "all": []}

        dates = [f"2025-01-{str(d).zfill(2)}" for d in range(1, 22)]  # 21 dates

        from nba.betting_xl.analyze_tv_impact import load_or_fetch_tv_cache

        result = load_or_fetch_tv_cache(dates)

        captured = capsys.readouterr()
        assert "...fetched 20/" in captured.out
        assert mock_fetch.call_count == 21


# ---------------------------------------------------------------------------
# load_all_picks
# ---------------------------------------------------------------------------
class TestLoadAllPicks:
    """Tests for loading picks from JSON files."""

    @patch("nba.betting_xl.analyze_tv_impact.PREDICTIONS_DIR")
    @patch("nba.betting_xl.analyze_tv_impact.glob.glob")
    @patch("builtins.open", new_callable=mock_open)
    def test_loads_xl_picks(self, mock_file, mock_glob, mock_dir):
        """Test loading XL picks from JSON files."""
        pick_data = {
            "date": "2025-01-15",
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
                }
            ],
        }
        mock_glob.side_effect = [
            ["/fake/xl_picks_2025-01-15.json"],
            [],  # No PRO picks
        ]
        mock_file.return_value.read.return_value = json.dumps(pick_data)

        with patch("json.load", return_value=pick_data):
            from nba.betting_xl.analyze_tv_impact import load_all_picks

            picks = load_all_picks()

        assert len(picks) == 1
        assert picks[0]["player_name"] == "LeBron James"
        assert picks[0]["system"] == "XL"
        assert picks[0]["game_date"] == "2025-01-15"

    @patch("nba.betting_xl.analyze_tv_impact.PREDICTIONS_DIR")
    @patch("nba.betting_xl.analyze_tv_impact.glob.glob")
    @patch("builtins.open", new_callable=mock_open)
    def test_skips_invalid_stat_types(self, mock_file, mock_glob, mock_dir):
        """Test that picks with invalid stat types are skipped."""
        pick_data = {
            "date": "2025-01-15",
            "picks": [
                {
                    "player_name": "LeBron James",
                    "stat_type": "STEALS",  # Not in STAT_COLUMN
                    "side": "OVER",
                    "best_line": 1.5,
                }
            ],
        }
        mock_glob.side_effect = [["/fake/xl_picks_2025-01-15.json"], []]
        with patch("json.load", return_value=pick_data):
            from nba.betting_xl.analyze_tv_impact import load_all_picks

            picks = load_all_picks()

        assert len(picks) == 0

    @patch("nba.betting_xl.analyze_tv_impact.PREDICTIONS_DIR")
    @patch("nba.betting_xl.analyze_tv_impact.glob.glob")
    @patch("builtins.open", new_callable=mock_open)
    def test_skips_missing_player_name(self, mock_file, mock_glob, mock_dir):
        """Test that picks without player_name are skipped."""
        pick_data = {
            "date": "2025-01-15",
            "picks": [
                {
                    "stat_type": "POINTS",
                    "side": "OVER",
                    "best_line": 25.5,
                }
            ],
        }
        mock_glob.side_effect = [["/fake/xl_picks_2025-01-15.json"], []]
        with patch("json.load", return_value=pick_data):
            from nba.betting_xl.analyze_tv_impact import load_all_picks

            picks = load_all_picks()

        assert len(picks) == 0

    @patch("nba.betting_xl.analyze_tv_impact.PREDICTIONS_DIR")
    @patch("nba.betting_xl.analyze_tv_impact.glob.glob")
    @patch("builtins.open", new_callable=mock_open)
    def test_date_format_normalization(self, mock_file, mock_glob, mock_dir):
        """Test that YYYYMMDD date format is normalized to YYYY-MM-DD."""
        pick_data = {
            "date": "20250115",
            "picks": [
                {
                    "player_name": "LeBron James",
                    "stat_type": "POINTS",
                    "side": "OVER",
                    "best_line": 25.5,
                }
            ],
        }
        mock_glob.side_effect = [["/fake/xl_picks_20250115.json"], []]
        with patch("json.load", return_value=pick_data):
            from nba.betting_xl.analyze_tv_impact import load_all_picks

            picks = load_all_picks()

        assert len(picks) == 1
        assert picks[0]["game_date"] == "2025-01-15"

    @patch("nba.betting_xl.analyze_tv_impact.PREDICTIONS_DIR")
    @patch("nba.betting_xl.analyze_tv_impact.glob.glob")
    @patch("builtins.open", new_callable=mock_open)
    def test_handles_json_decode_error(self, mock_file, mock_glob, mock_dir):
        """Test that invalid JSON files are skipped gracefully."""
        mock_glob.side_effect = [["/fake/xl_picks_2025-01-15.json"], []]
        with patch("json.load", side_effect=json.JSONDecodeError("err", "doc", 0)):
            from nba.betting_xl.analyze_tv_impact import load_all_picks

            picks = load_all_picks()

        assert len(picks) == 0

    @patch("nba.betting_xl.analyze_tv_impact.PREDICTIONS_DIR")
    @patch("nba.betting_xl.analyze_tv_impact.glob.glob")
    def test_no_files_returns_empty(self, mock_glob, mock_dir):
        """Test empty result when no pick files exist."""
        mock_glob.return_value = []

        from nba.betting_xl.analyze_tv_impact import load_all_picks

        picks = load_all_picks()
        assert picks == []

    @patch("nba.betting_xl.analyze_tv_impact.PREDICTIONS_DIR")
    @patch("nba.betting_xl.analyze_tv_impact.glob.glob")
    @patch("builtins.open", new_callable=mock_open)
    def test_uses_line_fallback(self, mock_file, mock_glob, mock_dir):
        """Test using 'line' when 'best_line' is not present."""
        pick_data = {
            "date": "2025-01-15",
            "picks": [
                {
                    "player_name": "LeBron James",
                    "stat_type": "POINTS",
                    "side": "OVER",
                    "line": 26.0,
                }
            ],
        }
        mock_glob.side_effect = [["/fake/xl_picks_2025-01-15.json"], []]
        with patch("json.load", return_value=pick_data):
            from nba.betting_xl.analyze_tv_impact import load_all_picks

            picks = load_all_picks()

        assert len(picks) == 1
        assert picks[0]["line"] == 26.0

    @patch("nba.betting_xl.analyze_tv_impact.PREDICTIONS_DIR")
    @patch("nba.betting_xl.analyze_tv_impact.glob.glob")
    @patch("builtins.open", new_callable=mock_open)
    def test_date_parsed_from_filename_yyyymmdd(self, mock_file, mock_glob, mock_dir):
        """Test date extraction from filename when JSON has no date field."""
        pick_data = {
            # No 'date' or 'game_date' key
            "picks": [
                {
                    "player_name": "LeBron James",
                    "stat_type": "POINTS",
                    "side": "OVER",
                    "best_line": 25.5,
                }
            ],
        }
        # Filename has date in YYYYMMDD format
        mock_glob.side_effect = [["/fake/predictions/xl_picks_20250115.json"], []]

        with (
            patch("json.load", return_value=pick_data),
            patch("os.path.basename", return_value="xl_picks_20250115.json"),
        ):
            from nba.betting_xl.analyze_tv_impact import load_all_picks

            picks = load_all_picks()

        assert len(picks) == 1
        assert picks[0]["game_date"] == "2025-01-15"

    @patch("nba.betting_xl.analyze_tv_impact.PREDICTIONS_DIR")
    @patch("nba.betting_xl.analyze_tv_impact.glob.glob")
    @patch("builtins.open", new_callable=mock_open)
    def test_date_parsed_from_filename_dashed(self, mock_file, mock_glob, mock_dir):
        """Test date extraction from filename with YYYY-MM-DD format."""
        pick_data = {
            "picks": [
                {
                    "player_name": "LeBron James",
                    "stat_type": "POINTS",
                    "side": "OVER",
                    "best_line": 25.5,
                }
            ],
        }
        mock_glob.side_effect = [["/fake/predictions/xl_picks_2025-01-15.json"], []]

        with (
            patch("json.load", return_value=pick_data),
            patch("os.path.basename", return_value="xl_picks_2025-01-15.json"),
        ):
            from nba.betting_xl.analyze_tv_impact import load_all_picks

            picks = load_all_picks()

        assert len(picks) == 1
        assert picks[0]["game_date"] == "2025-01-15"

    @patch("nba.betting_xl.analyze_tv_impact.PREDICTIONS_DIR")
    @patch("nba.betting_xl.analyze_tv_impact.glob.glob")
    @patch("builtins.open", new_callable=mock_open)
    def test_no_date_anywhere_skips_file(self, mock_file, mock_glob, mock_dir):
        """Test that files with no date in JSON or parseable filename are skipped."""
        pick_data = {
            "picks": [
                {
                    "player_name": "LeBron James",
                    "stat_type": "POINTS",
                    "side": "OVER",
                    "best_line": 25.5,
                }
            ],
        }
        mock_glob.side_effect = [["/fake/predictions/xl_picks_latest.json"], []]

        with (
            patch("json.load", return_value=pick_data),
            patch("os.path.basename", return_value="xl_picks_latest.json"),
        ):
            from nba.betting_xl.analyze_tv_impact import load_all_picks

            picks = load_all_picks()

        assert len(picks) == 0


# ---------------------------------------------------------------------------
# grade_picks
# ---------------------------------------------------------------------------
class TestGradePicks:
    """Tests for grading picks against actual game results."""

    @patch("nba.betting_xl.analyze_tv_impact.psycopg2.connect")
    def test_grades_over_win(self, mock_connect):
        """Test grading an OVER pick that wins."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            ("LeBron James", "2025-01-15", "LAL", "GSW", 30, 8, 7, 3, 35.5)
        ]
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        from nba.betting_xl.analyze_tv_impact import grade_picks

        picks = [
            {
                "player_name": "LeBron James",
                "stat_type": "POINTS",
                "side": "OVER",
                "line": 25.5,
                "game_date": "2025-01-15",
            }
        ]

        graded = grade_picks(picks)

        assert len(graded) == 1
        assert graded[0]["won"] is True
        assert graded[0]["actual"] == 30

    @patch("nba.betting_xl.analyze_tv_impact.psycopg2.connect")
    def test_grades_over_loss(self, mock_connect):
        """Test grading an OVER pick that loses."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            ("LeBron James", "2025-01-15", "LAL", "GSW", 20, 8, 7, 3, 35.5)
        ]
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        from nba.betting_xl.analyze_tv_impact import grade_picks

        picks = [
            {
                "player_name": "LeBron James",
                "stat_type": "POINTS",
                "side": "OVER",
                "line": 25.5,
                "game_date": "2025-01-15",
            }
        ]

        graded = grade_picks(picks)

        assert len(graded) == 1
        assert graded[0]["won"] is False

    @patch("nba.betting_xl.analyze_tv_impact.psycopg2.connect")
    def test_push_is_loss(self, mock_connect):
        """Test that a push (actual == line) is graded as loss."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            ("LeBron James", "2025-01-15", "LAL", "GSW", 25, 8, 7, 3, 35.5)
        ]
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        from nba.betting_xl.analyze_tv_impact import grade_picks

        picks = [
            {
                "player_name": "LeBron James",
                "stat_type": "POINTS",
                "side": "OVER",
                "line": 25.0,
                "game_date": "2025-01-15",
            }
        ]

        graded = grade_picks(picks)

        assert len(graded) == 1
        assert graded[0]["won"] is False

    @patch("nba.betting_xl.analyze_tv_impact.psycopg2.connect")
    def test_under_pick_graded(self, mock_connect):
        """Test grading an UNDER pick."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            ("LeBron James", "2025-01-15", "LAL", "GSW", 20, 8, 7, 3, 35.5)
        ]
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        from nba.betting_xl.analyze_tv_impact import grade_picks

        picks = [
            {
                "player_name": "LeBron James",
                "stat_type": "POINTS",
                "side": "UNDER",
                "line": 25.5,
                "game_date": "2025-01-15",
            }
        ]

        graded = grade_picks(picks)

        assert len(graded) == 1
        assert graded[0]["won"] is True

    @patch("nba.betting_xl.analyze_tv_impact.psycopg2.connect")
    def test_unmatched_picks_excluded(self, mock_connect):
        """Test that picks without matching game logs are excluded."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        from nba.betting_xl.analyze_tv_impact import grade_picks

        picks = [
            {
                "player_name": "Unknown Player",
                "stat_type": "POINTS",
                "side": "OVER",
                "line": 25.5,
                "game_date": "2025-01-15",
            }
        ]

        graded = grade_picks(picks)

        assert len(graded) == 0

    @patch("nba.betting_xl.analyze_tv_impact.psycopg2.connect")
    def test_empty_picks_returns_empty(self, mock_connect):
        """Test that empty picks list returns empty."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        from nba.betting_xl.analyze_tv_impact import grade_picks

        graded = grade_picks([])

        assert graded == []

    @patch("nba.betting_xl.analyze_tv_impact.psycopg2.connect")
    def test_rebounds_stat_column(self, mock_connect):
        """Test grading REBOUNDS stat type uses correct column."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            ("Anthony Davis", "2025-01-15", "LAL", "GSW", 22, 12, 3, 0, 36.0)
        ]
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        from nba.betting_xl.analyze_tv_impact import grade_picks

        picks = [
            {
                "player_name": "Anthony Davis",
                "stat_type": "REBOUNDS",
                "side": "OVER",
                "line": 9.5,
                "game_date": "2025-01-15",
            }
        ]

        graded = grade_picks(picks)

        assert len(graded) == 1
        assert graded[0]["actual"] == 12
        assert graded[0]["won"] is True

    @patch("nba.betting_xl.analyze_tv_impact.psycopg2.connect")
    def test_actual_none_excluded(self, mock_connect):
        """Test that picks where actual stat is None are excluded."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        # Return row where points is None
        mock_cursor.fetchall.return_value = [
            ("LeBron James", "2025-01-15", "LAL", "GSW", None, 8, 7, 3, 35.5)
        ]
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        from nba.betting_xl.analyze_tv_impact import grade_picks

        picks = [
            {
                "player_name": "LeBron James",
                "stat_type": "POINTS",
                "side": "OVER",
                "line": 25.5,
                "game_date": "2025-01-15",
            }
        ]

        graded = grade_picks(picks)

        assert len(graded) == 0


# ---------------------------------------------------------------------------
# tag_national_tv
# ---------------------------------------------------------------------------
class TestTagNationalTv:
    """Tests for tagging picks with national TV status."""

    def test_tags_national_tv_game(self):
        """Test tagging a pick in a national TV game."""
        from nba.betting_xl.analyze_tv_impact import tag_national_tv

        picks = [
            {
                "game_date": "2025-01-15",
                "team_abbrev": "LAL",
                "opponent_team": "BOS",
            }
        ]
        tv_cache = {
            "2025-01-15": {
                "national": {"LAL_BOS": ["ESPN"]},
                "all": ["LAL_BOS"],
            }
        }

        result = tag_national_tv(picks, tv_cache)

        assert result[0]["is_national_tv"] is True
        assert result[0]["tv_networks"] == ["ESPN"]

    def test_tags_local_game(self):
        """Test tagging a pick in a local-only game."""
        from nba.betting_xl.analyze_tv_impact import tag_national_tv

        picks = [
            {
                "game_date": "2025-01-15",
                "team_abbrev": "CLE",
                "opponent_team": "MEM",
            }
        ]
        tv_cache = {
            "2025-01-15": {
                "national": {"LAL_BOS": ["ESPN"]},
                "all": ["LAL_BOS", "CLE_MEM"],
            }
        }

        result = tag_national_tv(picks, tv_cache)

        assert result[0]["is_national_tv"] is False
        assert result[0]["tv_networks"] == []

    def test_missing_date_in_cache(self):
        """Test tagging when the game date is not in the cache."""
        from nba.betting_xl.analyze_tv_impact import tag_national_tv

        picks = [
            {
                "game_date": "2025-02-20",
                "team_abbrev": "LAL",
                "opponent_team": "BOS",
            }
        ]
        tv_cache = {}

        result = tag_national_tv(picks, tv_cache)

        assert result[0]["is_national_tv"] is False

    def test_match_by_opponent_team(self):
        """Test matching TV status using opponent team."""
        from nba.betting_xl.analyze_tv_impact import tag_national_tv

        picks = [
            {
                "game_date": "2025-01-15",
                "team_abbrev": "MIA",
                "opponent_team": "LAL",
            }
        ]
        tv_cache = {
            "2025-01-15": {
                "national": {"LAL_MIA": ["TNT"]},
                "all": ["LAL_MIA"],
            }
        }

        result = tag_national_tv(picks, tv_cache)

        assert result[0]["is_national_tv"] is True


# ---------------------------------------------------------------------------
# compute_stats
# ---------------------------------------------------------------------------
class TestComputeStats:
    """Tests for computing win rate and ROI stats."""

    def test_empty_picks(self):
        """Test stats computation with no picks."""
        from nba.betting_xl.analyze_tv_impact import compute_stats

        result = compute_stats([], "Empty")

        assert result["n"] == 0
        assert result["wins"] == 0
        assert result["wr"] == 0
        assert result["roi"] == 0

    def test_all_wins(self):
        """Test stats with all winning picks."""
        from nba.betting_xl.analyze_tv_impact import compute_stats

        picks = [{"won": True}, {"won": True}, {"won": True}]
        result = compute_stats(picks, "All Wins")

        assert result["n"] == 3
        assert result["wins"] == 3
        assert result["losses"] == 0
        assert result["wr"] == 100.0
        # ROI: 3 * 0.909 / 3 * 100 = 90.9%
        assert result["roi"] == 90.9

    def test_all_losses(self):
        """Test stats with all losing picks."""
        from nba.betting_xl.analyze_tv_impact import compute_stats

        picks = [{"won": False}, {"won": False}]
        result = compute_stats(picks, "All Losses")

        assert result["n"] == 2
        assert result["wins"] == 0
        assert result["losses"] == 2
        assert result["wr"] == 0.0
        # ROI: -2.0 / 2 * 100 = -100%
        assert result["roi"] == -100.0

    def test_mixed_results(self):
        """Test stats with mixed results."""
        from nba.betting_xl.analyze_tv_impact import compute_stats

        picks = [{"won": True}, {"won": False}, {"won": True}, {"won": False}]
        result = compute_stats(picks, "Mixed")

        assert result["n"] == 4
        assert result["wins"] == 2
        assert result["losses"] == 2
        assert result["wr"] == 50.0
        # ROI: (2 * 0.909 - 2 * 1.0) / 4 * 100 = (1.818 - 2.0)/4 * 100 = -4.55%
        assert result["roi"] == -4.55

    def test_label_preserved(self):
        """Test that the label is preserved in output."""
        from nba.betting_xl.analyze_tv_impact import compute_stats

        result = compute_stats([{"won": True}], "My Label")
        assert result["label"] == "My Label"

    def test_profit_computed(self):
        """Test that profit is computed correctly."""
        from nba.betting_xl.analyze_tv_impact import compute_stats

        picks = [{"won": True}]
        result = compute_stats(picks, "Single Win")

        expected_profit = 1 * 0.909 - 0 * 1.0  # 0.909
        assert result["profit"] == round(expected_profit, 2)


# ---------------------------------------------------------------------------
# chi_squared_test
# ---------------------------------------------------------------------------
class TestChiSquaredTest:
    """Tests for the chi-squared statistical test."""

    def test_empty_groups(self):
        """Test chi-squared with two empty groups."""
        from nba.betting_xl.analyze_tv_impact import chi_squared_test

        result = chi_squared_test([], [])

        assert result["chi2"] == 0
        assert result["p_value"] == 1.0
        assert result["significant"] is False

    def test_one_empty_group(self):
        """Test chi-squared when one group is empty."""
        from nba.betting_xl.analyze_tv_impact import chi_squared_test

        result = chi_squared_test([{"won": True}], [])

        assert result["chi2"] == 0
        assert result["p_value"] == 1.0
        assert result["significant"] is False

    def test_identical_groups(self):
        """Test chi-squared with identical win rates."""
        from nba.betting_xl.analyze_tv_impact import chi_squared_test

        group_a = [{"won": True}, {"won": False}]
        group_b = [{"won": True}, {"won": False}]

        result = chi_squared_test(group_a, group_b)

        assert result["chi2"] == 0.0
        assert result["significant"] is False

    def test_significant_difference(self):
        """Test chi-squared with significantly different groups."""
        from nba.betting_xl.analyze_tv_impact import chi_squared_test

        # Group A: 90% win rate (18 wins, 2 losses)
        group_a = [{"won": True}] * 18 + [{"won": False}] * 2
        # Group B: 30% win rate (6 wins, 14 losses)
        group_b = [{"won": True}] * 6 + [{"won": False}] * 14

        result = chi_squared_test(group_a, group_b)

        assert result["chi2"] > 0
        assert result["p_value"] < 0.05
        assert result["significant"] is True

    def test_not_significant_small_difference(self):
        """Test chi-squared with small non-significant difference."""
        from nba.betting_xl.analyze_tv_impact import chi_squared_test

        group_a = [{"won": True}] * 5 + [{"won": False}] * 5
        group_b = [{"won": True}] * 4 + [{"won": False}] * 6

        result = chi_squared_test(group_a, group_b)

        assert result["significant"] is False

    def test_p_value_uses_erfc(self):
        """Test that p-value uses math.erfc formula correctly."""
        from nba.betting_xl.analyze_tv_impact import chi_squared_test

        # Build groups that produce a known chi2
        group_a = [{"won": True}] * 10 + [{"won": False}] * 0
        group_b = [{"won": True}] * 0 + [{"won": False}] * 10

        result = chi_squared_test(group_a, group_b)

        # Verify chi2 > 0 and p_value is small for extreme difference
        assert result["chi2"] > 10.0
        assert result["p_value"] < 0.01


# ---------------------------------------------------------------------------
# print_section / print_comparison
# ---------------------------------------------------------------------------
class TestPrintFunctions:
    """Tests for print/formatting helper functions."""

    def test_print_section(self, capsys):
        """Test print_section outputs formatted section."""
        from nba.betting_xl.analyze_tv_impact import print_section

        print_section("TEST SECTION")
        captured = capsys.readouterr()

        assert "TEST SECTION" in captured.out
        assert "=" * 70 in captured.out

    def test_print_comparison(self, capsys):
        """Test print_comparison outputs formatted comparison."""
        from nba.betting_xl.analyze_tv_impact import print_comparison

        ntv = {"n": 100, "wins": 60, "losses": 40, "wr": 60.0, "roi": 5.45}
        local = {"n": 200, "wins": 100, "losses": 100, "wr": 50.0, "roi": -4.55}
        test = {"chi2": 3.45, "p_value": 0.06, "significant": False}

        print_comparison(ntv, local, test)
        captured = capsys.readouterr()

        assert "National TV" in captured.out
        assert "Local Only" in captured.out
        assert "Chi-squared" in captured.out
        assert "NO" in captured.out  # Not significant


# ---------------------------------------------------------------------------
# run_analysis (integration of all steps)
# ---------------------------------------------------------------------------
class TestRunAnalysis:
    """Tests for the full run_analysis orchestration."""

    @patch("nba.betting_xl.analyze_tv_impact.load_or_fetch_tv_cache")
    @patch("nba.betting_xl.analyze_tv_impact.grade_picks")
    @patch("nba.betting_xl.analyze_tv_impact.load_all_picks")
    def test_run_analysis_no_graded_picks(self, mock_load, mock_grade, mock_tv, capsys):
        """Test run_analysis handles no graded picks gracefully."""
        mock_load.return_value = [
            {"player_name": "Test", "system": "XL", "game_date": "2025-01-15"}
        ]
        mock_grade.return_value = []

        from nba.betting_xl.analyze_tv_impact import run_analysis

        run_analysis()

        captured = capsys.readouterr()
        assert "ERROR" in captured.out

    @patch("nba.betting_xl.analyze_tv_impact.load_or_fetch_tv_cache")
    @patch("nba.betting_xl.analyze_tv_impact.grade_picks")
    @patch("nba.betting_xl.analyze_tv_impact.load_all_picks")
    def test_run_analysis_full_flow(self, mock_load, mock_grade, mock_tv, capsys):
        """Test run_analysis completes full flow with graded picks."""
        mock_load.return_value = [
            {
                "player_name": "LeBron James",
                "system": "XL",
                "stat_type": "POINTS",
                "side": "OVER",
                "game_date": "2025-01-15",
                "confidence": "HIGH",
            }
        ]
        mock_grade.return_value = [
            {
                "player_name": "LeBron James",
                "system": "XL",
                "stat_type": "POINTS",
                "side": "OVER",
                "game_date": "2025-01-15",
                "team_abbrev": "LAL",
                "opponent_team": "BOS",
                "won": True,
                "actual": 30,
                "confidence": "HIGH",
            }
        ]
        mock_tv.return_value = {
            "2025-01-15": {"national": {"LAL_BOS": ["ESPN"]}, "all": ["LAL_BOS"]}
        }

        from nba.betting_xl.analyze_tv_impact import run_analysis

        run_analysis()

        captured = capsys.readouterr()
        assert "NATIONAL TV vs LOCAL BROADCAST" in captured.out
        assert "FEATURE EVALUATION SUMMARY" in captured.out

    @patch("nba.betting_xl.analyze_tv_impact.load_or_fetch_tv_cache")
    @patch("nba.betting_xl.analyze_tv_impact.grade_picks")
    @patch("nba.betting_xl.analyze_tv_impact.load_all_picks")
    def test_run_analysis_covers_all_sections(self, mock_load, mock_grade, mock_tv, capsys):
        """Test run_analysis covers BY SYSTEM, BY MARKET, BY SIDE, BY CONFIDENCE sections."""
        # Build 15+ picks per system/market/side/confidence to cover all analysis branches
        graded_picks = []
        for i in range(20):
            graded_picks.append(
                {
                    "player_name": f"Player {i}",
                    "system": "XL" if i < 12 else "PRO",
                    "stat_type": "POINTS" if i < 15 else "REBOUNDS",
                    "side": "OVER" if i < 15 else "UNDER",
                    "game_date": "2025-01-15",
                    "team_abbrev": "LAL",
                    "opponent_team": "BOS",
                    "won": i % 2 == 0,
                    "actual": 30 if i % 2 == 0 else 18,
                    "confidence": "HIGH" if i < 8 else "MEDIUM",
                    "is_national_tv": i < 10,
                    "tv_networks": ["ESPN"] if i < 10 else [],
                }
            )

        raw_picks = [
            {
                "player_name": f"Player {i}",
                "system": "XL" if i < 12 else "PRO",
                "game_date": "2025-01-15",
            }
            for i in range(20)
        ]
        mock_load.return_value = raw_picks
        mock_grade.return_value = graded_picks
        mock_tv.return_value = {
            "2025-01-15": {"national": {"LAL_BOS": ["ESPN"]}, "all": ["LAL_BOS"]}
        }

        from nba.betting_xl.analyze_tv_impact import run_analysis

        run_analysis()
        captured = capsys.readouterr()

        # Verify all sections are hit
        assert "BY SYSTEM" in captured.out
        assert "BY MARKET" in captured.out
        assert "BY SIDE" in captured.out
        assert "BY CONFIDENCE" in captured.out
        assert "WIN RATE BY NETWORK" in captured.out
        assert "BY DAY OF WEEK" in captured.out

    @patch("nba.betting_xl.analyze_tv_impact.load_or_fetch_tv_cache")
    @patch("nba.betting_xl.analyze_tv_impact.grade_picks")
    @patch("nba.betting_xl.analyze_tv_impact.load_all_picks")
    def test_run_analysis_significant_large_delta(self, mock_load, mock_grade, mock_tv, capsys):
        """Test recommendation for significant + large delta."""
        graded_picks = []
        # National TV: 90% WR (18/20)
        for i in range(20):
            graded_picks.append(
                {
                    "player_name": f"NTV Player {i}",
                    "system": "XL",
                    "stat_type": "POINTS",
                    "side": "OVER",
                    "game_date": "2025-01-15",
                    "team_abbrev": "LAL",
                    "opponent_team": "BOS",
                    "won": i < 18,
                    "actual": 30 if i < 18 else 15,
                    "confidence": "HIGH",
                }
            )
        # Local: 40% WR (8/20)
        for i in range(20):
            graded_picks.append(
                {
                    "player_name": f"Local Player {i}",
                    "system": "XL",
                    "stat_type": "POINTS",
                    "side": "OVER",
                    "game_date": "2025-01-15",
                    "team_abbrev": "CLE",
                    "opponent_team": "MEM",
                    "won": i < 8,
                    "actual": 30 if i < 8 else 15,
                    "confidence": "HIGH",
                }
            )

        mock_load.return_value = [
            {"player_name": "X", "system": "XL", "game_date": "2025-01-15"} for _ in range(40)
        ]
        mock_grade.return_value = graded_picks
        mock_tv.return_value = {
            "2025-01-15": {"national": {"LAL_BOS": ["ESPN"]}, "all": ["LAL_BOS", "CLE_MEM"]}
        }

        from nba.betting_xl.analyze_tv_impact import run_analysis

        run_analysis()
        captured = capsys.readouterr()

        assert "WORTH ADDING" in captured.out

    @patch("nba.betting_xl.analyze_tv_impact.load_or_fetch_tv_cache")
    @patch("nba.betting_xl.analyze_tv_impact.grade_picks")
    @patch("nba.betting_xl.analyze_tv_impact.load_all_picks")
    def test_run_analysis_marginal_recommendation(self, mock_load, mock_grade, mock_tv, capsys):
        """Test MARGINAL recommendation when significant but small delta (< 3%)."""
        # NTV: 52% WR, Local: 50% WR -- delta = 2% < 3.0
        # Use large sample to make chi_squared think it's significant
        graded_picks = []
        for i in range(100):
            graded_picks.append(
                {
                    "player_name": f"NTV Player {i}",
                    "system": "XL",
                    "stat_type": "POINTS",
                    "side": "OVER",
                    "game_date": "2025-01-15",
                    "team_abbrev": "LAL",
                    "opponent_team": "BOS",
                    "won": i < 52,  # 52% WR
                    "actual": 30 if i < 52 else 15,
                    "confidence": "HIGH",
                }
            )
        for i in range(100):
            graded_picks.append(
                {
                    "player_name": f"Local Player {i}",
                    "system": "XL",
                    "stat_type": "POINTS",
                    "side": "OVER",
                    "game_date": "2025-01-15",
                    "team_abbrev": "CLE",
                    "opponent_team": "MEM",
                    "won": i < 50,  # 50% WR
                    "actual": 30 if i < 50 else 15,
                    "confidence": "HIGH",
                }
            )

        mock_load.return_value = [
            {"player_name": "X", "system": "XL", "game_date": "2025-01-15"} for _ in range(200)
        ]
        mock_grade.return_value = graded_picks
        mock_tv.return_value = {
            "2025-01-15": {"national": {"LAL_BOS": ["ESPN"]}, "all": ["LAL_BOS", "CLE_MEM"]}
        }

        # Monkey-patch chi_squared_test so the OVERALL call returns significant
        # but the sub-section calls return the real results
        original_fn = None
        call_count = [0]

        import nba.betting_xl.analyze_tv_impact as tv_mod

        original_fn = tv_mod.chi_squared_test

        def patched_chi(group_a, group_b):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call is the overall comparison -> force significant
                return {"chi2": 5.0, "p_value": 0.02, "significant": True}
            return original_fn(group_a, group_b)

        tv_mod.chi_squared_test = patched_chi
        try:
            tv_mod.run_analysis()
        finally:
            tv_mod.chi_squared_test = original_fn

        captured = capsys.readouterr()
        assert "MARGINAL" in captured.out
