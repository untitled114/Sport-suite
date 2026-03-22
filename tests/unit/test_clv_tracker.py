"""
Tests for CLV (Closing Line Value) Tracker.

Tests the core CLV computation logic without requiring database connections.
"""

from unittest.mock import MagicMock, call, patch

import pytest

from nba.core.clv_tracker import CLVTracker, _american_to_implied_prob


class TestAmericanToImpliedProb:
    """Test odds conversion function."""

    def test_standard_favorite(self):
        prob = _american_to_implied_prob(-110)
        assert round(prob, 4) == 0.5238

    def test_heavy_favorite(self):
        prob = _american_to_implied_prob(-200)
        assert round(prob, 4) == 0.6667

    def test_underdog(self):
        prob = _american_to_implied_prob(150)
        assert round(prob, 4) == 0.4000

    def test_even_money(self):
        prob = _american_to_implied_prob(100)
        assert round(prob, 4) == 0.5000

    def test_none_returns_half(self):
        assert _american_to_implied_prob(None) == 0.5

    def test_zero_returns_half(self):
        assert _american_to_implied_prob(0) == 0.5

    def test_strong_underdog(self):
        prob = _american_to_implied_prob(300)
        assert round(prob, 4) == 0.2500

    def test_strong_favorite(self):
        prob = _american_to_implied_prob(-500)
        assert round(prob, 4) == 0.8333


class TestCLVTrackerComputeCLV:
    """Test CLVTracker.compute_clv."""

    def setup_method(self):
        self.tracker = CLVTracker()

    def test_line_drops_positive_clv(self):
        pick = {
            "player_name": "LeBron James",
            "stat_type": "POINTS",
            "run_date": "2026-03-20",
            "line": 25.5,
            "p_over": 0.72,
            "book": "draftkings",
        }
        opening = {
            "over_line": 25.5,
            "over_odds": -110,
            "under_odds": -110,
            "snapshot_at": "2026-03-20T10:00:00",
            "book_name": "draftkings",
        }
        closing = {
            "over_line": 24.5,
            "over_odds": -130,
            "under_odds": +110,
            "snapshot_at": "2026-03-20T18:00:00",
            "book_name": "draftkings",
        }

        with (
            patch.object(self.tracker, "get_opening_line", return_value=opening),
            patch.object(self.tracker, "get_closing_line", return_value=closing),
        ):
            result = self.tracker.compute_clv(pick)

        assert result is not None
        assert result["clv_cents"] > 0
        assert result["closing_line"] == 24.5
        assert result["opening_line"] == 25.5
        assert result["line_movement"] == -1.0

    def test_line_rises_negative_clv(self):
        pick = {
            "player_name": "Anthony Davis",
            "stat_type": "REBOUNDS",
            "run_date": "2026-03-20",
            "line": 10.5,
            "p_over": 0.65,
            "book": "fanduel",
        }
        opening = {
            "over_line": 10.5,
            "over_odds": -110,
            "under_odds": -110,
            "snapshot_at": "2026-03-20T10:00:00",
            "book_name": "fanduel",
        }
        closing = {
            "over_line": 11.5,
            "over_odds": +100,
            "under_odds": -120,
            "snapshot_at": "2026-03-20T18:00:00",
            "book_name": "fanduel",
        }

        with (
            patch.object(self.tracker, "get_opening_line", return_value=opening),
            patch.object(self.tracker, "get_closing_line", return_value=closing),
        ):
            result = self.tracker.compute_clv(pick)

        assert result is not None
        assert result["clv_cents"] < 0
        assert result["line_movement"] == 1.0

    def test_beat_close_true(self):
        pick = {
            "player_name": "Giannis",
            "stat_type": "POINTS",
            "run_date": "2026-03-20",
            "line": 30.5,
            "p_over": 0.75,
            "book": None,
        }
        opening = {
            "over_line": 30.5,
            "over_odds": -110,
            "under_odds": -110,
            "snapshot_at": "2026-03-20T10:00:00",
            "book_name": "dk",
        }
        closing = {
            "over_line": 30.5,
            "over_odds": -130,
            "under_odds": +110,
            "snapshot_at": "2026-03-20T18:00:00",
            "book_name": "dk",
        }

        with (
            patch.object(self.tracker, "get_opening_line", return_value=opening),
            patch.object(self.tracker, "get_closing_line", return_value=closing),
        ):
            result = self.tracker.compute_clv(pick)

        assert result["beat_close"] is True
        assert result["model_prob"] == 0.75

    def test_beat_close_false(self):
        pick = {
            "player_name": "Player",
            "stat_type": "POINTS",
            "run_date": "2026-03-20",
            "line": 20.0,
            "p_over": 0.50,
            "book": None,
        }
        opening = {
            "over_line": 20.0,
            "over_odds": -110,
            "under_odds": -110,
            "snapshot_at": "2026-03-20T10:00:00",
            "book_name": "dk",
        }
        closing = {
            "over_line": 20.0,
            "over_odds": -200,
            "under_odds": +170,
            "snapshot_at": "2026-03-20T18:00:00",
            "book_name": "dk",
        }

        with (
            patch.object(self.tracker, "get_opening_line", return_value=opening),
            patch.object(self.tracker, "get_closing_line", return_value=closing),
        ):
            result = self.tracker.compute_clv(pick)

        # model_prob 0.50 < closing_implied ~0.667 = did NOT beat close
        assert result["beat_close"] is False

    def test_no_snapshots_returns_none(self):
        pick = {
            "player_name": "X",
            "stat_type": "POINTS",
            "run_date": "2026-03-20",
            "line": 20.0,
            "p_over": 0.60,
            "book": None,
        }

        with (
            patch.object(self.tracker, "get_opening_line", return_value=None),
            patch.object(self.tracker, "get_closing_line", return_value=None),
        ):
            assert self.tracker.compute_clv(pick) is None

    def test_only_opening_missing_returns_none(self):
        pick = {
            "player_name": "X",
            "stat_type": "POINTS",
            "run_date": "2026-03-20",
            "line": 20.0,
            "p_over": 0.60,
            "book": None,
        }
        closing = {
            "over_line": 20.0,
            "over_odds": -110,
            "under_odds": -110,
            "snapshot_at": "2026-03-20T18:00:00",
            "book_name": "dk",
        }

        with (
            patch.object(self.tracker, "get_opening_line", return_value=None),
            patch.object(self.tracker, "get_closing_line", return_value=closing),
        ):
            assert self.tracker.compute_clv(pick) is None

    def test_stable_line_zero_clv(self):
        pick = {
            "player_name": "Tatum",
            "stat_type": "POINTS",
            "run_date": "2026-03-20",
            "line": 27.5,
            "p_over": 0.60,
            "book": "betmgm",
        }
        snapshot = {
            "over_line": 27.5,
            "over_odds": -110,
            "under_odds": -110,
            "snapshot_at": "2026-03-20T10:00:00",
            "book_name": "betmgm",
        }

        with (
            patch.object(self.tracker, "get_opening_line", return_value=snapshot),
            patch.object(self.tracker, "get_closing_line", return_value=snapshot),
        ):
            result = self.tracker.compute_clv(pick)

        assert result["clv_cents"] == 0.0
        assert result["line_movement"] == 0.0

    def test_null_odds_uses_default(self):
        """When odds are None, implied prob defaults to 0.5."""
        pick = {
            "player_name": "X",
            "stat_type": "POINTS",
            "run_date": "2026-03-20",
            "line": 20.0,
            "p_over": 0.60,
            "book": None,
        }
        snapshot = {
            "over_line": 20.0,
            "over_odds": None,
            "under_odds": None,
            "snapshot_at": "2026-03-20T10:00:00",
            "book_name": "dk",
        }

        with (
            patch.object(self.tracker, "get_opening_line", return_value=snapshot),
            patch.object(self.tracker, "get_closing_line", return_value=snapshot),
        ):
            result = self.tracker.compute_clv(pick)

        assert result["opening_implied_prob"] == 0.5
        assert result["closing_implied_prob"] == 0.5
        assert result["clv_cents"] == 0.0


class TestCLVTrackerDBMethods:
    """Test DB-hitting methods with mocked connections."""

    def setup_method(self):
        self.tracker = CLVTracker()

    @patch("nba.core.clv_tracker.psycopg2.connect")
    @patch("nba.core.clv_tracker.get_intelligence_db_config")
    def test_get_closing_line_with_book(self, mock_config, mock_connect):
        mock_config.return_value = {"host": "localhost", "port": 5539}
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = {
            "over_line": 25.5,
            "over_odds": -110,
            "under_odds": -110,
            "snapshot_at": "2026-03-20T18:00:00",
            "book_name": "draftkings",
        }
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value = mock_conn

        result = self.tracker.get_closing_line("LeBron James", "POINTS", "2026-03-20", "draftkings")

        assert result is not None
        assert result["over_line"] == 25.5
        mock_cursor.execute.assert_called_once()
        # Check book_name was passed (4 params = with book filter)
        args = mock_cursor.execute.call_args[0][1]
        assert len(args) == 4
        assert args[3] == "draftkings"

    @patch("nba.core.clv_tracker.psycopg2.connect")
    @patch("nba.core.clv_tracker.get_intelligence_db_config")
    def test_get_closing_line_no_book(self, mock_config, mock_connect):
        mock_config.return_value = {"host": "localhost", "port": 5539}
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = {
            "over_line": 25.5,
            "over_odds": -110,
            "under_odds": -110,
            "snapshot_at": "2026-03-20T18:00:00",
            "book_name": "fanduel",
        }
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value = mock_conn

        result = self.tracker.get_closing_line("LeBron James", "POINTS", "2026-03-20")

        assert result is not None
        # No book filter = 3 params
        args = mock_cursor.execute.call_args[0][1]
        assert len(args) == 3

    @patch("nba.core.clv_tracker.psycopg2.connect")
    @patch("nba.core.clv_tracker.get_intelligence_db_config")
    def test_get_closing_line_no_data(self, mock_config, mock_connect):
        mock_config.return_value = {"host": "localhost", "port": 5539}
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value = mock_conn

        result = self.tracker.get_closing_line("Nobody", "POINTS", "2026-03-20")
        assert result is None

    @patch("nba.core.clv_tracker.psycopg2.connect")
    @patch("nba.core.clv_tracker.get_intelligence_db_config")
    def test_get_opening_line_with_book(self, mock_config, mock_connect):
        mock_config.return_value = {"host": "localhost", "port": 5539}
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = {
            "over_line": 24.5,
            "over_odds": -115,
            "under_odds": -105,
            "snapshot_at": "2026-03-20T08:00:00",
            "book_name": "fanduel",
        }
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value = mock_conn

        result = self.tracker.get_opening_line("AD", "REBOUNDS", "2026-03-20", "fanduel")

        assert result is not None
        assert result["over_line"] == 24.5
        args = mock_cursor.execute.call_args[0][1]
        assert len(args) == 4

    @patch("nba.core.clv_tracker.psycopg2.connect")
    @patch("nba.core.clv_tracker.get_intelligence_db_config")
    def test_get_opening_line_no_book(self, mock_config, mock_connect):
        mock_config.return_value = {"host": "localhost", "port": 5539}
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = {
            "over_line": 10.5,
            "over_odds": -110,
            "under_odds": -110,
            "snapshot_at": "2026-03-20T08:00:00",
            "book_name": "dk",
        }
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value = mock_conn

        result = self.tracker.get_opening_line("AD", "REBOUNDS", "2026-03-20")

        assert result is not None
        args = mock_cursor.execute.call_args[0][1]
        assert len(args) == 3

    @patch("nba.core.clv_tracker.psycopg2.connect")
    @patch("nba.core.clv_tracker.get_intelligence_db_config")
    def test_get_opening_line_no_data(self, mock_config, mock_connect):
        mock_config.return_value = {"host": "localhost", "port": 5539}
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value = mock_conn

        result = self.tracker.get_opening_line("Nobody", "POINTS", "2026-03-20")
        assert result is None


class TestCLVTrackerDaily:
    """Test compute_daily_clv with mocked DB and CLV."""

    def setup_method(self):
        self.tracker = CLVTracker()

    @patch("nba.core.clv_tracker._connect_axiom")
    def test_daily_no_picks(self, mock_connect):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value = mock_conn

        result = self.tracker.compute_daily_clv("2026-03-20")
        assert result["total_picks"] == 0
        assert result["picks_with_clv"] == 0
        assert result["avg_clv_cents"] == 0.0

    @patch("nba.core.clv_tracker._connect_axiom")
    def test_daily_with_picks_but_no_snapshots(self, mock_connect):
        """Picks exist but no line snapshots -> picks_with_clv = 0."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            {
                "player_name": "LeBron",
                "stat_type": "POINTS",
                "line": 25.5,
                "p_over": 0.70,
                "book": "dk",
                "run_date": "2026-03-20",
                "model_version": "xl",
                "is_hit": True,
            },
        ]
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value = mock_conn

        # compute_clv returns None when no snapshots
        with patch.object(self.tracker, "compute_clv", return_value=None):
            result = self.tracker.compute_daily_clv("2026-03-20")

        assert result["total_picks"] == 1
        assert result["picks_with_clv"] == 0
        assert result["by_market"] == {}

    @patch("nba.core.clv_tracker._connect_axiom")
    def test_daily_with_clv_data(self, mock_connect):
        """Full path: picks + CLV data -> aggregated metrics."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            {
                "player_name": "LeBron",
                "stat_type": "POINTS",
                "line": 25.5,
                "p_over": 0.72,
                "book": "dk",
                "run_date": "2026-03-20",
                "model_version": "xl",
                "is_hit": True,
            },
            {
                "player_name": "AD",
                "stat_type": "REBOUNDS",
                "line": 10.5,
                "p_over": 0.68,
                "book": "fd",
                "run_date": "2026-03-20",
                "model_version": "v3",
                "is_hit": False,
            },
        ]
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value = mock_conn

        clv_values = [
            {
                "opening_line": 25.5,
                "closing_line": 24.5,
                "our_line": 25.5,
                "opening_implied_prob": 0.5238,
                "closing_implied_prob": 0.5652,
                "model_prob": 0.72,
                "clv_cents": 4.14,
                "beat_close": True,
                "line_movement": -1.0,
            },
            {
                "opening_line": 10.5,
                "closing_line": 11.5,
                "our_line": 10.5,
                "opening_implied_prob": 0.5238,
                "closing_implied_prob": 0.5000,
                "model_prob": 0.68,
                "clv_cents": -2.38,
                "beat_close": True,
                "line_movement": 1.0,
            },
        ]
        with patch.object(self.tracker, "compute_clv", side_effect=clv_values):
            result = self.tracker.compute_daily_clv("2026-03-20")

        assert result["total_picks"] == 2
        assert result["picks_with_clv"] == 2
        assert result["avg_clv_cents"] == round((4.14 + (-2.38)) / 2, 2)
        assert result["beat_close_rate"] == 1.0  # Both beat close
        assert result["clv_positive_rate"] == 0.5  # 1 positive, 1 negative
        assert "POINTS" in result["by_market"]
        assert "REBOUNDS" in result["by_market"]
        assert result["by_market"]["POINTS"]["count"] == 1
        assert result["by_market"]["REBOUNDS"]["count"] == 1


class TestCLVTrackerRolling:
    """Test compute_rolling_clv."""

    def setup_method(self):
        self.tracker = CLVTracker()

    def test_rolling_no_data(self):
        """Rolling CLV with no daily data returns zeros."""
        with patch.object(self.tracker, "compute_daily_clv") as mock_daily:
            mock_daily.return_value = {
                "date": "2026-03-20",
                "total_picks": 0,
                "picks_with_clv": 0,
                "avg_clv_cents": 0.0,
                "beat_close_rate": 0.0,
                "clv_positive_rate": 0.0,
                "by_market": {},
            }
            result = self.tracker.compute_rolling_clv(3)

        assert result["period"] == "3d"
        assert result["picks_with_clv"] == 0
        assert result["avg_clv_cents"] == 0.0

    def test_rolling_with_data(self):
        """Rolling CLV aggregates daily results correctly."""
        daily_results = [
            # Day 1: 2 picks with CLV
            {
                "date": "2026-03-21",
                "total_picks": 3,
                "picks_with_clv": 2,
                "avg_clv_cents": 3.0,
                "beat_close_rate": 0.75,
                "clv_positive_rate": 0.60,
                "by_market": {
                    "POINTS": {"count": 2, "avg_clv_cents": 3.0, "beat_close_rate": 0.75}
                },
            },
            # Day 2: 1 pick with CLV
            {
                "date": "2026-03-20",
                "total_picks": 2,
                "picks_with_clv": 1,
                "avg_clv_cents": -1.5,
                "beat_close_rate": 0.50,
                "clv_positive_rate": 0.0,
                "by_market": {
                    "REBOUNDS": {"count": 1, "avg_clv_cents": -1.5, "beat_close_rate": 0.50}
                },
            },
            # Day 3: no CLV data
            {
                "date": "2026-03-19",
                "total_picks": 0,
                "picks_with_clv": 0,
                "avg_clv_cents": 0.0,
                "beat_close_rate": 0.0,
                "clv_positive_rate": 0.0,
                "by_market": {},
            },
        ]

        call_idx = [0]

        def mock_daily(date_str):
            idx = call_idx[0]
            call_idx[0] += 1
            return daily_results[idx] if idx < len(daily_results) else daily_results[-1]

        with patch.object(self.tracker, "compute_daily_clv", side_effect=mock_daily):
            result = self.tracker.compute_rolling_clv(3)

        assert result["period"] == "3d"
        assert result["picks_with_clv"] == 3  # 2 + 1
        assert (
            result["total_picks"] == 5
        )  # 3 + 2 (day 3 has 0 picks_with_clv so not in daily_results)
        assert "POINTS" in result["by_market"]
        assert "REBOUNDS" in result["by_market"]
        assert len(result["daily"]) == 2  # Only days with CLV data

    def test_rolling_single_day(self):
        """Rolling CLV with 1 day lookback."""
        with patch.object(self.tracker, "compute_daily_clv") as mock_daily:
            mock_daily.return_value = {
                "date": "2026-03-21",
                "total_picks": 5,
                "picks_with_clv": 4,
                "avg_clv_cents": 2.5,
                "beat_close_rate": 0.65,
                "clv_positive_rate": 0.55,
                "by_market": {
                    "POINTS": {"count": 4, "avg_clv_cents": 2.5, "beat_close_rate": 0.65}
                },
            }
            result = self.tracker.compute_rolling_clv(1)

        assert result["period"] == "1d"
        assert result["picks_with_clv"] == 4
        assert result["avg_clv_cents"] == 2.5
