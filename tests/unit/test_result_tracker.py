"""
Tests for Result Tracker.

Tests performance computation, anomaly detection, and edge bucketing
without requiring database connections.
"""

from unittest.mock import MagicMock, patch

import pytest

from nba.core.result_tracker import ResultTracker, _edge_bucket, _grade


class TestGrade:
    """Test standard -110 juice grading."""

    def test_win_profit(self):
        result = _grade(True)
        assert result["outcome"] == "WIN"
        assert result["profit"] == 0.909

    def test_loss_profit(self):
        result = _grade(False)
        assert result["outcome"] == "LOSS"
        assert result["profit"] == -1.0


class TestEdgeBucket:
    """Test edge classification into buckets."""

    def test_large_edge(self):
        assert _edge_bucket(12.5) == "10%+"

    def test_medium_edge(self):
        assert _edge_bucket(8.0) == "7-10%"

    def test_small_edge(self):
        assert _edge_bucket(5.5) == "5-7%"

    def test_moderate_edge(self):
        assert _edge_bucket(4.0) == "3-5%"

    def test_low_edge(self):
        assert _edge_bucket(1.5) == "0-3%"

    def test_negative_edge(self):
        assert _edge_bucket(-2.0) == "negative"

    def test_none_edge(self):
        assert _edge_bucket(None) == "unknown"

    def test_boundary_10(self):
        assert _edge_bucket(10.0) == "10%+"

    def test_boundary_7(self):
        assert _edge_bucket(7.0) == "7-10%"

    def test_boundary_5(self):
        assert _edge_bucket(5.0) == "5-7%"

    def test_boundary_3(self):
        assert _edge_bucket(3.0) == "3-5%"

    def test_boundary_0(self):
        assert _edge_bucket(0.0) == "0-3%"


def _make_picks(count, wins, market="POINTS", model="xl", tier="META", edge=5.0):
    """Helper to generate mock pick lists."""
    picks = []
    for i in range(count):
        picks.append(
            {
                "player_name": f"Player {i}",
                "stat_type": market,
                "model_version": model,
                "tier": tier,
                "line": 25.5,
                "p_over": 0.65,
                "edge": edge,
                "spread": 2.0,
                "book": "draftkings",
                "run_date": "2026-03-19",
                "is_hit": i < wins,
                "context_snapshot": None,
            }
        )
    return picks


def _mock_connect_with_picks(picks):
    """Create a mock connection that returns the given picks."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = picks
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    return mock_conn


class TestResultTrackerComputeMetrics:
    """Test ResultTracker.compute_metrics."""

    def setup_method(self):
        self.tracker = ResultTracker()

    @patch("nba.core.result_tracker._connect_axiom")
    def test_no_picks(self, mock_connect):
        mock_connect.return_value = _mock_connect_with_picks([])

        result = self.tracker.compute_metrics("2026-03-01", "2026-03-20")
        assert result["total"] == 0
        assert result["wins"] == 0
        assert result["losses"] == 0
        assert result["win_rate"] == 0.0
        assert result["roi"] == 0.0

    @patch("nba.core.result_tracker._connect_axiom")
    def test_all_wins(self, mock_connect):
        picks = _make_picks(5, 5)
        mock_connect.return_value = _mock_connect_with_picks(picks)

        result = self.tracker.compute_metrics("2026-03-19", "2026-03-19")
        assert result["total"] == 5
        assert result["wins"] == 5
        assert result["losses"] == 0
        assert result["win_rate"] == 100.0

    @patch("nba.core.result_tracker._connect_axiom")
    def test_all_losses(self, mock_connect):
        picks = _make_picks(5, 0)
        mock_connect.return_value = _mock_connect_with_picks(picks)

        result = self.tracker.compute_metrics("2026-03-19", "2026-03-19")
        assert result["total"] == 5
        assert result["wins"] == 0
        assert result["losses"] == 5
        assert result["win_rate"] == 0.0
        assert result["roi"] < 0

    @patch("nba.core.result_tracker._connect_axiom")
    def test_mixed_results_aggregation(self, mock_connect):
        # 2 wins, 1 loss
        picks = _make_picks(3, 2)
        mock_connect.return_value = _mock_connect_with_picks(picks)

        result = self.tracker.compute_metrics("2026-03-19", "2026-03-19")
        assert result["total"] == 3
        assert result["wins"] == 2
        assert result["losses"] == 1
        expected_profit = 2 * 0.909 + 1 * (-1.0)
        assert abs(result["profit"] - round(expected_profit, 2)) < 0.01

    @patch("nba.core.result_tracker._connect_axiom")
    def test_by_market_breakdown(self, mock_connect):
        picks = _make_picks(2, 1, market="POINTS") + _make_picks(3, 2, market="REBOUNDS")
        mock_connect.return_value = _mock_connect_with_picks(picks)

        result = self.tracker.compute_metrics("2026-03-19", "2026-03-19")
        assert "POINTS" in result["by_market"]
        assert "REBOUNDS" in result["by_market"]
        assert result["by_market"]["POINTS"]["w"] == 1
        assert result["by_market"]["POINTS"]["l"] == 1
        assert result["by_market"]["REBOUNDS"]["w"] == 2
        assert result["by_market"]["REBOUNDS"]["l"] == 1

    @patch("nba.core.result_tracker._connect_axiom")
    def test_by_model_breakdown(self, mock_connect):
        picks = _make_picks(2, 1, model="xl") + _make_picks(2, 2, model="v3")
        mock_connect.return_value = _mock_connect_with_picks(picks)

        result = self.tracker.compute_metrics("2026-03-19", "2026-03-19")
        assert "xl" in result["by_model"]
        assert "v3" in result["by_model"]
        assert result["by_model"]["xl"]["w"] == 1
        assert result["by_model"]["v3"]["w"] == 2

    @patch("nba.core.result_tracker._connect_axiom")
    def test_by_tier_breakdown(self, mock_connect):
        picks = _make_picks(2, 2, tier="X") + _make_picks(3, 1, tier="META")
        mock_connect.return_value = _mock_connect_with_picks(picks)

        result = self.tracker.compute_metrics("2026-03-19", "2026-03-19")
        assert "X" in result["by_tier"]
        assert "META" in result["by_tier"]
        assert result["by_tier"]["X"]["win_rate"] == 100.0

    @patch("nba.core.result_tracker._connect_axiom")
    def test_by_edge_bucket_breakdown(self, mock_connect):
        picks = _make_picks(2, 2, edge=12.0) + _make_picks(2, 0, edge=4.0)
        mock_connect.return_value = _mock_connect_with_picks(picks)

        result = self.tracker.compute_metrics("2026-03-19", "2026-03-19")
        assert "10%+" in result["by_edge_bucket"]
        assert "3-5%" in result["by_edge_bucket"]

    @patch("nba.core.result_tracker._connect_axiom")
    def test_none_tier_defaults(self, mock_connect):
        picks = _make_picks(1, 1, tier="META")
        picks[0]["tier"] = None
        mock_connect.return_value = _mock_connect_with_picks(picks)

        result = self.tracker.compute_metrics("2026-03-19", "2026-03-19")
        assert "unknown" in result["by_tier"]

    @patch("nba.core.result_tracker._connect_axiom")
    def test_none_edge(self, mock_connect):
        picks = _make_picks(1, 1)
        picks[0]["edge"] = None
        mock_connect.return_value = _mock_connect_with_picks(picks)

        result = self.tracker.compute_metrics("2026-03-19", "2026-03-19")
        assert "unknown" in result["by_edge_bucket"]


class TestResultTrackerRolling:
    """Test ResultTracker.compute_rolling."""

    def setup_method(self):
        self.tracker = ResultTracker()

    @patch("nba.core.result_tracker._connect_axiom")
    def test_rolling_includes_period(self, mock_connect):
        mock_connect.return_value = _mock_connect_with_picks([])

        result = self.tracker.compute_rolling(7)
        assert result["period"] == "7d"
        assert result["days"] == 7

    @patch("nba.core.result_tracker._connect_axiom")
    def test_rolling_30d(self, mock_connect):
        mock_connect.return_value = _mock_connect_with_picks([])

        result = self.tracker.compute_rolling(30)
        assert result["period"] == "30d"
        assert result["days"] == 30


class TestResultTrackerAnomalies:
    """Test ResultTracker.check_anomalies with various scenarios."""

    def setup_method(self):
        self.tracker = ResultTracker()

    @patch("nba.core.result_tracker._connect_axiom")
    def test_healthy_no_anomalies(self, mock_connect):
        """No alerts when no data (threshold not met)."""
        mock_connect.return_value = _mock_connect_with_picks([])
        anomalies = self.tracker.check_anomalies()
        assert anomalies == []

    @patch.object(ResultTracker, "compute_rolling")
    def test_low_7d_win_rate(self, mock_rolling):
        """Detects 7-day WR below 52%."""
        mock_rolling.side_effect = [
            # 7-day: 40% WR, 10 picks
            {
                "total": 10,
                "wins": 4,
                "losses": 6,
                "win_rate": 40.0,
                "roi": -10.0,
                "by_market": {},
                "by_model": {},
            },
            # 30-day: healthy
            {
                "total": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0,
                "roi": 0,
                "by_market": {},
                "by_model": {},
            },
            # 90-day: empty
            {
                "total": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0,
                "roi": 0,
                "by_market": {},
                "by_model": {},
            },
        ]
        anomalies = self.tracker.check_anomalies()
        wr_alerts = [a for a in anomalies if "7-day WR" in a]
        assert len(wr_alerts) >= 1

    @patch.object(ResultTracker, "compute_rolling")
    def test_low_7d_roi(self, mock_rolling):
        """Detects 7-day ROI below -5%."""
        mock_rolling.side_effect = [
            # 7-day: good WR but bad ROI
            {
                "total": 10,
                "wins": 6,
                "losses": 4,
                "win_rate": 60.0,
                "roi": -8.0,
                "by_market": {},
                "by_model": {},
            },
            # 30-day: empty
            {
                "total": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0,
                "roi": 0,
                "by_market": {},
                "by_model": {},
            },
            # 90-day: empty
            {
                "total": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0,
                "roi": 0,
                "by_market": {},
                "by_model": {},
            },
        ]
        anomalies = self.tracker.check_anomalies()
        roi_alerts = [a for a in anomalies if "ROI" in a]
        assert len(roi_alerts) >= 1

    @patch.object(ResultTracker, "compute_rolling")
    def test_low_30d_win_rate(self, mock_rolling):
        """Detects 30-day WR below 50%."""
        mock_rolling.side_effect = [
            # 7-day: not enough picks
            {
                "total": 3,
                "wins": 2,
                "losses": 1,
                "win_rate": 66.7,
                "roi": 5.0,
                "by_market": {},
                "by_model": {},
            },
            # 30-day: 45% WR, 20 picks
            {
                "total": 20,
                "wins": 9,
                "losses": 11,
                "win_rate": 45.0,
                "roi": -3.0,
                "by_market": {},
                "by_model": {},
            },
            # 90-day: empty
            {
                "total": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0,
                "roi": 0,
                "by_market": {},
                "by_model": {},
            },
        ]
        anomalies = self.tracker.check_anomalies()
        wr30_alerts = [a for a in anomalies if "30-day" in a]
        assert len(wr30_alerts) >= 1

    @patch.object(ResultTracker, "compute_rolling")
    def test_market_anomaly(self, mock_rolling):
        """Detects per-market WR critically low."""
        mock_rolling.side_effect = [
            # 7-day: POINTS at 25% WR (4 picks)
            {
                "total": 8,
                "wins": 3,
                "losses": 5,
                "win_rate": 37.5,
                "roi": -10.0,
                "by_market": {"POINTS": {"w": 1, "l": 3, "total": 4, "win_rate": 25.0}},
                "by_model": {},
            },
            # 30-day
            {
                "total": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0,
                "roi": 0,
                "by_market": {},
                "by_model": {},
            },
            # 90-day
            {
                "total": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0,
                "roi": 0,
                "by_market": {},
                "by_model": {},
            },
        ]
        anomalies = self.tracker.check_anomalies()
        market_alerts = [a for a in anomalies if "POINTS" in a]
        assert len(market_alerts) >= 1

    @patch.object(ResultTracker, "compute_rolling")
    def test_model_anomaly(self, mock_rolling):
        """Detects per-model WR anomaly."""
        mock_rolling.side_effect = [
            # 7-day: xl at 33% WR (3 picks)
            {
                "total": 6,
                "wins": 3,
                "losses": 3,
                "win_rate": 50.0,
                "roi": 0,
                "by_market": {},
                "by_model": {"xl": {"w": 1, "l": 2, "total": 3, "win_rate": 33.3}},
            },
            # 30-day
            {
                "total": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0,
                "roi": 0,
                "by_market": {},
                "by_model": {},
            },
            # 90-day
            {
                "total": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0,
                "roi": 0,
                "by_market": {},
                "by_model": {},
            },
        ]
        anomalies = self.tracker.check_anomalies()
        model_alerts = [a for a in anomalies if "Model xl" in a]
        assert len(model_alerts) >= 1

    @patch.object(ResultTracker, "compute_rolling")
    def test_season_anomaly(self, mock_rolling):
        """Detects season-long WR decline."""
        mock_rolling.side_effect = [
            # 7-day: fine
            {
                "total": 3,
                "wins": 2,
                "losses": 1,
                "win_rate": 66.7,
                "roi": 5.0,
                "by_market": {},
                "by_model": {},
            },
            # 30-day: fine
            {
                "total": 5,
                "wins": 3,
                "losses": 2,
                "win_rate": 60.0,
                "roi": 3.0,
                "by_market": {},
                "by_model": {},
            },
            # 90-day: 50% WR with 60 picks
            {
                "total": 60,
                "wins": 30,
                "losses": 30,
                "win_rate": 50.0,
                "roi": -2.0,
                "by_market": {},
                "by_model": {},
            },
        ]
        anomalies = self.tracker.check_anomalies()
        season_alerts = [a for a in anomalies if "Season" in a]
        assert len(season_alerts) >= 1

    @patch.object(ResultTracker, "compute_rolling")
    def test_multiple_anomalies(self, mock_rolling):
        """Multiple anomalies detected simultaneously."""
        mock_rolling.side_effect = [
            # 7-day: bad WR + bad ROI
            {
                "total": 10,
                "wins": 3,
                "losses": 7,
                "win_rate": 30.0,
                "roi": -30.0,
                "by_market": {"POINTS": {"w": 1, "l": 4, "total": 5, "win_rate": 20.0}},
                "by_model": {"xl": {"w": 1, "l": 4, "total": 5, "win_rate": 20.0}},
            },
            # 30-day
            {
                "total": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0,
                "roi": 0,
                "by_market": {},
                "by_model": {},
            },
            # 90-day
            {
                "total": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0,
                "roi": 0,
                "by_market": {},
                "by_model": {},
            },
        ]
        anomalies = self.tracker.check_anomalies()
        # Should have WR + ROI + market + model = at least 4 alerts
        assert len(anomalies) >= 4


class TestResultTrackerPerformanceSummary:
    """Test ResultTracker.get_performance_summary."""

    def setup_method(self):
        self.tracker = ResultTracker()

    @patch.object(ResultTracker, "check_anomalies", return_value=[])
    @patch.object(ResultTracker, "compute_rolling")
    def test_summary_structure(self, mock_rolling, mock_anomalies):
        mock_rolling.return_value = {
            "total": 10,
            "wins": 6,
            "losses": 4,
            "win_rate": 60.0,
            "roi": 5.0,
            "profit": 0.5,
            "by_market": {},
            "by_tier": {},
            "by_edge_bucket": {},
            "by_model": {},
            "period": "7d",
            "days": 7,
        }

        result = self.tracker.get_performance_summary()

        assert "rolling_7d" in result
        assert "rolling_30d" in result
        assert "anomalies" in result
        assert "computed_at" in result
        assert result["anomalies"] == []

    @patch.object(ResultTracker, "check_anomalies", return_value=["alert1"])
    @patch.object(ResultTracker, "compute_rolling")
    def test_summary_with_anomalies(self, mock_rolling, mock_anomalies):
        mock_rolling.return_value = {
            "total": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0,
            "roi": 0,
            "profit": 0,
            "by_market": {},
            "by_tier": {},
            "by_edge_bucket": {},
            "by_model": {},
        }

        result = self.tracker.get_performance_summary()
        assert result["anomalies"] == ["alert1"]

    @patch("nba.core.clv_tracker.CLVTracker")
    @patch.object(ResultTracker, "check_anomalies", return_value=[])
    @patch.object(ResultTracker, "compute_rolling")
    def test_summary_with_clv(self, mock_rolling, mock_anomalies, MockCLV):
        mock_rolling.return_value = {
            "total": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0,
            "roi": 0,
            "profit": 0,
            "by_market": {},
            "by_tier": {},
            "by_edge_bucket": {},
            "by_model": {},
        }
        mock_clv = MagicMock()
        mock_clv.compute_rolling_clv.return_value = {"avg_clv_cents": 2.5}
        MockCLV.return_value = mock_clv

        result = self.tracker.get_performance_summary()
        assert result["clv_7d"] == {"avg_clv_cents": 2.5}

    @patch.object(ResultTracker, "check_anomalies", return_value=[])
    @patch.object(ResultTracker, "compute_rolling")
    def test_summary_clv_unavailable(self, mock_rolling, mock_anomalies):
        mock_rolling.return_value = {
            "total": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0,
            "roi": 0,
            "profit": 0,
            "by_market": {},
            "by_tier": {},
            "by_edge_bucket": {},
            "by_model": {},
        }

        # Patch the import path within get_performance_summary's local scope
        import nba.core.clv_tracker as clv_mod

        original = clv_mod.CLVTracker
        clv_mod.CLVTracker = MagicMock(side_effect=Exception("no DB"))
        try:
            result = self.tracker.get_performance_summary()
        finally:
            clv_mod.CLVTracker = original

        assert result["clv_7d"] is None


class TestPersistMetrics:
    """Tests for ResultTracker.persist_metrics()."""

    def setup_method(self):
        self.tracker = ResultTracker()

    @patch("nba.core.result_tracker.get_connection")
    @patch.object(ResultTracker, "check_anomalies", return_value=[])
    @patch.object(ResultTracker, "compute_rolling")
    def test_persist_metrics_creates_table(self, mock_rolling, mock_anomalies, mock_get_conn):
        """Verify CREATE TABLE IF NOT EXISTS is called."""
        mock_rolling.return_value = {
            "total": 5,
            "wins": 3,
            "losses": 2,
            "win_rate": 60.0,
            "roi": 5.0,
            "profit": 0.55,
            "by_market": {},
            "by_tier": {},
            "by_edge_bucket": {},
            "by_model": {},
        }
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 1
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_get_conn.return_value = mock_conn

        self.tracker.persist_metrics("2026-03-22")

        # First cursor call should be CREATE TABLE
        calls = mock_cursor.execute.call_args_list
        assert len(calls) >= 1
        create_sql = calls[0][0][0]
        assert "CREATE TABLE IF NOT EXISTS" in create_sql
        assert "performance_metrics" in create_sql

    @patch("nba.core.result_tracker.get_connection")
    @patch.object(ResultTracker, "check_anomalies", return_value=["alert1"])
    @patch.object(ResultTracker, "compute_rolling")
    def test_persist_metrics_inserts_rows(self, mock_rolling, mock_anomalies, mock_get_conn):
        """Verify INSERT is called with correct data for each period."""
        mock_rolling.return_value = {
            "total": 10,
            "wins": 6,
            "losses": 4,
            "win_rate": 60.0,
            "roi": 5.5,
            "profit": 1.45,
            "by_market": {"POINTS": {"w": 6, "l": 4}},
            "by_tier": {"META": {"w": 6, "l": 4}},
            "by_edge_bucket": {"5-7%": {"w": 6, "l": 4}},
            "by_model": {"xl": {"w": 6, "l": 4}},
        }
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 1
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_get_conn.return_value = mock_conn

        rows = self.tracker.persist_metrics("2026-03-22")

        # Should have written rows for both 7d and 30d periods
        assert rows == 2  # 1 row per period * 2 periods

        # Verify INSERT calls (after the CREATE TABLE call)
        insert_calls = [
            c
            for c in mock_cursor.execute.call_args_list
            if "INSERT INTO performance_metrics" in str(c)
        ]
        assert len(insert_calls) == 2

        # Verify the data in the first INSERT call
        first_insert_params = insert_calls[0][0][1]
        assert first_insert_params[0] == "2026-03-22"  # metric_date
        assert first_insert_params[1] == "7d"  # period
        assert first_insert_params[2] == 10  # total_picks

        # Verify commit was called
        mock_conn.commit.assert_called_once()

    @patch("nba.core.result_tracker.get_connection")
    def test_persist_metrics_handles_error(self, mock_get_conn):
        """On DB error, returns 0 (doesn't raise)."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = Exception("DB connection failed")
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_get_conn.return_value = mock_conn

        result = self.tracker.persist_metrics("2026-03-22")

        assert result == 0
        mock_conn.rollback.assert_called_once()
        mock_conn.close.assert_called_once()

    @patch("nba.core.result_tracker.get_connection")
    @patch.object(ResultTracker, "check_anomalies", return_value=[])
    @patch.object(ResultTracker, "compute_rolling")
    def test_persist_metrics_skips_empty_periods(self, mock_rolling, mock_anomalies, mock_get_conn):
        """Periods with 0 total picks are skipped (no INSERT)."""
        mock_rolling.return_value = {
            "total": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0,
            "roi": 0,
            "profit": 0,
            "by_market": {},
            "by_tier": {},
            "by_edge_bucket": {},
            "by_model": {},
        }
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_get_conn.return_value = mock_conn

        rows = self.tracker.persist_metrics("2026-03-22")

        assert rows == 0
        # Only CREATE TABLE should have been called, no INSERTs
        insert_calls = [c for c in mock_cursor.execute.call_args_list if "INSERT" in str(c)]
        assert len(insert_calls) == 0
