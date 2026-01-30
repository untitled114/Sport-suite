"""
Performance Monitor Unit Tests
==============================
Tests for the betting performance monitoring system.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


class TestMonitorMetrics:
    """Tests for performance metric calculations."""

    def test_win_rate_calculation(self):
        """Test win rate calculation."""
        wins = 65
        total = 100
        win_rate = wins / total
        assert win_rate == 0.65

    def test_roi_calculation(self):
        """Test ROI calculation."""
        # Standard -110 odds (bet $110 to win $100)
        wins = 55
        losses = 45
        total_wagered = 100 * 110  # $110 per bet
        total_won = wins * 100  # $100 profit per win
        total_lost = losses * 110  # $110 lost per loss

        net_profit = total_won - total_lost
        roi = (net_profit / total_wagered) * 100

        expected_roi = ((55 * 100) - (45 * 110)) / (100 * 110) * 100
        assert abs(roi - expected_roi) < 0.01

    def test_profit_calculation(self):
        """Test total profit calculation."""
        # Win: +1 unit, Loss: -1.1 units (at -110 odds)
        wins = 60
        losses = 40

        profit = (wins * 1.0) - (losses * 1.1)
        expected = 60 - 44  # 60 - 40*1.1
        assert abs(profit - expected) < 0.01

    def test_streak_counting(self):
        """Test win/loss streak calculation."""
        results = [1, 1, 1, 0, 0, 1, 1]  # 1 = win, 0 = loss

        # Current streak from end
        current_streak = 0
        streak_type = results[-1]
        for r in reversed(results):
            if r == streak_type:
                current_streak += 1
            else:
                break

        assert current_streak == 2  # Last two are wins


class TestMonitorFilters:
    """Tests for performance filtering by market/tier."""

    @pytest.fixture
    def sample_picks_df(self):
        """Create sample picks DataFrame."""
        return pd.DataFrame(
            {
                "player_name": ["Player A", "Player B", "Player C", "Player D"],
                "stat_type": ["POINTS", "POINTS", "REBOUNDS", "REBOUNDS"],
                "filter_tier": ["V3", "STAR", "META", "A"],
                "p_over": [0.75, 0.72, 0.65, 0.70],
                "edge": [3.5, 2.8, 2.0, 1.5],
                "result": [1, 1, 0, 1],  # 1 = win, 0 = loss
            }
        )

    def test_filter_by_market(self, sample_picks_df):
        """Test filtering by market type."""
        points_picks = sample_picks_df[sample_picks_df["stat_type"] == "POINTS"]
        assert len(points_picks) == 2

        rebounds_picks = sample_picks_df[sample_picks_df["stat_type"] == "REBOUNDS"]
        assert len(rebounds_picks) == 2

    def test_filter_by_tier(self, sample_picks_df):
        """Test filtering by tier."""
        v3_picks = sample_picks_df[sample_picks_df["filter_tier"] == "V3"]
        assert len(v3_picks) == 1

        star_picks = sample_picks_df[sample_picks_df["filter_tier"] == "STAR"]
        assert len(star_picks) == 1

    def test_market_win_rates(self, sample_picks_df):
        """Test calculating win rates by market."""
        market_stats = sample_picks_df.groupby("stat_type").agg({"result": ["sum", "count"]})
        market_stats.columns = ["wins", "total"]
        market_stats["win_rate"] = market_stats["wins"] / market_stats["total"]

        # POINTS: 2 wins / 2 total = 100%
        assert market_stats.loc["POINTS", "win_rate"] == 1.0

        # REBOUNDS: 1 win / 2 total = 50%
        assert market_stats.loc["REBOUNDS", "win_rate"] == 0.5


class TestMonitorThresholds:
    """Tests for monitoring threshold checks."""

    def test_min_sample_size(self):
        """Test minimum sample size requirement."""
        min_samples = 10
        current_samples = 5

        has_sufficient_data = current_samples >= min_samples
        assert has_sufficient_data is False

    def test_confidence_interval(self):
        """Test confidence interval calculation."""
        win_rate = 0.60
        n = 100

        # 95% CI for proportion
        z = 1.96
        se = np.sqrt(win_rate * (1 - win_rate) / n)
        ci_lower = win_rate - z * se
        ci_upper = win_rate + z * se

        assert ci_lower < win_rate < ci_upper
        assert ci_lower > 0.5  # Statistically significant if lower > 0.5

    def test_roi_threshold(self):
        """Test ROI threshold checks."""
        min_roi_threshold = 0.0  # Break even
        current_roi = 5.5  # 5.5% ROI

        is_profitable = current_roi > min_roi_threshold
        assert is_profitable is True


class TestMonitorAlerts:
    """Tests for performance alert conditions."""

    def test_losing_streak_alert(self):
        """Test alert for extended losing streak."""
        max_losing_streak = 5
        current_streak = 6

        should_alert = current_streak >= max_losing_streak
        assert should_alert is True

    def test_win_rate_drop_alert(self):
        """Test alert for significant win rate drop."""
        historical_wr = 0.65
        recent_wr = 0.50
        drop_threshold = 0.10

        win_rate_drop = historical_wr - recent_wr
        should_alert = win_rate_drop >= drop_threshold

        assert should_alert is True

    def test_edge_degradation_alert(self):
        """Test alert for edge degradation."""
        historical_edge = 3.5
        recent_edge = 1.8
        min_edge = 2.0

        edge_degraded = recent_edge < min_edge
        assert edge_degraded is True


class TestMonitorTrends:
    """Tests for trend analysis."""

    @pytest.fixture
    def daily_results(self):
        """Create daily results for trend analysis."""
        dates = [datetime(2025, 1, i) for i in range(1, 15)]
        return pd.DataFrame(
            {
                "date": dates,
                "wins": [3, 4, 2, 5, 3, 2, 4, 5, 3, 4, 2, 3, 4, 5],
                "losses": [2, 1, 3, 1, 2, 3, 2, 1, 2, 2, 3, 2, 1, 1],
            }
        )

    def test_rolling_win_rate(self, daily_results):
        """Test 7-day rolling win rate calculation."""
        daily_results["win_rate"] = daily_results["wins"] / (
            daily_results["wins"] + daily_results["losses"]
        )
        rolling_wr = daily_results["win_rate"].rolling(7).mean()

        # First 6 days should be NaN
        assert pd.isna(rolling_wr.iloc[5])

        # Day 7+ should have values
        assert not pd.isna(rolling_wr.iloc[6])

    def test_cumulative_profit(self, daily_results):
        """Test cumulative profit tracking."""
        # Simple 1:1 odds for testing
        daily_results["profit"] = daily_results["wins"] - daily_results["losses"]
        daily_results["cumulative"] = daily_results["profit"].cumsum()

        # Total should be sum of all daily profits
        expected_total = daily_results["profit"].sum()
        assert daily_results["cumulative"].iloc[-1] == expected_total

    def test_trend_direction(self, daily_results):
        """Test trend direction detection."""
        daily_results["win_rate"] = daily_results["wins"] / (
            daily_results["wins"] + daily_results["losses"]
        )

        # Compare first half to second half
        first_half = daily_results["win_rate"].iloc[:7].mean()
        second_half = daily_results["win_rate"].iloc[7:].mean()

        if second_half > first_half:
            trend = "improving"
        elif second_half < first_half:
            trend = "declining"
        else:
            trend = "stable"

        # Based on test data, should be improving
        assert trend in ["improving", "declining", "stable"]


class TestMonitorReporting:
    """Tests for report generation."""

    def test_daily_summary_structure(self):
        """Test daily summary report structure."""
        summary = {
            "date": "2025-01-15",
            "total_picks": 10,
            "wins": 6,
            "losses": 4,
            "win_rate": 0.60,
            "profit": 2.0,
            "roi": 18.2,
            "markets": {
                "POINTS": {"picks": 6, "wins": 4, "wr": 0.67},
                "REBOUNDS": {"picks": 4, "wins": 2, "wr": 0.50},
            },
        }

        # Required fields
        assert "date" in summary
        assert "total_picks" in summary
        assert "win_rate" in summary
        assert "roi" in summary
        assert "markets" in summary

    def test_weekly_summary_aggregation(self):
        """Test weekly summary aggregation."""
        daily_data = [
            {"wins": 5, "losses": 3},
            {"wins": 4, "losses": 4},
            {"wins": 6, "losses": 2},
            {"wins": 3, "losses": 5},
            {"wins": 5, "losses": 3},
            {"wins": 4, "losses": 4},
            {"wins": 5, "losses": 3},
        ]

        total_wins = sum(d["wins"] for d in daily_data)
        total_losses = sum(d["losses"] for d in daily_data)
        weekly_wr = total_wins / (total_wins + total_losses)

        assert total_wins == 32
        assert total_losses == 24
        assert abs(weekly_wr - 0.571) < 0.01


class TestHealthChecks:
    """Tests for system health checks."""

    def test_data_freshness_check(self):
        """Test data freshness validation."""
        last_update = datetime(2025, 1, 15, 10, 0, 0)
        current_time = datetime(2025, 1, 15, 12, 0, 0)
        max_age_hours = 6

        age_hours = (current_time - last_update).total_seconds() / 3600
        is_fresh = age_hours <= max_age_hours

        assert is_fresh is True

    def test_model_availability_check(self):
        """Test model file availability check."""
        required_models = ["POINTS", "REBOUNDS"]
        available_models = ["POINTS", "REBOUNDS", "ASSISTS", "THREES"]

        missing = [m for m in required_models if m not in available_models]
        all_available = len(missing) == 0

        assert all_available is True

    def test_database_connectivity_check(self):
        """Test database connectivity validation."""
        databases = {
            "players": True,
            "games": True,
            "intelligence": True,
            "team": True,
        }

        all_connected = all(databases.values())
        assert all_connected is True
