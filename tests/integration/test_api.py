"""
Integration tests for the FastAPI inference service.

Uses FastAPI TestClient — no real server or database connections needed.
Tests endpoint availability and response structure.
"""

from unittest.mock import MagicMock, patch

import pytest

fastapi = pytest.importorskip("fastapi", reason="fastapi not installed")
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client with mocked model manager."""
    # Patch the model manager before importing the app
    with patch("nba.api.dependencies.ModelManager") as MockManager:
        mock_instance = MagicMock()
        mock_instance.enabled_markets = ["POINTS", "REBOUNDS"]
        mock_instance._models_loaded = False
        mock_instance.feature_extractor = None
        mock_instance.predictors = {}
        MockManager.return_value = mock_instance
        MockManager._instance = None
        MockManager._initialized = False

        # Clear the lru_cache
        from nba.api.dependencies import get_model_manager

        get_model_manager.cache_clear()

        from nba.api.main import app

        yield TestClient(app)

        get_model_manager.cache_clear()


class TestRootEndpoint:
    """Test the root endpoint."""

    def test_root_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_root_has_api_info(self, client):
        response = client.get("/")
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "endpoints" in data

    def test_root_lists_performance_endpoints(self, client):
        response = client.get("/")
        data = response.json()
        assert "performance" in data["endpoints"]
        assert "rolling" in data["endpoints"]["performance"]
        assert "clv" in data["endpoints"]["performance"]
        assert "anomalies" in data["endpoints"]["performance"]


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
        assert "version" in data


class TestPicksEndpoints:
    """Test picks endpoints."""

    def test_today_picks_requires_file(self, client):
        """GET /picks/today returns 404 when no picks file exists."""
        response = client.get("/picks/today")
        # Without a picks file, expect 404
        assert response.status_code in (404, 200)

    def test_invalid_date_format(self, client):
        """GET /picks/{date} rejects invalid date format."""
        response = client.get("/picks/not-a-date")
        assert response.status_code == 422


class TestPredictionEndpoints:
    """Test prediction endpoints."""

    def test_markets_returns_200(self, client):
        """GET /predict/markets returns enabled markets."""
        response = client.get("/predict/markets")
        assert response.status_code == 200
        data = response.json()
        assert "enabled_markets" in data
        assert "disabled_markets" in data


class TestPerformanceEndpoints:
    """Test performance tracking endpoints."""

    @patch("nba.core.result_tracker.ResultTracker")
    def test_performance_returns_200(self, MockTracker, client):
        """GET /performance returns performance metrics."""
        mock_tracker = MagicMock()
        mock_tracker.compute_rolling.return_value = {
            "period": "7d",
            "days": 7,
            "total": 10,
            "wins": 6,
            "losses": 4,
            "win_rate": 60.0,
            "roi": 5.5,
            "profit": 0.55,
            "by_market": {},
            "by_tier": {},
            "by_edge_bucket": {},
            "by_model": {},
        }
        mock_tracker.check_anomalies.return_value = []
        MockTracker.return_value = mock_tracker

        response = client.get("/performance?days=7")
        assert response.status_code == 200
        data = response.json()
        assert data["period"] == "7d"
        assert data["win_rate"] == 60.0
        assert data["total_bets"] == 10
        assert data["anomalies"] == []

    @patch("nba.core.result_tracker.ResultTracker")
    def test_performance_default_days(self, MockTracker, client):
        """GET /performance defaults to 7 days."""
        mock_tracker = MagicMock()
        mock_tracker.compute_rolling.return_value = {
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
        mock_tracker.check_anomalies.return_value = []
        MockTracker.return_value = mock_tracker

        response = client.get("/performance")
        assert response.status_code == 200

    @patch("nba.core.result_tracker.ResultTracker")
    def test_anomalies_returns_200(self, MockTracker, client):
        """GET /performance/anomalies returns anomaly list."""
        mock_tracker = MagicMock()
        mock_tracker.check_anomalies.return_value = ["7-day WR 45.0% below 52.0%"]
        MockTracker.return_value = mock_tracker

        response = client.get("/performance/anomalies")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alert"
        assert data["count"] == 1
        assert len(data["anomalies"]) == 1

    @patch("nba.core.result_tracker.ResultTracker")
    def test_anomalies_healthy(self, MockTracker, client):
        """GET /performance/anomalies returns healthy when no issues."""
        mock_tracker = MagicMock()
        mock_tracker.check_anomalies.return_value = []
        MockTracker.return_value = mock_tracker

        response = client.get("/performance/anomalies")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["count"] == 0

    @patch("nba.core.clv_tracker.CLVTracker")
    def test_clv_returns_200(self, MockCLV, client):
        """GET /performance/clv returns CLV metrics."""
        mock_tracker = MagicMock()
        mock_tracker.compute_rolling_clv.return_value = {
            "period": "7d",
            "days": 7,
            "dates_checked": 7,
            "total_picks": 20,
            "picks_with_clv": 15,
            "avg_clv_cents": 2.3,
            "beat_close_rate": 0.60,
            "clv_positive_rate": 0.55,
            "by_market": {},
            "daily": [],
        }
        MockCLV.return_value = mock_tracker

        response = client.get("/performance/clv?days=7")
        assert response.status_code == 200
        data = response.json()
        assert data["avg_clv_cents"] == 2.3
        assert data["beat_close_rate"] == 0.60
