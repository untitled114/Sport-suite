"""
API Routes
==========
FastAPI routers for predictions, health checks, picks, and performance.
"""

from nba.api.routes.health import router as health_router
from nba.api.routes.performance import router as performance_router
from nba.api.routes.picks import router as picks_router
from nba.api.routes.predictions import router as predictions_router

__all__ = ["health_router", "performance_router", "picks_router", "predictions_router"]
