"""
API Routes
==========
FastAPI routers for predictions and health checks.
"""

from nba.api.routes.health import router as health_router
from nba.api.routes.picks import router as picks_router
from nba.api.routes.predictions import router as predictions_router

__all__ = ["health_router", "picks_router", "predictions_router"]
