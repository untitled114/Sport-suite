"""
NBA Props Prediction API
========================
FastAPI application for NBA player props predictions.

Features:
- Single and batch prediction endpoints
- Health checks (API, models, databases)
- Lazy model loading for fast startup
- CORS support for web clients

Usage:
    # Development
    uvicorn nba.api.main:app --reload --port 8000

    # Production
    uvicorn nba.api.main:app --host 0.0.0.0 --port 8000 --workers 4

API Documentation:
    - Swagger UI: http://localhost:8000/docs
    - ReDoc: http://localhost:8000/redoc
    - OpenAPI JSON: http://localhost:8000/openapi.json
"""

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime

import psycopg2
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from nba.api import __version__
from nba.api.dependencies import get_model_manager
from nba.api.routes import health_router, predictions_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ==============================================================================
# Application Lifespan
# ==============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Handles startup and shutdown events.
    Models are loaded lazily on first request, not at startup.
    """
    # Startup
    logger.info("=" * 60)
    logger.info("NBA Props Prediction API starting...")
    logger.info(f"Version: {__version__}")
    logger.info("=" * 60)

    # Initialize model manager (does not load models yet - lazy loading)
    manager = get_model_manager()
    logger.info(f"Enabled markets: {manager.enabled_markets}")
    logger.info("Models will load on first prediction request (lazy loading)")

    # Optional: Preload models at startup (comment out for faster startup)
    preload = os.getenv("API_PRELOAD_MODELS", "false").lower() == "true"
    if preload:
        logger.info("Preloading models (API_PRELOAD_MODELS=true)...")
        try:
            manager.ensure_loaded()
            logger.info("Models preloaded successfully")
        except (psycopg2.Error, KeyError, TypeError, ValueError) as e:
            logger.error(f"Failed to preload models: {e}")

    yield

    # Shutdown
    logger.info("NBA Props Prediction API shutting down...")

    # Cleanup feature extractor database connections
    try:
        if manager.feature_extractor is not None:
            manager.feature_extractor.close()
            logger.info("Feature extractor connections closed")
    except (psycopg2.Error, KeyError, TypeError, ValueError) as e:
        logger.error(f"Error closing connections: {e}")


# ==============================================================================
# Create FastAPI Application
# ==============================================================================

app = FastAPI(
    title="NBA Props Prediction API",
    description="""
## NBA Player Props Prediction API

This API provides machine learning-powered predictions for NBA player props betting.

### Features

- **Single Predictions**: Get predictions for individual player props
- **Batch Predictions**: Process multiple predictions in a single request
- **Health Checks**: Monitor API, model, and database status

### Enabled Markets

Currently, only **POINTS** and **REBOUNDS** markets are enabled based on
validation performance:

| Market | Status | Win Rate | ROI |
|--------|--------|----------|-----|
| POINTS | Enabled | 56.7% | +8.27% |
| REBOUNDS | Enabled | 61.2% | +16.96% |
| ASSISTS | Disabled | 14.6% | -72.05% |
| THREES | Disabled | 46.5% | -11.23% |

### Model Architecture

The API uses XL (Extra Large) models with a two-head stacked architecture:
- **Head 1 (Regressor)**: Predicts the stat value
- **Head 2 (Classifier)**: Predicts P(OVER)

Features include:
- 78 player features (rolling stats, team context, matchup history)
- 20 book disagreement features (line spread, deviations)
- 2 computed features (is_home, expected_diff)
- 2 base features (line, opponent_team)

### Authentication

This API does not currently require authentication.
    """,
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    openapi_tags=[
        {
            "name": "Predictions",
            "description": "Generate predictions for player props",
        },
        {
            "name": "Health",
            "description": "Health checks and status monitoring",
        },
    ],
    contact={
        "name": "Sports Suite",
        "url": "https://github.com/sports-suite",
    },
    license_info={
        "name": "MIT",
    },
)


# ==============================================================================
# CORS Middleware
# ==============================================================================

# Configure CORS
cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# Exception Handlers
# ==============================================================================


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler for unhandled errors.

    Logs the error and returns a standardized error response.
    """
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred",
            "detail": str(exc) if os.getenv("DEBUG", "false").lower() == "true" else None,
            "timestamp": datetime.now().isoformat(),
        },
    )


# ==============================================================================
# Include Routers
# ==============================================================================

app.include_router(predictions_router)
app.include_router(health_router)


# ==============================================================================
# Root Endpoint
# ==============================================================================


@app.get(
    "/",
    tags=["Root"],
    summary="API Information",
    description="Returns basic API information and links to documentation.",
)
async def root():
    """
    Root endpoint.

    Returns API information and documentation links.
    """
    return {
        "name": "NBA Props Prediction API",
        "version": __version__,
        "description": "Machine learning predictions for NBA player props",
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc",
            "openapi": "/openapi.json",
        },
        "endpoints": {
            "predictions": {
                "single": "POST /predict",
                "batch": "POST /predict/batch",
                "markets": "GET /predict/markets",
            },
            "health": {
                "basic": "GET /health",
                "models": "GET /health/models",
                "database": "GET /health/db",
                "ready": "GET /health/ready",
            },
        },
        "enabled_markets": ["POINTS", "REBOUNDS"],
        "timestamp": datetime.now().isoformat(),
    }


# ==============================================================================
# Main Entry Point
# ==============================================================================

if __name__ == "__main__":
    import uvicorn

    # Get configuration from environment
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    workers = int(os.getenv("API_WORKERS", 1))
    reload = os.getenv("API_RELOAD", "false").lower() == "true"

    uvicorn.run(
        "nba.api.main:app",
        host=host,
        port=port,
        workers=workers,
        reload=reload,
        log_level="info",
    )
