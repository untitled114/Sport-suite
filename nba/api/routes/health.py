"""
Health Check Endpoints
======================
Endpoints for monitoring API and system health.
"""

import logging
from datetime import datetime

import psycopg2
from fastapi import APIRouter, Depends

from nba.api import __version__
from nba.api.dependencies import DatabaseConfig, check_all_databases, get_model_manager
from nba.api.schemas import (
    DatabaseHealthResponse,
    DatabaseStatus,
    HealthResponse,
    ModelsHealthResponse,
    ModelStatus,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["Health"])


@router.get(
    "",
    response_model=HealthResponse,
    summary="Basic health check",
    description="Returns basic health status and API version. Used for liveness probes.",
)
async def health_check() -> HealthResponse:
    """
    Basic health check endpoint.

    Returns API status and version. Suitable for Kubernetes liveness probes.
    """
    return HealthResponse(
        status="healthy",
        version=__version__,
        timestamp=datetime.now(),
    )


@router.get(
    "/models",
    response_model=ModelsHealthResponse,
    summary="Check model status",
    description="Returns status of all loaded ML models. Used for readiness probes.",
)
async def models_health_check() -> ModelsHealthResponse:
    """
    Model health check endpoint.

    Returns status of all prediction models. Suitable for readiness probes.
    Models are loaded lazily on first request.
    """
    manager = get_model_manager()

    # Get model status (may trigger lazy loading)
    try:
        manager.ensure_loaded()
    except (psycopg2.Error, KeyError, TypeError, ValueError) as e:
        logger.error(f"Error loading models: {e}")

    status_info = manager.get_model_status()

    # Build model status list
    models = []
    for market in manager.enabled_markets:
        model_info = status_info["models"].get(market, {})
        models.append(
            ModelStatus(
                market=market,
                loaded=model_info.get("loaded", False),
                feature_count=model_info.get("feature_count"),
                model_version=model_info.get("model_version"),
            )
        )

    total_loaded = sum(1 for m in models if m.loaded)
    overall_status = "healthy" if total_loaded == len(manager.enabled_markets) else "degraded"

    return ModelsHealthResponse(
        status=overall_status,
        models=models,
        total_loaded=total_loaded,
        enabled_markets=manager.enabled_markets,
        timestamp=datetime.now(),
    )


@router.get(
    "/db",
    response_model=DatabaseHealthResponse,
    summary="Check database connectivity",
    description="Returns connectivity status for all NBA databases.",
)
async def database_health_check() -> DatabaseHealthResponse:
    """
    Database health check endpoint.

    Checks connectivity to all NBA databases (players, games, team, intelligence).
    """
    result = check_all_databases()

    databases = [
        DatabaseStatus(
            name=db["name"],
            connected=db["connected"],
            host=db["host"],
            port=db["port"],
            error=db.get("error"),
        )
        for db in result["databases"]
    ]

    return DatabaseHealthResponse(
        status=result["status"],
        databases=databases,
        total_connected=result["total_connected"],
        timestamp=datetime.now(),
    )


@router.get(
    "/ready",
    response_model=HealthResponse,
    summary="Readiness check",
    description="Comprehensive readiness check - verifies models and databases are available.",
)
async def readiness_check() -> HealthResponse:
    """
    Readiness check endpoint.

    Verifies that:
    1. At least one model is loaded
    2. At least one database is connected

    Suitable for Kubernetes readiness probes.
    """
    manager = get_model_manager()

    # Check models
    try:
        manager.ensure_loaded()
        model_status = manager.get_model_status()
        models_ok = any(m.get("loaded", False) for m in model_status["models"].values())
    except (psycopg2.Error, KeyError, TypeError, ValueError) as e:
        logger.error(f"Readiness check - model error: {e}")
        models_ok = False

    # Check databases
    try:
        db_result = check_all_databases()
        db_ok = db_result["total_connected"] >= 1
    except (psycopg2.Error, KeyError, TypeError, ValueError) as e:
        logger.error(f"Readiness check - database error: {e}")
        db_ok = False

    # Determine overall status
    if models_ok and db_ok:
        status = "healthy"
    elif models_ok or db_ok:
        status = "degraded"
    else:
        status = "unhealthy"

    return HealthResponse(
        status=status,
        version=__version__,
        timestamp=datetime.now(),
    )
