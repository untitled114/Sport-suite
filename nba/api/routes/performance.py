"""
Performance Endpoints
=====================
Endpoints for querying pick performance, CLV, and anomaly detection.

These endpoints read from nba_prediction_history (graded picks)
and nba_line_snapshots (closing line data).
"""

import logging
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

from fastapi import APIRouter, HTTPException, Query, status

EST = ZoneInfo("America/New_York")

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/performance", tags=["Performance"])


@router.get(
    "",
    summary="Rolling performance metrics",
    description="Returns win rate, ROI, CLV, and anomalies for the last N days.",
)
async def get_performance(
    days: int = 7,
) -> dict:
    """Get rolling performance metrics.

    Returns win rate, ROI, by-market/tier/model breakdown,
    CLV metrics, and any active anomalies.
    """
    try:
        from nba.core.result_tracker import ResultTracker

        tracker = ResultTracker()
        metrics = tracker.compute_rolling(days)
        anomalies = tracker.check_anomalies()

        # Try CLV
        clv_data = None
        try:
            from nba.core.clv_tracker import CLVTracker

            clv_tracker = CLVTracker()
            clv_data = clv_tracker.compute_rolling_clv(min(days, 14))
            # Remove daily detail for API response brevity
            if clv_data:
                clv_data.pop("daily", None)
        except Exception as e:
            logger.debug(f"CLV unavailable: {e}")

        return {
            "period": f"{days}d",
            "win_rate": metrics.get("win_rate", 0),
            "roi": metrics.get("roi", 0),
            "total_bets": metrics.get("total", 0),
            "wins": metrics.get("wins", 0),
            "losses": metrics.get("losses", 0),
            "profit": metrics.get("profit", 0),
            "by_market": metrics.get("by_market", {}),
            "by_tier": metrics.get("by_tier", {}),
            "by_edge_bucket": metrics.get("by_edge_bucket", {}),
            "by_model": metrics.get("by_model", {}),
            "clv": clv_data,
            "anomalies": anomalies,
            "timestamp": datetime.now(EST).isoformat(),
        }
    except Exception as e:
        logger.error(f"Performance query failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Performance query failed: {str(e)}",
        )


@router.get(
    "/clv",
    summary="Closing Line Value metrics",
    description="CLV measures whether lines moved in your direction after your pick.",
)
async def get_clv(
    days: int = 7,
) -> dict:
    """Get CLV (Closing Line Value) metrics.

    Positive CLV means you're consistently getting better lines than the market,
    which is the strongest indicator of long-term profitability.
    """
    try:
        from nba.core.clv_tracker import CLVTracker

        tracker = CLVTracker()
        result = tracker.compute_rolling_clv(days)
        return result
    except Exception as e:
        logger.error(f"CLV query failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"CLV query failed: {str(e)}",
        )


@router.get(
    "/anomalies",
    summary="Active performance anomalies",
    description="Check for anomalies: WR drops, ROI decline, model-specific issues.",
)
async def get_anomalies() -> dict:
    """Get active performance anomalies.

    Returns a list of alerts if any metrics are below thresholds.
    Empty list means all metrics are healthy.
    """
    try:
        from nba.core.result_tracker import ResultTracker

        tracker = ResultTracker()
        anomalies = tracker.check_anomalies()
        return {
            "status": "alert" if anomalies else "healthy",
            "anomalies": anomalies,
            "count": len(anomalies),
            "timestamp": datetime.now(EST).isoformat(),
        }
    except Exception as e:
        logger.error(f"Anomaly check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Anomaly check failed: {str(e)}",
        )
