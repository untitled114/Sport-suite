"""
Prediction Endpoints
====================
Endpoints for generating NBA player prop predictions.
"""

import logging
import time
from datetime import date, datetime
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status

from nba.api.dependencies import ModelManager, get_model_manager
from nba.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    Confidence,
    ErrorResponse,
    MarketType,
    PredictionRequest,
    PredictionResponse,
    Side,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predict", tags=["Predictions"])


def generate_reasoning(
    predicted_value: float,
    line: float,
    probability: float,
    side: str,
    confidence: str,
) -> str:
    """
    Generate a human-readable explanation for the prediction.

    Args:
        predicted_value: Model's predicted stat value
        line: Input prop line
        probability: P(OVER)
        side: Recommended side (OVER/UNDER)
        confidence: Confidence level

    Returns:
        Reasoning string
    """
    edge = predicted_value - line
    edge_direction = "above" if edge > 0 else "below"

    if side == "OVER":
        prob_str = f"{probability * 100:.0f}%"
        return (
            f"Model predicts {predicted_value:.1f} ({abs(edge):.1f} {edge_direction} line). "
            f"P(OVER): {prob_str}. Confidence: {confidence}."
        )
    else:
        prob_under = (1 - probability) * 100
        return (
            f"Model predicts {predicted_value:.1f} ({abs(edge):.1f} {edge_direction} line). "
            f"P(UNDER): {prob_under:.0f}%. Confidence: {confidence}."
        )


def make_prediction(
    request: PredictionRequest,
    manager: ModelManager,
) -> PredictionResponse:
    """
    Generate a single prediction.

    Args:
        request: Prediction request
        manager: Model manager instance

    Returns:
        PredictionResponse

    Raises:
        HTTPException: If prediction fails
    """
    market = request.stat_type.value

    # Validate market is enabled
    if not manager.is_market_enabled(market):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Market '{market}' is not enabled. "
            f"Enabled markets: {manager.enabled_markets}",
        )

    # Get predictor
    predictor = manager.get_predictor(market)
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model for {market} is not loaded",
        )

    # Get feature extractor
    extractor = manager.get_feature_extractor()
    if extractor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Feature extractor is not loaded",
        )

    # Determine game date (default to today)
    game_date = request.game_date or date.today()

    try:
        # Extract features
        features = extractor.extract_features(
            player_name=request.player_name,
            game_date=game_date,
            is_home=request.is_home,
            opponent_team=request.opponent_team,
            line=request.line,
            stat_type=market,
            include_book_features=True,
        )

        if features is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Could not extract features for player '{request.player_name}'",
            )

        # Generate prediction
        result = predictor.predict(
            features_dict=features,
            line=request.line,
            player_name=request.player_name,
            game_date=str(game_date),
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Prediction failed for {request.player_name}",
            )

        # Extract prediction values
        predicted_value = result.get("predicted_value", 0.0)
        p_over = result.get("p_over", 0.5)
        edge = result.get("edge", 0.0)
        confidence_str = result.get("confidence", "LOW")

        # Map confidence
        confidence_map = {
            "HIGH": Confidence.HIGH,
            "MEDIUM": Confidence.MEDIUM,
            "LOW": Confidence.LOW,
        }
        confidence = confidence_map.get(confidence_str, Confidence.LOW)

        # Determine side (based on probability, not edge)
        side = Side.OVER if p_over >= 0.5 else Side.UNDER

        # Generate reasoning
        reasoning = generate_reasoning(
            predicted_value=predicted_value,
            line=request.line,
            probability=p_over,
            side=side.value,
            confidence=confidence.value,
        )

        return PredictionResponse(
            player_name=request.player_name,
            stat_type=request.stat_type,
            line=request.line,
            predicted_value=round(predicted_value, 2),
            probability=round(p_over, 4),
            edge=round(edge, 2),
            confidence=confidence,
            side=side,
            reasoning=reasoning,
            opponent_team=request.opponent_team,
            game_date=game_date,
            model_version="xl",
            timestamp=datetime.now(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error for {request.player_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


@router.post(
    "",
    response_model=PredictionResponse,
    summary="Generate single prediction",
    description="Generate a prediction for a single player prop.",
    responses={
        200: {"description": "Successful prediction"},
        400: {"description": "Invalid request (e.g., market not enabled)"},
        404: {"description": "Player not found"},
        500: {"description": "Prediction error"},
        503: {"description": "Model not loaded"},
    },
)
async def predict_single(
    request: PredictionRequest,
) -> PredictionResponse:
    """
    Generate a single prediction.

    Takes a player name, stat type, and line value, returns the model's
    prediction including probability, edge, and recommendation.

    Example request:
    ```json
    {
        "player_name": "LeBron James",
        "stat_type": "POINTS",
        "line": 25.5,
        "opponent_team": "BOS",
        "game_date": "2025-01-30",
        "is_home": true
    }
    ```
    """
    manager = get_model_manager()
    manager.ensure_loaded()

    return make_prediction(request, manager)


@router.post(
    "/batch",
    response_model=BatchPredictionResponse,
    summary="Generate batch predictions",
    description="Generate predictions for multiple player props in a single request.",
    responses={
        200: {"description": "Batch predictions (may include partial failures)"},
        400: {"description": "Invalid request"},
        503: {"description": "Models not loaded"},
    },
)
async def predict_batch(
    request: BatchPredictionRequest,
) -> BatchPredictionResponse:
    """
    Generate batch predictions.

    Takes a list of prediction requests and returns predictions for all.
    Individual failures do not fail the entire batch.

    Example request:
    ```json
    {
        "predictions": [
            {"player_name": "LeBron James", "stat_type": "POINTS", "line": 25.5},
            {"player_name": "Anthony Davis", "stat_type": "REBOUNDS", "line": 10.5}
        ]
    }
    ```
    """
    start_time = time.time()

    manager = get_model_manager()
    manager.ensure_loaded()

    predictions: List[PredictionResponse] = []
    failed = 0

    for req in request.predictions:
        try:
            prediction = make_prediction(req, manager)
            predictions.append(prediction)
        except HTTPException as e:
            logger.warning(f"Batch prediction failed for {req.player_name}: {e.detail}")
            failed += 1
        except Exception as e:
            logger.error(f"Batch prediction error for {req.player_name}: {e}")
            failed += 1

    processing_time_ms = (time.time() - start_time) * 1000

    return BatchPredictionResponse(
        predictions=predictions,
        total=len(request.predictions),
        successful=len(predictions),
        failed=failed,
        processing_time_ms=round(processing_time_ms, 2),
        timestamp=datetime.now(),
    )


@router.get(
    "/markets",
    summary="Get enabled markets",
    description="Returns list of markets enabled for predictions.",
)
async def get_enabled_markets() -> dict:
    """
    Get list of enabled markets.

    Returns the markets that are currently enabled for predictions.
    Per system configuration, only POINTS and REBOUNDS are enabled.
    """
    manager = get_model_manager()

    return {
        "enabled_markets": manager.enabled_markets,
        "disabled_markets": ["ASSISTS", "THREES"],
        "note": "ASSISTS and THREES are disabled due to poor validation performance",
    }
