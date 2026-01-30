"""
Pydantic Schemas for NBA Props Prediction API
==============================================
Request and response models for the API endpoints.
"""

from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class MarketType(str, Enum):
    """Supported prop markets."""

    POINTS = "POINTS"
    REBOUNDS = "REBOUNDS"
    ASSISTS = "ASSISTS"
    THREES = "THREES"


class Side(str, Enum):
    """Bet side recommendation."""

    OVER = "OVER"
    UNDER = "UNDER"


class Confidence(str, Enum):
    """Prediction confidence level."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class BookName(str, Enum):
    """Supported sportsbooks."""

    DRAFTKINGS = "draftkings"
    FANDUEL = "fanduel"
    BETMGM = "betmgm"
    CAESARS = "caesars"
    BET365 = "bet365"
    BETRIVERS = "betrivers"
    ESPNBET = "espnbet"
    FANATICS = "fanatics"
    UNDERDOG = "underdog"
    PRIZEPICKS = "prizepicks"


# ==============================================================================
# Request Models
# ==============================================================================


class PredictionRequest(BaseModel):
    """
    Request body for single prediction endpoint.

    Example:
        {
            "player_name": "LeBron James",
            "stat_type": "POINTS",
            "line": 25.5,
            "opponent_team": "BOS",
            "game_date": "2025-01-30",
            "is_home": true
        }
    """

    player_name: str = Field(..., min_length=2, max_length=100, description="Player's full name")
    stat_type: MarketType = Field(..., description="Prop market type")
    line: float = Field(..., ge=0.5, le=100, description="Prop line value")
    opponent_team: Optional[str] = Field(
        None, min_length=2, max_length=3, description="Opponent team code (e.g., 'BOS')"
    )
    game_date: Optional[date] = Field(None, description="Game date (defaults to today)")
    is_home: Optional[bool] = Field(None, description="Is player's team at home")
    book_name: Optional[BookName] = Field(None, description="Sportsbook offering the line")

    @field_validator("player_name")
    @classmethod
    def normalize_player_name(cls, v: str) -> str:
        """Strip whitespace and normalize."""
        return " ".join(v.split())

    @field_validator("opponent_team")
    @classmethod
    def normalize_opponent_team(cls, v: Optional[str]) -> Optional[str]:
        """Normalize team code to uppercase."""
        if v is not None:
            return v.upper()
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "player_name": "LeBron James",
                "stat_type": "POINTS",
                "line": 25.5,
                "opponent_team": "BOS",
                "game_date": "2025-01-30",
                "is_home": True,
            }
        }


class BatchPredictionRequest(BaseModel):
    """
    Request body for batch prediction endpoint.

    Example:
        {
            "predictions": [
                {"player_name": "LeBron James", "stat_type": "POINTS", "line": 25.5},
                {"player_name": "Anthony Davis", "stat_type": "REBOUNDS", "line": 10.5}
            ]
        }
    """

    predictions: List[PredictionRequest] = Field(
        ..., min_length=1, max_length=100, description="List of prediction requests"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "predictions": [
                    {
                        "player_name": "LeBron James",
                        "stat_type": "POINTS",
                        "line": 25.5,
                        "opponent_team": "BOS",
                    },
                    {
                        "player_name": "Anthony Davis",
                        "stat_type": "REBOUNDS",
                        "line": 10.5,
                        "opponent_team": "BOS",
                    },
                ]
            }
        }


# ==============================================================================
# Response Models
# ==============================================================================


class PredictionResponse(BaseModel):
    """
    Response body for a single prediction.

    Contains the model's prediction, probability, and recommendation.
    """

    player_name: str = Field(..., description="Player's full name")
    stat_type: MarketType = Field(..., description="Prop market type")
    line: float = Field(..., description="Input prop line")
    predicted_value: float = Field(..., description="Model's predicted stat value")
    probability: float = Field(..., ge=0, le=1, description="Probability of hitting OVER")
    edge: float = Field(..., description="Predicted value minus line")
    confidence: Confidence = Field(..., description="Prediction confidence level")
    side: Side = Field(..., description="Recommended bet side")
    reasoning: str = Field(..., description="Brief explanation of the prediction")
    opponent_team: Optional[str] = Field(None, description="Opponent team code")
    game_date: Optional[date] = Field(None, description="Game date")
    model_version: str = Field(default="xl", description="Model version used")
    timestamp: datetime = Field(default_factory=datetime.now, description="Prediction timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "player_name": "LeBron James",
                "stat_type": "POINTS",
                "line": 25.5,
                "predicted_value": 28.3,
                "probability": 0.72,
                "edge": 2.8,
                "confidence": "MEDIUM",
                "side": "OVER",
                "reasoning": "Model predicts 28.3 points (edge: +2.8). Probability of OVER: 72%.",
                "opponent_team": "BOS",
                "game_date": "2025-01-30",
                "model_version": "xl",
                "timestamp": "2025-01-30T14:30:00",
            }
        }


class BatchPredictionResponse(BaseModel):
    """
    Response body for batch predictions.

    Contains list of predictions and summary statistics.
    """

    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    total: int = Field(..., description="Total number of predictions requested")
    successful: int = Field(..., description="Number of successful predictions")
    failed: int = Field(..., description="Number of failed predictions")
    processing_time_ms: float = Field(..., description="Total processing time in ms")
    timestamp: datetime = Field(default_factory=datetime.now, description="Batch timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "predictions": [],
                "total": 2,
                "successful": 2,
                "failed": 0,
                "processing_time_ms": 150.5,
                "timestamp": "2025-01-30T14:30:00",
            }
        }


class ErrorResponse(BaseModel):
    """
    Error response body.

    Returned when a request fails.
    """

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Invalid player name",
                "detail": "Player 'Unknown Player' not found in database",
                "timestamp": "2025-01-30T14:30:00",
            }
        }


# ==============================================================================
# Health Check Response Models
# ==============================================================================


class ModelStatus(BaseModel):
    """Status of a single model."""

    market: MarketType = Field(..., description="Market type")
    loaded: bool = Field(..., description="Whether model is loaded")
    feature_count: Optional[int] = Field(None, description="Number of features")
    model_version: Optional[str] = Field(None, description="Model version")


class HealthResponse(BaseModel):
    """
    Basic health check response.

    Returns API status and version information.
    """

    status: str = Field(..., description="Health status ('healthy' or 'unhealthy')")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(default_factory=datetime.now, description="Check timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "timestamp": "2025-01-30T14:30:00",
            }
        }


class ModelsHealthResponse(BaseModel):
    """
    Models health check response.

    Returns status of all loaded models.
    """

    status: str = Field(..., description="Overall models status")
    models: List[ModelStatus] = Field(..., description="Status of each model")
    total_loaded: int = Field(..., description="Number of models loaded")
    enabled_markets: List[str] = Field(..., description="Markets enabled for predictions")
    timestamp: datetime = Field(default_factory=datetime.now, description="Check timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "models": [
                    {
                        "market": "POINTS",
                        "loaded": True,
                        "feature_count": 102,
                        "model_version": "xl",
                    },
                    {
                        "market": "REBOUNDS",
                        "loaded": True,
                        "feature_count": 102,
                        "model_version": "xl",
                    },
                ],
                "total_loaded": 2,
                "enabled_markets": ["POINTS", "REBOUNDS"],
                "timestamp": "2025-01-30T14:30:00",
            }
        }


class DatabaseStatus(BaseModel):
    """Status of a single database connection."""

    name: str = Field(..., description="Database name")
    connected: bool = Field(..., description="Whether connection is active")
    host: str = Field(..., description="Database host")
    port: int = Field(..., description="Database port")
    error: Optional[str] = Field(None, description="Connection error if any")


class DatabaseHealthResponse(BaseModel):
    """
    Database health check response.

    Returns connectivity status for all databases.
    """

    status: str = Field(..., description="Overall database status")
    databases: List[DatabaseStatus] = Field(..., description="Status of each database")
    total_connected: int = Field(..., description="Number of connected databases")
    timestamp: datetime = Field(default_factory=datetime.now, description="Check timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "databases": [
                    {
                        "name": "nba_players",
                        "connected": True,
                        "host": "localhost",
                        "port": 5536,
                    },
                    {
                        "name": "nba_intelligence",
                        "connected": True,
                        "host": "localhost",
                        "port": 5539,
                    },
                ],
                "total_connected": 2,
                "timestamp": "2025-01-30T14:30:00",
            }
        }
