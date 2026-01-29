"""
Pydantic Models for Data Validation
====================================
Type-safe data structures for the NBA props prediction system.

Usage:
    from nba.core.schemas import PropLine, Prediction

    # Validate prop data
    prop = PropLine(
        player_name="LeBron James",
        stat_type="POINTS",
        line=25.5,
        book_name="draftkings",
        game_date="2025-11-06"
    )

    # Validate prediction output
    pred = Prediction(
        player_name="LeBron James",
        stat_type="POINTS",
        prediction=28.5,
        p_over=0.72,
        side="OVER",
        edge=3.0
    )
"""

import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class MarketType(str, Enum):
    """Supported prop markets."""

    POINTS = "POINTS"
    REBOUNDS = "REBOUNDS"
    ASSISTS = "ASSISTS"
    THREES = "THREES"


class Side(str, Enum):
    """Bet side."""

    OVER = "OVER"
    UNDER = "UNDER"


class Confidence(str, Enum):
    """Prediction confidence level."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    STANDARD = "STANDARD"
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


class PropLine(BaseModel):
    """
    A single prop line from a sportsbook.

    Represents one book's offering for a player prop.
    """

    player_name: str = Field(..., min_length=1, description="Player's full name")
    stat_type: MarketType = Field(..., description="Prop market type")
    line: float = Field(..., ge=0, le=200, description="Prop line value")
    book_name: BookName = Field(..., description="Sportsbook name")
    game_date: datetime.date = Field(..., description="Game date")
    opponent_team: Optional[str] = Field(None, max_length=3, description="Opponent team code")
    is_home: Optional[bool] = Field(None, description="Is player's team at home")
    over_odds: Optional[int] = Field(None, description="Over odds (American format)")
    under_odds: Optional[int] = Field(None, description="Under odds (American format)")

    @field_validator("player_name")
    @classmethod
    def normalize_player_name(cls, v: str) -> str:
        """Strip whitespace and normalize."""
        return " ".join(v.split())

    @field_validator("game_date", mode="before")
    @classmethod
    def parse_game_date(cls, v):
        """Accept string or date."""
        if isinstance(v, str):
            return datetime.datetime.strptime(v, "%Y-%m-%d").date()
        return v

    @field_validator("book_name", mode="before")
    @classmethod
    def normalize_book_name(cls, v) -> BookName:
        """Accept string or BookName enum, normalize to BookName."""
        if isinstance(v, BookName):
            return v
        # Normalize string to lowercase and match to enum
        normalized = v.lower().replace(" ", "").replace("_", "")
        for book in BookName:
            if book.value == normalized or book.name.lower() == normalized:
                return book
        # Try partial match for common variations
        book_map = {
            "dk": BookName.DRAFTKINGS,
            "fd": BookName.FANDUEL,
            "mgm": BookName.BETMGM,
            "espn": BookName.ESPNBET,
            "rivers": BookName.BETRIVERS,
            "ud": BookName.UNDERDOG,
            "pp": BookName.PRIZEPICKS,
        }
        if normalized in book_map:
            return book_map[normalized]
        raise ValueError(f"Unknown book name: {v}. Valid books: {[b.value for b in BookName]}")


class PropLineCollection(BaseModel):
    """
    Collection of prop lines for a single player/stat/game.

    Represents all available lines across books for line shopping.
    """

    player_name: str
    stat_type: MarketType
    game_date: datetime.date
    lines: List[PropLine] = Field(..., min_length=1)

    @property
    def softest_line(self) -> PropLine:
        """Get the softest (lowest) line for OVER bets."""
        return min(self.lines, key=lambda x: x.line)

    @property
    def hardest_line(self) -> PropLine:
        """Get the hardest (highest) line for UNDER bets."""
        return max(self.lines, key=lambda x: x.line)

    @property
    def consensus_line(self) -> float:
        """Get median line across all books."""
        sorted_lines = sorted(p.line for p in self.lines)
        n = len(sorted_lines)
        if n % 2 == 0:
            return (sorted_lines[n // 2 - 1] + sorted_lines[n // 2]) / 2
        return sorted_lines[n // 2]

    @property
    def line_spread(self) -> float:
        """Get spread between highest and lowest lines."""
        lines = [p.line for p in self.lines]
        return max(lines) - min(lines)


class Prediction(BaseModel):
    """
    A single model prediction.

    Represents the model's output for one prop.
    """

    player_name: str = Field(..., description="Player's full name")
    stat_type: MarketType = Field(..., description="Prop market type")
    prediction: float = Field(..., ge=0, description="Predicted stat value")
    p_over: float = Field(..., ge=0, le=1, description="Probability of hitting OVER")
    side: Side = Field(..., description="Recommended bet side")
    edge: float = Field(..., ge=0, description="Edge over the line")
    best_book: str = Field(..., description="Book with best line for this side")
    best_line: float = Field(..., ge=0, description="Best available line")
    confidence: Confidence = Field(Confidence.STANDARD, description="Prediction confidence")
    game_date: Optional[datetime.date] = Field(None, description="Game date")
    opponent_team: Optional[str] = Field(None, description="Opponent team code")

    @model_validator(mode="after")
    def validate_side_consistency(self):
        """Ensure side matches probability."""
        if self.side == Side.OVER and self.p_over < 0.5:
            raise ValueError("Side is OVER but p_over < 0.5")
        if self.side == Side.UNDER and self.p_over > 0.5:
            raise ValueError("Side is UNDER but p_over > 0.5")
        return self


class PredictionBatch(BaseModel):
    """
    A batch of predictions (daily output).

    Represents all predictions for a given day.
    """

    generated_at: datetime.datetime = Field(default_factory=datetime.datetime.now)
    date: datetime.date = Field(..., description="Prediction date")
    strategy: str = Field("XL Line Shopping", description="Strategy used")
    markets_enabled: List[MarketType] = Field(..., description="Markets included")
    total_picks: int = Field(..., ge=0, description="Total number of picks")
    picks: List[Prediction] = Field(default_factory=list)
    summary: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_pick_count(self):
        """Ensure total_picks matches picks list."""
        if self.total_picks != len(self.picks):
            self.total_picks = len(self.picks)
        return self


class FeatureVector(BaseModel):
    """
    Feature vector for model input.

    Contains the 102 features used by the XL model. The schema explicitly defines
    the most common features for validation, but uses `extra = "allow"` to support
    additional dynamic features from the feature extractor.

    Feature Categories (102 total):
    - Core (2): is_home, line
    - Player rolling stats (36): EMA stats for points/rebounds/assists/threes/
      steals/blocks/turnovers/minutes/fg_pct across L3/L5/L10/L20 windows
    - Team context (8): pace, offensive/defensive ratings, projected possessions
    - Advanced context (6): rest days, back-to-back, travel distance, altitude, season phase
    - Usage (5): starter flag, bench points ratio, position encoded, avg teammate usage, injured teammates
    - Matchup (6): head-to-head stats, matchup advantage score
    - Recent performance (3): points per minute L5, player last game minutes, days since last 30pt game
    - Book disagreement (20): line spread, consensus, deviations, etc.

    Design Decision:
    The `extra = "allow"` setting means additional features beyond those explicitly
    defined here are accepted. This allows the model to evolve without breaking
    validation, while still enforcing constraints on the most critical features.
    """

    # Core identifiers (not used in model, for tracking)
    player_name: Optional[str] = None
    game_date: Optional[datetime.date] = None
    stat_type: Optional[MarketType] = None

    # Home/away and line (2 features)
    is_home: float = Field(..., ge=0, le=1, description="1 if home, 0 if away")
    line: float = Field(..., ge=0, le=200, description="Prop line")

    # Player rolling stats - Points (4 features)
    ema_points_L3: float = Field(default=0.0, ge=0)
    ema_points_L5: float = Field(default=0.0, ge=0)
    ema_points_L10: float = Field(default=0.0, ge=0)
    ema_points_L20: float = Field(default=0.0, ge=0)

    # Player rolling stats - Rebounds (4 features)
    ema_rebounds_L3: float = Field(default=0.0, ge=0)
    ema_rebounds_L5: float = Field(default=0.0, ge=0)
    ema_rebounds_L10: float = Field(default=0.0, ge=0)
    ema_rebounds_L20: float = Field(default=0.0, ge=0)

    # Player rolling stats - Assists (4 features)
    ema_assists_L3: float = Field(default=0.0, ge=0)
    ema_assists_L5: float = Field(default=0.0, ge=0)
    ema_assists_L10: float = Field(default=0.0, ge=0)
    ema_assists_L20: float = Field(default=0.0, ge=0)

    # Player rolling stats - Threes (4 features)
    ema_threes_L3: float = Field(default=0.0, ge=0)
    ema_threes_L5: float = Field(default=0.0, ge=0)
    ema_threes_L10: float = Field(default=0.0, ge=0)
    ema_threes_L20: float = Field(default=0.0, ge=0)

    # Player rolling stats - Steals (4 features)
    ema_steals_L3: float = Field(default=0.0, ge=0)
    ema_steals_L5: float = Field(default=0.0, ge=0)
    ema_steals_L10: float = Field(default=0.0, ge=0)
    ema_steals_L20: float = Field(default=0.0, ge=0)

    # Player rolling stats - Blocks (4 features)
    ema_blocks_L3: float = Field(default=0.0, ge=0)
    ema_blocks_L5: float = Field(default=0.0, ge=0)
    ema_blocks_L10: float = Field(default=0.0, ge=0)
    ema_blocks_L20: float = Field(default=0.0, ge=0)

    # Player rolling stats - Turnovers (4 features)
    ema_turnovers_L3: float = Field(default=0.0, ge=0)
    ema_turnovers_L5: float = Field(default=0.0, ge=0)
    ema_turnovers_L10: float = Field(default=0.0, ge=0)
    ema_turnovers_L20: float = Field(default=0.0, ge=0)

    # Player rolling stats - Minutes (4 features)
    ema_minutes_L3: float = Field(default=0.0, ge=0)
    ema_minutes_L5: float = Field(default=0.0, ge=0)
    ema_minutes_L10: float = Field(default=0.0, ge=0)
    ema_minutes_L20: float = Field(default=0.0, ge=0)

    # Player rolling stats - FG% (4 features)
    ema_fg_pct_L3: float = Field(default=0.0, ge=0, le=1)
    ema_fg_pct_L5: float = Field(default=0.0, ge=0, le=1)
    ema_fg_pct_L10: float = Field(default=0.0, ge=0, le=1)
    ema_fg_pct_L20: float = Field(default=0.0, ge=0, le=1)

    # Team context (8 features)
    team_pace: float = Field(default=100.0, ge=80, le=120)
    opp_defensive_rating: float = Field(default=110.0, ge=90, le=130)
    team_offensive_rating: float = Field(default=110.0, ge=90, le=130)
    projected_possessions: float = Field(default=100.0, ge=80, le=120)
    opp_pace: float = Field(default=100.0, ge=80, le=120)
    team_vs_position_def_rank: float = Field(default=15.0, ge=1, le=30)
    opp_vs_position_pts_allowed: float = Field(default=25.0, ge=0)
    game_total: float = Field(default=220.0, ge=180, le=280)

    # Advanced context (6 features)
    rest_days: float = Field(default=1.0, ge=0)
    is_back_to_back: float = Field(default=0.0, ge=0, le=1)
    travel_distance: float = Field(default=0.0, ge=0)
    altitude: float = Field(default=0.0, ge=0)
    season_phase: float = Field(default=0.5, ge=0, le=1)  # 0=early, 1=late
    days_into_season: float = Field(default=50.0, ge=0)

    # Usage features (5 features)
    is_starter: float = Field(default=1.0, ge=0, le=1)
    bench_points_ratio: float = Field(default=0.0, ge=0, le=1)
    position_encoded: float = Field(default=2.0, ge=0)  # 1=PG, 2=SG, 3=SF, 4=PF, 5=C
    avg_teammate_usage: float = Field(default=0.2, ge=0, le=1)
    injured_teammates_count: float = Field(default=0.0, ge=0)

    # Matchup features (6 features)
    h2h_avg_stat: float = Field(default=0.0, ge=0)
    h2h_games_played: float = Field(default=0.0, ge=0)
    matchup_advantage_score: float = Field(default=0.0)
    opp_def_vs_pos_rank: float = Field(default=15.0, ge=1, le=30)
    opp_recent_def_rating: float = Field(default=110.0, ge=90, le=130)
    last_meeting_stat: float = Field(default=0.0, ge=0)

    # Recent performance (3 features)
    pts_per_min_L5: float = Field(default=0.5, ge=0)
    last_game_minutes: float = Field(default=30.0, ge=0)
    days_since_last_30pt_game: float = Field(default=10.0, ge=0)

    # Book disagreement features (20 features)
    line_spread: float = Field(default=0.0, ge=0)
    consensus_line: float = Field(default=0.0, ge=0)
    line_std_dev: float = Field(default=0.0, ge=0)
    num_books_offering: float = Field(default=1.0, ge=1)
    line_coef_variation: float = Field(default=0.0, ge=0)
    draftkings_deviation: float = Field(default=0.0)
    fanduel_deviation: float = Field(default=0.0)
    betmgm_deviation: float = Field(default=0.0)
    caesars_deviation: float = Field(default=0.0)
    bet365_deviation: float = Field(default=0.0)
    betrivers_deviation: float = Field(default=0.0)
    espnbet_deviation: float = Field(default=0.0)
    fanatics_deviation: float = Field(default=0.0)
    softest_book_id: float = Field(default=0.0, ge=0)
    hardest_book_id: float = Field(default=0.0, ge=0)
    line_spread_percentile: float = Field(default=0.5, ge=0, le=1)
    books_agree: float = Field(default=1.0, ge=0, le=1)
    books_disagree: float = Field(default=0.0, ge=0, le=1)
    softest_vs_consensus: float = Field(default=0.0)
    hardest_vs_consensus: float = Field(default=0.0)

    class Config:
        """Pydantic config."""

        extra = "allow"  # Allow additional features beyond those explicitly defined

    def to_array(self, feature_names: List[str]) -> List[float]:
        """Convert to array in specified feature order."""
        data = self.model_dump()
        return [data.get(name, 0.0) for name in feature_names]

    @classmethod
    def explicit_feature_count(cls) -> int:
        """Return count of explicitly defined features (excluding identifiers)."""
        excluded = {"player_name", "game_date", "stat_type"}
        return len([f for f in cls.model_fields.keys() if f not in excluded])


class ValidationResult(BaseModel):
    """
    Result of a validated prediction.

    Tracks actual outcomes for backtesting.
    """

    prediction: Prediction
    actual_value: float = Field(..., ge=0, description="Actual stat value")
    result: str = Field(..., pattern="^(WIN|LOSS|PUSH)$", description="Bet result")
    profit: float = Field(..., description="Profit/loss in units")
    validated_at: datetime.datetime = Field(default_factory=datetime.datetime.now)

    @property
    def hit_over(self) -> bool:
        """Did the player hit the over?"""
        return self.actual_value > self.prediction.best_line

    @model_validator(mode="after")
    def compute_result(self):
        """Auto-compute result if not provided."""
        if self.actual_value > self.prediction.best_line:
            expected = "WIN" if self.prediction.side == Side.OVER else "LOSS"
        elif self.actual_value < self.prediction.best_line:
            expected = "WIN" if self.prediction.side == Side.UNDER else "LOSS"
        else:
            expected = "PUSH"

        if self.result != expected:
            raise ValueError(f"Result mismatch: expected {expected}, got {self.result}")
        return self


class TrainingMetrics(BaseModel):
    """
    Model training metrics.

    Captures all metrics from a training run.
    """

    market: MarketType
    trained_at: datetime.datetime = Field(default_factory=datetime.datetime.now)
    samples_train: int = Field(..., ge=0)
    samples_test: int = Field(..., ge=0)

    # Regressor metrics
    rmse_train: float = Field(..., ge=0)
    rmse_test: float = Field(..., ge=0)
    mae_test: float = Field(..., ge=0)
    r2_test: float = Field(..., ge=-1, le=1)

    # Classifier metrics
    accuracy_train: float = Field(..., ge=0, le=1)
    accuracy_test: float = Field(..., ge=0, le=1)
    auc_test: float = Field(..., ge=0, le=1)
    auc_calibrated: float = Field(..., ge=0, le=1)
    auc_blended: float = Field(..., ge=0, le=1)
    brier_before: float = Field(..., ge=0, le=1)
    brier_after: float = Field(..., ge=0, le=1)

    # Feature info
    feature_count: int = Field(..., ge=1)
    top_features: List[str] = Field(default_factory=list)
