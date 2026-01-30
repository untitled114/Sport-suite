# Model Card: NBA POINTS Predictor

## Model Details

| Property | Value |
|----------|-------|
| **Model Name** | POINTS Stacked Two-Head |
| **Version** | 2.0.0 |
| **Trained Date** | 2026-01-11 08:19:11 |
| **Architecture** | Stacked Two-Head (Regressor + Classifier) |
| **Framework** | LightGBM 4.x |
| **Features** | 166 |
| **Production Status** | DEPLOYED |

## Intended Use

### Primary Use Case
Predict NBA player scoring prop outcomes (OVER/UNDER) with probability estimates for sports analytics and betting research.

### Intended Users
- Sports analysts
- Quantitative researchers
- Betting professionals

### Out-of-Scope Uses
- This model should NOT be used for:
  - Financial advice or guaranteed betting returns
  - Player evaluation for team management decisions
  - Real-time in-game predictions (designed for pre-game only)

## Model Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     INPUT: 166 Features                      │
│  Player (42) + Team (28) + H2H (36) + Book (22) + More      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              HEAD 1: LightGBM Regressor                      │
│              Predicts: Absolute stat value                   │
│              Output: e.g., 25.3 points                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  expected_diff  │
                    │  = pred - line  │
                    └─────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              HEAD 2: LightGBM Classifier                     │
│              Input: 166 features + expected_diff            │
│              Output: P(actual > line)                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Isotonic Calibration                           │
│              Adjusts probabilities for reliability          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Ensemble Blending                              │
│              60% classifier + 40% residual-based            │
│              Scale factor: 5.0                              │
└─────────────────────────────────────────────────────────────┘
```

## Training Data

| Property | Value |
|----------|-------|
| **Source** | BettingPros historical props (2023-2025) |
| **Samples** | ~24,316 POINTS props |
| **Date Range** | Oct 2023 - Jan 2025 |
| **Train/Test Split** | 70/30 temporal split |
| **Home/Away Distribution** | 54.5% home / 45.5% away |

### Data Pipeline
1. Props fetched from BettingPros API
2. Enriched with home/away status from `player_game_logs` table
3. Features extracted using `LiveFeatureExtractorXL`
4. Temporal split (no shuffle) to prevent data leakage

## Performance Metrics

### Regressor (Value Prediction)
| Metric | Train | Test |
|--------|-------|------|
| RMSE | 4.78 | 6.13 |
| MAE | - | 4.55 |
| R² | - | 0.537 |

### Classifier (OVER/UNDER Prediction)
| Metric | Train | Test |
|--------|-------|------|
| Accuracy | 80.4% | 67.0% |
| AUC (raw) | - | 0.732 |
| AUC (calibrated) | - | 0.735 |
| AUC (blended) | - | 0.736 |
| Brier Score (before) | - | 0.209 |
| Brier Score (after) | - | 0.207 |

### Live Validation (Jan 1-28, 2026)
| Metric | Value |
|--------|-------|
| Total Bets | 115 |
| Win Rate | 60.0% |
| ROI | +14.5% |

## Feature Categories (166 total)

| Category | Count | Examples |
|----------|-------|----------|
| Player Rolling Stats | 42 | ema_points_L3/L5/L10/L20, minutes, FG% |
| Team & Game Context | 28 | pace, ratings, rest, B2B, travel |
| Head-to-Head History | 36 | h2h_avg_points, L3/L5 vs opponent |
| Book Disagreement | 22 | line_spread, consensus, deviations |
| Prop History | 12 | hit_rate_L20, bayesian_confidence |
| Vegas & BettingPros | 17 | vegas_total, bp_projection |
| Computed | 1 | expected_diff |

## Feature Importance

Feature importance analysis available via SHAP (SHapley Additive exPlanations).

### Generate Analysis
```bash
python -m nba.models.generate_feature_importance --market POINTS
```

### Output Files
- `nba/models/model_cards/images/POINTS_shap_summary.png` - Beeswarm plot
- `nba/models/model_cards/images/POINTS_shap_bar.png` - Bar chart

### Top Features (Regressor)
| Rank | Feature | Importance |
|------|---------|------------|
| 1 | ema_points_L5 | Player's 5-game scoring average |
| 2 | line | Prop line value |
| 3 | ema_minutes_L5 | Recent minutes played |
| 4 | team_pace | Team's pace rating |
| 5 | h2h_avg_points | Historical scoring vs opponent |

*Full SHAP analysis generated on demand. Values reflect feature contribution to model predictions.*

## Limitations

### Known Limitations
- **Player injuries**: Model does not account for in-game injuries
- **Lineup changes**: Late scratches can invalidate predictions
- **Back-to-back variance**: Higher uncertainty on B2B games
- **Blowout risk**: Garbage time can significantly affect stats
- **New players**: Limited data for rookies and recent trades

### Failure Modes
- Extreme outlier games (career highs, injuries)
- Games with significant pace changes
- Players returning from extended absence
- Changed team context (new coach, key trades)

## Bias & Fairness Analysis

### Demographic Considerations
This model predicts basketball statistics based on player performance data. While the model does not use demographic attributes as direct features, potential bias sources include:

| Bias Type | Risk Level | Description |
|-----------|------------|-------------|
| **Sample Bias** | Medium | Training data weighted toward star players |
| **Position Bias** | Low | Guards may differ from centers |
| **Team Context Bias** | Medium | High-scoring teams may inflate features |
| **Recency Bias** | Low | EMA features weight recent games |

### Mitigation Strategies
1. **Class weighting**: Balanced class weights during training
2. **Feature normalization**: All features scaled
3. **Temporal validation**: Test set temporally separated
4. **Regular monitoring**: Performance tracked across player tiers

## Model Files

```
nba/models/saved_xl/
├── points_market_regressor.pkl     # LightGBM regressor
├── points_market_classifier.pkl    # LightGBM classifier
├── points_market_calibrator.pkl    # Isotonic calibration
├── points_market_imputer.pkl       # Feature imputer
├── points_market_scaler.pkl        # Feature scaler
├── points_market_features.pkl      # Feature name list (166)
└── points_market_metadata.json     # Model metadata
```

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 2.0.0 | 2026-01-11 | Upgraded to 166 features, added H2H/prop history |
| 1.0.0 | 2025-11-06 | Initial production release (102 features) |
