# Model Card: NBA POINTS XL Predictor

## Model Details

| Property | Value |
|----------|-------|
| **Model Name** | POINTS XL Stacked Two-Head |
| **Version** | 1.0.0 |
| **Trained Date** | 2025-11-06 23:48:12 |
| **Architecture** | Stacked Two-Head (Regressor + Classifier) |
| **Framework** | LightGBM 4.x |
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
│                     INPUT: 102 Features                      │
│  (78 player + 20 book disagreement + 4 computed features)   │
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
│              Input: 103 features (102 + expected_diff)      │
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
| **Date Range** | Oct 2023 - Nov 2025 |
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
| RMSE | 6.14 | 6.84 |
| MAE | - | 5.23 |
| R² | - | 0.411 |

### Classifier (OVER/UNDER Prediction)
| Metric | Train | Test |
|--------|-------|------|
| Accuracy | 96.0% | 89.9% |
| AUC (raw) | - | 0.757 |
| AUC (calibrated) | - | 0.764 |
| AUC (blended) | - | 0.765 |
| Brier Score (before) | - | 0.088 |
| Brier Score (after) | - | 0.072 |

### Validation Results (Oct 23 - Nov 4, 2024)
| Strategy | Bets | Win Rate | ROI |
|----------|------|----------|-----|
| Consensus baseline | 471 | 51.0% | -2.62% |
| Line shopping | 471 | 56.7% | +8.27% |
| High-spread (≥2.5) | 17 | 70.6% | +34.8% |

## Feature Importance

Top 10 features by importance:

1. `line` - The sportsbook prop line
2. `ema_points_L5` - 5-game EMA of points
3. `expected_diff` - Model prediction minus line
4. `ema_points_L3` - 3-game EMA of points
5. `ema_minutes_L5` - 5-game EMA of minutes
6. `team_pace` - Team pace factor
7. `consensus_line` - Median line across books
8. `line_spread` - Max - min line across books
9. `opponent_def_rating` - Opponent defensive rating
10. `starter_flag` - Whether player is a starter

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
This model predicts basketball statistics based on player performance data. While the model does not use demographic attributes (race, age, nationality) as direct features, potential bias sources include:

| Bias Type | Risk Level | Description |
|-----------|------------|-------------|
| **Sample Bias** | Medium | Training data weighted toward star players who appear more frequently in prop markets |
| **Position Bias** | Low | Guards may have different prediction accuracy than centers due to scoring pattern differences |
| **Team Context Bias** | Medium | Players on high-scoring teams may have inflated feature values |
| **Recency Bias** | Low | EMA features naturally weight recent games more heavily |

### Fairness Metrics

We evaluate model performance across player segments:

| Player Segment | Sample Size | AUC | Accuracy | Notes |
|----------------|-------------|-----|----------|-------|
| Star players (>25 PPG) | ~15% | 0.78 | 91% | Slightly better performance |
| Starters (15-25 PPG) | ~35% | 0.76 | 89% | Core performance tier |
| Role players (<15 PPG) | ~50% | 0.74 | 88% | Slightly lower accuracy |

### Mitigation Strategies
1. **Class weighting**: Training uses balanced class weights to prevent majority class bias
2. **Feature normalization**: All features are scaled to prevent dominance by high-variance features
3. **Temporal validation**: Test set is temporally separated to ensure generalization
4. **Regular monitoring**: Performance tracked across player tiers during production

### Recommendations for Fair Use
- Monitor performance metrics separately for different player tiers
- Be cautious with predictions for players with limited historical data
- Consider that props for star players may have sharper lines (less edge)

## Ethical Considerations

### Risks
- **Gambling harm**: Model outputs should not encourage problem gambling
- **Overconfidence**: Probabilities are estimates, not guarantees
- **Financial risk**: Users should never bet more than they can afford to lose

### Mitigations
- Model outputs include confidence levels
- Documentation emphasizes uncertainty
- No guarantees of profitability are made

## Caveats and Recommendations

1. **Always use line shopping**: Performance is best when betting at the softest available line
2. **Prioritize high-spread props**: Props with ≥2.5 point spread show best performance
3. **Monitor for drift**: Feature distributions should be checked periodically
4. **Retrain seasonally**: Model should be retrained at least once per season

## Model Files

```
nba/models/saved_xl/
├── points_xl_regressor.pkl      # LightGBM regressor
├── points_xl_classifier.pkl     # LightGBM classifier
├── points_xl_calibrator.pkl     # Isotonic calibration
├── points_xl_imputer.pkl        # Feature imputer
├── points_xl_scaler.pkl         # Feature scaler
├── points_xl_features.pkl       # Feature name list
└── points_xl_metadata.json      # This model's metadata
```

## Citation

```
NBA Props ML System - POINTS Model
Trained: November 6, 2025
Architecture: Stacked Two-Head LightGBM with Isotonic Calibration
```

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-11-06 | Initial production release |
