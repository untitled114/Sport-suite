# Model Card: NBA REBOUNDS XL Predictor

## Model Details

| Property | Value |
|----------|-------|
| **Model Name** | REBOUNDS XL Stacked Two-Head |
| **Version** | 1.0.0 |
| **Trained Date** | 2025-11-06 23:48:15 |
| **Architecture** | Stacked Two-Head (Regressor + Classifier) |
| **Framework** | LightGBM 4.x |
| **Production Status** | DEPLOYED |

## Intended Use

### Primary Use Case
Predict NBA player rebounding prop outcomes (OVER/UNDER) with probability estimates for sports analytics and betting research.

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
│              Output: e.g., 8.7 rebounds                     │
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
│              Scale factor: 2.0 (smaller range than points)  │
└─────────────────────────────────────────────────────────────┘
```

## Training Data

| Property | Value |
|----------|-------|
| **Source** | BettingPros historical props (2023-2025) |
| **Samples** | ~24,241 REBOUNDS props |
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
| RMSE | 2.60 | 2.80 |
| MAE | - | 2.13 |
| R² | - | 0.402 |

### Classifier (OVER/UNDER Prediction)
| Metric | Train | Test |
|--------|-------|------|
| Accuracy | 95.8% | 89.0% |
| AUC (raw) | - | 0.745 |
| AUC (calibrated) | - | 0.751 |
| AUC (blended) | - | 0.748 |
| Brier Score (before) | - | 0.094 |
| Brier Score (after) | - | 0.076 |

### Validation Results (Oct 23 - Nov 4, 2024)
| Strategy | Bets | Win Rate | ROI |
|----------|------|----------|-----|
| Consensus baseline | 178 | 51.0% | -2.62% |
| Line shopping | 178 | 61.2% | +16.96% |

**Note**: REBOUNDS shows the best validation performance of all markets, likely due to more predictable rebounding patterns compared to scoring.

## Feature Importance

Top 10 features by importance:

1. `line` - The sportsbook prop line
2. `ema_rebounds_L5` - 5-game EMA of rebounds
3. `expected_diff` - Model prediction minus line
4. `ema_rebounds_L3` - 3-game EMA of rebounds
5. `ema_minutes_L5` - 5-game EMA of minutes
6. `position_encoded` - Player position (centers rebound more)
7. `consensus_line` - Median line across books
8. `opponent_def_rating` - Opponent defensive rating
9. `starter_flag` - Whether player is a starter
10. `team_pace` - Team pace factor

## Limitations

### Known Limitations
- **Foul trouble**: Players in foul trouble may get fewer minutes
- **Blowouts**: Reduced minutes in blowout games
- **Opponent size**: Model may underweight opponent's height advantages
- **Offensive rebounds**: More volatile than defensive rebounds
- **Pace variance**: Fast-paced games create more rebound opportunities

### Failure Modes
- Games with unusual pace (very slow or very fast)
- Players matched against dominant rebounders
- Games with high foul counts (more free throws = fewer rebounds)
- Double-overtime games (stat inflation)

## Bias & Fairness Analysis

### Demographic Considerations
This model predicts basketball rebounding statistics based on player performance data. While the model does not use demographic attributes (race, age, nationality) as direct features, potential bias sources include:

| Bias Type | Risk Level | Description |
|-----------|------------|-------------|
| **Position Bias** | High | Centers and power forwards naturally dominate rebounding data; guards may have lower accuracy |
| **Sample Bias** | Medium | Training data weighted toward high-usage rebounders who appear more frequently in prop markets |
| **Height Bias** | Medium | Taller players may have more predictable rebounding patterns |
| **Team Context Bias** | Low | Players on teams with fewer rebounders may have inflated opportunities |

### Fairness Metrics

We evaluate model performance across player segments:

| Player Segment | Sample Size | AUC | Accuracy | Notes |
|----------------|-------------|-----|----------|-------|
| Centers (>8 RPG) | ~25% | 0.78 | 91% | Best performance - most predictable |
| Power Forwards (5-8 RPG) | ~30% | 0.75 | 89% | Strong performance |
| Guards/Wings (<5 RPG) | ~45% | 0.71 | 86% | Lower accuracy - more volatile |

### Mitigation Strategies
1. **Position-aware features**: Model includes position encoding to account for role differences
2. **Class weighting**: Training uses balanced class weights to prevent majority class bias
3. **Feature normalization**: All features are scaled to prevent dominance by high-variance features
4. **Temporal validation**: Test set is temporally separated to ensure generalization

### Recommendations for Fair Use
- Be cautious with predictions for guards and small forwards (higher variance)
- Monitor performance metrics separately for different positions
- Consider that big men's props may have sharper lines due to predictability

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

1. **Best performing market**: REBOUNDS shows highest validation win rate (61.2%)
2. **Focus on big men**: Model performs better for centers/power forwards
3. **Check opponent**: Opponent's rebounding rank affects accuracy
4. **Monitor minutes**: Any minute restriction significantly impacts rebounds

## Model Files

```
nba/models/saved_xl/
├── rebounds_xl_regressor.pkl    # LightGBM regressor
├── rebounds_xl_classifier.pkl   # LightGBM classifier
├── rebounds_xl_calibrator.pkl   # Isotonic calibration
├── rebounds_xl_imputer.pkl      # Feature imputer
├── rebounds_xl_scaler.pkl       # Feature scaler
├── rebounds_xl_features.pkl     # Feature name list
└── rebounds_xl_metadata.json    # This model's metadata
```

## Citation

```
NBA Props ML System - REBOUNDS Model
Trained: November 6, 2025
Architecture: Stacked Two-Head LightGBM with Isotonic Calibration
```

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-11-06 | Initial production release |
