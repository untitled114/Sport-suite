# Model Card: NBA THREES XL Predictor

## Model Details

| Property | Value |
|----------|-------|
| **Model Name** | THREES XL Stacked Two-Head |
| **Version** | 1.0.0 |
| **Trained Date** | 2025-11-06 23:48:16 |
| **Architecture** | Stacked Two-Head (Regressor + Classifier) |
| **Framework** | LightGBM 4.x |
| **Production Status** | DISABLED |

## Intended Use

### Primary Use Case
Predict NBA player three-pointer made prop outcomes (OVER/UNDER) with probability estimates.

### Current Status
**This model is DISABLED in production** due to poor validation performance. It is retained for research purposes.

### Why Disabled?
- Validation win rate: 46.5% (below 52.4% breakeven)
- Validation ROI: -11.23% (consistent losses)
- High variance in three-point shooting makes prediction difficult

## Model Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     INPUT: 102 Features                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              HEAD 1: LightGBM Regressor                      │
│              R² = 0.178 (weak explanatory power)            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              HEAD 2: LightGBM Classifier                     │
│              AUC = 0.717 (moderate but insufficient)        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Ensemble Blending                              │
│              Scale factor: 2.0 (small range stats)          │
└─────────────────────────────────────────────────────────────┘
```

## Training Data

| Property | Value |
|----------|-------|
| **Source** | BettingPros historical props (2023-2025) |
| **Samples** | ~22,230 THREES props |
| **Date Range** | Oct 2023 - Nov 2025 |
| **Train/Test Split** | 70/30 temporal split |

## Performance Metrics

### Regressor (Value Prediction)
| Metric | Train | Test |
|--------|-------|------|
| RMSE | 1.36 | 1.43 |
| MAE | - | 1.13 |
| R² | - | **0.178** |

**Note**: R² of 0.178 is weak. Three-point shooting has high game-to-game variance.

### Classifier (OVER/UNDER Prediction)
| Metric | Train | Test |
|--------|-------|------|
| Accuracy | 73.9% | 67.8% |
| AUC (raw) | - | 0.709 |
| AUC (calibrated) | - | 0.713 |
| AUC (blended) | - | **0.717** |
| Brier Score (before) | - | 0.208 |
| Brier Score (after) | - | 0.182 |

**Note**: AUC of 0.717 looks decent, but doesn't translate to profitable betting.

### Validation Results (Oct 23 - Nov 4, 2024)
| Strategy | Bets | Win Rate | ROI |
|----------|------|----------|-----|
| Line shopping | 71 | **46.5%** | **-11.23%** |

**Warning**: Below breakeven win rate results in consistent losses.

## Why Does This Model Underperform?

### The Three-Point Paradox

The model has reasonable AUC (0.717) but poor betting performance. Why?

1. **Inherent randomness**: Three-point shooting has ~36% success rate per attempt
   - Even elite shooters have 30-50% variance game-to-game
   - This randomness is irreducible

2. **Low line values**: Props are typically 1.5-4.5 threes
   - Missing one shot can swing the result
   - Binary classification is very noisy

3. **Hot hand fallacy**: Model may overfit to recent shooting streaks
   - Shooting streaks don't persist reliably
   - Markets may already price this in

4. **Sample selection**: Sportsbooks offer props selectively
   - They avoid offering props on the most predictable situations
   - Remaining props are harder to beat

### Comparison to Points

| Aspect | POINTS | THREES |
|--------|--------|--------|
| Line range | 10-35 | 1.5-4.5 |
| Per-attempt success | ~45% FG | ~36% 3PT |
| Game variance | ±5-8 pts | ±2-3 threes |
| Model AUC | 0.765 | 0.717 |
| Validation WR | 56.7% | 46.5% |

Three-point props are fundamentally harder to predict due to lower volume and higher variance.

## Limitations

### Known Limitations
- High inherent randomness in three-point shooting
- Low line values amplify single-shot variance
- Model cannot predict hot/cold shooting nights

### When This Model Might Work (Research)
- Very high line spreads (≥1.5 threes across books)
- Players with extremely stable shot volumes
- Games with specific pace/defensive matchups

## Bias & Fairness Analysis

### Demographic Considerations
This model predicts three-pointers made. Despite being disabled for production, bias analysis is documented for research purposes.

| Bias Type | Risk Level | Description |
|-----------|------------|-------------|
| **Volume Bias** | High | High-volume shooters (8+ attempts) have more predictable patterns |
| **Role Bias** | Medium | Catch-and-shoot specialists vs off-the-dribble shooters behave differently |
| **Team Context Bias** | Medium | Players on teams with many shooters may have varied opportunities |
| **Hot Hand Bias** | High | Model may overfit to recent shooting streaks that don't persist |

### Fairness Metrics

| Player Segment | Sample Size | AUC | Accuracy | Notes |
|----------------|-------------|-----|----------|-------|
| High volume (>6 3PA/game) | ~25% | 0.74 | 69% | Best performance tier |
| Medium volume (3-6 3PA/game) | ~45% | 0.71 | 67% | Core tier |
| Low volume (<3 3PA/game) | ~30% | 0.68 | 65% | Highest variance |

### Key Finding
The model shows better AUC (0.717) but fails to translate to profitable betting. The inherent randomness of three-point shooting (~36% success rate) creates irreducible variance that affects all player segments similarly.

### Potential Research Direction
A fairness-aware improvement might stratify by shot volume, creating separate models for high-volume shooters where patterns are more stable.

## Ethical Considerations

### Warning
**DO NOT USE THIS MODEL FOR BETTING**

While this model performs better than ASSISTS, it still produces losses. The negative ROI is consistent and significant.

## Model Files

```
nba/models/saved_xl/
├── threes_xl_regressor.pkl      # LightGBM regressor
├── threes_xl_classifier.pkl     # LightGBM classifier
├── threes_xl_calibrator.pkl     # Isotonic calibration
├── threes_xl_imputer.pkl        # Feature imputer
├── threes_xl_scaler.pkl         # Feature scaler
├── threes_xl_features.pkl       # Feature name list
└── threes_xl_metadata.json      # This model's metadata
```

## Citation

```
NBA Props ML System - THREES Model
Trained: November 6, 2025
Status: DISABLED (below breakeven)
```

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-11-06 | Initial training, disabled after validation |

## Research Notes

Potential improvements to explore:

1. **Shot attempt prediction**: Predict attempts, then apply shooting percentage

2. **Game context**:
   - Blowout probability (affects shot distribution)
   - Opponent 3PT defense ranking
   - Pace impact on attempt volume

3. **Player segmentation**:
   - High-volume shooters (>8 attempts/game) may be more predictable
   - Role players have higher variance

4. **Line-specific models**:
   - Different models for low lines (1.5-2.5) vs high lines (3.5+)

5. **Alternative targets**:
   - Predict 3PA (attempts) instead of 3PM (makes)
   - More stable target variable
