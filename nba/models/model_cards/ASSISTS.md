# Model Card: NBA ASSISTS XL Predictor

## Model Details

| Property | Value |
|----------|-------|
| **Model Name** | ASSISTS XL Stacked Two-Head |
| **Version** | 1.0.0 |
| **Trained Date** | 2025-11-06 23:48:15 |
| **Architecture** | Stacked Two-Head (Regressor + Classifier) |
| **Framework** | LightGBM 4.x |
| **Production Status** | DISABLED |

## Intended Use

### Primary Use Case
Predict NBA player assist prop outcomes (OVER/UNDER) with probability estimates.

### Current Status
**This model is DISABLED in production** due to poor validation performance. It is retained for research purposes and potential future improvement.

### Why Disabled?
- Validation win rate: 14.6% (significantly below breakeven)
- Validation ROI: -72.05% (severe underperformance)
- AUC: 0.588 (near random chance)

## Model Architecture

Same as other markets (stacked two-head with blending), but with worse performance characteristics.

```
┌─────────────────────────────────────────────────────────────┐
│                     INPUT: 102 Features                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              HEAD 1: LightGBM Regressor                      │
│              R² = 0.062 (very weak explanatory power)       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              HEAD 2: LightGBM Classifier                     │
│              AUC = 0.588 (barely above random)              │
└─────────────────────────────────────────────────────────────┘
```

## Training Data

| Property | Value |
|----------|-------|
| **Source** | BettingPros historical props (2023-2025) |
| **Samples** | ~24,352 ASSISTS props |
| **Date Range** | Oct 2023 - Nov 2025 |
| **Train/Test Split** | 70/30 temporal split |

## Performance Metrics

### Regressor (Value Prediction)
| Metric | Train | Test |
|--------|-------|------|
| RMSE | 0.95 | 0.99 |
| MAE | - | 0.78 |
| R² | - | **0.062** |

**Warning**: R² of 0.062 indicates the model explains only 6.2% of variance. The regressor is essentially not learning meaningful patterns.

### Classifier (OVER/UNDER Prediction)
| Metric | Train | Test |
|--------|-------|------|
| Accuracy | 63.6% | 54.4% |
| AUC (raw) | - | 0.572 |
| AUC (calibrated) | - | 0.578 |
| AUC (blended) | - | **0.588** |
| Brier Score (before) | - | 0.246 |
| Brier Score (after) | - | 0.243 |

**Warning**: AUC of 0.588 is barely above random chance (0.5). The classifier is not reliable.

### Validation Results (Oct 23 - Nov 4, 2024)
| Strategy | Bets | Win Rate | ROI |
|----------|------|----------|-----|
| Line shopping | 41 | **14.6%** | **-72.05%** |

**Critical Failure**: This model loses money at an alarming rate. Do NOT use for betting.

## Why Does This Model Fail?

### Hypotheses

1. **High variance stat**: Assists are highly dependent on teammate shooting
   - A player can make 10 perfect passes but get 0 assists if teammates miss

2. **Game flow dependency**: Assists correlate with close games
   - Blowouts reduce assist opportunities
   - Model can't predict game flow

3. **Lineup dependencies**: Assists depend heavily on who else is playing
   - Injured scorers = fewer assist opportunities
   - Model may not capture these dynamics well

4. **Small sample effects**: Assist props often have lower lines (3-8)
   - Binary outcomes more sensitive to variance
   - One or two possessions can swing the result

### Potential Improvements (Research Only)

1. Add teammate shooting percentage features
2. Add game flow prediction (blowout probability)
3. Include lineup-specific features
4. Use smaller rolling windows for volatility

## Limitations

### Known Limitations
- Model is fundamentally unreliable for this market
- Should NOT be used for any betting decisions
- Retained only for research and comparison purposes

### When This Model Might Improve
- If teammate context features are added
- If game flow prediction is incorporated
- If lineup-specific training data is available

## Bias & Fairness Analysis

### Demographic Considerations
This model predicts basketball assist statistics. Due to the model's poor overall performance, bias analysis is limited in practical value, but documented for completeness.

| Bias Type | Risk Level | Description |
|-----------|------------|-------------|
| **Position Bias** | High | Point guards dominate assist data; model may perform differently across positions |
| **Role Bias** | High | Primary ball handlers have more predictable patterns than secondary players |
| **Teammate Dependency** | Very High | Assists depend heavily on teammate shooting, creating confounding bias |
| **Usage Rate Bias** | Medium | High-usage players may have more stable patterns |

### Fairness Metrics (Limited Reliability)

Due to poor model performance (AUC 0.588), these metrics should be interpreted cautiously:

| Player Segment | Sample Size | AUC | Accuracy | Notes |
|----------------|-------------|-----|----------|-------|
| Point Guards (>7 APG) | ~20% | 0.61 | 56% | Still near random |
| Secondary handlers (4-7 APG) | ~35% | 0.58 | 54% | Poor across the board |
| Role players (<4 APG) | ~45% | 0.55 | 52% | Essentially random |

### Key Finding
The model's fundamental failure appears to be stat-specific (assist variance), not bias-related. Fairness analysis will be more meaningful if the underlying prediction quality improves.

## Ethical Considerations

### Critical Warning
**DO NOT USE THIS MODEL FOR BETTING**

Using this model for actual betting decisions would result in significant financial losses. It is retained in the codebase only for:
- Research purposes
- A/B testing of future improvements
- Understanding model failure modes

## Model Files

```
nba/models/saved_xl/
├── assists_xl_regressor.pkl     # LightGBM regressor (WEAK)
├── assists_xl_classifier.pkl    # LightGBM classifier (WEAK)
├── assists_xl_calibrator.pkl    # Isotonic calibration
├── assists_xl_imputer.pkl       # Feature imputer
├── assists_xl_scaler.pkl        # Feature scaler
├── assists_xl_features.pkl      # Feature name list
└── assists_xl_metadata.json     # This model's metadata
```

## Citation

```
NBA Props ML System - ASSISTS Model
Trained: November 6, 2025
Status: DISABLED (validation failure)
```

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-11-06 | Initial training, immediately disabled |

## Research Notes

If attempting to improve this model, consider:

1. **Increase feature engineering for assists**:
   - Teammate 3PT% (assist opportunities)
   - Ball handler vs off-ball classification
   - Play-by-play assist patterns

2. **Different model architecture**:
   - May need point guard-specific model
   - Consider position-stratified training

3. **Alternative target**:
   - Predict assist rate (assists per minute) instead
   - Then scale by predicted minutes
