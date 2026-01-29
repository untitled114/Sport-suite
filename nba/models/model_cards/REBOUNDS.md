# Model Card: NBA REBOUNDS Predictor

## Model Details

| Property | Value |
|----------|-------|
| **Model Name** | REBOUNDS Stacked Two-Head |
| **Version** | 2.0.0 |
| **Trained Date** | 2026-01-11 08:19:57 |
| **Architecture** | Stacked Two-Head (Regressor + Classifier) |
| **Framework** | LightGBM 4.x |
| **Features** | 166 |
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
│                     INPUT: 166 Features                      │
│  Player (42) + Team (28) + H2H (36) + Book (22) + More      │
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
│              Scale factor: 2.0                              │
└─────────────────────────────────────────────────────────────┘
```

## Training Data

| Property | Value |
|----------|-------|
| **Source** | BettingPros historical props (2023-2025) |
| **Samples** | ~24,241 REBOUNDS props |
| **Date Range** | Oct 2023 - Jan 2025 |
| **Train/Test Split** | 70/30 temporal split |
| **Home/Away Distribution** | 54.5% home / 45.5% away |

## Performance Metrics

### Regressor (Value Prediction)
| Metric | Train | Test |
|--------|-------|------|
| RMSE | 1.80 | 2.42 |
| MAE | - | 1.79 |
| R² | - | 0.531 |

### Classifier (OVER/UNDER Prediction)
| Metric | Train | Test |
|--------|-------|------|
| Accuracy | 82.9% | 67.7% |
| AUC (raw) | - | 0.747 |
| AUC (calibrated) | - | 0.749 |
| AUC (blended) | - | 0.748 |
| Brier Score (before) | - | 0.204 |
| Brier Score (after) | - | 0.201 |

### Live Validation (Jan 1-28, 2026)
| Metric | Value |
|--------|-------|
| Total Bets | 89 |
| Win Rate | 59.6% |
| ROI | +13.7% |

## Feature Categories (166 total)

| Category | Count | Examples |
|----------|-------|----------|
| Player Rolling Stats | 42 | ema_rebounds_L3/L5/L10/L20, minutes |
| Team & Game Context | 28 | pace, ratings, rest, B2B, travel |
| Head-to-Head History | 36 | h2h_avg_rebounds, L3/L5 vs opponent |
| Book Disagreement | 22 | line_spread, consensus, deviations |
| Prop History | 12 | hit_rate_L20, bayesian_confidence |
| Vegas & BettingPros | 17 | vegas_total, bp_projection |
| Computed | 1 | expected_diff |

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

### Position Considerations
| Player Segment | AUC | Notes |
|----------------|-----|-------|
| Centers (>8 RPG) | 0.78 | Most predictable |
| Power Forwards (5-8 RPG) | 0.75 | Strong performance |
| Guards/Wings (<5 RPG) | 0.71 | Higher variance |

## Model Files

```
nba/models/saved_xl/
├── rebounds_market_regressor.pkl     # LightGBM regressor
├── rebounds_market_classifier.pkl    # LightGBM classifier
├── rebounds_market_calibrator.pkl    # Isotonic calibration
├── rebounds_market_imputer.pkl       # Feature imputer
├── rebounds_market_scaler.pkl        # Feature scaler
├── rebounds_market_features.pkl      # Feature name list (166)
└── rebounds_market_metadata.json     # Model metadata
```

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 2.0.0 | 2026-01-11 | Upgraded to 166 features, added H2H/prop history |
| 1.0.0 | 2025-11-06 | Initial production release (102 features) |
