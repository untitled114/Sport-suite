# Model Card: NBA POINTS Predictor

## Model Details

| Property | Value |
|----------|-------|
| **Model Name** | POINTS V5 Stacked Two-Head |
| **Version** | 5.0.0 |
| **Trained Date** | 2026-03-24 |
| **Architecture** | Stacked Two-Head (Regressor + Classifier) |
| **Framework** | LightGBM 4.x |
| **Features** | 134 |
| **Production Status** | DEPLOYED |

## Model Architecture

```
                        INPUT: 134 Features
  Player EMAs (32) + Team (6) + H2H (12) + Book (18) + BP (8) + More

                              |
                              v
              HEAD 1: LightGBM Regressor
              lr=0.05, ~76 trees (early-stopped)
              Output: predicted stat value (e.g., 25.3 points)
                              |
                              v
                   expected_diff (OOF)
                   = prediction - line
                   (5-fold CV + noise injection)
                              |
                              v
              HEAD 2: LightGBM Classifier
              lr=0.05, ~105 trees, ALL features + expected_diff
              Output: P(actual > line)
                              |
                              v
              Platt Scaling Calibration
              LogisticRegression on 15% holdout
              (skipped if raw probs already calibrated)
```

## Training Data

| Property | Value |
|----------|-------|
| **Source** | BettingPros historical props |
| **Samples** | 26,306 |
| **Date Range** | Oct 2024 - Mar 2026 |
| **Train/Test Split** | 70/30 temporal split (no shuffle) |
| **Builder** | `build_xl_training_dataset_batched.py` |

## Performance Metrics

### Single-Split Training
| Metric | Train | Test |
|--------|-------|------|
| Regressor RMSE | 5.44 | 5.53 |
| Regressor R2 | - | 0.586 |
| Classifier AUC | - | **0.731** |
| Classifier Accuracy | 71.9% | 67.1% |
| Brier Score | - | 0.210 |
| Log Loss | - | 0.608 |

### Walk-Forward Validation (3 folds, 2-month test windows)

| Fold | Train Period | Test Period | AUC | Win Rate | Edge WR | ROI |
|------|-------------|-------------|-----|----------|---------|-----|
| 1 | 2024-10-22 to 2025-04-13 | 2025-10-21 to 2025-10-21 | 0.753 | 60.0% | 70.6% | — |
| 2 | 2024-10-22 to 2025-10-21 | 2025-10-22 to 2025-12-21 | 0.712 | 65.1% | 67.7% | — |
| 3 | 2024-10-22 to 2025-12-21 | 2025-12-22 to 2026-02-21 | 0.724 | 62.6% | 67.8% | — |

**Mean AUC: 0.730 (+/- 0.017) | Win Rate: 62.6% | Edge WR: 68.7% | ROI: +25.3%**

![AUC by Fold](images/POINTS_walkforward_auc.png)
![Win Rate](images/POINTS_walkforward_winrate.png)
![Cumulative ROI](images/POINTS_walkforward_roi.png)

## Top Features

### Regressor (Top 10)
| Rank | Feature | Splits |
|------|---------|--------|
| 1 | h2h_std_points | 197 |
| 2 | h2h_away_avg_points | 190 |
| 3 | h2h_home_avg_points | 185 |
| 4 | h2h_L3_points | 183 |
| 5 | h2h_L5_points | 164 |
| 6 | line | 152 |
| 7 | prop_hit_rate_context | 132 |
| 8 | opp_positional_def | 103 |
| 9 | bp_projection_diff | 80 |
| 10 | bp_hit_rate_season | 75 |

### Classifier (Top 10)
| Rank | Feature | Splits |
|------|---------|--------|
| 1 | expected_diff | 134 |
| 2 | prop_hit_rate_context | 107 |
| 3 | h2h_std_points | 95 |
| 4 | h2h_L5_points | 95 |
| 5 | opp_positional_def | 80 |
| 6 | bp_hit_rate_season | 78 |
| 7 | h2h_away_avg_points | 77 |
| 8 | h2h_trend_points | 74 |
| 9 | prop_days_since_last_hit | 63 |
| 10 | h2h_home_avg_points | 60 |

## Feature Categories (134 total)

| Category | Count | Examples |
|----------|-------|----------|
| Player Rolling Stats (all stats) | 38 | ema_points/rebounds/assists/threes/steals/blocks/turnovers/minutes L3-L20 |
| Shooting & Advanced | 8 | fg_pct L3-L20, ft_rate_L10, true_shooting_L10, plus_minus |
| Team & Game Context | 10 | pace, off/def rating, travel_distance_km, altitude |
| Head-to-Head (primary stat) | 12 | h2h_avg/std/L3/L5/L10/L20, home/away splits |
| Book Disagreement | 18 | line_spread, consensus, deviations per book |
| Prop History | 9 | hit_rate_L20/context, bayesian_confidence |
| BettingPros | 8 | bp_projection_diff, bp_probability, bp_hit_rate |
| Vegas | 2 | vegas_total, vegas_spread |
| Situational | 6 | days_rest, starter_flag, bench_points_ratio |
| Computed | 1 | expected_diff (OOF, noise-injected) |

## Model Files

```
nba/models/saved_xl/
  points_v5_regressor.pkl
  points_v5_classifier.pkl
  points_v5_calibrator.pkl
  points_v5_imputer.pkl
  points_v5_scaler.pkl
  points_v5_features.pkl
  points_v5_metadata.json
```

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 5.0.0 | 2026-03-24 | OOF expected_diff, Platt calibration, cross-stat EMAs, 134 features, AUC 0.731 |
| 2.0.0 | 2026-01-11 | 166 features, H2H/prop history (retired) |
| 1.0.0 | 2025-11-06 | Initial XL release, 102 features (retired) |
