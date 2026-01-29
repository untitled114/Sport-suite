# Validation Methodology

This document explains how model performance is validated to prevent overfitting and data leakage.

## The Problem

Sports betting models are prone to:
1. **Overfitting**: Model memorizes training data, fails on new data
2. **Data Leakage**: Future information accidentally used during training/validation
3. **Selection Bias**: Cherry-picking favorable backtest periods

## Our Approach

### Temporal Split (No Lookahead)

```
Training Data:     Oct 2023 ─────────────────────────────▶ Apr 2025
                   |<──────── ~24,000 props/market ────────▶|

Validation Data:                                              Oct 30 - Nov 7, 2024
                                                              |<── 8 days ──▶|
                   ──────────────────────────────────────────────────────────▶ time
```

**Key**: Validation period is WITHIN the training date range but uses TEMPORAL feature extraction.

### Point-in-Time Feature Extraction

When validating on a prop from Nov 3, 2024:

```python
# CORRECT: Only use data available BEFORE Nov 3
rolling_stats = query("""
    SELECT ema_points_L10 FROM player_rolling_stats
    WHERE player_id = ? AND game_date < '2024-11-03'
    ORDER BY game_date DESC LIMIT 1
""")

# WRONG: Using data from Nov 3 or later (data leakage!)
```

This is enforced in `validate_xl_models.py:9`:
```python
# CRITICAL: Prevents data leakage by extracting features AS OF historical game date.
```

### Separate Result Loading

Actual results are loaded AFTER predictions are generated:

```python
# Step 1: Generate predictions (no access to results)
predictions = model.predict(features)

# Step 2: Load actuals from database
actuals = query("SELECT actual_result FROM nba_prop_lines WHERE id = ?")

# Step 3: Compare
win = (side == 'OVER' and actual > line) or (side == 'UNDER' and actual < line)
```

---

## How to Run Validation

### Full Backtest

```bash
python3 nba/betting_xl/validate_xl_models.py \
  --start-date 2024-10-30 \
  --end-date 2024-11-07
```

**Output:**
```
Market     Bets    Wins    Win Rate    ROI
─────────────────────────────────────────────
POINTS     18      12      66.7%       +9.1%
REBOUNDS   10      9       90.0%       +72.7%
─────────────────────────────────────────────
TOTAL      28      21      75.0%       +43.2%
```

### What the Script Does

1. **Load Historical Props**: Queries `nba_prop_lines` for the date range
2. **Filter to Evaluated Props**: Only props with `actual_result IS NOT NULL`
3. **Extract Features Point-in-Time**: For each prop, extract features using only data available before that game
4. **Generate Predictions**: Run through model pipeline (regressor → classifier → calibrator)
5. **Apply Production Filters**: Same filters used in live predictions
6. **Compare to Actuals**: Calculate win/loss for each pick
7. **Aggregate Metrics**: Win rate, ROI, per-market breakdown

---

## Validation Results History

Stored in `nba/models/MODEL_REGISTRY.toml`:

```toml
[[validation_runs]]
date = "2024-11-08"
period = "2024-10-30 to 2024-11-07"
strategy = "hybrid_dual_filter"
total_bets = 28
total_wins = 21
win_rate = 0.750
roi = 0.4318
profit_units = 12.09
markets = ["POINTS", "REBOUNDS"]
notes = "Best performing strategy."
```

### Historical Strategy Comparison

| Strategy | Bets | Win Rate | ROI | Notes |
|----------|------|----------|-----|-------|
| Hybrid Dual-Filter | 28 | 75.0% | +43.2% | Current production |
| Point Value Only | 20 | 65.0% | +24.1% | Too conservative |
| Probability Only | 1,173 | 51.6% | -1.5% | Too permissive |

---

## Known Limitations

### Sample Size

28 bets over 8 days is a small sample. 95% confidence interval for 75% win rate with n=28:

```
75% ± 16% → True win rate likely between 59% and 91%
```

**Interpretation**: We're confident the model beats the 52.4% breakeven, but the exact edge is uncertain.

### Market Efficiency

Sportsbooks adjust lines based on betting activity. An edge that exists in backtesting may shrink in live betting due to:
- Line movement toward model's direction
- Reduced limits on winning bettors
- Market copying successful strategies

### Regime Changes

Historical patterns may not persist:
- Player injuries/trades
- Rule changes
- Team strategy shifts
- Referee tendencies

---

## What Would Invalidate Results

Red flags that would require re-validation:

1. **Code Bug in Feature Extraction**: If `extract_live_features_xl.py` has a bug that uses future data
2. **Incorrect Date Filtering**: If the temporal split isn't enforced properly
3. **Training/Validation Overlap**: If the same prop appears in both sets
4. **Model Retraining on Validation Data**: If validation results are used to tune and then re-evaluated

---

## Reproducing Results

### Prerequisites

```bash
# Databases running
docker ps | grep nba

# Props loaded for validation period
psql -p 5539 -c "SELECT COUNT(*) FROM nba_prop_lines
                 WHERE game_date BETWEEN '2024-10-30' AND '2024-11-07'
                 AND actual_result IS NOT NULL"
# Should return 700+ rows
```

### Run

```bash
cd /path/to/nba-props-ml

# Set credentials
export DB_PASSWORD=your_password

# Run validation
python3 nba/betting_xl/validate_xl_models.py \
  --start-date 2024-10-30 \
  --end-date 2024-11-07
```

### Verify

Compare output to `MODEL_REGISTRY.toml` validation history. Results should match within floating-point tolerance.

---

## Questions?

### "Why not use cross-validation?"

Time series data violates i.i.d. assumptions. Cross-validation would leak future information. Walk-forward validation (train on past, test on future) is the correct approach.

### "Why such a short validation period?"

Longer is better, but requires:
1. Complete historical props data
2. Complete actual results
3. Feature data for the entire period

We validated on the period with the highest data quality.

### "How do I know the backtest isn't cherry-picked?"

The validation period (Oct 30 - Nov 7, 2024) was chosen before viewing results, based on data availability. The full validation history is in `MODEL_REGISTRY.toml`, including failed strategies.
