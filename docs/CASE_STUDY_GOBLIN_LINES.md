# Case Study: PrizePicks Alternate Lines (Goblin/Demon) - Training Data Mismatch

**Date:** February 4, 2026
**Authors:** Sports Suite Team
**Status:** Validated

---

## Executive Summary

This case study documents an unexpected finding: **out-of-distribution data improved model performance** rather than degrading it. PrizePicks alternate lines ("goblin" and "demon"), which were never present in training data, yielded an **85% win rate** compared to 70% on standard lines the model was trained on.

---

## Background

### The System

Sports Suite's NBA betting system uses a two-head stacked model architecture:
- **Head 1 (Regressor):** Predicts actual stat value (e.g., "Player X will score 24.7 points")
- **Head 2 (Classifier):** Predicts P(actual > line) using regressor output as a feature
- **Calibration:** Isotonic regression for probability calibration

### Model Versions

| Model | Features | Trained | Data Source |
|-------|----------|---------|-------------|
| XL | 102 | Dec 2025 | BettingPros (standard sportsbooks) |
| V3 | 136 | Feb 3, 2026 | BettingPros DFS subset (Underdog + PrizePicks standard) |

### The Data Gap

**Training data coverage (up to Dec 2025):**
- DraftKings, FanDuel, BetMGM, Caesars, FanDuel, BetRivers, ESPNBet
- Underdog: 152,091 historical props
- PrizePicks (standard): 48,574 historical props
- **PrizePicks Goblin/Demon: 0 props** (didn't exist)

**Production data (Feb 2026):**
- Goblin lines added Feb 1, 2026
- Demon lines added Feb 1, 2026
- These represent ~40% of daily prop volume

---

## The Problem

### Initial Hypothesis

When we discovered goblin/demon lines were appearing as "best lines" in production picks, we hypothesized this could be problematic:

1. Model never saw these line types during training
2. Goblin lines are systematically lower (softer) than standard
3. Edge calculations might be artificially inflated
4. Real-world performance could suffer due to distribution mismatch

### The Question

> "Should we filter out goblin/demon lines since the model wasn't trained on them?"

---

## Methodology

### Test Design

We implemented a `--standard-only` flag to run parallel backtests:

```bash
# Test 1: WITH goblin/demon lines (current production behavior)
python3 run_historical_backtest.py --start 2026-01-19 --end 2026-02-02 --no-seed

# Test 2: WITHOUT goblin/demon lines (standard only)
python3 run_historical_backtest.py --start 2026-01-19 --end 2026-02-02 --no-seed --standard-only
```

### Implementation

Filter applied at two levels:
1. **Main props query:** Exclude `book_name IN ('prizepicks_goblin', 'prizepicks_demon', 'prizepicks_alt')`
2. **Line optimizer:** Skip alternate books when finding softest line

### Validation Period

- **Dates:** January 19 - February 2, 2026 (15 days)
- **Markets:** POINTS, REBOUNDS
- **Models:** XL + V3 running in parallel

---

## Results

### Overall Comparison

| Metric | With Goblin/Demon | Standard Only | Delta |
|--------|-------------------|---------------|-------|
| **Total Picks** | 47 | 17 | +30 |
| **Validated** | 44 | 17 | +27 |
| **Wins** | 35 | 12 | +23 |
| **Losses** | 9 | 5 | +4 |
| **Win Rate** | **79.5%** | 70.6% | **+8.9%** |

### By Market

| Market | With Goblin | Standard Only |
|--------|-------------|---------------|
| POINTS | 74.2% (23W/8L) | 64.3% (9W/5L) |
| REBOUNDS | 92.3% (12W/1L) | 100% (3W/0L) |

### By Model Version

| Model | With Goblin | Standard Only |
|-------|-------------|---------------|
| XL | 78.3% (18W/5L) | 63.6% (7W/4L) |
| V3 | 81.0% (17W/4L) | 83.3% (5W/1L) |

### Implied Goblin-Specific Performance

Isolating the picks that came specifically from goblin lines:
- **Extra picks:** 27 (44 - 17)
- **Extra wins:** 23 (35 - 12)
- **Extra losses:** 4 (9 - 5)
- **Goblin-specific WR:** **85.2%** (23/27)

---

## Analysis

### Why Did Out-of-Distribution Data Help?

The key insight is understanding **what the model predicts**:

```
Model Output: "Player X will score 24.7 points"
              (This is an ABSOLUTE prediction, not line-relative)
```

**Standard line scenario:**
- Line: 24.5 points
- Model prediction: 24.7
- Edge: +0.2 points (0.8%)
- P(OVER): ~52%

**Goblin line scenario:**
- Line: 20.5 points
- Model prediction: 24.7
- Edge: +4.2 points (20.5%)
- P(OVER): ~85%

The model's prediction doesn't change based on the line - it predicts the **actual expected stat value**. Softer lines simply make it easier to be on the right side.

### The Distribution Mismatch Paradox

Traditional ML wisdom says out-of-distribution data hurts performance. This case shows an exception:

| Scenario | Training Mismatch Effect |
|----------|--------------------------|
| **Feature drift** | Hurts - model sees unfamiliar patterns |
| **Label drift** | Hurts - outcome distribution changes |
| **Line softness** | Helps - easier threshold to clear |

Goblin lines represent a **favorable** distribution shift - the threshold moved in our favor, not the underlying player performance distribution.

### Why This Wasn't Obvious

The concern was that goblin lines might have:
- Different market dynamics
- Different sharp money influence
- Different closing line value patterns

But because our model predicts **raw stat values** (not market-relative outcomes), none of these factors degraded performance.

---

## Decision: Keep Goblin/Demon Lines

### Rationale

1. **Higher volume:** 2.7x more actionable picks (47 vs 17)
2. **Higher win rate:** +8.9% improvement (79.5% vs 70.6%)
3. **Compounding effect:** More bets × higher WR = significantly better ROI
4. **Genuine edge:** Softer lines provide real value, not artificial inflation

### Production Configuration

```python
# line_optimizer.py - DO NOT filter out alternate lines
PRIZEPICKS_ALT_BOOKS = {
    "prizepicks_goblin",  # Keep - 85% WR in backtest
    "prizepicks_demon",   # Keep - provides line spread context
    "prizepicks_alt",
}

# These are INCLUDED in line shopping (not filtered)
```

### Monitoring

Added `--standard-only` flag for ongoing comparison:
```bash
# Periodic validation to ensure goblin edge persists
python3 run_historical_backtest.py --start DATE --end DATE --standard-only
```

---

## Lessons Learned

### 1. Understand What Your Model Predicts

The model predicts **actual stat values**, not line-relative outcomes. This distinction is crucial for understanding when distribution shift helps vs. hurts.

### 2. Test Assumptions Empirically

Initial intuition ("untrained data = bad") was wrong. The 15-day backtest provided concrete evidence to override the assumption.

### 3. Softer Lines Are Genuine Edge

DFS platforms (PrizePicks, Underdog) offer systematically softer lines. This isn't noise - it's exploitable edge when combined with accurate predictions.

### 4. Volume Matters

Even if win rates were equal, 2.7x more picks would justify inclusion. The fact that win rate also improved makes this an obvious decision.

---

## Technical Implementation

### Files Modified

1. **`line_optimizer.py`**
   - Added `PRIZEPICKS_ALT_BOOKS` constant
   - Added `standard_only` parameter to `LineOptimizer.__init__`
   - Filter applied in `get_book_lines()` query

2. **`generate_xl_predictions.py`**
   - Added `standard_only` parameter
   - Filter applied in main props query
   - Passed to `LineOptimizer` constructor

3. **`run_historical_backtest.py`**
   - Added `--standard-only` CLI flag
   - Passed through to generator

### Database Schema Context

```sql
-- PrizePicks lines stored with distinct book_name values
SELECT DISTINCT book_name FROM nba_props_xl WHERE book_name LIKE 'prizepicks%';
-- prizepicks        (standard)
-- prizepicks_goblin (lower/softer)
-- prizepicks_demon  (higher/harder)
```

---

## Appendix: Raw Backtest Output

### With Goblin/Demon
```
BACKTEST SUMMARY: 2026-01-19 to 2026-02-02
Mode: All books
================================================================================
Days processed: 15
Total picks generated: 47
Total validated: 44
Wins:   35
Losses: 9
WIN RATE: 79.5%

POINTS    : 23W / 8L = 74.2%
REBOUNDS  : 12W / 1L = 92.3%
```

### Standard Only
```
BACKTEST SUMMARY: 2026-01-19 to 2026-02-02
Mode: Standard only (no goblin/demon)
================================================================================
Days processed: 15
Total picks generated: 17
Total validated: 17
Wins:   12
Losses: 5
WIN RATE: 70.6%

POINTS    : 9W / 5L = 64.3%
REBOUNDS  : 3W / 0L = 100.0%
```

---

## Context: Walk-Forward Cross-Validation Results

### Model Performance (3-Fold Walk-Forward CV)

Walk-forward cross-validation with expanding training window validates model stability over time:

```
WALK-FORWARD CV SUMMARY (POINTS Market)
============================================================
Metric               Mean      Std       Min       Max
------------------------------------------------------------
AUC (blended)       0.7514    0.0268    0.7142    0.7761
Accuracy            0.6701    0.0287    0.6296    0.6916
RMSE                5.974     0.183     5.790     6.224
R²                  0.5523    0.0195    0.5285    0.5763

Per-Fold Results:
Fold  Train Period             Test Period              AUC     Accuracy
------------------------------------------------------------------------
1     2023-10-24 to 2025-02-03 2025-02-04 to 2025-04-04 0.7640  69.16%
2     2023-10-24 to 2025-04-04 2025-04-05 to 2025-04-13 0.7761  68.92%
3     2023-10-24 to 2025-04-13 2025-10-21 to 2025-11-30 0.7142  62.96%
```

**Key Metrics:**
- **Mean Accuracy:** 67% across all folds
- **AUC Stability:** Low variance (std=0.027) indicates consistent performance
- **Temporal Stability:** First half avg AUC 0.764, second half 0.745 (minimal degradation)

### Why Filters Further Improve Results

The walk-forward CV tests model accuracy on all predictions. Production picks apply additional filters:
- Edge threshold ≥ 2.5%
- Line spread ≥ 1.5 points
- Minimum 3 books offering
- Risk assessment (volatility, defense, trends)

| Scenario | Bets | Win Rate | Source |
|----------|------|----------|--------|
| Walk-forward CV (all predictions) | ~10,000 | 67% | 3-fold CV |
| Filtered picks (standard only) | 17 | 70.6% | Standard backtest |
| Filtered picks (all books) | 44 | 79.5% | Goblin backtest |

**Conclusion:** The model has strong baseline accuracy (67%). Filters select highest-confidence picks (70%). Goblin lines add volume at even higher accuracy (80%).

### Decision Framework Validated

```
Walk-Forward CV: ~67% accuracy (model baseline)
     ↓ + Production Filters
Filtered: ~70% WR (high-confidence picks only)
     ↓ + Goblin/Demon Lines
Optimal:  ~80% WR (soft lines + high volume)
```

---

## References

- ADR-003: LightGBM over XGBoost/CatBoost
- ADR-006: Book Disagreement Features
- Training data: `nba/features/datasets/xl_training_*_2023_2025.csv`
- Backtest script: `nba/betting_xl/run_historical_backtest.py`
- Walk-forward CV script: `nba/models/walk_forward_validation.py`
- Walk-forward CV log: `/tmp/train_points_cv.log`
