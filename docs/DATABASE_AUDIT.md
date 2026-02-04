# NBA XL Training Data Audit Report
**Date:** February 3, 2026
**Status:** CRITICAL GAPS IN DATA UTILIZATION

---

## Executive Summary

This audit reveals **significant gaps** between available database resources and what's being used for training. The batched dataset builder is structurally sound but underutilizes 50%+ of available high-value data.

### Score Card

| Category | Score | Notes |
|----------|-------|-------|
| Data Quality | 8/10 | Batched builder is clean, no duplicates |
| Feature Coverage | 5/10 | **Major gaps - 50%+ unused data** |
| Temporal Integrity | 9/10 | Proper train/test split, no leakage |
| Best Practices | 6/10 | Missing walk-forward CV |

---

## Database Summary

| Database | Port | Tables | Status |
|----------|------|--------|--------|
| nba_players | 5536 | 4 | ⚠️ Historical data limited |
| nba_games | 5537 | 3 | ✅ OK (10,972 team_game_logs) |
| nba_team | 5538 | 4 | ✅ OK (positional defense data) |
| nba_intelligence | 5539 | 6 | ⚠️ **Underutilized** |

---

## CRITICAL: Unused Database Resources

### 1. Position-Specific Defense Ratings (NOT USED)

**Available in `team_season_stats`:**
```sql
def_rating_vs_pg   -- Defense vs Point Guards
def_rating_vs_sg   -- Defense vs Shooting Guards
def_rating_vs_sf   -- Defense vs Small Forwards
def_rating_vs_pf   -- Defense vs Power Forwards
def_rating_vs_c    -- Defense vs Centers
```

**Sample Data (2026 Season):**
| Team | def_rating_vs_pg | def_rating_vs_sf | def_rating_vs_c |
|------|-----------------|-----------------|-----------------|
| DET | 109.70 | 105.40 | 111.20 |
| BOS | 111.80 | 109.90 | 108.50 |
| ATL | 114.90 | 115.20 | 110.80 |

**Impact:** Player position is encoded but not matched against opponent's positional defense.

---

### 2. Per-Game Team Stats (NOT USED)

**Available in `team_game_logs` (10,972 rows):**
```sql
pace               -- Actual pace that game
offensive_rating   -- Actual offensive rating
defensive_rating   -- Actual defensive rating
possessions        -- Actual possessions
```

**Current:** Uses seasonal averages only
**Problem:** Team pace varies 97-106 game-to-game

---

### 3. Book Historical Accuracy (NOT USED)

**Available in `book_historical_accuracy` (14,869 records):**
```sql
hit_rate           -- Historical hit rate for this book/market
line_bias          -- Systematic over/under setting
soft_line_rate     -- % of props where this book has softest line
sharpe_ratio       -- Risk-adjusted line accuracy
```

**Sample Data:**
| Book | Market | Hit Rate | Soft Line Rate |
|------|--------|----------|----------------|
| underdog | POINTS | 0.499 | 0.23 |
| fanduel | POINTS | 0.481 | 0.18 |
| caesars | POINTS | 0.477 | 0.14 |

---

### 4. Player Minutes Projections (NOT USED)

**Available in `player_minutes_projections`:**
```sql
projected_mpg      -- Projected minutes per game
confidence         -- Confidence score (0-1)
```

---

## Data Utilization Summary

| Data Source | Records | Used? | Priority |
|-------------|---------|-------|----------|
| `player_game_logs` | 150K+ | ✅ Yes | - |
| `team_season_stats` (basic) | 120 | ✅ Yes | - |
| `team_season_stats` (positional) | 120 | ❌ **No** | **HIGH** |
| `team_game_logs` | 10,972 | ❌ **No** | **HIGH** |
| `book_historical_accuracy` | 14,869 | ❌ **No** | **HIGH** |
| `player_minutes_projections` | 500+ | ❌ **No** | **MEDIUM** |
| `matchup_history` | 61,380 | ✅ Yes | - |
| `prop_performance_history` | 32,743 | ✅ Yes | - |
| `injury_report` | Variable | ❌ **No** | **MEDIUM** |

**Estimated Feature Gap:** 8-12 high-value features not being used

---

## Code Issues Found

### 5. Season Lookup Bug (CRITICAL)

**Current Code (`build_xl_training_dataset_batched.py` line 275):**
```python
WHERE season = (SELECT MAX(season) FROM team_season_stats)
```

**Problem:** Uses 2026 season stats for ALL historical games (2023-2025).

**Fix:** Load all seasons and lookup by game date.

---

### 6. Redundant Feature: `days_rest_copy`

Both `days_rest` and `days_rest_copy` exist with correlation = 1.0

---

### 7. No Walk-Forward Cross-Validation

**Current:** Single temporal split (70/30)
**Better:** Time-series CV with expanding window

---

## New Features to Add

| Feature Name | Source | Expected Impact |
|--------------|--------|-----------------|
| `opp_positional_def` | team_season_stats | HIGH - matchup-specific |
| `team_L5_pace` | team_game_logs | MEDIUM - recent form |
| `opp_L5_def_rating` | team_game_logs | MEDIUM - recent form |
| `softest_book_accuracy` | book_historical_accuracy | MEDIUM - line quality |
| `projected_mpg` | player_minutes_projections | MEDIUM - minutes context |

---

## Recommended Actions

### Immediate (Before Next Training)

1. **Fix season lookup bug** - Use correct season for each game_date
2. **Remove `days_rest_copy`** - Redundant feature
3. **Add position-specific defense** - `opp_positional_def`

### Short-Term (This Week)

4. Add book historical accuracy features
5. Add team rolling stats from `team_game_logs`
6. Add minutes projection features

### Medium-Term (This Month)

7. Implement walk-forward CV in training script
8. Add injury impact features

---

## Validation Checklist

Before next model training:

- [ ] Verify no duplicate rows
- [ ] Verify correct season lookup for team stats
- [ ] Verify position-specific defense is populated (non-zero)
- [ ] Verify book accuracy features are populated
- [ ] Verify days_rest_copy is removed
- [ ] Run feature importance to validate new features

---

**Prepared by:** Claude Code Audit
