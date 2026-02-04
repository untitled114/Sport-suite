# NBA Feature Engineering

**Status:** Production Ready
**XL Features:** 102
**V3 Features:** 136
**Last Updated:** February 2026

---

## Overview

Feature extraction for the NBA props prediction system. Supports both XL (102 features) and V3 (136 features) model architectures.

---

## Core Scripts

### `build_xl_training_dataset.py`

Standard dataset builder:
```bash
python3 build_xl_training_dataset.py --output datasets/
```

### `build_xl_training_dataset_batched.py`

10x faster batched version with in-memory caching:
```bash
# All books
python3 build_xl_training_dataset_batched.py --output datasets/

# DFS only (PrizePicks + Underdog)
python3 build_xl_training_dataset_batched.py --output datasets/ --dfs-only
```

### `extract_live_features_xl.py`

Real-time feature extraction for predictions:
- Supports both XL (102) and V3 (136) feature sets
- Hybrid PostgreSQL + MongoDB fallback
- Handles missing data gracefully

---

## Feature Categories

### XL Model (102 Features)

**Player Features (78):**
- Rolling stats: EMA L3/L5/L10/L20 for points, rebounds, assists, etc.
- Team context: pace, offensive/defensive ratings, projected possessions
- Advanced: rest days, B2B, travel distance, altitude, season phase
- Usage: starter flag, bench points ratio, position, teammate usage
- Matchup: H2H stats, matchup advantage score
- Recent performance: points per minute L5, days since last 30pt game

**Book Features (20):**
- Line variance: spread, consensus, std dev, num books, coef variation
- Per-book deviations: DraftKings, FanDuel, BetMGM, Caesars, etc.
- Softest/hardest book IDs, line spread percentile
- Min/max line, softest vs consensus

**Computed (4):**
- `is_home`: Home/away status (from player_game_logs)
- `line`: Sportsbook prop line
- `opponent_team`: Opponent team code
- `expected_diff`: Regressor prediction - line

### V3 Model (136 Features = 102 + 34)

All XL features plus:

**Season/Temporal (6):**
- `days_into_season`, `season_phase_encoded`
- `is_early_season`, `is_mid_season`, `is_late_season`, `is_playoffs`

**Volatility (8):**
- `{stat}_std_L5`, `{stat}_std_L10`
- `minutes_std_L5`, `minutes_std_L10`, `fga_std_L5`
- `{stat}_trend_ratio`, `minutes_trend_ratio`
- `usage_volatility_score`

**H2H Decay (5):**
- `h2h_decayed_avg_{stat}`, `h2h_trend_{stat}`
- `h2h_recency_adjusted_{stat}`, `h2h_time_decay_factor`
- `h2h_reliability`

**Line/Book (9):**
- `line_std`, `softest_book_hit_rate`, `softest_book_soft_rate`
- `softest_book_line_bias`, `line_source_reliability`
- `line_delta`, `line_movement_std`, `consensus_strength`
- `snapshot_count`, `hours_tracked`

**Matchup/Other (6):**
- `efficiency_vs_context`, `game_velocity`, `season_phase`
- `resistance_adjusted_L3`, `volume_proxy`, `momentum_short_term`

---

## Extractors Directory

Modular feature extractors in `extractors/`:

| File | Features | Description |
|------|----------|-------------|
| `base.py` | - | Abstract base class |
| `book_features.py` | 23 | Book disagreement features |
| `h2h_features.py` | 36 | Head-to-head matchup features |
| `prop_history_features.py` | 12 | Prop history (hit rates) |
| `vegas_features.py` | 2 | Vegas total/spread context |
| `team_betting_features.py` | 5 | ATS/OU percentages |
| `cheatsheet_features.py` | 8 | BettingPros projections |

---

## Output Datasets

Located in `datasets/`:

### Standard (All Books)
```
xl_training_POINTS_2023_2025.csv      (51,639 rows, 102 cols)
xl_training_REBOUNDS_2023_2025.csv    (similar)
```

### DFS Only (PrizePicks + Underdog)
```
xl_training_POINTS_dfs_2023_2025_batched.csv    (22,671 rows, 136 cols)
xl_training_REBOUNDS_dfs_2023_2025_batched.csv  (similar)
```

### V3 (Batched with 34 Additional Features)
```
xl_training_POINTS_2023_2025_batched.csv    (38MB, 136 cols)
xl_training_REBOUNDS_2023_2025_batched.csv  (36MB, 136 cols)
```

---

## Data Sources

| Table | Database | Port | Purpose |
|-------|----------|------|---------|
| `nba_props_xl` | nba_intelligence | 5539 | Prop lines + outcomes |
| `player_game_logs` | nba_players | 5536 | is_home enrichment |
| `player_rolling_stats` | nba_players | 5536 | EMA rolling stats |
| `team_stats` | nba_team | 5538 | Pace/ratings |

---

## Key Decisions

### Home/Away Enrichment

The `is_home` feature is enriched from `player_game_logs`, not inferred:
- 90.5% match rate from database
- 9.5% fallback to True (home)
- Critical fix: Old models had 100% home games (INVALID)

### EMA vs Simple Rolling

Exponential Moving Average (EMA) used for all rolling stats:
- More weight on recent games
- Smoother trends
- Better captures form changes

### Book Disagreement Features

20 features capture line shopping opportunities:
- Spread between softest/hardest books
- Per-book deviations from consensus
- Line variance metrics

---

## Related

- [Main README](../../README.md) - Project overview
- [betting_xl README](../betting_xl/README.md) - Prediction system
- [models README](../models/) - Model training
