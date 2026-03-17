# V4 Model Retraining Plan

**Created:** March 16, 2026
**Status:** Planning → Backfill → Build → Deploy

---

## Overview

V4 adds **~15 BettingPros analytics features** to the existing XL (102) and V3 (136) feature sets.
Unlike direct sportsbook line features (cold start), BP analytics features have **full historical coverage back to 2023** via the `/v3/props` API and server-rendered DVP pages.

**Auto-retraining trigger:** 7-day rolling win rate drops below **60%**.

---

## Phase 1: BP Historical Backfill

### 1A: Props API Backfill (~2,000 API calls)

**Endpoint:** `GET https://api.bettingpros.com/v3/props`

**Confirmed:** API returns full analytics for any historical date, including:

```json
{
  "projection": {
    "recommended_side": "under",
    "value": 13.7,
    "probability": 0.68,
    "expected_value": 0.363,
    "bet_rating": 5,
    "diff": -3.8
  },
  "extra": {
    "opposition_rank": { "rank": 16, "value": 114.2 }
  },
  "scoring": {
    "is_scored": true,
    "actual": 19
  },
  "performance": {
    "last_1":  { "over": 1, "under": 0, "push": 0 },
    "last_5":  { "over": 3, "under": 2, "push": 0 },
    "last_10": { "over": 5, "under": 5, "push": 0 },
    "last_15": { "over": 6, "under": 9, "push": 0 },
    "last_20": { "over": 7, "under": 13, "push": 0 },
    "season":  { "over": 16, "under": 56, "push": 0 }
  }
}
```

**Parameters:**
```
sport=NBA
date=YYYY-MM-DD
market_id=156 (POINTS) | 157 (REBOUNDS)
limit=500
page=1,2,...
include_markets=true
include_counts=true
```

**Backfill scope:**
- Date range: Oct 24, 2023 → Mar 15, 2026 (~500 game dates)
- Markets: POINTS (156), REBOUNDS (157) = 2 per date
- ~1,000 API calls total (some dates may need pagination → ~2,000 max)
- Rate: 1 req/sec → ~33 minutes total
- **Auth:** `BETTINGPROS_API_KEY` + `x-level: cHJlbWl1bQ==` (Premium tier)

**Storage:** New table `bp_historical_analytics` in nba_intelligence (port 5539):
```sql
CREATE TABLE bp_historical_analytics (
    id SERIAL PRIMARY KEY,
    player_name VARCHAR(255) NOT NULL,
    stat_type VARCHAR(50) NOT NULL,
    game_date DATE NOT NULL,
    -- Projection
    bp_projection DECIMAL(5,2),
    bp_projection_diff DECIMAL(5,2),
    bp_probability DECIMAL(5,4),
    bp_expected_value DECIMAL(6,4),
    bp_bet_rating INTEGER,              -- 1-5 stars
    bp_recommended_side VARCHAR(10),     -- 'over' | 'under'
    -- Opposition
    bp_opposition_rank INTEGER,          -- 1-30
    bp_opposition_value DECIMAL(6,2),    -- Defensive rating
    -- Hit Rates (at time of prop)
    bp_hit_rate_L1 DECIMAL(4,3),
    bp_hit_rate_L5 DECIMAL(4,3),
    bp_hit_rate_L10 DECIMAL(4,3),
    bp_hit_rate_L15 DECIMAL(4,3),
    bp_hit_rate_L20 DECIMAL(4,3),
    bp_hit_rate_season DECIMAL(4,3),
    -- Lines
    bp_over_line DECIMAL(5,2),
    bp_consensus_line DECIMAL(5,2),
    bp_over_odds INTEGER,
    bp_consensus_odds INTEGER,
    -- Actual (from BP)
    bp_actual_value DECIMAL(5,2),
    bp_is_scored BOOLEAN,
    -- Metadata
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (player_name, game_date, stat_type)
);
```

**Script:** `nba/betting_xl/scripts/backfill_bp_analytics.py`

### 1B: DVP Historical Backfill (3 page fetches)

**Endpoint:** `https://www.bettingpros.com/nba/defense-vs-position/?season=XXXX`

**Confirmed availability:**
| Season Param | NBA Season | avgGamesPlayed | Status |
|-------------|-----------|----------------|--------|
| 2023 | 2023-24 | 82 | Full season |
| 2024 | 2024-25 | 82 | Full season |
| 2025 | 2025-26 | 67 | In progress |

**Data per season:** 30 teams x 7 positions x 9 stats = 1,890 values

**Stats available per position:**
- points, rebounds, assists, three_points_made
- steals, blocks, turnovers
- free_throw_perc, field_goals_perc

**Storage:** New table `bp_dvp_historical` in nba_intelligence:
```sql
CREATE TABLE bp_dvp_historical (
    id SERIAL PRIMARY KEY,
    season INTEGER NOT NULL,         -- 2023, 2024, 2025
    team VARCHAR(10) NOT NULL,       -- Normalized (GSW, not GS)
    position VARCHAR(5) NOT NULL,    -- ALL, PG, SG, SF, PF, C, TM
    stat_name VARCHAR(30) NOT NULL,
    value DECIMAL(6,2),
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (season, team, position, stat_name)
);
```

**Script:** `nba/betting_xl/scripts/backfill_bp_dvp.py`

---

## Phase 2: V4 Feature Extractor

### New Features (~15, from BP historical data)

**File:** `nba/features/extractors/bp_analytics_features.py`

| # | Feature Name | Source | Description |
|---|-------------|--------|-------------|
| 1 | `bp_projection_diff` | API | BP projection - line (how far BP thinks from line) |
| 2 | `bp_expected_value` | API | BP's computed EV for the over |
| 3 | `bp_bet_rating` | API | 1-5 star rating (integer) |
| 4 | `bp_probability` | API | BP's computed P(over) |
| 5 | `bp_opposition_rank` | API | Opponent defensive rank for this stat (1-30) |
| 6 | `bp_opposition_value` | API | Opponent defensive value (raw) |
| 7 | `bp_hit_rate_L5` | API | Over hit rate last 5 games |
| 8 | `bp_hit_rate_L10` | API | Over hit rate last 10 games |
| 9 | `bp_hit_rate_L15` | API | Over hit rate last 15 games |
| 10 | `bp_hit_rate_season` | API | Over hit rate for season |
| 11 | `bp_hit_rate_trend` | Computed | L5 - L15 (recent vs longer-term momentum) |
| 12 | `dvp_stat_allowed` | DVP | Opponent's DVP stat allowed for player's position |
| 13 | `dvp_stat_rank` | DVP | Opponent's DVP rank for player's position (1-30) |
| 14 | `bp_recommended_over` | API | 1 if BP recommends over, 0 otherwise |
| 15 | `bp_projection_vs_consensus` | API | BP projection - consensus line |

**V4 total features:** 136 (V3) + 15 (BP analytics) = **~151 features**

### Feature Extraction for Training

For historical training data, features are looked up from `bp_historical_analytics` and `bp_dvp_historical` tables by joining on `(player_name, game_date, stat_type)`.

For live prediction, features come from the same BP API calls already made during the pipeline (cheatsheet, hit rates fetchers).

---

## Phase 3: Auto-Retraining System

### Trigger: 7-Day Rolling WR < 60%

**Performance Monitor** runs daily via Airflow (after validation pipeline):

```python
# Query nba_prediction_history for graded picks from last 7 days
SELECT COUNT(*) as total,
       SUM(CASE WHEN is_hit THEN 1 ELSE 0 END) as wins
FROM nba_prediction_history
WHERE run_date >= CURRENT_DATE - INTERVAL '7 days'
  AND actual_result IS NOT NULL
  AND is_hit IS NOT NULL
  AND run_number = (
      SELECT MAX(run_number) FROM nba_prediction_history p2
      WHERE p2.run_date = nba_prediction_history.run_date
        AND p2.player_name = nba_prediction_history.player_name
        AND p2.stat_type = nba_prediction_history.stat_type
  )
```

**Trigger conditions (ANY fires retraining):**
1. 7-day rolling WR < 60% (with minimum 20 graded picks)
2. Drift detection > 15% of features drifted (existing `drift_service.py`)
3. Scheduled bi-weekly (1st and 15th of month)

### Retraining Pipeline

```
TRIGGER
  │
  ▼
┌─────────────────────────────────┐
│ 1. BUILD TRAINING DATASET       │
│    - Expanding window:          │
│      Oct 2023 → present - 14d  │
│    - Recency weight: 2x for    │
│      last 90 days via          │
│      sample_weight in LightGBM │
│    - Include BP analytics       │
│      features from backfill     │
│    - Include DVP features       │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│ 2. WALK-FORWARD VALIDATION      │
│    - 6 folds, 2-month test      │
│    - Metrics: AUC, WR, ROI      │
│    - Must beat current model:   │
│      AUC improvement > 0.005    │
│      WR not regressed > 2%      │
│      ROI not regressed > 3%     │
└────────────┬────────────────────┘
             │ passes?
             ▼
┌─────────────────────────────────┐
│ 3. COMPARE VS PRODUCTION        │
│    - Run candidate on latest    │
│      14-day holdout data        │
│    - Side-by-side metrics table │
│    - Log to MODEL_REGISTRY.toml │
└────────────┬────────────────────┘
             │ better?
             ▼
┌─────────────────────────────────┐
│ 4. SHADOW MODE (3 days)         │
│    - Save candidate model to    │
│      saved_xl/{market}_v4_*.pkl │
│    - Run candidate in parallel  │
│      (model_version="v4")       │
│    - Compare live picks vs      │
│      production picks daily     │
│    - Auto-promote after 3 days  │
│      if WR >= production WR     │
│    - Auto-rollback if WR drops  │
│      below production - 5%      │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│ 5. PROMOTE / ROLLBACK           │
│    - Copy candidate → production│
│    - Update MODEL_REGISTRY.toml │
│    - Regenerate reference       │
│      distributions for drift    │
│    - Discord alert via Axiom    │
│    - Keep last 3 model versions │
│      for rollback               │
└─────────────────────────────────┘
```

### Airflow DAG: `nba_retraining_pipeline`

```
Schedule: Daily at 10:00 AM EST (after validation at 9:30 AM)
          Only proceeds past gate if trigger condition met.

Tasks:
  check_retraining_trigger
    └── build_training_dataset (if triggered)
          └── train_candidate_models (POINTS + REBOUNDS in parallel)
                └── walk_forward_validate
                      └── compare_vs_production
                            └── enter_shadow_mode (or reject)
```

### Model Versioning

```
nba/models/saved_xl/
├── points_xl_*.pkl           # Current XL production
├── points_v3_*.pkl           # Current V3 production
├── points_v4_*.pkl           # V4 candidate (during shadow)
├── archive/
│   ├── points_v3_20260203/   # Previous V3 (rollback target)
│   └── points_v4_20260501/   # Previous V4 attempt
└── MODEL_REGISTRY.toml       # Version history + metrics
```

### Recency Weighting

LightGBM supports `sample_weight` in `Dataset()`. Weight scheme:

```python
def compute_sample_weights(dates: pd.Series) -> np.ndarray:
    """2x weight for last 90 days, 1.5x for 91-180 days, 1x for older."""
    now = pd.Timestamp.now()
    days_ago = (now - dates).dt.days
    weights = np.where(days_ago <= 90, 2.0,
              np.where(days_ago <= 180, 1.5, 1.0))
    return weights
```

---

## Phase 4: Direct Line Features (Future, Cold Start)

These 19 features from `nba/features/extractors/direct_line_features.py` require accumulated direct sportsbook data:

| Feature | Available When |
|---------|---------------|
| num_direct_sources | Now (accumulating) |
| line_movement_velocity | 2-3 months of snapshots |
| bp_discrepancy_avg | Now (accumulating) |
| cross_platform_agreement | Now (accumulating) |
| ... (19 total) | V4 Phase 2: ~June 2026 |

**V4 Phase 2 total features:** 151 (Phase 1) + 19 (direct) = **~170 features**

---

## Build Order

| Step | Task | Estimated Effort | Dependency |
|------|------|-----------------|------------|
| 1 | Update CLAUDE.md | 30 min | None |
| 2 | Write this plan | 30 min | None |
| 3 | `backfill_bp_analytics.py` | 2 hrs | None |
| 4 | `backfill_bp_dvp.py` | 1 hr | None |
| 5 | Run backfill (API calls) | 33 min | Steps 3-4 |
| 6 | `bp_analytics_features.py` extractor | 2 hrs | Step 5 |
| 7 | Update `build_xl_training_dataset.py` | 2 hrs | Step 6 |
| 8 | Add recency weighting to `train_market.py` | 1 hr | None |
| 9 | Build V4 training dataset | 30 min (runtime) | Steps 7-8 |
| 10 | Train V4 models | 20 min (runtime) | Step 9 |
| 11 | Walk-forward validation | 30 min (runtime) | Step 10 |
| 12 | Performance monitor DAG | 2 hrs | None |
| 13 | Retraining DAG | 3 hrs | Steps 7-12 |
| 14 | Shadow mode logic | 2 hrs | Step 13 |
| 15 | Model versioning + rollback | 1 hr | Step 14 |

**Critical path:** Steps 3-5 (backfill) → 6-7 (features) → 9-11 (train + validate)

---

## Data Volume Estimates

| Dataset | Current | After Backfill |
|---------|---------|---------------|
| Training props (POINTS) | 51,639 | 51,639 (same props, richer features) |
| Training props (REBOUNDS) | 50,756 | 50,756 |
| Features per prop (V3) | 136 | ~151 (V4 Phase 1) |
| BP analytics rows | 0 | ~200K+ (all historical props with BP data) |
| DVP rows | 0 | ~5,670 (3 seasons x 30 teams x 7 pos x 9 stats) |
| API calls needed | - | ~2,000 |
| Backfill runtime | - | ~33 minutes |

---

## Success Criteria

1. **V4 AUC > V3 AUC** on walk-forward validation (target: >0.005 improvement)
2. **V4 WR >= V3 WR** on 14-day holdout
3. **Auto-retraining fires correctly** when WR dips below 60%
4. **Shadow mode** correctly promotes/rejects candidate models
5. **Rollback** works within 1 minute if production degrades
6. **BP analytics features** show non-zero SHAP importance (are actually useful)

---

## Risk Mitigation

- **BP API rate limit:** 1 req/sec is conservative; if throttled, add exponential backoff
- **BP API changes:** Cache raw JSON responses locally for replay; don't re-fetch if already cached
- **Overfitting on new features:** Walk-forward validation catches this; recency weighting helps
- **Model bloat (151 features):** Monitor SHAP; drop features with zero importance after first V4 training
- **Stale DVP data:** DVP is season-level only; good enough for training, live features use current-season DVP
- **Backfill data quality:** Cross-validate BP `scoring.actual` vs our `player_game_logs.actual_value`
