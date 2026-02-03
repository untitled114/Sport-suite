# System Design

This document explains how data flows through the system, why architectural decisions were made, and how to trace a prediction from raw data to output.

## High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              PHASE 1: INGESTION                             │
│                              (Morning Pipeline)                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   External APIs                                                             │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐    │
│   │ BettingPros │   │ ESPN API    │   │ NBA Stats   │   │ Underdog    │    │
│   │ Premium     │   │ (schedule)  │   │ (box scores)│   │ (DFS odds)  │    │
│   └──────┬──────┘   └──────┬──────┘   └──────┬──────┘   └──────┬──────┘    │
│          │                 │                 │                 │            │
│          ▼                 ▼                 ▼                 ▼            │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                         Fetcher Layer                               │  │
│   │  nba/betting_xl/fetchers/                                           │  │
│   │  ├── fetch_bettingpros.py  # 7 sportsbooks (DK, FD, MGM, etc.)     │  │
│   │  ├── fetch_cheatsheet.py   # Underdog, analytics data              │  │
│   │  └── fetch_all.py          # Orchestrator                          │  │
│   └──────────────────────────────┬──────────────────────────────────────┘  │
│                                  │                                          │
│                                  ▼                                          │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                          Loader Layer                               │  │
│   │  nba/betting_xl/loaders/                                            │  │
│   │  ├── load_props_to_db.py       # Props → nba_prop_lines table      │  │
│   │  └── load_cheatsheet_to_db.py  # DFS data → database               │  │
│   └──────────────────────────────┬──────────────────────────────────────┘  │
│                                  │                                          │
│                                  ▼                                          │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                         PostgreSQL                                  │  │
│   │                                                                     │  │
│   │   nba_players:5536          nba_games:5537                          │  │
│   │   ├── player_profile        ├── games                               │  │
│   │   ├── player_game_logs      ├── box_scores                          │  │
│   │   └── player_rolling_stats  └── game_context                        │  │
│   │                                                                     │  │
│   │   nba_team:5538             nba_intelligence:5539                   │  │
│   │   ├── teams                 ├── nba_prop_lines  (42K+ props)        │  │
│   │   ├── team_stats            ├── injuries                            │  │
│   │   └── team_rolling_stats    └── matchup_history                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │ Game date arrives
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            PHASE 2: FEATURE EXTRACTION                      │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   For each prop (player + stat_type + line):                               │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │              LiveFeatureExtractorXL                                 │  │
│   │              nba/features/extract_live_features_xl.py               │  │
│   │                                                                     │  │
│   │   Input: player_name, game_date, opponent, is_home, line            │  │
│   │                                                                     │  │
│   │   Queries:                                                          │  │
│   │   ├── player_rolling_stats  → L3/L5/L10/L20 EMA stats              │  │
│   │   ├── team_stats            → pace, off_rating, def_rating          │  │
│   │   ├── matchup_history       → H2H performance vs opponent           │  │
│   │   ├── injuries              → teammate injury impact                │  │
│   │   └── nba_prop_lines        → multi-book line data                  │  │
│   │                                                                     │  │
│   │   Output: XL (102) or V3 (136) feature vector                       │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   Feature Categories (XL: 102, V3: 136 total):                              │
│   ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐           │
│   │ Player (42)      │ │ Team/Game (28)   │ │ H2H History (36) │           │
│   │ ema_points_L3/5  │ │ team_pace        │ │ h2h_avg_points   │           │
│   │ ema_rebounds_L10 │ │ opp_def_rating   │ │ h2h_L3_rebounds  │           │
│   │ ema_minutes_L20  │ │ rest_days, B2B   │ │ h2h_home/away    │           │
│   │ fg_pct, ft_rate  │ │ travel_distance  │ │ sample_quality   │           │
│   └──────────────────┘ └──────────────────┘ └──────────────────┘           │
│   ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐           │
│   │ Book (22)        │ │ Prop History (12)│ │ Vegas/BP (17+1)  │           │
│   │ line_spread      │ │ hit_rate_L20     │ │ vegas_total      │           │
│   │ dk/fd_deviation  │ │ bayesian_conf    │ │ bp_projection    │           │
│   │ softest_vs_cons  │ │ consecutive_hits │ │ expected_diff    │           │
│   └──────────────────┘ └──────────────────┘ └──────────────────┘           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PHASE 3: MODEL INFERENCE                          │
│                           (Evening Pipeline)                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                     Two-Head Stacked Model                          │  │
│   │                     nba/betting_xl/xl_predictor.py                  │  │
│   │                                                                     │  │
│   │                                                                     │  │
│   │   Feature Vector (XL: 102 or V3: 136)                               │  │
│   │          │                                                          │  │
│   │          ▼                                                          │  │
│   │   ┌──────────────┐                                                  │  │
│   │   │   Imputer    │  Fill missing values (median strategy)           │  │
│   │   └──────┬───────┘                                                  │  │
│   │          ▼                                                          │  │
│   │   ┌──────────────┐                                                  │  │
│   │   │   Scaler     │  StandardScaler normalization                    │  │
│   │   └──────┬───────┘                                                  │  │
│   │          │                                                          │  │
│   │          ├─────────────────────────┐                                │  │
│   │          ▼                         ▼                                │  │
│   │   ┌──────────────┐          ┌──────────────┐                        │  │
│   │   │  REGRESSOR   │          │  CLASSIFIER  │                        │  │
│   │   │  (LightGBM)  │          │  (LightGBM)  │                        │  │
│   │   │              │          │              │                        │  │
│   │   │  Predicts    │ ───────▶ │  Input: N    │                        │  │
│   │   │  stat value  │ exp_diff │  + exp_diff  │                        │  │
│   │   │              │          │              │                        │  │
│   │   │  Output:     │          │  Output:     │                        │  │
│   │   │  27.19 pts   │          │  P(OVER)=0.74│                        │  │
│   │   └──────────────┘          └──────┬───────┘                        │  │
│   │                                    │                                │  │
│   │                                    ▼                                │  │
│   │                            ┌──────────────┐                         │  │
│   │                            │ CALIBRATOR   │                         │  │
│   │                            │ (Isotonic)   │                         │  │
│   │                            │              │                         │  │
│   │                            │ Maps raw     │                         │  │
│   │                            │ prob to      │                         │  │
│   │                            │ calibrated   │                         │  │
│   │                            │ probability  │                         │  │
│   │                            └──────┬───────┘                         │  │
│   │                                   │                                 │  │
│   │                                   ▼                                 │  │
│   │                            Calibrated P(OVER)                       │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PHASE 4: BET FILTERING                             │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                      Line Optimizer                                 │  │
│   │                      nba/betting_xl/line_optimizer.py               │  │
│   │                                                                     │  │
│   │   Filter Criteria (Hybrid Dual-Filter):                             │  │
│   │                                                                     │  │
│   │   POINTS:                                                           │  │
│   │   ├── p_over >= 0.65  (probability gate)                            │  │
│   │   └── AND (edge >= 2.0 pts OR line_spread >= 2.5)                   │  │
│   │                                                                     │  │
│   │   REBOUNDS:                                                         │  │
│   │   ├── p_over >= 0.65  (probability gate)                            │  │
│   │   └── AND (edge >= 1.0 pts OR line_spread >= 2.5)                   │  │
│   │                                                                     │  │
│   │   ASSISTS/THREES: DISABLED (poor backtest performance)              │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                    Production Safeguards                            │  │
│   │                    nba/betting_xl/config/production_policies.py     │  │
│   │                                                                     │  │
│   │   ├── Data Freshness: Props must be < 6 hours old                   │  │
│   │   ├── Stop-Loss: Pause after 3 consecutive losing days              │  │
│   │   ├── Max Bets/Day: 10 (prevents overexposure)                      │  │
│   │   └── Min Books: Require 3+ books for line spread calculation       │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            PHASE 5: OUTPUT                                  │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Output File: nba/betting_xl/predictions/xl_picks_YYYY-MM-DD.json         │
│                                                                             │
│   {                                                                         │
│     "generated_at": "2026-01-15T16:53:34",                                  │
│     "date": "2026-01-07",                                                   │
│     "total_picks": 6,                                                       │
│     "picks": [                                                              │
│       {                                                                     │
│         "player_name": "Stephen Curry",                                     │
│         "stat_type": "POINTS",                                              │
│         "side": "OVER",                                                     │
│         "prediction": 31.46,        ← Regressor output                      │
│         "p_over": 0.740,            ← Calibrated probability                │
│         "best_book": "underdog",    ← Line shopping result                  │
│         "best_line": 27.5,                                                  │
│         "edge": 3.96,               ← prediction - line                     │
│         "line_distribution": [...]  ← All books + their lines               │
│       }                                                                     │
│     ]                                                                       │
│   }                                                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Database Schema

### nba_players (Port 5536)

```sql
-- Core player data
player_profile (player_id, player_name, team, position, height, weight)
player_game_logs (player_id, game_date, opponent, is_home, pts, reb, ast, ...)
player_rolling_stats (player_id, game_date, ema_points_L3, ema_points_L5, ...)
```

### nba_intelligence (Port 5539)

```sql
-- Prop lines from 7 sportsbooks
nba_prop_lines (
  id,
  player_name,
  stat_type,        -- 'POINTS', 'REBOUNDS', etc.
  game_date,
  line,             -- The prop line (e.g., 27.5)
  source,           -- 'draftkings', 'fanduel', etc.
  book_id,
  actual_result,    -- Filled after game (for validation)
  created_at
)

-- Indexes for performance
CREATE INDEX idx_props_player_date ON nba_prop_lines(player_name, game_date);
CREATE INDEX idx_props_date_stat ON nba_prop_lines(game_date, stat_type);
```

---

## Model Training Pipeline

### Step 1: Build Training Dataset

```bash
cd nba/features
python3 build_xl_training_dataset.py --output datasets/
```

**What it does:**
1. Queries `nba_prop_lines` for historical props with `actual_result IS NOT NULL`
2. Enriches `is_home` from `player_game_logs` (90.5% match rate)
3. Extracts 166 features for each prop using point-in-time data
4. Outputs: `xl_training_POINTS_2023_2025.csv` (24,316 rows)

**Data Leakage Prevention:**
```python
# From build_xl_training_dataset.py
# Only use data available BEFORE the game date
WHERE game_date < prop_date
```

### Step 2: Train Two-Head Model

```bash
cd nba/models
python3 train_market.py --market POINTS \
  --data ../features/datasets/xl_training_POINTS_2023_2025.csv
```

**What it does:**
1. Temporal split: 70% train (earlier), 30% test (later)
2. Train regressor on (features → actual_result)
3. Train classifier on (features + expected_diff → did_go_over)
4. Fit IsotonicRegression calibrator on test set
5. Save 6 artifacts per market: regressor, classifier, calibrator, imputer, scaler, features

### Step 3: Validate on Holdout

```bash
python3 nba/betting_xl/validate_xl_models.py \
  --start-date 2024-10-30 --end-date 2024-11-07
```

**What it does:**
1. Load historical props from validation period
2. Extract features AS OF each game date (no lookahead)
3. Generate predictions using trained models
4. Apply production filters
5. Compare predictions to actual results
6. Report win rate, ROI, per-market breakdown

---

## Why This Architecture?

### Why Two Heads?

**Problem**: Directly predicting P(OVER) from features loses information about the magnitude of edge.

**Solution**:
- **Regressor** learns to predict the actual stat value
- **Classifier** learns when the regressor's edge is exploitable
- The `expected_diff` feature (regressor_prediction - line) bridges them

This separation allows the classifier to learn: "When the model thinks Curry will score 5 points more than the line, how often does he actually go over?"

### Why Isotonic Calibration?

Raw classifier probabilities are often overconfident. If the model outputs 0.80, it might only be correct 65% of the time.

Isotonic regression learns a monotonic mapping from raw → calibrated probabilities using the test set. This improves betting decisions because:
- A bet at 0.70 calibrated probability has ~70% expected win rate
- Proper calibration enables accurate EV calculations

### Why Disable ASSISTS/THREES?

Backtesting revealed:
- **ASSISTS**: 14.6% win rate (worse than random)
- **THREES**: 46.5% win rate (losing after vig)

Rather than deploy models that lose money, these markets are disabled. Potential causes:
- Higher variance in these stats
- Lines are sharper (less inefficiency to exploit)
- Feature set may not capture relevant signals

### Why Multi-Book Line Shopping?

Different sportsbooks post different lines for the same player:

```
Stephen Curry POINTS:
  Underdog:   27.5
  FanDuel:    28.5
  BetMGM:     29.5
```

If the model predicts 31.5 points, betting Underdog (27.5) provides 4 points of edge vs. BetMGM (29.5) which only provides 2 points.

Backtest showed line shopping improves ROI by +6.8% vs. betting consensus.

---

## Tracing a Prediction End-to-End

**Scenario**: Generate prediction for LeBron James REBOUNDS on Jan 15, 2026

### 1. Data Ingestion (morning)

```bash
./nba/nba-predictions.sh morning
```

- Fetches props from BettingPros API
- Loads to `nba_prop_lines` table
- Result: 7 sportsbooks have LeBron REBOUNDS lines ranging from 7.5 to 8.5

### 2. Feature Extraction

```python
# In generate_xl_predictions.py
extractor = LiveFeatureExtractorXL()
features = extractor.extract(
    player_name="LeBron James",
    game_date="2026-01-15",
    opponent="GSW",
    is_home=True,
    stat_type="REBOUNDS"
)
# Returns 166-dim vector
```

Key features extracted:
- `ema_rebounds_L3`: 8.2 (last 3 games EMA)
- `ema_rebounds_L10`: 7.8 (last 10 games EMA)
- `opp_reb_allowed`: 44.2 (GSW allows 44.2 reb/game)
- `line_spread`: 1.0 (8.5 - 7.5)
- `num_books`: 7

### 3. Model Inference

```python
# In xl_predictor.py
predictor = XLPredictor('rebounds')
prediction, p_over, side, edge = predictor.predict(features, line=7.5)

# Results:
# prediction = 8.34 (regressor)
# p_over = 0.72 (calibrated)
# side = "OVER"
# edge = 0.84 points
```

### 4. Filtering

```python
# In line_optimizer.py
# Rebounds filter: p_over >= 0.65 AND (edge >= 1.0 OR spread >= 2.5)

p_over = 0.72    # ✓ >= 0.65
edge = 0.84      # ✗ < 1.0
spread = 1.0     # ✗ < 2.5

# FILTERED OUT (doesn't meet edge OR spread condition)
```

This prop would NOT appear in the output because it doesn't pass the filter threshold.

### 5. If it passed:

```json
{
  "player_name": "LeBron James",
  "stat_type": "REBOUNDS",
  "side": "OVER",
  "prediction": 8.34,
  "p_over": 0.72,
  "best_book": "fanduel",
  "best_line": 7.5,
  "edge": 0.84
}
```

---

## Monitoring and Safeguards

### Daily Performance Check

```python
# In nba/betting_xl/config/production_policies.py

STOP_LOSS_CONFIG = {
    "consecutive_losing_days": 3,    # Pause after 3 red days
    "min_win_rate_7d": 0.50,         # Alert if WR drops below 50%
    "max_bets_per_day": 10           # Prevent overexposure
}
```

### Data Freshness Validation

```python
# In nba/betting_xl/config/data_freshness_validator.py

def validate_props_freshness(game_date):
    # Props must be fetched within 6 hours
    # Prevents betting on stale lines
    ...
```

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `nba/nba-predictions.sh` | Pipeline orchestrator (morning/evening workflows) |
| `nba/betting_xl/generate_xl_predictions.py` | Main prediction generator |
| `nba/betting_xl/xl_predictor.py` | Model loading and inference |
| `nba/betting_xl/line_optimizer.py` | Bet filtering logic |
| `nba/betting_xl/validate_xl_models.py` | Historical backtesting |
| `nba/features/extract_live_features_xl.py` | Feature extraction (166 features) |
| `nba/features/build_xl_training_dataset.py` | Training data builder |
| `nba/models/train_market.py` | Model training script |
| `nba/models/MODEL_REGISTRY.toml` | Validation history and metrics |
