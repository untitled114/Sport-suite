# NBA Player Props ML System

End-to-end machine learning pipeline for NBA player prop betting. Ingests live odds from 7 sportsbooks, extracts 102 features per prop, and generates calibrated predictions using a two-head LightGBM architecture.

## What This Project Demonstrates

- **ML Pipeline Design**: Feature extraction, model training, probability calibration
- **Data Engineering**: Multi-source ingestion, PostgreSQL schema design, automated pipelines
- **Production Thinking**: Validation methodology, stop-loss safeguards, monitoring

## Results (Honest Framing)

### Backtested Performance

| Metric | Value | Sample | Notes |
|--------|-------|--------|-------|
| **Backtest Win Rate** | 75.0% | 28 bets | 8-day holdout (Oct 30 - Nov 7, 2024) |
| **Backtest ROI** | +43.2% | 28 bets | Temporal split, no lookahead |
| **Line Shopping Lift** | +6.8% | 761 bets | vs. consensus baseline |

**Important caveats:**
- Backtesting ≠ live performance. Markets adapt, edges decay.
- Small sample sizes (28 bets) have high variance. True edge likely lower.
- Results are exploratory. No claim of guaranteed profitability.

### Methodology

```
Training:   Oct 2023 - Apr 2025 (~24,000 props per market)
Validation: Oct 30 - Nov 7, 2024 (8 days, 28 filtered bets)
Split:      Temporal (no future data leakage)
```

Full validation details: [`nba/models/MODEL_REGISTRY.toml`](nba/models/MODEL_REGISTRY.toml)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           DATA INGESTION                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                   │
│  │ BettingPros  │  │ ESPN API     │  │ NBA Stats    │                   │
│  │ (7 books)    │  │ (schedule)   │  │ (box scores) │                   │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                   │
│         └─────────────────┼─────────────────┘                           │
│                           ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    PostgreSQL (4 databases)                      │   │
│  │  nba_players:5536  nba_games:5537  nba_team:5538  nba_intel:5539│   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        FEATURE EXTRACTION (102 features)                │
│                                                                         │
│  Player (78)              Book (20)              Computed (4)           │
│  ├─ Rolling stats L3/5/10 ├─ Line spread         ├─ is_home             │
│  ├─ Team pace/ratings     ├─ Book deviations     ├─ line                │
│  ├─ Matchup history       ├─ Consensus variance  ├─ opponent_team       │
│  └─ Rest days, B2B        └─ Softest/hardest     └─ expected_diff       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         TWO-HEAD MODEL                                  │
│                                                                         │
│    ┌────────────────────┐        ┌────────────────────┐                │
│    │  Regressor Head    │        │  Classifier Head   │                │
│    │  (LightGBM)        │        │  (LightGBM)        │                │
│    │                    │        │                    │                │
│    │  Input: 102 feat   │───────▶│  Input: 102 feat   │                │
│    │  Output: predicted │  diff  │  + expected_diff   │                │
│    │          value     │        │  Output: P(OVER)   │                │
│    └────────────────────┘        └─────────┬──────────┘                │
│                                            │                            │
│                                            ▼                            │
│                              ┌────────────────────┐                     │
│                              │  Isotonic          │                     │
│                              │  Calibration       │                     │
│                              └─────────┬──────────┘                     │
│                                        │                                │
│                                        ▼                                │
│                           Calibrated P(OVER) → Bet decision             │
└─────────────────────────────────────────────────────────────────────────┘
```

For detailed architecture: [`SYSTEM_DESIGN.md`](SYSTEM_DESIGN.md)

---

## File Map

```
nba/
├── betting_xl/
│   ├── generate_xl_predictions.py   # Main prediction generator
│   ├── xl_predictor.py              # Model loading + inference
│   ├── validate_xl_models.py        # Backtest on historical data
│   ├── line_optimizer.py            # Bet filtering logic
│   │
│   ├── predictions/                 # Daily picks output
│   │   └── xl_picks_YYYY-MM-DD.json # ← Model inference results
│   ├── fetchers/                    # Data collection
│   │   └── fetch_bettingpros.py     # 7-book prop fetcher
│   └── logs/                        # Pipeline execution logs
│
├── models/
│   ├── saved_xl/                    # Production model files
│   │   ├── points_xl_regressor.pkl  # Regressor head
│   │   ├── points_xl_classifier.pkl # Classifier head
│   │   ├── points_xl_calibrator.pkl # Isotonic calibration
│   │   ├── points_xl_features.pkl   # Feature names (102)
│   │   └── points_xl_metadata.json  # Training metrics
│   ├── train_market.py              # Model training script
│   └── MODEL_REGISTRY.toml          # Validation history
│
├── features/
│   ├── build_xl_training_dataset.py # Creates training CSVs
│   ├── extract_live_features_xl.py  # Live feature extraction
│   └── datasets/                    # Training data (95k props)
│
└── nba-predictions.sh               # Pipeline orchestrator
```

---

## Concrete Example

**Input**: Stephen Curry POINTS prop, Jan 7, 2026

```
Player:   Stephen Curry
Market:   POINTS
Opponent: MIL (home game)
Lines:    Underdog=27.5, FanDuel=28.5, BetMGM=29.5, ...
```

**Model Output**:

```json
{
  "player_name": "Stephen Curry",
  "stat_type": "POINTS",
  "side": "OVER",
  "prediction": 31.46,           // Regressor predicts 31.5 points
  "p_over": 0.740,               // 74% probability of going over
  "best_book": "underdog",       // Softest line found
  "best_line": 27.5,             // 4 points below prediction
  "edge": 3.96,                  // Points of edge
  "reasoning": "Model predicts 31.5 vs softest line 27.5"
}
```

**Interpretation**:
- Model expects Curry to score ~31.5 points
- Underdog has him at 27.5 (4 points lower than other books)
- This creates a 14.4% edge opportunity
- Bet OVER at Underdog if it passes filter thresholds

Full schema: [`docs/PREDICTION_SCHEMA.md`](docs/PREDICTION_SCHEMA.md)

---

## Validation Methodology

### Data Leakage Prevention

```python
# From validate_xl_models.py:9
# CRITICAL: Prevents data leakage by extracting features AS OF historical game date.
```

1. **Temporal Split**: Training data ends before validation period begins
2. **Point-in-Time Features**: Rolling stats computed only with games before prediction date
3. **No Lookahead**: Actual results loaded separately after predictions generated

### How to Reproduce

```bash
# Run validation on historical holdout
python3 nba/betting_xl/validate_xl_models.py \
  --start-date 2024-10-30 \
  --end-date 2024-11-07

# Output: Market-by-market win rates, ROI, edge tier breakdowns
```

### Validation Results

```
Period: Oct 30 - Nov 7, 2024 (8 days)
Strategy: Hybrid dual-filter (p_over >= 0.65 AND edge conditions)

Market     Bets    Wins    Win Rate    Notes
─────────────────────────────────────────────
POINTS     18      12      66.7%       Strong
REBOUNDS   10      9       90.0%       Excellent (small sample)
ASSISTS    -       -       DISABLED    14.6% WR in testing
THREES     -       -       DISABLED    46.5% WR in testing
─────────────────────────────────────────────
TOTAL      28      21      75.0%       +43.2% ROI
```

---

## Quick Start

### Prerequisites
- Python 3.10+
- Docker (for PostgreSQL databases)
- BettingPros API key (for live props)

### Setup

```bash
# 1. Clone and configure
git clone https://github.com/untitled114/nba-props-betting.git
cd nba-props-ml
cp .env.example .env
# Edit .env with your credentials

# 2. Start databases
cd docker && docker-compose up -d

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run predictions
./nba/nba-predictions.sh evening
```

### Daily Workflow

```bash
# Morning: Fetch latest props + update stats
./nba/nba-predictions.sh morning

# Evening: Generate predictions
./nba/nba-predictions.sh evening

# Output: nba/betting_xl/predictions/xl_picks_YYYY-MM-DD.json
```

---

## Technical Details

### Model Architecture

| Component | Implementation |
|-----------|----------------|
| Regressor | LightGBM (predicts stat value) |
| Classifier | LightGBM (predicts P(OVER)) |
| Calibration | IsotonicRegression |
| Blending | 60% classifier + 40% residual |

### Training Metrics (POINTS market)

| Metric | Train | Test |
|--------|-------|------|
| RMSE | 6.13 | 6.84 |
| MAE | - | 5.23 |
| R² | - | 0.41 |
| AUC | 0.96 | 0.77 |

### Feature Engineering

- **78 player features**: EMA-weighted rolling stats (L3/L5/L10/L20), team context, matchup history
- **20 book features**: Line spread, variance, per-book deviations from consensus
- **4 computed features**: is_home, line, opponent_team, expected_diff

Full feature list: [`nba/features/README.md`](nba/features/README.md)

---

## Technologies

- **ML**: LightGBM, scikit-learn, pandas
- **Database**: PostgreSQL 15 (4 databases, Docker)
- **Data Sources**: BettingPros API (7 books), ESPN API, NBA Stats
- **Pipeline**: Bash orchestration, Python data processing

---

## Why Certain Design Choices

**Why two-head architecture?**
Separating value prediction (regressor) from probability estimation (classifier) allows better calibration. The regressor learns the expected stat value; the classifier learns when the model's edge is actually exploitable.

**Why disable ASSISTS/THREES?**
Backtesting showed these markets underperform (ASSISTS: 14.6% WR, THREES: 46.5% WR). Rather than deploy losing models, they're disabled pending retraining.

**Why Isotonic Calibration?**
Raw classifier probabilities are overconfident. Isotonic regression maps them to empirically accurate probabilities.

---

## License

MIT License

---

**Author**: [@untitled114](https://github.com/untitled114)
