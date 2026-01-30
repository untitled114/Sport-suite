# NBA Player Props ML System

End-to-end machine learning pipeline for NBA player prop betting. Ingests live odds from 7 sportsbooks, extracts 166 features per prop, and generates calibrated predictions using stacked LightGBM architectures.

## What This Project Demonstrates

- **ML Pipeline Design**: Feature extraction, model training, probability calibration
- **Data Engineering**: Multi-source ingestion, PostgreSQL schema design, automated pipelines
- **Production Thinking**: Validation methodology, filtering strategies, monitoring

## Results (January 2026)

### Live Performance

| Metric | Value | Sample | Notes |
|--------|-------|--------|-------|
| **Win Rate** | 60.3% | 232 bets | Jan 1-28, 2026 (filtered picks) |
| **ROI** | +15.2% | 232 bets | At standard -110 odds |
| **Units Profit** | +35.26u | 28 days | ~8 picks/day average |

### By Market

| Market | Bets | Win Rate | ROI |
|--------|------|----------|-----|
| POINTS | 115 | 60.0% | +14.5% |
| REBOUNDS | 89 | 59.6% | +13.7% |
| ASSISTS | 28 | 64.3% | +22.7% |

**Important caveats:**
- Results from filtered picks only (edge thresholds, probability filters)
- Past performance does not guarantee future results
- Markets adapt; edges can decay over time

### Methodology

```
Training:   Oct 2023 - Jan 2025 (~24,000 props per market)
Validation: Jan 1-28, 2026 (232 filtered bets)
Split:      Temporal (no future data leakage)
```

---

## Model Architecture

### Production Models (Jan 11, 2026)

| Market | Features | R² | AUC | Architecture |
|--------|----------|-----|-----|--------------|
| **POINTS** | 166 | 0.537 | 0.736 | Two-head stacked |
| **REBOUNDS** | 166 | 0.531 | 0.748 | Two-head stacked |
| **ASSISTS** | 102 | - | - | Legacy XL (disabled) |
| **THREES** | 102 | - | - | Legacy XL (disabled) |

### Feature Breakdown (166 features)

```
Player Rolling Stats (42):
├── EMA-weighted L3/L5/L10/L20 for 9 stats
├── Plus/minus, FT rate, true shooting
└── Points per minute, momentum, efficiency

Team & Game Context (28):
├── Team/opponent pace, ratings
├── Rest days, B2B, games in L7
├── Travel distance, altitude, season phase
└── Starter flag, position, teammate usage

Head-to-Head History (36):
├── H2H averages for points/rebounds/assists/threes
├── L3/L5/L10/L20 windows per stat
├── Home/away splits
└── Recency weight, sample quality

Book Disagreement (22):
├── Line spread, consensus, std dev
├── Per-book deviations (8 books)
├── Softest/hardest book IDs
└── Books agree/disagree flags

Prop History (12):
├── Hit rates: L20, context, vs defense
├── Line vs season avg, percentile
└── Bayesian confidence, streaks

Vegas & BettingPros (17):
├── Vegas total, spread
├── Team ATS/OU percentages
└── BP projections, ratings, hit rates

Computed (1):
└── expected_diff (regressor prediction - line)
```

### Two-Head Stacked Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      TWO-HEAD STACKED MODEL                             │
│                                                                         │
│    ┌────────────────────┐        ┌────────────────────┐                │
│    │  HEAD 1: Regressor │        │  HEAD 2: Classifier │                │
│    │  (LightGBM)        │        │  (LightGBM)         │                │
│    │                    │        │                     │                │
│    │  Input: 166 feat   │───────▶│  Input: 166 feat    │                │
│    │  Output: predicted │  diff  │  + expected_diff    │                │
│    │          value     │        │  Output: P(OVER)    │                │
│    └────────────────────┘        └─────────┬──────────┘                │
│                                            │                            │
│                                            ▼                            │
│                              ┌────────────────────┐                     │
│                              │  Isotonic          │                     │
│                              │  Calibration       │                     │
│                              └─────────┬──────────┘                     │
│                                        │                                │
│                                        ▼                                │
│                    Blended: 60% classifier + 40% residual               │
└─────────────────────────────────────────────────────────────────────────┘
```

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
```

---

## File Map

```
nba/
├── betting_xl/
│   ├── xl_predictor.py              # Model loading + inference
│   ├── line_optimizer.py            # Bet filtering logic
│   ├── validate_predictions.py      # Results validation (all systems)
│   ├── validate_xl_models.py        # Historical model backtesting
│   ├── predictions/                 # Daily picks output
│   └── fetchers/                    # Data collection (7 books)
│       ├── fetch_bettingpros.py     # Main odds fetcher
│       ├── fetch_cheatsheet.py      # BettingPros projections
│       ├── fetch_pick6_live.py      # Odds API integration
│       └── fetch_vegas_lines.py     # Vegas totals/spreads
│
├── models/
│   ├── saved_xl/                    # Production models
│   │   ├── {market}_xl_*.pkl        # 102-feature XL models (CURRENT PRODUCTION)
│   │   ├── {market}_v3_*.pkl        # 166-feature V3 models (available, not deployed)
│   │   └── {market}_matchup_*.pkl   # 44-feature matchup add-on models
│   ├── train_market.py              # Model training
│   └── model_cards/                 # Model documentation
│
├── features/
│   ├── extract_live_features_xl.py  # 166-feature extraction
│   ├── extractors/                  # Modular feature extractors
│   │   ├── book_features.py         # Book disagreement features
│   │   ├── h2h_features.py          # Head-to-head features
│   │   └── prop_history_features.py # Prop history features
│   └── datasets/                    # Training data
│
├── core/
│   ├── schemas.py                   # Pydantic validation
│   ├── exceptions.py                # Custom exception hierarchy
│   ├── drift_detection.py           # Feature drift monitoring
│   ├── drift_service.py             # Production drift service
│   ├── experiment_tracking.py       # MLflow integration
│   └── logging_config.py            # Structured logging
│
├── config/
│   ├── thresholds.py                # Betting thresholds config
│   ├── constants.py                 # Shared constants
│   └── database.py                  # Database connection config
│
└── nba-predictions.sh               # Pipeline orchestrator
```

---

## Quick Start

### Prerequisites
- Python 3.10+
- Docker (for PostgreSQL databases)
- BettingPros API key

### Setup

```bash
# Clone and configure
git clone https://github.com/untitled114/Sport-suite.git
cd Sport-suite
cp .env.example .env  # Add your credentials

# Start databases
cd docker && docker-compose up -d

# Install
pip install -e ".[dev]"

# Run predictions
./nba/nba-predictions.sh evening
```

### Validate Results

```bash
# Using the CLI (recommended)
./nba/nba-predictions.sh validate           # Yesterday's results
./nba/nba-predictions.sh validate --7d      # Last 7 days
./nba/nba-predictions.sh validate --30d     # Last 30 days

# Or directly with Python
cd nba/betting_xl
python3 validate_predictions.py --start-date 2026-01-01 --end-date 2026-01-28
```

---

## Technical Details

### Training Metrics (Jan 11, 2026)

**POINTS Market:**
| Metric | Train | Test |
|--------|-------|------|
| RMSE | 4.78 | 6.13 |
| MAE | - | 4.55 |
| R² | - | 0.537 |
| AUC | 0.80 | 0.736 |

**REBOUNDS Market:**
| Metric | Train | Test |
|--------|-------|------|
| RMSE | 1.80 | 2.42 |
| MAE | - | 1.79 |
| R² | - | 0.531 |
| AUC | 0.83 | 0.748 |

### Data Sources

- **BettingPros API**: 7 sportsbooks (DraftKings, FanDuel, BetMGM, Caesars, BetRivers, ESPNBet, Underdog)
- **ESPN API**: Game schedules, matchup data (fallback for BettingPros)
- **NBA Stats**: Box scores, player game logs

---

## Engineering Highlights

- **512 tests** with pytest (unit + integration), 96% coverage
- **Pydantic schemas** for data validation
- **Custom exception hierarchy** for error handling
- **Feature drift detection** with KS tests and z-score monitoring
- **MLflow integration** for experiment tracking (optional)
- **Pre-commit hooks** (black, isort, flake8, bandit)
- **GitHub Actions CI/CD** for automated testing
- **Airflow DAGs** for pipeline orchestration

---

## License

MIT License

---

**Author**: [@untitled114](https://github.com/untitled114)
