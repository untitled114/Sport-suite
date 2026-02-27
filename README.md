# NBA Player Props ML System

![Tests](https://github.com/untitled114/Sport-suite/actions/workflows/test.yml/badge.svg)
![Lint](https://github.com/untitled114/Sport-suite/actions/workflows/lint.yml/badge.svg)
![Security](https://github.com/untitled114/Sport-suite/actions/workflows/security.yml/badge.svg)
![Deploy](https://github.com/untitled114/Sport-suite/actions/workflows/deploy.yml/badge.svg)
![Coverage](https://img.shields.io/badge/coverage-70%25+-brightgreen)
![Python](https://img.shields.io/badge/python-3.10+-blue)

End-to-end machine learning pipeline for NBA player prop predictions. Ingests live data from 10 REST API sources, stores it across a 5-database PostgreSQL architecture, engineers 136 features per prediction, and generates calibrated probabilities using stacked LightGBM models — all orchestrated via Airflow with automated validation and drift monitoring.

## What This Project Demonstrates

- **Data Pipeline Development**: Multi-source ETL from 10 REST APIs into PostgreSQL (5 databases) and MongoDB, with deduplication, schema enforcement, and incremental loads
- **SQL & Data Modeling**: Normalized relational schema across player, game, team, and intelligence databases; dbt transformation layer (staging → intermediate → marts)
- **Data Quality & Governance**: Feature drift detection (KS tests, z-score monitoring), Pydantic schema validation, automated result verification, data freshness checks
- **Performance Optimization**: Indexed query patterns, parallel feature extraction via ThreadPoolExecutor, batched dataset builds, incremental prop loading with upsert logic
- **Orchestration & Monitoring**: 4 Airflow DAGs with retry logic, health checks, stop-loss safeguards, and structured logging
- **ML Integration**: Stacked two-head LightGBM (regressor + classifier) with isotonic probability calibration, walk-forward cross-validation, and ensemble blending

---

## Architecture Overview

```
10 REST APIs (BettingPros, ESPN, NBA Stats, Underdog, PrizePicks, ...)
        │
        ▼
┌──────────────────────┐
│    Fetcher Layer      │  nba/betting_xl/fetchers/
│    (10 sources)       │  Rate limiting, retries, normalization
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│    Loader Layer       │  nba/betting_xl/loaders/
│    (Upsert + Dedup)   │  PostgreSQL + MongoDB dual-write
└──────────┬───────────┘
           ▼
┌──────────────────────────────────────────────────┐
│  PostgreSQL (5 databases)         MongoDB         │
│  ├── nba_players:5536             nba_betting_xl  │
│  ├── nba_games:5537                               │
│  ├── nba_team:5538                                │
│  └── nba_intelligence:5539                        │
└──────────┬───────────────────────────────────────┘
           ▼
┌──────────────────────┐
│  dbt Transformations  │  staging → intermediate → marts
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  Feature Engineering  │  136 features per prediction
│  (6 modular extractors)│  Player, Book, H2H, Vegas, Team, Prop History
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  Stacked LightGBM     │  Regressor (predicted value)
│  Two-Head Model       │  + Classifier (P(OVER))
│  + Isotonic Calibration│  + Ensemble blending (60/40)
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  Calibrated Predictions│  → JSON output
│  + Automated Validation│  → Result tracking
└──────────────────────┘
```

---

## Results (February 2026)

### Filtered Backtest Performance (Jan 19 - Feb 2, 2026)

| Metric | Combined | Standard Only |
|--------|----------|---------------|
| **Picks** | 47 | 17 |
| **Win Rate** | **79.5%** | 70.6% |
| **POINTS** | 74.2% | 64.3% |
| **REBOUNDS** | 92.3% | 100% |

### Walk-Forward Cross-Validation (3-Fold)

| Metric | POINTS | Notes |
|--------|--------|-------|
| Mean Accuracy | 67.0% | Model baseline |
| Mean AUC | 0.751 | Consistent across folds |
| AUC Std Dev | 0.027 | Low variance = stable |

**Active Markets**: POINTS & REBOUNDS only (ASSISTS/THREES disabled — see [ADR-004](docs/adr/ADR-004-disabling-assists-threes-markets.md))

---

## Model Architecture

### Production Models

Both XL and V3 models run in parallel with independent feature sets.

| Model | Market | Features | R² | AUC | Trained | Status |
|-------|--------|----------|-----|-----|---------|--------|
| **XL** | POINTS | 102 | 0.410 | 0.767 | Dec 2025 | Deployed |
| **XL** | REBOUNDS | 102 | 0.403 | 0.749 | Dec 2025 | Deployed |
| **V3** | POINTS | 136 | 0.548 | 0.740 | Feb 2026 | Deployed |
| **V3** | REBOUNDS | 136 | 0.530 | 0.739 | Feb 2026 | Deployed |

### Two-Head Stacked Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      TWO-HEAD STACKED MODEL                             │
│                                                                         │
│    ┌────────────────────┐        ┌────────────────────┐                │
│    │  HEAD 1: Regressor │        │  HEAD 2: Classifier │                │
│    │  (LightGBM)        │        │  (LightGBM)         │                │
│    │                    │        │                     │                │
│    │  Input: N features │───────▶│  Input: N features  │                │
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

### Feature Breakdown

**XL Model (102 features):**
- Player rolling stats (78): EMA L3/L5/L10/L20 for points, rebounds, assists, etc.
- Book disagreement (20): Line spread, per-book deviations, softest/hardest tracking
- Computed (4): is_home, line, opponent_team, expected_diff

**V3 Model (136 features = 102 + 34 additional):**
- Season/Temporal (6): days_into_season, season_phase, early/mid/late/playoffs flags
- Volatility (8): stat std devs, trend ratios, usage volatility
- H2H Decay (5): decayed averages, trends, recency-adjusted matchup stats
- Line/Book (9): line movement, consensus strength, softest book metrics
- Matchup (6): efficiency_vs_context, game_velocity, resistance_adjusted

---

## Data Sources

### Sportsbooks (via BettingPros Premium API)
- DraftKings, FanDuel, BetMGM, Caesars, BetRivers, ESPNBet, Fanatics

### DFS Platforms
- Underdog Fantasy, PrizePicks (standard + alternate lines)

### Supporting Data
- **ESPN API**: Game schedules, matchup data
- **NBA Stats**: Box scores, player game logs

**Daily Volume**: ~2,600 props/day across 10 sources, loaded with upsert deduplication

---

## Orchestration (Airflow)

| DAG | Schedule (EST) | Purpose |
|-----|----------------|---------|
| `nba_full_pipeline` | 9:00 AM | Full ETL: fetch → load → enrich → predict |
| `nba_validation_pipeline` | 9:30 AM | Validate yesterday's picks against actuals |
| `nba_health_check` | Every 6h | Data freshness, model drift, DB connectivity |

---

## Quick Start

### Prerequisites
- Python 3.10+
- Docker (for PostgreSQL databases)
- API keys (BettingPros, Odds API)

### Setup

```bash
git clone https://github.com/untitled114/Sport-suite.git
cd Sport-suite

# 1. Create environment file
cp .env.example .env  # Set DB_PASSWORD, API keys

# 2. Install and start databases
make install    # pip install -e ".[dev]"
make db-up      # docker-compose up -d (5 PostgreSQL + MongoDB)

# 3. Run dbt transformations
cd dbt && dbt run && cd ..

# 4. Run the pipeline
make run        # Full pipeline (fetch + predict)
```

### Common Commands

```bash
make run          # Full pipeline (data + predictions)
make refresh      # Quick refresh (line movements only)
make picks        # Show current picks
make validate     # Validate yesterday's results
make validate-7d  # Validate last 7 days

make test         # Run tests with coverage
make lint         # Run linters (black, isort, flake8, mypy)
```

---

## Project Structure

```
nba/
├── betting_xl/                     # Core prediction system
│   ├── xl_predictor.py             # Model loading + inference
│   ├── line_optimizer.py           # Multi-book line shopping + filtering
│   ├── generate_xl_predictions.py  # Main prediction generator
│   ├── run_historical_backtest.py  # Walk-forward backtesting
│   ├── validate_predictions.py     # Results validation
│   ├── show_picks.py              # CLI pick display
│   ├── fetchers/                   # Data collection (10 API sources)
│   ├── loaders/                    # PostgreSQL + MongoDB loaders
│   └── config/                     # Production policies, freshness validation
│
├── models/
│   ├── saved_xl/                   # Production model artifacts (XL + V3)
│   ├── train_market.py             # Two-head stacked model training
│   ├── generate_feature_importance.py  # SHAP analysis
│   └── model_cards/                # Model documentation with SHAP plots
│
├── features/
│   ├── extract_live_features_xl.py # Live feature extraction (102/136 features)
│   ├── build_xl_training_dataset.py # Training dataset builder
│   └── extractors/                 # Modular extractors (book, h2h, vegas, team, prop)
│
├── config/                         # Centralized config (frozen dataclasses)
├── core/                           # Drift detection, logging, exceptions, schemas
└── scripts/                        # Data loaders (stats, injuries, teams)

dbt/                                # dbt transformation layer
├── models/
│   ├── staging/                    # Raw → cleaned (sources.yml)
│   ├── intermediate/               # Business logic transforms
│   └── marts/                      # Analytics-ready tables
└── README.md

notebooks/                          # Exploratory analysis
├── 01_feature_importance_analysis.ipynb
├── 02_validation_deep_dive.ipynb
├── 03_live_prediction_example.ipynb
└── model_retraining_guide.ipynb

airflow/                            # DAG definitions
docker/                             # Database containers + backup scripts
docs/                               # ADRs, case studies, schema docs
tests/                              # Unit + integration tests (70%+ coverage)
```

---

## Engineering Practices

### CI/CD (GitHub Actions)
- **Lint**: black, isort, flake8, mypy
- **Tests**: pytest with 70%+ coverage threshold
- **Security**: Gitleaks secret scanning, Trivy container scanning, Bandit static analysis
- **Deploy**: Automated deployment on push to main

### Data Quality
- Feature drift detection (KS tests, z-score monitoring)
- Pydantic schemas for runtime data validation
- Data freshness enforcement (stale prop rejection)
- Pre-training data quality checks

### Architecture
- Centralized config via frozen dataclasses (no magic numbers)
- Modular feature extractors with dependency injection
- Custom exception hierarchy for specific failure modes
- Conventional commits, Architecture Decision Records

---

## Documentation

- [System Design](SYSTEM_DESIGN.md) — End-to-end data flow with architecture diagrams
- [Architecture Decision Records](docs/adr/) — Design rationale (6 ADRs)
- [Database Audit](docs/DATABASE_AUDIT.md) — Schema documentation and data utilization analysis
- [dbt Transformations](dbt/README.md) — Staging, intermediate, and mart layer documentation
- [Notebooks](notebooks/) — Feature importance, validation deep-dive, live prediction examples
- [Validation Methodology](docs/VALIDATION.md) — Backtesting approach and metrics
- [Server Operations](docs/SERVER_OPERATIONS.md) — Deployment and monitoring runbook
---

## License

MIT License

---

**Author**: [@untitled114](https://github.com/untitled114)
