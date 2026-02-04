# NBA Player Props ML System

![Tests](https://github.com/untitled114/Sport-suite/actions/workflows/test.yml/badge.svg)
![Lint](https://github.com/untitled114/Sport-suite/actions/workflows/lint.yml/badge.svg)
![Security](https://github.com/untitled114/Sport-suite/actions/workflows/security.yml/badge.svg)
![Deploy](https://github.com/untitled114/Sport-suite/actions/workflows/deploy.yml/badge.svg)
![Coverage](https://img.shields.io/badge/coverage-70%25+-brightgreen)
![Python](https://img.shields.io/badge/python-3.10+-blue)

End-to-end machine learning pipeline for NBA player prop betting. Ingests live odds from 10 sources (7 sportsbooks + 3 DFS platforms), extracts features per prop (102 for XL, 136 for V3), and generates calibrated predictions using stacked LightGBM architectures.

## What This Project Demonstrates

- **ML Pipeline Design**: Feature extraction, model training, probability calibration
- **Data Engineering**: Multi-source ingestion, PostgreSQL schema design, automated pipelines
- **Production Thinking**: Validation methodology, drift monitoring, CI/CD, Airflow orchestration

---

## Results (February 2026)

### Filtered Backtest Performance (Jan 19 - Feb 2, 2026)

| Metric | With Goblin Lines | Standard Only |
|--------|-------------------|---------------|
| **Picks** | 47 | 17 |
| **Win Rate** | **79.5%** | 70.6% |
| **POINTS** | 74.2% | 64.3% |
| **REBOUNDS** | 92.3% | 100% |

**Key Finding:** PrizePicks alternate lines ("goblin") improve performance despite not being in training data. See [Case Study: Goblin Lines](docs/CASE_STUDY_GOBLIN_LINES.md).

### Walk-Forward Cross-Validation (3-Fold)

| Metric | POINTS | Notes |
|--------|--------|-------|
| Mean Accuracy | 67.0% | Model baseline |
| Mean AUC | 0.751 | Consistent across folds |
| AUC Std Dev | 0.027 | Low variance = stable |

**Fold-by-fold:** 69.2% → 68.9% → 63.0% (slight degradation over time, expected)

### Decision Framework

```
Walk-Forward CV: ~67% accuracy (model baseline)
+ Filters:       ~70% WR (high-confidence picks)
+ Soft Lines:    ~80% WR (maximized edge + volume)
```

**Active Markets**: POINTS & REBOUNDS only (ASSISTS/THREES disabled due to poor AUC)

---

## Model Architecture

### Production Models

Both XL and V3 models run in parallel. Picks include `model_version: "xl"` or `"v3"`.

| Model | Market | Features | R² | AUC | Trained | Status |
|-------|--------|----------|-----|-----|---------|--------|
| **XL** | POINTS | 102 | 0.410 | 0.767 | Dec 2025 | ✅ DEPLOYED |
| **XL** | REBOUNDS | 102 | 0.403 | 0.749 | Dec 2025 | ✅ DEPLOYED |
| **V3** | POINTS | 136 | 0.548 | 0.740 | Feb 2026 | ✅ DEPLOYED |
| **V3** | REBOUNDS | 136 | 0.530 | 0.739 | Feb 2026 | ✅ DEPLOYED |

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

**V3 Model (136 features):**
- All XL features (102) plus:
- Season/Temporal (6): days_into_season, season_phase, early/mid/late/playoffs flags
- Volatility (8): stat std devs, trend ratios, usage volatility
- H2H Decay (5): decayed averages, trends, recency-adjusted matchup stats
- Line/Book (9): line movement, consensus strength, softest book metrics
- Matchup (6): efficiency_vs_context, game_velocity, resistance_adjusted

---

## Data Sources

### Sportsbooks (via BettingPros API)
- DraftKings, FanDuel, BetMGM, Caesars, BetRivers, ESPNBet, Fanatics

### DFS Platforms (Softer Lines)
- **Underdog Fantasy** - Standard DFS lines
- **PrizePicks** - Standard lines
- **PrizePicks Goblin** - Lower lines (easier threshold, validated 85% WR)
- **PrizePicks Demon** - Higher lines (harder threshold)

### Supporting Data
- **ESPN API**: Game schedules, matchup data
- **NBA Stats**: Box scores, player game logs

---

## Production Deployment

### Server
- **Hetzner VPS**: `5.161.239.229`
- **User**: `sportsuite`
- **Path**: `/home/sportsuite/sport-suite`

### Airflow Pipelines

| DAG | Schedule (EST) | Purpose |
|-----|----------------|---------|
| `nba_full_pipeline` | 9:00 AM | Full data collection + predictions |
| `nba_validation_pipeline` | 9:30 AM | Validate yesterday's picks |
| `nba_health_check` | Every 6h | System health monitoring |

### Discord Bot (Cephalon Axiom)

| Command | Description |
|---------|-------------|
| `/nba` | Show today's picks summary |
| `/nba-detail` | Detailed pick cards with lines |
| `/nba-refresh` | Quick refresh (~1 min) |
| `/nba-run` | Full pipeline run (~5 min) |

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

# 1. Create environment file
cp .env.example .env
nano .env  # Set DB_PASSWORD, BETTINGPROS_API_KEY, ODDS_API_KEY

# 2. Load environment variables
source .env && export DB_USER DB_PASSWORD

# 3. Install and start
make install
make db-up

# Or manually:
pip install -e ".[dev]"
cd docker && docker-compose up -d
```

### Common Commands

```bash
make run          # Full pipeline (data + predictions)
make refresh      # Quick refresh (line movements only)
make picks        # Show current picks
make validate     # Validate yesterday's results
make validate-7d  # Validate last 7 days

make test         # Run tests with coverage
make lint         # Run linters
make deploy       # Deploy to production server
```

---

## Project Structure

```
nba/
├── betting_xl/                 # Core prediction system
│   ├── xl_predictor.py         # Model loading + inference
│   ├── line_optimizer.py       # Line shopping + filtering
│   ├── generate_xl_predictions.py  # Main prediction generator
│   ├── run_historical_backtest.py  # Walk-forward backtesting
│   ├── validate_predictions.py # Results validation
│   ├── show_picks.py           # CLI pick display
│   └── fetchers/               # Data collection (10 sources)
│
├── models/
│   ├── saved_xl/               # Production models (XL + V3)
│   ├── train_market.py         # Model training script
│   └── model_cards/            # SHAP documentation
│
├── features/
│   ├── extract_live_features_xl.py  # Feature extraction
│   ├── build_xl_training_dataset.py # Dataset builder
│   └── extractors/             # Modular extractors
│
├── config/                     # Centralized configuration
├── core/                       # Utilities (drift, logging, exceptions)
└── nba-predictions.sh          # Pipeline orchestrator

airflow/                        # DAG definitions
docker/                         # Database containers
docs/                           # ADRs, case studies
tests/                          # Unit + integration tests
```

---

## Engineering Practices

### CI/CD (GitHub Actions)
- **Lint**: black, isort, flake8, mypy
- **Tests**: pytest with 70%+ coverage
- **Security**: Gitleaks, Trivy, Bandit scanning
- **Deploy**: Auto-deploy to production on push to main

### Data Quality
- Feature drift detection (KS tests, z-score monitoring)
- Pydantic schemas for runtime validation
- Pre-training data quality checks

### Architecture
- Centralized config (frozen dataclasses, no magic numbers)
- Modular feature extractors (dependency injection)
- Custom exception hierarchy
- Conventional commits

---

## Documentation

- [Case Study: Goblin Lines](docs/CASE_STUDY_GOBLIN_LINES.md) - Training data mismatch analysis
- [Architecture Decision Records](docs/adr/) - Design decisions
- [Database Audit](docs/DATABASE_AUDIT.md) - Schema documentation
- [CLAUDE.md](.claude/CLAUDE.md) - Complete system reference

---

## License

MIT License

---

**Author**: [@untitled114](https://github.com/untitled114)
