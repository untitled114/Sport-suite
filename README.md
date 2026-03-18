# Sport-Suite: NBA Player Props Prediction Platform

![Tests](https://github.com/untitled114/Sport-suite/actions/workflows/test.yml/badge.svg)
![Lint](https://github.com/untitled114/Sport-suite/actions/workflows/lint.yml/badge.svg)
![Security](https://github.com/untitled114/Sport-suite/actions/workflows/security.yml/badge.svg)
![Coverage](https://img.shields.io/badge/coverage-97%25-brightgreen)
![Python](https://img.shields.io/badge/python-3.10+-blue)

Production ML platform for NBA player prop predictions. Ingests live data from **11 sportsbook APIs** (4 direct + 7 via aggregator), stores across a **5-database PostgreSQL architecture**, engineers **188 features** per prediction using 10 modular extractors, and generates calibrated probabilities with stacked LightGBM models — orchestrated via 6 Airflow DAGs with automated validation, drift detection, and an AI-powered Discord bot (22 tools) for real-time intelligence.

**30-day performance: 69% WR, +32.3% ROI** (as of March 2026)

## What This Project Demonstrates

- **Multi-Source Data Engineering**: 11 REST API integrations (4 direct sportsbook scrapers via residential proxy + 7 via BettingPros), with append-only line snapshot tracking (15K+ snapshots/day), cross-source deduplication, and historical backfill (67K+ analytics records)
- **SQL & Data Modeling**: 5 PostgreSQL databases (3.4M+ prop lines, 108K game logs, 67K BP analytics, 5.7K DVP records), normalized schemas, migration scripts, backup automation
- **Feature Engineering at Scale**: 188 features across 10 modular extractors — player rolling stats, book disagreement signals, H2H matchups, BettingPros analytics (projections, EV, hit rates), DVP rankings, game context (pace, blowout risk), temporal milestones (trade deadline, All-Star break, team tenure)
- **ML Pipeline**: Two-head stacked LightGBM (regressor + classifier) with isotonic calibration, temporal decay weighting, walk-forward cross-validation, and auto-retraining triggers
- **Production Operations**: Multi-cloud deployment (AWS + GCP), 6 Airflow DAGs, systemd service management, rsync deployment, 97% test coverage (1,459 tests)
- **AI-Powered Intelligence**: Discord bot with 22 Claude-powered tools — natural language queries, pick explanation, line movement tracking, performance analytics, bankroll management

## Architecture

```
11 Sportsbook APIs                    BettingPros Analytics
├── DraftKings (direct, CO proxy)     ├── DVP (30 teams x 7 pos x 9 stats)
├── FanDuel (direct, CO proxy)        ├── League Trends (17 categories)
├── BetMGM (direct, CO proxy)         ├── Matchups (handle %, expert picks)
├── Underdog (direct, CO proxy)       └── Hit Rates, Projections, EV
├── PrizePicks (direct, FL proxy)
└── 7 books via BettingPros API
        │                                      │
        ▼                                      ▼
┌──────────────────────────────────────────────────────┐
│  Fetcher Layer (10+ sources)                         │
│  Rate limiting, proxy routing, retry, validation     │
│  → nba_props_xl (3.4M rows)                         │
│  → nba_line_snapshots (15K+/day, append-only)        │
│  → bp_historical_analytics (67K backfilled)          │
│  → bp_dvp_historical (5.7K, 3 seasons)               │
└──────────────────┬───────────────────────────────────┘
                   ▼
┌──────────────────────────────────────────────────────┐
│  Feature Engineering (188 features, 10 extractors)   │
│  ├── Player rolling stats (78)                       │
│  ├── Book disagreement (23)                          │
│  ├── H2H matchup (36)                               │
│  ├── BP analytics (15): projection, EV, hit rates    │
│  ├── Game context (8): pace, blowout, usage          │
│  ├── Temporal (10): trade deadline, team tenure       │
│  ├── Direct line (19): movement, cross-source        │
│  └── Vegas, team betting, prop history, cheatsheet    │
└──────────────────┬───────────────────────────────────┘
                   ▼
┌──────────────────────────────────────────────────────┐
│  Stacked LightGBM (Two-Head)                         │
│  HEAD 1: Regressor → predicted stat value            │
│  HEAD 2: Classifier → P(OVER) with expected_diff     │
│  Isotonic calibration → 60/40 ensemble blending      │
│  XL (102 feat) + V3 (136 feat) in parallel           │
└──────────────────┬───────────────────────────────────┘
                   ▼
┌──────────────────────────────────────────────────────┐
│  Conviction Engine                                    │
│  Aggregates signals across 7 daily pipeline runs     │
│  Tracks pick stability, line movement, BP alignment  │
│  Labels: LOCKED | STRONG | WATCH | SKIP              │
└──────────────────┬───────────────────────────────────┘
                   ▼
┌──────────────────────────────────────────────────────┐
│  Output                                               │
│  ├── xl_picks_{date}.json → Lunara Sports (GCP)      │
│  ├── nba_prediction_history → PostgreSQL              │
│  └── Cephalon Axiom → Discord (22 AI tools)          │
└──────────────────────────────────────────────────────┘
```

## Results (March 2026)

### 30-Day Rolling Performance

| Metric | Value |
|--------|-------|
| **Win Rate** | **69%** (88W-39L) |
| **ROI** | **+32.3%** |
| REBOUNDS | 100% WR (9-0 last 7 days) |
| XL Model | 83% WR (5-1) |
| V3 Model | 60% WR (12-8) |
| LOCKED Conviction | 70% WR |

### Production Models

| Model | Market | Features | R² | AUC | Trained | Status |
|-------|--------|----------|-----|-----|---------|--------|
| XL | POINTS | 102 | 0.410 | 0.767 | Dec 2025 | Deployed |
| XL | REBOUNDS | 102 | 0.403 | 0.749 | Dec 2025 | Deployed |
| V3 | POINTS | 136 | 0.548 | 0.740 | Feb 2026 | Deployed |
| V3 | REBOUNDS | 136 | 0.530 | 0.739 | Feb 2026 | Deployed |
| V4 | ALL | 188 | — | — | Training | Dataset building |

## Data Sources (11 APIs)

### Direct Sportsbook Fetchers (via Colorado/Florida residential proxy)
- **DraftKings** — `sportsbook-nash.draftkings.com` API
- **FanDuel** — `sbapi.il.sportsbook.fanduel.com` API
- **BetMGM** — CDS-API with fixture-view endpoint
- **Underdog Fantasy** — `api.underdogfantasy.com` (UUID-based)
- **PrizePicks** — Direct API via Florida proxy

### BettingPros Premium API (7 additional books)
DraftKings, FanDuel, BetMGM, Caesars, BetRivers, ESPNBet, Underdog

### BettingPros Analytics (server-rendered scraping)
- **Defense vs Position**: 30 teams x 7 positions x 9 stats (historical: 2023-2025)
- **League Trends**: 17 situational categories
- **Matchups**: Handle %, expert picks, line movement, EV

### Supporting Data
- **ESPN API**: Game schedules, live scores
- **NBA Stats**: Box scores, player game logs (108K+)

**Daily volume**: ~8,200 direct props + ~6,800 BettingPros props + 15K+ line snapshots

## Feature Engineering (188 Features)

| Extractor | Count | Description |
|-----------|-------|-------------|
| Player Rolling Stats | 78 | EMA L3/L5/L10/L20, minutes, FG%, plus/minus |
| Book Disagreement | 23 | Line spread, per-book deviations, softest/hardest |
| H2H Matchup | 36 | Head-to-head stats with time decay |
| BP Analytics | 15 | Projection, EV, bet rating, hit rates (L5/L10/L15/season), opposition rank |
| Game Context | 8 | Pace, blowout risk, plus/minus, usage proxy, scoring efficiency |
| Temporal Milestones | 10 | Trade deadline, All-Star break, playoff push, team tenure, trade detection |
| Direct Line | 19 | Line movement velocity, cross-source discrepancy, freshness |
| Prop History | 12 | Hit rates by context (home/away, defense, rest) |
| Vegas Context | 2 | Game total, spread |
| Team/Cheatsheet | 13 | Team betting ATS/O/U, BP cheatsheet data |

## Discord Bot (Cephalon Axiom)

AI-powered sports intelligence assistant with 22 tools, running on Claude Sonnet.

### Natural Language Intelligence
No slash commands needed. DM Axiom casually:
- *"How did we do?"* → Pulls yesterday's results
- *"Should I trust this Jokic pick?"* → Explains the pick with full context
- *"How have we done on Jokic?"* → Historical W/L track record
- *"Any line movement?"* → Real-time snapshot comparison across books
- *"What changed since this morning?"* → Pick evolution (added/dropped/changed)
- *"What's tomorrow?"* → ESPN schedule preview
- *"How's V3 POINTS doing?"* → Flexible performance breakdown

### Tool Categories

| Category | Tools | Description |
|----------|-------|-------------|
| Intelligence | 7 | explain_pick, player_track_record, performance_breakdown, line_movement (timeline), pick_evolution, tomorrow_preview, bankroll |
| Analytics | 2 | DVP lookup (defense vs position), BP analytics (projection/EV/hit rates) |
| Picks | 5 | Current picks, pipeline status, conviction, pick detail, prop lines |
| Database | 6 | Player stats, injuries, matchups, team stats, pick history, validation |
| Admin | 2 | Pipeline refresh, full pipeline run |

### Proactive Intelligence
- **Morning Brief** (7 AM): Yesterday's autopsy + rolling stats + today's outlook
- **Pipeline Alerts**: Line movements, pick evaporations, health issues
- **Injury Monitor**: Cross-references injuries with active picks
- **Pre-Game Brief**: T-60 mini-brief per game
- **Live Game Monitor**: Real-time pick tracking during games

## Orchestration (6 Airflow DAGs)

| DAG | Schedule (EST) | Purpose |
|-----|---------------|---------|
| `nba_full_pipeline` | Every 3hr, 2:30AM-8:30PM | Full ETL + predictions (T-120 gated) |
| `nba_refresh_pipeline` | On-demand | Quick line refresh (~3-4 min) |
| `nba_validation_pipeline` | 3:30 AM | Grade yesterday's picks (after game logs load) |
| `nba_daily_card` | Every 30min, 8AM-10:30PM | Conviction card delivery |
| `nba_health_check` | Every 6h | Data freshness, DB connectivity |
| `nba_retraining` | Triggered | Auto-retrain when 7-day WR < 60% |

## V4 Retraining System

Auto-retraining pipeline triggered when 7-day rolling win rate drops below 60%.

```
Performance Monitor (daily)
         │ WR < 60%?
         ▼
Build Training Dataset (expanding window + temporal decay)
         │ 60K+ props, 188 features
         ▼
Walk-Forward Validation (6 folds, 2-month test windows)
         │ Must beat current AUC + WR
         ▼
Shadow Mode (3 days parallel)
         │ Compare live picks
         ▼
Promote or Rollback
```

**Historical backfill**: 67,519 BP analytics records + 5,670 DVP values across 3 NBA seasons, enabling V4 training on the full 2023-2026 dataset.

## Infrastructure

### Multi-Cloud

| Cloud | Services |
|-------|----------|
| **AWS** (EC2) | ML pipeline, Airflow, 5 PostgreSQL DBs, FastAPI API, Axiom bot |
| **GCP** (Cloud Run) | Lunara Sports (real-time streaming platform) |
| **Vercel** | Lunara frontend (React) |

### Databases (5 PostgreSQL)

| Database | Port | Key Tables | Rows |
|----------|------|-----------|------|
| nba_players | 5536 | player_game_logs, player_profile | 108K |
| nba_games | 5537 | games, team_game_logs | 6.4K |
| nba_team | 5538 | teams, team_season_stats | 4 tables |
| nba_intelligence | 5539 | nba_props_xl, nba_line_snapshots, bp_historical_analytics, bp_dvp_historical | 3.4M+ |
| cephalon_axiom | 5541 | nba_prediction_history, axiom_conviction | 1K+ |

## Quick Start

```bash
git clone https://github.com/untitled114/Sport-suite.git
cd Sport-suite

# 1. Environment
cp .env.example .env  # Set DB_PASSWORD, BETTINGPROS_API_KEY

# 2. Install and start databases
make install    # pip install -e ".[dev]"
make db-up      # docker-compose up -d (5 PostgreSQL containers)

# 3. Run the pipeline
make run        # Full pipeline (fetch + predict)

# 4. Check results
make picks      # Show current picks
make validate   # Validate yesterday's results
```

### Common Commands

```bash
make run              # Full pipeline
make refresh          # Quick line refresh only
make picks            # Show current picks
make validate         # Validate yesterday's results

make test             # Run tests (97% coverage, 1,459 tests)
make lint             # black, isort, flake8
make deploy           # Deploy to AWS server
make deploy-restart   # Deploy + restart Airflow
```

## Engineering Practices

- **Testing**: 1,459 tests, 97% coverage (pytest + pre-commit hooks)
- **CI/CD**: black, isort, flake8, bandit (security), gitleaks (secrets)
- **Architecture**: Frozen dataclasses for config, modular extractors with dependency injection, custom exception hierarchy, 6 Architecture Decision Records
- **Data Quality**: Feature drift detection (KS tests), Pydantic validation, data freshness enforcement, autocommit on read-only connections to prevent transaction cascading
- **Conventional commits**, structured JSON logging, automated database backups with 7-day rotation

## Project Structure

```
nba/
├── betting_xl/                     # Core prediction system
│   ├── fetchers/                   # 10+ API sources (direct + aggregator)
│   ├── loaders/                    # PostgreSQL loaders with upsert
│   ├── scripts/                    # Backfill, session capture, validation
│   ├── analysis/                   # Line discrepancy analysis
│   ├── xl_predictor.py             # Model loading + inference
│   ├── line_optimizer.py           # Multi-book line shopping
│   └── generate_xl_predictions.py  # Main prediction orchestrator
│
├── models/
│   ├── saved_xl/                   # Production artifacts (XL + V3 + V4)
│   ├── train_market.py             # Two-head stacked training
│   └── walk_forward_validation.py  # Temporal cross-validation
│
├── features/
│   ├── extract_live_features_xl.py # Live extraction (188 features)
│   ├── build_xl_training_dataset.py # Training dataset builder
│   └── extractors/                 # 10 modular extractors
│       ├── base.py                 # BaseFeatureExtractor ABC
│       ├── book_features.py        # 23 book disagreement features
│       ├── h2h_features.py         # 36 H2H matchup features
│       ├── bp_analytics_features.py # 15 BP projection/EV/hit rate features
│       ├── game_context_features.py # 8 pace/blowout/usage features
│       ├── temporal_features.py    # 10 milestone/trade detection features
│       ├── direct_line_features.py # 19 cross-source features
│       └── ...                     # prop_history, vegas, team, cheatsheet
│
├── core/                           # Services
│   ├── conviction_engine.py        # Multi-run signal aggregation
│   ├── daily_card.py               # Discord embed builder
│   ├── axiom_writer.py             # Prediction history writer
│   ├── drift_service.py            # Feature drift detection
│   └── data_registry.py            # Ingestion tracking
│
├── config/                         # Frozen dataclasses
│   ├── thresholds.py               # All thresholds and hyperparams
│   └── database.py                 # Connection configs
│
└── V4_RETRAINING_PLAN.md           # V4 training architecture

cephalon/                           # AI bot modules (separate repo)
├── axiom_tools.py                  # 22 tool definitions (2,800 lines)
├── brain.py                        # Claude API integration
├── personalities.py                # System prompts
└── axiom_db.py                     # Read-only DB manager

airflow/dags/                       # 6 DAG definitions
docker/                             # Database containers + migrations
tests/                              # 1,459 tests (97% coverage)
docs/adr/                           # 6 Architecture Decision Records
```

## Related Projects

- **[Lunara Sports](https://github.com/untitled114/play-by-play)** — Real-time NBA streaming platform (Kafka + Java Streams + FastAPI + React)
- **[SQL Server Sentinel](https://github.com/untitled114/sql-server-sentinel)** — Production SQL Server monitoring + chaos engineering
