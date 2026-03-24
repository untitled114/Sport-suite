# Sport-Suite: NBA Player Props Prediction Platform

![Tests](https://github.com/untitled114/Sport-suite/actions/workflows/test.yml/badge.svg)
![Lint](https://github.com/untitled114/Sport-suite/actions/workflows/lint.yml/badge.svg)
![Security](https://github.com/untitled114/Sport-suite/actions/workflows/security.yml/badge.svg)
![Deploy](https://github.com/untitled114/Sport-suite/actions/workflows/deploy.yml/badge.svg)
![Coverage](https://img.shields.io/badge/coverage-98%25-brightgreen)
![Python](https://img.shields.io/badge/python-3.10+-blue)

Production ML platform for NBA player prop predictions. Ingests live data from **11 sportsbook APIs** (4 direct + 7 via aggregator), stores in a **consolidated TimescaleDB** with 6 schemas, engineers **134 features** per prediction using 10 modular extractors, and generates calibrated probabilities with stacked LightGBM models — orchestrated via 6 Airflow DAGs with automated validation, drift detection, model registry with auto-rollback, and an AI-powered Discord bot (22 tools) for real-time intelligence.

### Tech Stack
`Python` · `PostgreSQL/TimescaleDB` · `LightGBM` · `Airflow` · `FastAPI` · `Docker` · `AWS EC2` · `GCP Cloud Run` · `Grafana` · `Metabase` · `MLflow` · `GitHub Actions (self-hosted runner)` · `Claude API (Anthropic)` · `psycopg2` · `scikit-learn` · `pandas` · `Pydantic v2`

## Walk-Forward Validated Results

Models are validated using **expanding-window walk-forward cross-validation** — trained on historical data, tested on future periods never seen during training. No data leakage, no lookahead bias.

### POINTS

**Mean AUC: 0.730 (+/- 0.017) | Win Rate: 62.6% | Edge WR: 68.7% | ROI: +25.3%**

| Fold | Train Period | Test Period | AUC | Win Rate | Edge WR |
|------|-------------|-------------|-----|----------|---------|
| 1 | Oct 2024 - Apr 2025 | Oct 2025 | 0.753 | 60.0% | 70.6% |
| 2 | Oct 2024 - Oct 2025 | Oct - Dec 2025 | 0.712 | 65.1% | 67.7% |
| 3 | Oct 2024 - Dec 2025 | Dec 2025 - Feb 2026 | 0.724 | 62.6% | 67.8% |

![POINTS AUC](nba/models/model_cards/images/POINTS_walkforward_auc.png)
![POINTS Win Rate](nba/models/model_cards/images/POINTS_walkforward_winrate.png)
![POINTS ROI](nba/models/model_cards/images/POINTS_walkforward_roi.png)

### REBOUNDS

**Mean AUC: 0.729 (+/- 0.029) | Win Rate: 65.6% | Edge WR: 69.3% | ROI: +25.8%**

| Fold | Train Period | Test Period | AUC | Win Rate | Edge WR |
|------|-------------|-------------|-----|----------|---------|
| 1 | Oct 2024 - Apr 2025 | Oct 2025 | 0.754 | 71.4% | 72.7% |
| 2 | Oct 2024 - Oct 2025 | Oct - Dec 2025 | 0.689 | 60.3% | 65.2% |
| 3 | Oct 2024 - Dec 2025 | Dec 2025 - Feb 2026 | 0.743 | 65.2% | 70.0% |

![REBOUNDS AUC](nba/models/model_cards/images/REBOUNDS_walkforward_auc.png)
![REBOUNDS Win Rate](nba/models/model_cards/images/REBOUNDS_walkforward_winrate.png)
![REBOUNDS ROI](nba/models/model_cards/images/REBOUNDS_walkforward_roi.png)

### Production Models (V5)

| Model | Market | Features | R² | AUC | WF AUC | Trained | Status |
|-------|--------|----------|-----|------|--------|---------|--------|
| V5 | POINTS | 134 | 0.586 | 0.731 | 0.730 | Mar 2026 | Deployed |
| V5 | REBOUNDS | 134 | 0.534 | 0.729 | 0.729 | Mar 2026 | Deployed |

## What This Project Demonstrates

- **Multi-Source Data Engineering**: 11 REST API integrations (4 direct sportsbook scrapers via residential proxy + 7 via BettingPros), with append-only line snapshot tracking (15K+ snapshots/day), cross-source deduplication, and historical backfill (67K+ analytics records)
- **SQL & Data Modeling**: Consolidated TimescaleDB with 6 schemas (3.7M+ prop lines, 108K game logs, 67K BP analytics), schema-based routing via `search_path`, migration tooling
- **Feature Engineering at Scale**: 134 features across 10 modular extractors — player rolling stats (all 8 stat types), book disagreement signals, H2H matchups, BettingPros analytics (projections, EV, hit rates), DVP rankings, game context (pace, travel distance, blowout risk)
- **ML Pipeline**: Two-head stacked LightGBM with out-of-fold expected_diff, Platt scaling calibration, noise injection for variance matching, walk-forward validation, MLflow tracking, and auto-retraining triggers
- **Production Operations**: Multi-cloud deployment (AWS + GCP), 6 Airflow DAGs, self-hosted GitHub Actions runner, model registry with promotion gate and auto-rollback, 98% test coverage (2,096 tests)
- **Five-Pillar Observability**: Pipeline telemetry, model registry, validation runs, dataset fingerprinting, feature default rate tracking — all queryable via Grafana and Metabase dashboards
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
│  → intelligence.nba_props_xl (3.7M rows)             │
│  → intelligence.nba_line_snapshots (15K+/day)        │
│  → intelligence.bp_historical_analytics (67K)        │
│  → intelligence.bp_dvp_historical (5.7K, 3 seasons)  │
└──────────────────┬───────────────────────────────────┘
                   ▼
┌──────────────────────────────────────────────────────┐
│  Feature Engineering (134 features, 10 extractors)   │
│  ├── Player rolling stats (38): all 8 stats x L3-L20│
│  ├── Book disagreement (18): per-book deviations     │
│  ├── H2H matchup (12): primary stat only             │
│  ├── BP analytics (8): projection, EV, hit rates     │
│  ├── Prop history (9): context-adjusted hit rates    │
│  ├── Shooting/advanced (8): FG%, FT rate, TS%, +/-   │
│  └── Team, game context, vegas, situational          │
└──────────────────┬───────────────────────────────────┘
                   ▼
┌──────────────────────────────────────────────────────┐
│  Stacked LightGBM (Two-Head, V5)                     │
│  HEAD 1: Regressor → predicted stat value            │
│  HEAD 2: Classifier → P(OVER) with expected_diff     │
│  ├── expected_diff via 5-fold OOF (no leakage)       │
│  ├── Noise injection (train/test variance matching)  │
│  ├── Platt scaling on 15% holdout                    │
│  └── MLflow tracking: params, metrics, git hash      │
└──────────────────┬───────────────────────────────────┘
                   ▼
┌──────────────────────────────────────────────────────┐
│  Model Registry + Promotion Gate                      │
│  training → shadow → production → rolled_back        │
│  AUC improvement ≥ 0.005, std < 0.03, WR ≥ prod     │
│  Auto-rollback when 7d WR < 60%                      │
└──────────────────┬───────────────────────────────────┘
                   ▼
┌──────────────────────────────────────────────────────┐
│  Conviction Engine                                    │
│  Aggregates signals across 6 daily pipeline runs     │
│  Tracks pick stability, line movement, BP alignment  │
│  Labels: LOCKED | STRONG | WATCH | SKIP              │
└──────────────────┬───────────────────────────────────┘
                   ▼
┌──────────────────────────────────────────────────────┐
│  Output                                               │
│  ├── xl_picks_{date}.json → Lunara Sports (GCP)      │
│  ├── axiom.nba_prediction_history → PostgreSQL        │
│  ├── Cephalon Axiom → Discord (22 AI tools)          │
│  └── Grafana + Metabase dashboards                    │
└──────────────────────────────────────────────────────┘
```

## Data Sources (11 APIs)

**Direct Sportsbook Fetchers** (via Colorado/Florida residential proxy):
- **DraftKings** — `sportsbook-nash.draftkings.com` API
- **FanDuel** — `sbapi.il.sportsbook.fanduel.com` API
- **BetMGM** — CDS-API with fixture-view endpoint
- **Underdog Fantasy** — `api.underdogfantasy.com` (UUID-based)
- **PrizePicks** — Direct API via Florida proxy

**BettingPros Premium API** (7 additional books): DraftKings, FanDuel, BetMGM, Caesars, BetRivers, ESPNBet, Underdog

**BettingPros Analytics** (server-rendered scraping):
- Defense vs Position: 30 teams x 7 positions x 9 stats (historical: 2023-2025)
- League Trends: 17 situational categories
- Matchups: Handle %, expert picks, line movement, EV

**Daily volume**: ~8,200 direct props + ~6,800 BettingPros props + 15K+ line snapshots

## Feature Engineering (134 Features)

| Extractor | Count | Description |
|-----------|-------|-------------|
| Player Rolling Stats | 38 | EMA for all 8 stats (pts/reb/ast/3pm/stl/blk/tov/min) at L3/L5/L10/L20 + FG% |
| Shooting & Advanced | 8 | FG% L3-L20, ft_rate_L10, true_shooting_L10, plus_minus L5/L10 |
| Book Disagreement | 18 | Line spread, per-book deviations (DK/FD/MGM/CZR/BR), softest/hardest |
| H2H Matchup | 12 | Primary stat avg/std/L3/L5/L10/L20, home/away splits (no cross-stat) |
| Prop History | 9 | Hit rates (L20, context-adjusted), bayesian confidence, streaks |
| BettingPros | 8 | Projection diff, probability, EV%, hit rates, opponent rank |
| Team & Game Context | 10 | Pace, off/def ratings, travel_distance_km, altitude, projected poss |
| Situational | 6 | Days rest, starter flag, bench_points_ratio, avg_teammate_usage |
| Vegas Context | 2 | Game total, spread |
| Computed | 1 | expected_diff (5-fold OOF, noise-injected) |

**32 features pruned** via importance audit: zero-signal features (injury flags, season temporals, redundant BP analytics, `days_into_season` temporal leakage) dropped in `FEATURE_PREPROCESSING.common_cols_to_drop`.

## Five-Pillar Observability

| Pillar | Table | What it tracks |
|--------|-------|---------------|
| Operational Excellence | `axiom.pipeline_runs` | Per-task telemetry, anomaly detection, feature counts |
| Reliability | `axiom.model_registry` | Model lifecycle: training → shadow → production → rolled_back |
| Performance Efficiency | `axiom.validation_runs` | Walk-forward results, promotion gate decisions |
| Security/Integrity | metadata JSON | Dataset MD5 hash, feature default rates |
| Cost Optimization | MLflow (`ModelTracker`) | Training duration, git hash, feature importance |

**Dashboards**: Grafana (Pipeline Operations, Model Performance) + Metabase (analytics) at `admin.lunara-app.com`

## Discord Bot (Cephalon Axiom)

AI-powered sports intelligence assistant with 22 tools, running on Claude Sonnet 4.6. No slash commands — just DM Axiom naturally.

**Example conversations**:
- *"How did we do?"* → Pulls yesterday's results with W/L breakdown
- *"Should I trust this Jokic pick?"* → Explains the pick with full context
- *"Any line movement?"* → Real-time snapshot comparison across books
- *"What changed since this morning?"* → Pick evolution (added/dropped/changed)
- *"How's V5 POINTS doing?"* → Flexible performance breakdown by market/model

**Tool Categories**:

| Category | Tools | Description |
|----------|-------|-------------|
| Intelligence | 7 | explain_pick, player_track_record, performance_breakdown, line_movement, pick_evolution, tomorrow_preview, bankroll |
| Analytics | 2 | DVP lookup, BP analytics (projection/EV/hit rates) |
| Picks | 5 | Current picks, pipeline status, conviction, pick detail, prop lines |
| Database | 6 | Player stats, injuries, matchups, team stats, pick history, validation |
| Admin | 2 | Pipeline refresh, full pipeline run |

**Proactive Intelligence** (background tasks):
- Morning Brief (7 AM): Yesterday's autopsy + rolling stats + today's outlook
- Pipeline Alerts: Line movements, pick evaporations, health issues
- Injury Monitor: Cross-references injuries with active picks (every 5 min)
- Pre-Game Brief: T-60 mini-brief per game
- Live Game Monitor: Real-time pick tracking during games

## Orchestration (6 Airflow DAGs)

| DAG | Schedule (EST) | Purpose |
|-----|---------------|---------|
| `nba_full_pipeline` | Every 3hr, 2:30AM-8:30PM | Full ETL + predictions (T-120 gated) |
| `nba_refresh_pipeline` | On-demand | Quick line refresh (~3-4 min) |
| `nba_validation_pipeline` | 3:30 AM | Grade picks + CLV tracking + result tracker |
| `nba_daily_card` | Every 30min, 8AM-10:30PM | Conviction card delivery |
| `nba_drift_detection` | Weekly, Sunday 4AM | Feature drift PSI/KS-test |
| `nba_retraining` | Triggered | Auto-retrain when 7-day WR < 60% |

## Database (Consolidated TimescaleDB)

Single instance on port 5500, database `sportsuite`, 6 schemas:

| Schema | Key Tables | Rows |
|--------|-----------|------|
| `players` | player_game_logs, player_profile | 108K |
| `games` | games, team_game_logs | 6.4K |
| `teams` | teams, team_season_stats, team_betting_performance | 30 |
| `intelligence` | nba_props_xl, nba_line_snapshots, bp_historical_analytics | 3.7M+ |
| `axiom` | nba_prediction_history, axiom_conviction, pipeline_runs, model_registry, validation_runs | 1K+ |
| `features` | feature_sets, computed_features, clv_tracking, performance_metrics | — |

## Quick Start

```bash
git clone https://github.com/untitled114/Sport-suite.git
cd Sport-suite

# 1. Environment
cp .env.example .env  # Set DB_PASSWORD, BETTINGPROS_API_KEY

# 2. Install and start database
make install    # pip install -e ".[dev]"
make db-up      # Consolidated TimescaleDB on port 5500

# 3. Run the pipeline
make run        # Full pipeline (fetch + predict)

# 4. Check results
make picks      # Show current picks
make validate   # Validate yesterday's results
```

## Common Commands

```bash
make run              # Full pipeline
make refresh          # Quick line refresh only
make picks            # Show current picks
make validate         # Validate yesterday's results

make test             # Run tests (98% coverage, 2,096 tests)
make lint             # black, isort, flake8
make deploy           # Auto-deploys via self-hosted runner on push
```

## Infrastructure

| Platform | Services |
|----------|----------|
| **AWS** (EC2) | ML pipeline, Airflow, TimescaleDB, FastAPI API, Cephalon Axiom + Atlas |
| **GCP** (Cloud Run) | Lunara Sports (lunara-app.com), Cephalon Lumen |
| **Grafana** | Pipeline Operations + Model Performance dashboards |
| **Metabase** | Analytics dashboards (3 schema connections) |
| **CI/CD** | Self-hosted GitHub Actions runner on EC2, auto-deploy on push |

## Cephalon Fleet

| Bot | Domain | Tools | AI Model |
|-----|--------|-------|----------|
| **Axiom** | NBA predictions, picks, line shopping | 22 | Claude Sonnet 4.6 |
| **Lumen** | Lunara Sports live tracking | — | Claude Sonnet 4.6 |
| **Solace** | FTMO trading, strategy execution | 14 | Claude Sonnet 4.6 |
| **Atlas** | Fleet overwatch, DB health, pipeline, Sentinel | 18 | Claude Sonnet 4.6 |

## Engineering Practices

- **Testing**: 2,096 tests, 98% coverage (pytest + pre-commit hooks)
- **CI/CD**: black, isort, flake8, bandit (security), gitleaks (secrets), auto-deploy via self-hosted runner
- **Architecture**: Frozen dataclasses for config, modular extractors with dependency injection, custom exception hierarchy
- **Data Quality**: Feature drift detection (KS tests), pipeline telemetry with anomaly detection, autocommit on read-only connections
- **Model Management**: Registry with promotion gate (AUC >= 0.005 improvement, std < 0.03, WR check), auto-rollback, dataset fingerprinting (MD5), MLflow experiment tracking with git hash tagging
- **Training Integrity**: Out-of-fold expected_diff (prevents classifier early-stopping), noise injection (variance matching), Platt scaling on holdout (not in-sample), 32 zero-importance features pruned
- Conventional commits, structured JSON logging, automated database backups

## Related Projects

- [Lunara Sports](https://lunara-app.com) — Real-time NBA streaming platform (Kafka + Java Streams + FastAPI + React)
- [PostgreSQL Sentinel](https://github.com/untitled114/postgres-sentinel) — Production database monitoring, chaos engineering, and incident response
