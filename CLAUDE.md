# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Critical Rules

**SYNTH DATA IS EXTREMELY PROHIBITED.** Never generate synthetic/fake data for training, testing predictions, or any pipeline output.

**Timezone: EST (America/New_York)** is the standard everywhere — fetchers, APIs, loaders, DBs, timestamps, logs. Always use `ZoneInfo("America/New_York")`.

**No Co-Authored-By in commits.** Claude is already listed as a contributor on GitHub.

## Session Workflow

**Start of session**: Run `/session-start` to verify system health before writing code.

**Slash commands available**:
- `/session-start` — Pre-flight checklist (tests, DB, model registry, anomalies)
- `/status` — Quick system overview (version, tests, models, DB)
- `/diagnose` — Deep diagnostic when something is wrong
- `/release [major|minor|patch]` — Bump version, update changelog, tag

**After completing work**:
1. All tests must pass (`python3 -m pytest tests/unit/ -q`)
2. Update `CHANGELOG.md` under `[Unreleased]` for user-facing changes
3. Commit with conventional commits (`feat|fix|refactor|test|docs(scope): message`)

**Version**: Read from `nba/__init__.__version__`. Bump via `/release`.

## Commands

```bash
# Setup
make install          # pip install -e ".[dev]" + pre-commit
make db-up            # Start 5 PostgreSQL containers (ports 5536-5541)

# Development
make test             # pytest with coverage (70% threshold)
make lint             # black, isort, flake8 checks
make fmt              # Auto-format (black + isort)
make typecheck        # mypy (non-blocking)

# Run single test file
pytest tests/unit/test_conviction_engine.py -v

# Run tests by marker
pytest -m unit        # Fast, no external deps
pytest -m integration # May need mocked services
pytest -m "not slow"  # Skip DB/API tests (CI default)

# Pipeline
make run              # Full pipeline (./nba/nba-predictions.sh full)
make refresh          # Quick line refresh only
make validate         # Validate yesterday's picks
make picks            # Show current picks

# Training
make train            # Retrain XL models (POINTS, REBOUNDS)
make train-v3         # Retrain V3 models (136 features)
make walk-forward     # Walk-forward validation

# Deploy (rsync to AWS server, no git on server)
make deploy           # Code only
make deploy-restart   # Code + restart Airflow
```

## Architecture

### Data Flow
```
BettingPros API (7 books) ──────────┐
Direct Fetchers (DK, FD, BetMGM, UD)┤→ fetchers/ → loaders/ → PostgreSQL
PrizePicks (FL proxy) ──────────────┘                         ├── nba_props_xl (3.4M+)
                                                              └── nba_line_snapshots (26K+/day)
BP Analytics (DVP, Trends, Matchups) → lines/*.json
ESPN/nba_api → scripts/ → PostgreSQL (players, games, teams)
                                                          ↓
                                              extract_live_features_xl.py
                                              (102 XL / 136 V3 / ~188 V4 features)
                                                          ↓
                                              xl_predictor.py (LightGBM)
                                              Two-head: Regressor + Classifier
                                              Blending: 60% classifier + 40% residual
                                                          ↓
                                              line_optimizer.py (tier filtering)
                                                          ↓
                                         xl_picks_{date}.json → Discord bot (Axiom)
                                                               → Lunara (GCP)
                                                               → nba_prediction_history
```

### Model System

Two models run in parallel for each market (POINTS, REBOUNDS):
- **XL** (102 features): `nba/models/saved_xl/{market}_xl_*.pkl`
- **V3** (136 features): `nba/models/saved_xl/{market}_v3_*.pkl`
- **V4** (~188 features): In training — adds BP analytics, game context, temporal milestones

Each model is a **two-head stacked LightGBM**:
1. HEAD 1 (Regressor) → predicted stat value
2. HEAD 2 (Classifier) → P(OVER), using features + `expected_diff` from HEAD 1
3. IsotonicRegression calibration on classifier output
4. Blending: 60% classifier probability + 40% residual-based probability

6 pickle files per model: `{market}_{version}_{regressor|classifier|calibrator|imputer|scaler|features}.pkl`

ASSISTS and THREES markets are **disabled** (poor ROI).

### Tier System

Tiers are filter criteria applied after prediction, not model types:

| Tier | Key Criteria |
|------|-------------|
| X | p_over >= 0.85, edge >= 3.0 |
| Z | p_over >= 0.70, edge >= 3.0 |
| META | spread >= 1.5, edge_pct >= 20% |
| Goldmine | High line spread (soft lines) |
| star_tier | Relaxed filters for star players |

### Databases (Docker, ports 5536-5541)

| DB | Port | Key Tables |
|----|------|-----------|
| nba_players | 5536 | player_game_logs (107K), player_profile (5K, has diacritics — use `unaccent()`) |
| nba_games | 5537 | games (6.4K, has pace/blowout_flag), team_game_logs |
| nba_team | 5538 | teams, team_season_stats, team_rolling_stats, team_betting_performance |
| nba_intelligence | 5539 | nba_props_xl (3.4M), nba_line_snapshots (26K+/day), bp_historical_analytics (67K), bp_dvp_historical (5.7K), cheatsheet_data (216K), injury_report, matchup_history |
| cephalon_axiom | 5541 | nba_prediction_history (1K+), axiom_conviction, axiom_pipeline_audit, axiom_posts, axiom_memory |

Credentials: `DB_USER`/`DB_PASSWORD` from `.env` (actual user: `mlb_user`).

**IMPORTANT:** `player_profile.full_name` uses diacritics (Jokić, Dončić, Schröder). Always use `unaccent(pp.full_name)` when joining with ASCII player names from props data.

**IMPORTANT:** Set `autocommit = True` on all read-only DB connections. Without this, a single failed query aborts the transaction and ALL subsequent queries on that connection return empty results silently.

### Airflow DAGs

| DAG | Schedule (EST) | Purpose |
|-----|---------------|---------|
| nba_full_pipeline | Every 3hr, 2:30AM-8:30PM | Full data + predictions (T-120 gated) |
| nba_refresh_pipeline | On-demand | Quick line refresh via ThreadPoolExecutor |
| nba_validation_pipeline | **3:30 AM** | Grade yesterday's picks (runs AFTER full pipeline loads game logs) |
| nba_daily_card | Every 30min, 8AM-10:30PM | Send Discord card at T-120 to T-5 before tip |

The full pipeline has a **T-120 gate**: skips if within 120 minutes of first tip or games already started.

### Direct Sportsbook Fetchers (Mar 2026)

Fetch player props directly from sportsbooks via Colorado residential proxy (IPRoyal, 3GB):

| Fetcher | Book | Status | Props/Day |
|---------|------|--------|-----------|
| `fetch_draftkings_direct.py` | DraftKings | Working | ~400 |
| `fetch_fanduel_direct.py` | FanDuel | Working | ~350 |
| `fetch_betmgm_direct.py` | BetMGM | Working (no Points O/U) | ~300 |
| `fetch_underdog_direct.py` | Underdog Fantasy | Working | ~350 |
| `fetch_betrivers_direct.py` | BetRivers (Kambi) | Rate-limited | 0 |
| `fetch_hardrock_direct.py` | Hard Rock (Kambi) | Blocked (GeoComply) | 0 |

**Proxy routing:** `proxy_manager.py` — "sportsbooks" profile (Colorado), "prizepicks" profile (Florida).

**Line snapshots:** Every fetch appends to `nba_line_snapshots` (append-only, 15K+/day). Tracks per-book line movement across the day.

### BettingPros Analytics (Mar 2026)

`fetch_bettingpros_analytics.py` scrapes server-rendered BP pages (no API key needed):

| Data Source | Content | Output |
|-------------|---------|--------|
| DVP | 30 teams x 7 positions x 9 stats | `bp_dvp_{date}.json` |
| League Trends | 17 situational categories (ATS, ML, O/U) | `bp_league_trends_{date}.json` |
| Matchups | Per-game odds, handle %, expert picks, EV | `bp_matchups_{date}.json` |

**Historical backfill complete:** 67,519 props + 5,670 DVP values stored in `bp_historical_analytics` and `bp_dvp_historical` tables. BP API returns full analytics for any date back to Oct 2023.

### Feature Extractors (`nba/features/extractors/`)

Modular design — each inherits `BaseFeatureExtractor`:

| Extractor | Features | Status |
|-----------|----------|--------|
| `BookFeatureExtractor` | 23 | Production (XL/V3) |
| `H2HFeatureExtractor` | 36 | Production (XL/V3) |
| `PropHistoryExtractor` | 12 | Production (XL/V3) |
| `VegasContextExtractor` | 2 | Production (XL/V3) |
| `TeamBettingExtractor` | 5 | Production (XL/V3) |
| `CheatsheetExtractor` | 8 | Production (XL/V3) |
| `BPAnalyticsFeatureExtractor` | 15 | V4 — BP projections, hit rates, DVP, opp rank |
| `GameContextFeatureExtractor` | 8 | V4 — pace, blowout risk, plus/minus, usage, efficiency |
| `TemporalFeatureExtractor` | 10 | V4 — trade deadline, All-Star, playoff push, team tenure |
| `DirectLineFeatureExtractor` | 19 | V4 — line movement velocity, cross-source discrepancy |

**V4 total: ~188 features** (136 V3 + 15 BP + 8 game context + 10 temporal + 19 direct lines)

### Key Modules

- **`generate_xl_predictions.py`** — Main orchestrator: loads props, extracts features, runs models, applies line optimizer, writes picks JSON
- **`line_optimizer.py`** — Line shopping across 7+ books, tier filtering (PRODUCTION_CONFIG dict), edge/probability thresholds
- **`conviction_engine.py`** — Aggregates signals across multiple pipeline runs, BettingPros pick recommendations, computes conviction scores
- **`daily_card.py`** — Builds Discord embed cards from conviction-scored picks
- **`axiom_writer.py`** — Writes picks to `nba_prediction_history`, manages pipeline audit
- **`xl_predictor.py`** — Loads model pickles, runs two-head inference with calibration

### Discord Bot (Cephalon Axiom)

Runs as `cephalon-axiom.service` on AWS. Code in `discord/bot.py` + `nba_commands.py`.
AI modules in `/home/untitled/Cephalons/cephalon/` (deployed to `/home/cephalons/cephalon/` on server).

**Slash Commands:**
- `/nba` — Today's picks summary
- `/nba-detail` — Full pick cards
- `/nba-refresh` — Quick line refresh (~3-4 min via `quick_refresh.py`)
- `/nba-run` — Full pipeline run
- `/nba-settings` — Configure proactive intelligence features
- Auto-DM at 9:15 AM EST with conviction card

**Proactive Intelligence (background tasks in `bot.py`):**

| Task | Schedule | Purpose |
|------|----------|---------|
| Morning Brief | 7:00 AM EST daily | Yesterday's autopsy + rolling performance + today's outlook |
| Pipeline Alert Engine | Post-pipeline (30s poll) | Line movements, evaporations, pipeline health |
| Injury Monitor | Every 5 min (8AM-11PM) | Cross-ref injuries with active picks |
| Pre-Game Brief | Every 5 min (noon-11PM) | T-60 min mini-brief per game |
| Live Game Monitor | Every 2 min (5PM-1AM) | Real-time pick tracking |
| Self-Healing | Post-pipeline | Auto-refresh on <100 props or <4 books |

**AI Tool System (22 tools in `axiom_tools.py`):**

| Category | Tools |
|----------|-------|
| Picks & Pipeline | `get_current_picks`, `get_pipeline_status`, `lookup_conviction`, `lookup_pick_detail`, `lookup_prop_lines` |
| Intelligence (Jarvis-tier) | `explain_pick`, `query_player_track_record`, `performance_breakdown`, `check_line_movement` (timeline + per-book), `pick_evolution` (added/dropped/changed across runs), `tomorrow_preview`, `bankroll_update` |
| Analytics | `lookup_dvp` (defense vs position), `lookup_bp_analytics` (BP projection/EV/hit rates) |
| Database | `lookup_player_stats`, `lookup_injuries`, `lookup_matchup_history`, `lookup_team_stats`, `search_pick_history`, `lookup_validation` |
| Admin | `run_nba_refresh`, `run_full_pipeline` |

**Natural language:** Axiom understands casual DMs ("how did we do?", "what's tonight?", "should I trust Jokic?") and uses tools proactively without slash commands.

**Cost:** ~$15-25/month (Claude Sonnet 4.5, 50 msg/day cap, ~$0.023/message).

**Cephalon AI Modules (`/home/untitled/Cephalons/cephalon/`):**

| Module | Purpose |
|--------|---------|
| `brain.py` | Claude Sonnet API with tool loop (max 5 turns), history, rate limiting |
| `axiom_tools.py` | 22 tool definitions + handlers (2,800 lines) |
| `axiom_db.py` | Read-only PostgreSQL connection manager + `execute_write` for axiom_memory |
| `context.py` | Live data providers (ESPN schedule, performance, injuries) |
| `intelligence.py` | Morning brief analyzer + post-game autopsy |
| `alerts.py` | Alert engine (line movement, evaporation, pipeline, injury, performance) |
| `live_tracker.py` | Live ESPN game tracking during active games |
| `personalities.py` | System prompts (AXIOM, LUMEN, SOLACE, ATLAS) |
| `persistence.py` | SQLite + PostgreSQL conversation/knowledge persistence |

### Multi-Cloud Setup

- **AWS** (16.58.146.197): Sport-Suite pipeline, Airflow, databases, Axiom bot, API on port 8000
- **GCP** (Cloud Run): Lunara/play-by-play platform (lunara-app.com), syncs picks via Sport-Suite API
- **Hetzner** (5.161.239.229): **DECOMMISSIONED** — all services stopped Mar 16, 2026. Data-only, do NOT run anything there. NEVER connect services to this server.

### Lunara Integration

Lunara's `pick_sync_poller` (GCP) calls `GET /picks/today` on AWS:8000 every 5 minutes.
Auth: `LUNARA_API_KEY` env var, Bearer token.
The API reads `xl_picks_{date}.json` from the predictions directory.

## Code Style

- **Black** formatting, 100 char line length
- **isort** with black profile
- Conventional commits: `feat|fix|refactor|test|docs(scope): message`
- Custom exception hierarchy in `nba/core/exceptions.py`
- Frozen dataclasses for thresholds in `nba/config/thresholds.py`
- Structured logging (JSON in production)

### V4 Retraining System

**Full plan:** `nba/V4_RETRAINING_PLAN.md`

**Auto-retraining trigger:** 7-day rolling WR < 60% (from `nba_prediction_history`).

**V4 features (~188 total):**
- V3 base: 136 features (XL 102 + V3 34)
- BP analytics: 15 features (projection, EV, hit rates, DVP, opposition rank) — backfilled 67K props
- Game context: 8 features (pace, blowout risk, plus/minus, usage, efficiency)
- Temporal milestones: 10 features (trade deadline, All-Star, playoff push, team tenure, trade detection)
- Direct line: 19 features (line movement, cross-source — accumulating)

**BP historical backfill complete:** 67,519 props + 5,670 DVP values across seasons 2023-2026.

**NBA milestone dates** defined in `nba/features/extractors/temporal_features.py`:
- 2023-24: TD Feb 8, ASB Feb 16-18, Season end Apr 14
- 2024-25: TD Feb 6, ASB Feb 14-16, Season end Apr 13
- 2025-26: TD Feb 5, ASB Feb 13-15, Season end Apr 12

**Pipeline:** Trigger → Build dataset (expanding window + exponential decay weighting) → Walk-forward validate → Compare vs production → Shadow mode (3 days) → Promote/rollback.

**Dataset building:** `python3 features/build_xl_training_dataset.py --output features/datasets/` (~3 hours, ~60K props, 239 columns). Run from `nba/` directory with `PYTHONPATH=.../nba`.

## Testing

Tests in `tests/unit/` (33 files) and `tests/integration/`. Fixtures in `tests/conftest.py` provide mock DB cursors, sample feature vectors (102-dim), mock LightGBM models, and environment variables.

Coverage: **97%** (1,459 tests). Integration modules (fetchers, loaders, DB-heavy code) are excluded — they need real service connections.

## Known Issues

- **`player_profile.full_name` diacritics:** Jokić, Dončić, Nurkić, Valančiūnas, Schröder — use `unaccent()` in SQL joins
- **`team_betting_performance` table may not exist** on some DB instances — causes transaction abort if autocommit not set
- **BP API curated subset:** `/v3/props` returns ~80-100 players per date, not all players. Some stars (Curry) excluded from BP's feed. 68% match rate with our props data.
- **Games table stale:** Only goes to Apr 2025 on server — game context features may return defaults for 2025-26 games without pace data
