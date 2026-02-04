# NBA XL Betting System

**Status:** Production Ready (XL + V3 models in parallel)
**Last Updated:** February 4, 2026
**Validated Performance:** 79.5% WR on filtered picks (Jan 19 - Feb 2, 2026)

---

## Quick Start

```bash
# Full pipeline (morning)
./nba/nba-predictions.sh run

# Quick refresh (line movements only)
./nba/nba-predictions.sh refresh

# Show today's picks
./nba/nba-predictions.sh picks

# Health check
./nba/nba-predictions.sh health
```

---

## Model Versions

Both XL and V3 models run in parallel. Picks include `model_version: "xl"` or `"v3"`.

| Model | Features | R² | AUC | Trained |
|-------|----------|-----|-----|---------|
| **XL** | 102 | 0.41 | 0.77 | Dec 2025 |
| **V3** | 136 | 0.55 | 0.74 | Feb 2026 |

**When both models agree:** Picks show "✓ BOTH MODELS AGREE" with individual probabilities.

---

## Data Sources (10 Total)

### Sportsbooks (via BettingPros)
- DraftKings, FanDuel, BetMGM, Caesars, BetRivers, ESPNBet, Fanatics

### DFS Platforms (Softer Lines)
- **Underdog Fantasy** - Standard DFS
- **PrizePicks** - Standard lines
- **PrizePicks Goblin** - Lower lines (85% WR in backtest)
- **PrizePicks Demon** - Higher lines

**Key Finding:** Goblin lines outperform standard lines despite not being in training data.
See [Case Study](../../docs/CASE_STUDY_GOBLIN_LINES.md).

---

## Directory Structure

```
betting_xl/
├── fetchers/                     # Data fetchers
│   ├── fetch_bettingpros.py      # 7 sportsbooks via API
│   ├── fetch_prizepicks_direct.py # PrizePicks alternates
│   ├── fetch_cheatsheet.py       # BettingPros projections
│   └── fetch_vegas_lines.py      # Vegas totals/spreads
│
├── loaders/                      # Database loaders
│   ├── load_props_to_db.py       # Main props loader
│   ├── load_prizepicks_to_db.py  # PrizePicks loader
│   └── load_cheatsheet_to_db.py  # Cheatsheet loader
│
├── predictions/                  # Daily picks (JSON output)
│   └── xl_picks_YYYY-MM-DD.json
│
├── backtest/                     # Backtest results
│   └── backtest_summary_*.json
│
├── config/                       # Configuration
│   └── README.md
│
├── generate_xl_predictions.py    # Main prediction generator
├── xl_predictor.py               # Model loading + inference
├── line_optimizer.py             # Line shopping + filtering
├── show_picks.py                 # CLI display
├── validate_predictions.py       # Multi-system validation
├── validate_xl_models.py         # Historical backtesting
├── run_historical_backtest.py    # Walk-forward backtest harness
└── enrich_props_with_matchups.py # Matchup data enrichment
```

---

## Production Filters

Picks are filtered to high-confidence scenarios:

### Filter Criteria
- **Edge threshold:** ≥ 2.5%
- **Line spread:** ≥ 1.5 points (difference between softest/hardest)
- **Minimum books:** ≥ 3 offering the prop
- **Risk assessment:** Volatility, defense matchup, trends

### Filter Impact

| Scenario | Win Rate |
|----------|----------|
| Walk-forward CV (model baseline) | ~67% |
| With filters (standard lines) | ~70% |
| With filters + goblin lines | ~80% |

---

## Key Components

### `generate_xl_predictions.py`
Main prediction generator:
- Queries props from database
- Extracts features (102 for XL, 136 for V3)
- Runs both model versions
- Applies line optimization
- Outputs filtered picks to JSON

### `xl_predictor.py`
Model loading and inference:
- Loads models from `nba/models/saved_xl/`
- Handles XL vs V3 feature sets
- Applies JSONCalibrator for probability adjustment
- Returns predictions with p_over, edge, confidence

### `line_optimizer.py`
Line shopping and filtering:
- Finds softest line across all books
- Calculates edge vs prediction
- Applies tier-based filtering
- Blacklists trap books (configurable)
- Supports `--standard-only` mode (excludes goblin/demon)

### `show_picks.py`
CLI display:
- Shows picks grouped by market
- Displays stake sizing and risk level
- Highlights model agreement
- Shows line distribution across books

### `run_historical_backtest.py`
Walk-forward backtesting:
- Temporal safety (no future data leakage)
- Seed period for calibrator warmup
- Supports `--standard-only` flag for comparison
- Outputs summary with daily breakdown

---

## Validation

```bash
# Validate specific date range
python3 validate_predictions.py --start-date 2026-01-19 --end-date 2026-02-02

# Run historical backtest (15 days)
python3 run_historical_backtest.py --start 2026-01-19 --end 2026-02-02 --no-seed

# Compare standard-only vs all books
python3 run_historical_backtest.py --start 2026-01-19 --end 2026-02-02 --no-seed --standard-only
```

---

## Performance Summary

### Filtered Backtest (Jan 19 - Feb 2, 2026)

```
WITH GOBLIN LINES:
  Picks: 47 (44 validated)
  Win Rate: 79.5%
  POINTS: 74.2% (23W/8L)
  REBOUNDS: 92.3% (12W/1L)

STANDARD ONLY:
  Picks: 17 validated
  Win Rate: 70.6%
  POINTS: 64.3% (9W/5L)
  REBOUNDS: 100% (3W/0L)
```

### By Model Version
```
XL:  78.3% (18W/5L)
V3:  81.0% (17W/4L)
```

---

## Troubleshooting

### No picks generated
```bash
# Check props exist
./nba/nba-predictions.sh health

# Check database
source docker/.env
PGPASSWORD=$DB_PASSWORD psql -h localhost -p 5539 -U $DB_USER -d nba_intelligence \
  -c "SELECT COUNT(*) FROM nba_props_xl WHERE game_date = CURRENT_DATE;"
```

### Missing matchup data
```bash
# Enrich props with opponent/is_home data
python3 nba/betting_xl/enrich_props_with_matchups.py --date 2026-02-04
```

### Database issues
```bash
cd docker && docker-compose restart
docker ps | grep nba
```

---

## Related Documentation

- [Main README](../../README.md) - Project overview
- [Case Study: Goblin Lines](../../docs/CASE_STUDY_GOBLIN_LINES.md) - Training data mismatch analysis
- [CLAUDE.md](../../.claude/CLAUDE.md) - Complete system reference
- [ADRs](../../docs/adr/) - Architecture decisions
