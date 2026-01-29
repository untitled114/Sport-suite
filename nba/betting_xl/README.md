# NBA XL Betting System

**Status:** Production Ready
**Last Updated:** January 3, 2026
**Performance:** 68.2% WR (58W/27L) validated Nov 18 - Dec 31, 2025

---

## Quick Start

```bash
# Morning workflow (data collection)
./nba/nba-predictions.sh morning

# Evening workflow (generate picks)
./nba/nba-predictions.sh evening

# Health check
./nba/nba-predictions.sh health
```

---

## Tier System

| Tier | Market | Win Rate | Description |
|------|--------|----------|-------------|
| **Tier X** | POINTS | 75% | High-confidence (p >= 0.65) |
| **Tier A** | REBOUNDS | 68% | Standard threshold (p >= 0.60) |
| **Star Tier** | Both | 65% | Profitable star players |
| **Pro Tier** | POINTS/ASSISTS | 76-95% | BettingPros cheatsheet filters |

---

## Directory Structure

```
betting_xl/
├── fetchers/                 # Data fetchers
│   ├── fetch_all.py          # Orchestrator
│   ├── fetch_bettingpros.py  # Main sportsbook props
│   └── fetch_cheatsheet.py   # BettingPros cheatsheet
│
├── loaders/                  # Database loaders
│   ├── load_props_to_db.py   # Props loader
│   └── load_cheatsheet_to_db.py
│
├── predictions/              # Daily picks (JSON)
│   ├── xl_picks_YYYY-MM-DD.json
│   └── pro_picks_YYYYMMDD.json
│
├── config/                   # Configuration files
│   └── production_policies.py
│
├── generate_xl_predictions.py  # XL model predictions
├── generate_cheatsheet_picks.py # Pro tier predictions
├── xl_predictor.py            # Model loading + calibration
├── line_optimizer.py          # Filtering logic + tiers
├── validate_predictions.py    # Performance validation
├── populate_actual_values.py  # Backfill results
└── enrich_props_with_matchups.py # Matchup enrichment
```

---

## Core Components

### XL Predictor (`xl_predictor.py`)
- Loads trained models from `nba/models/saved_xl/`
- Applies JSONCalibrator for probability adjustment
- Returns predictions with p_over, edge, tier

### Line Optimizer (`line_optimizer.py`)
- Tier X/A/Star filtering logic
- Book blacklists (FanDuel for POINTS)
- Pseudo-book filtering (consensus)
- Min/max line gates

### Cheatsheet Picks (`generate_cheatsheet_picks.py`)
- Pro tier using BettingPros analytics
- L5/L15 hit rate filters
- Opponent defense rank filters
- 76-95% expected WR

---

## Data Sources

### Sportsbooks (7 active)
- DraftKings, FanDuel, BetMGM, Caesars
- BetRivers, ESPNBet, Underdog

### Analytics
- BettingPros Premium API
- Cheatsheet projections
- Hit rate data (L5/L15/Season)

---

## Filter Configuration

### POINTS (Tier X)
```python
min_probability: 0.65
min_line: 12.0
max_line: 24.0
max_edge_points: 5.0
require_soft_book: True  # Underdog/ESPNBet
```

### REBOUNDS (Tier A)
```python
min_probability: 0.60
min_line: 3.0  # Filters bad ESPNBet data
min_spread: 2.0
min_edge_points: 1.0
```

---

## Validation

```bash
# Validate specific date
cd nba/betting_xl
PYTHONPATH=$(pwd) python3 validate_predictions.py --date 2026-01-02

# Check calibrator status
cd nba/models
python3 json_calibrator.py status --lookback 21
```

---

## Performance History

### Backtest (Nov 18 - Dec 31, 2025)
```
POINTS (Tier X):  17W / 7L  = 70.8% WR
REBOUNDS (Tier A): 41W / 20L = 67.2% WR
OVERALL:          58W / 27L = 68.2% WR
```

### By Tier
```
Tier X:    75.0% WR
Tier A:    68.0% WR
Star Tier: 65.0% WR
```

---

## Troubleshooting

### No picks generated
- Check props exist: `./nba/nba-predictions.sh health`
- Verify filters aren't too aggressive
- Check JSONCalibrator has enough samples

### Low win rate
- Validate filters match sweet spot
- Check for pattern in losses (line? edge? book?)
- Review calibrator adjustments

### Database issues
```bash
# Restart databases
cd docker && docker-compose restart

# Check connection
PGPASSWORD=${DB_PASSWORD} psql -h localhost -p 5539 -U ${DB_USER} -d nba_intelligence -c "SELECT 1;"
```

---

**See also:** Main `README.md` for project overview
