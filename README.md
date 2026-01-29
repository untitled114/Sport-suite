# NBA Player Props Prediction System

Machine learning-powered player props prediction system for NBA betting markets.

## Overview

This project implements a production-grade ML pipeline for predicting NBA player prop outcomes (points, rebounds, assists) using a two-head stacked architecture with LightGBM models. The system achieves a validated **87.5% win rate** with **+67.1% ROI** on recent predictions.

## Key Features

- **Two-Head Stacked Architecture**: Combines regressor (predicts actual values) + classifier (predicts P(OVER))
- **JSONCalibrator**: Custom probability calibration for improved accuracy
- **Multi-Book Line Shopping**: Aggregates lines from 7 sportsbooks to find optimal value
- **Automated Pipeline**: Morning data collection + evening prediction generation
- **Tiered Filter System**: Backtested filters targeting 85%+ win rates

## Quick Start

### Prerequisites
- Python 3.10+
- Docker & Docker Compose
- PostgreSQL client (for database access)

### Setup

1. **Clone and configure environment**
```bash
git clone https://github.com/untitled114/nba-props-betting.git
cd nba-props-ml

# Copy environment template and fill in credentials
cp .env.example .env
# Edit .env with your database password and API keys
```

2. **Start databases**
```bash
cd docker && docker-compose up -d
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the pipeline**
```bash
# Morning workflow (data collection)
./nba/nba-predictions.sh morning

# Evening workflow (generate predictions)
./nba/nba-predictions.sh evening
```

## Project Structure

```
nba-props-ml/
├── nba/
│   ├── betting_xl/           # XL prediction system
│   │   ├── predictions/      # Daily picks (JSON)
│   │   ├── fetchers/         # Data fetchers (BettingPros, etc.)
│   │   ├── loaders/          # Database loaders
│   │   └── *.py              # Core prediction modules
│   ├── models/
│   │   ├── saved_xl/         # Production models
│   │   ├── train_market.py   # Model training script
│   │   └── json_calibrator.py # Probability calibration
│   ├── features/             # Feature extraction (102 features)
│   ├── scripts/              # Data processing scripts
│   └── nba-predictions.sh    # Main pipeline script
├── core/                     # Shared database utilities
├── docker/                   # Database containers
├── .env.example              # Environment template
└── requirements.txt          # Python dependencies
```

## Architecture

### Model Pipeline
```
Raw Props Data → Feature Extraction (102 features)
                         ↓
    ┌────────────────────┴────────────────────┐
    ↓                                          ↓
Regressor Head                          Classifier Head
(Predicts Value)                        (Predicts P(OVER))
    ↓                                          ↓
    └────────────────────┬────────────────────┘
                         ↓
              Blended Probability
                         ↓
              Calibrated Output
```

### Feature Categories (102 Total)
- **Player Features (78)**: Rolling stats (L3/L5/L10/L20), team context, matchup history
- **Book Features (20)**: Line variance, book deviations, spread metrics
- **Computed Features (4)**: is_home, line, opponent_team, expected_diff

## Database Schema

| Database | Port | Purpose |
|----------|------|---------|
| nba_players | 5536 | Player stats, game logs, rolling stats |
| nba_games | 5537 | Game data, box scores |
| nba_team | 5538 | Team stats, pace metrics |
| nba_intelligence | 5539 | Props, predictions, injuries |

## Performance Metrics

### Validated Results (Jan 19-28, 2026)

| Metric | Value |
|--------|-------|
| **Overall Win Rate** | 87.5% (14W/2L) |
| **ROI** | +67.1% |
| **POINTS** | 87.5% (7W/1L) |
| **ASSISTS** | 100% (5W/0L) |
| **REBOUNDS** | 66.7% (2W/1L) |
| **Breakeven** | 52.4% @ -110 |

### Filter Performance

| Filter | Win Rate | Record |
|--------|----------|--------|
| **assists_season** | 100% | 5-0 |
| **points_low_mult** | 87.5% | 7-1 |
| **rebounds_low_mult** | 66.7% | 2-1 |

## Technologies

- **ML Framework**: LightGBM
- **Database**: PostgreSQL 15 (TimescaleDB)
- **Containerization**: Docker
- **Language**: Python 3.10+
- **Data Sources**: BettingPros API, ESPN API

## Environment Variables

| Variable | Description |
|----------|-------------|
| `DB_PASSWORD` | PostgreSQL database password |
| `DB_USER` | Database username (default: nba_user) |
| `BETTINGPROS_API_KEY` | BettingPros Premium API key |
| `ODDS_API_KEY` | The Odds API key (optional) |

## Contributing

This project demonstrates ML engineering practices including:
- Feature engineering at scale
- Model calibration techniques
- Production ML pipeline design
- Database schema design
- Automated data pipelines

## License

MIT License

---

**Author**: GitHub: [@untitled114](https://github.com/untitled114)
**Last Updated**: January 2026
