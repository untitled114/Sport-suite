# NBA Feature Engineering

**Status:** ✅ Complete - Production Ready

This directory contains feature extraction for the NBA props prediction system.

## Core Scripts

### `build_xl_training_dataset.py`
Main dataset builder that creates training data with 102 features:
- Queries historical props from `nba_prop_lines`
- Enriches with `is_home` from `player_game_logs`
- Extracts all player, team, and book features
- Outputs CSVs for model training

### `extract_live_features.py`
Real-time feature extractor for live predictions:
- 78 player features (rolling stats, team context, matchup history)
- 20 book features (line variance, deviations)
- 4 computed features (is_home, line, opponent, expected_diff)

### `extract_live_features_xl.py`
XL variant with hybrid PostgreSQL + MongoDB support:
- Falls back to PostgreSQL when MongoDB unavailable
- Same 102 features in all scenarios

## Feature Categories (102 Total)

### Player Features (78)
- **Rolling Stats**: EMA-weighted L3/L5/L10/L20 averages
- **Team Context**: Pace, offensive/defensive ratings
- **Matchup History**: Head-to-head stats vs opponent
- **Usage**: Starter flag, minutes trends, position encoding

### Book Features (20)
- **Line Variance**: Spread, consensus, std dev, coefficient of variation
- **Book Deviations**: Per-book deviation from consensus
- **Line Shopping**: Softest/hardest book identification

### Computed Features (4)
- `is_home`: Home/away status
- `line`: Sportsbook prop line
- `opponent_team`: Opponent team code
- `expected_diff`: Regressor prediction - line

## Output Datasets

Located in `datasets/`:
- `xl_training_POINTS_2023_2025.csv`
- `xl_training_REBOUNDS_2023_2025.csv`
- `xl_training_ASSISTS_2023_2025.csv`
- `xl_training_THREES_2023_2025.csv`

---

**Last Updated:** January 2026
