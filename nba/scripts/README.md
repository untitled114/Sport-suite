# NBA Data Loading Scripts

**Status:** Production Ready
**Last Updated:** February 2026

Data loaders for populating NBA databases from NBA Stats API and ESPN.

---

## Directory Structure

```
scripts/
├── loaders/                        # Database population scripts
│   ├── load_nba_players.py         # Player profiles + season stats
│   ├── load_nba_games.py           # Game schedules
│   ├── load_nba_teams.py           # Team profiles
│   ├── load_player_gamelogs_bulk.py # Player game logs (is_home source)
│   ├── load_team_boxscores.py      # Team box scores
│   ├── calculate_rolling_stats.py  # EMA rolling averages
│   ├── calculate_team_stats.py     # Team season stats
│   ├── fetch_espn_games.py         # ESPN game data (fallback)
│   ├── fetch_espn_injuries_fixed.py # Injury reports
│   └── master_reload.sh            # Full database reload
│
├── utilities/                      # Shared utilities
│   └── nba_api_wrapper.py          # NBA API client
│
├── backfill_*.py                   # Historical data scripts
├── fetch_daily_*.py                # Daily update scripts
└── update_*.py                     # Incremental update scripts
```

---

## Data Sources

### Primary: NBA Stats API (`nba_api` library)
- Player profiles and season stats
- Game schedules and box scores
- Team statistics
- Player game logs (**critical for is_home feature**)

### Secondary: ESPN API
- Game schedules (fallback when NBA API fails)
- Injury reports
- Real-time game status

---

## Key Scripts

### Initial Load (Full Reload)

```bash
# Full database reload (30-40 minutes)
cd loaders && ./master_reload.sh
```

### Daily Updates

```bash
# Update rolling stats (run after games complete)
python3 loaders/calculate_rolling_stats.py

# Fetch today's injuries
python3 fetch_daily_injury_report.py

# Update player game logs (incremental)
python3 loaders/load_player_gamelogs_bulk.py --incremental

# Update team stats
python3 loaders/calculate_team_stats.py
```

### Airflow Integration

These scripts are called by Airflow DAGs:
- `nba_full_pipeline` - Morning data collection
- `nba_health_check` - Verify data freshness

---

## Database Targets

| Script | Database | Port | Key Tables |
|--------|----------|------|------------|
| load_nba_players.py | nba_players | 5536 | player_profile, player_season_stats |
| load_player_gamelogs_bulk.py | nba_players | 5536 | player_game_logs |
| calculate_rolling_stats.py | nba_players | 5536 | player_rolling_stats |
| load_nba_games.py | nba_games | 5537 | games, box_scores |
| load_nba_teams.py | nba_team | 5538 | teams, team_stats |
| fetch_espn_injuries_fixed.py | nba_intelligence | 5539 | injury_report |

---

## Critical: Home/Away Data

The `player_game_logs` table is the source of truth for `is_home`:

```sql
-- Check home/away distribution
SELECT is_home, COUNT(*)
FROM player_game_logs
WHERE game_date >= '2023-10-24'
GROUP BY is_home;

-- Expected: ~54% home, ~46% away
```

This data is used by `build_xl_training_dataset.py` to enrich training data.

---

## Rate Limiting

NBA Stats API is rate-limited. Scripts include:
- Built-in delays (0.6s between requests)
- Retry logic with exponential backoff
- Connection pooling

---

## Related

- [Main README](../../README.md) - Project overview
- [features README](../features/README.md) - Feature extraction
- [betting_xl README](../betting_xl/README.md) - Prediction system
