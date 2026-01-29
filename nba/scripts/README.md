# NBA Data Loading Scripts

**Status:** ✅ Complete - Production Ready

This directory contains data loaders for populating NBA databases.

## Directory Structure

```
scripts/
├── loaders/                    # Database population scripts
│   ├── load_nba_players.py     # Player profiles + season stats
│   ├── load_nba_games.py       # Game schedules
│   ├── load_nba_teams.py       # Team profiles
│   ├── load_player_gamelogs_bulk.py  # Player game logs
│   ├── load_team_boxscores.py  # Team box scores
│   ├── calculate_rolling_stats.py    # Rolling averages
│   ├── calculate_team_stats.py       # Team season stats
│   ├── fetch_espn_games.py     # ESPN game data
│   ├── fetch_espn_injuries_fixed.py  # Injury reports
│   └── master_reload.sh        # Full database reload
│
├── utilities/                  # Shared utilities
│   └── nba_api_wrapper.py      # NBA API client
│
├── backfill_*.py               # Historical data scripts
├── fetch_daily_*.py            # Daily update scripts
└── update_*.py                 # Incremental update scripts
```

## Data Sources

### Primary: NBA Stats API (`nba_api` library)
- Player profiles and season stats
- Game schedules and box scores
- Team statistics

### Secondary: ESPN API
- Game schedules (fallback)
- Injury reports
- Real-time game status

## Key Scripts

### Initial Load
```bash
# Full database reload (30-40 minutes)
cd loaders && ./master_reload.sh
```

### Daily Updates
```bash
# Update rolling stats
python3 loaders/calculate_rolling_stats.py

# Fetch today's injuries
python3 fetch_daily_injury_report.py

# Update current rosters
python3 loaders/update_current_rosters.py
```

### Monitoring
```bash
# Check loading progress
./loaders/monitor_loading.sh
```

## Database Targets

| Script | Database | Port |
|--------|----------|------|
| load_nba_players.py | nba_players | 5536 |
| load_nba_games.py | nba_games | 5537 |
| load_nba_teams.py | nba_team | 5538 |
| fetch_espn_injuries_fixed.py | nba_intelligence | 5539 |

---

**Last Updated:** January 2026
