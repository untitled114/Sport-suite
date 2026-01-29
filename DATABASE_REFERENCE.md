# DATABASE REFERENCE - NBA Betting System
**Last Updated:** January 22, 2026

## Quick Start

```bash
# Start all databases
cd ~/Sport-suite/docker && docker-compose up -d

# Launch Jupyter notebook for system review
cd ~/Sport-suite
jupyter notebook nba_system_review.ipynb

# Or use JupyterLab
jupyter lab nba_system_review.ipynb
```

---

## DATABASE INVENTORY

### NBA Databases (Active)

| Database | Port | Connection String | Purpose |
|----------|------|-------------------|---------|
| **nba_players** | 5536 | `PGPASSWORD=${DB_PASSWORD} psql -h localhost -p 5536 -U ${DB_USER} -d nba_players` | Player stats, game logs, rolling stats |
| **nba_games** | 5537 | `PGPASSWORD=${DB_PASSWORD} psql -h localhost -p 5537 -U ${DB_USER} -d nba_games` | Game-level data, box scores |
| **nba_team** | 5538 | `PGPASSWORD=${DB_PASSWORD} psql -h localhost -p 5538 -U ${DB_USER} -d nba_team` | Team stats, season aggregates |
| **nba_intelligence** | 5539 | `PGPASSWORD=${DB_PASSWORD} psql -h localhost -p 5539 -U ${DB_USER} -d nba_intelligence` | Props, cheatsheet, injuries, predictions |
| **nba_reference** | 5540 | `PGPASSWORD=${DB_PASSWORD} psql -h localhost -p 5540 -U ${DB_USER} -d nba_reference` | Reference data |

### MongoDB

| Database | Port | Connection String | Purpose |
|----------|------|-------------------|---------|
| **nba_mongodb** | 27017 | `mongo localhost:27017/nba` | Document storage (if used) |

---

## KEY TABLES BY DATABASE

### nba_players (Port 5536)

**Main Tables:**
- `player_profile` - Player bio data, IDs
- `player_game_logs` - Individual game performances (CRITICAL for actual results)
- `player_rolling_stats` - L3/L5/L10/L20 rolling averages with EMA
- `player_season_stats` - Season aggregates

**Key Columns in player_game_logs:**
```sql
player_id, game_id, game_date, season, team_abbrev, opponent_abbrev,
is_home,  -- CRITICAL: home/away status
points, rebounds, assists, steals, blocks, turnovers,
three_pointers_made, minutes_played, fg_made, fg_attempted,
ft_made, ft_attempted, plus_minus
```

**Query Example:**
```sql
-- Get actual results for validation
SELECT
    pp.full_name as player_name,
    pgl.game_date,
    pgl.points,
    pgl.rebounds,
    pgl.assists,
    pgl.points + pgl.rebounds as pr,
    pgl.rebounds + pgl.assists as ra,
    pgl.is_home
FROM player_game_logs pgl
JOIN player_profile pp ON pgl.player_id = pp.player_id
WHERE pgl.game_date = '2026-01-21'
ORDER BY pp.full_name;
```

### nba_intelligence (Port 5539)

**Main Tables:**
- `cheatsheet_data` - Underdog props with projections (PRODUCTION DATA SOURCE)
- `nba_prop_lines` - Historical props from all books
- `injuries` - Injury reports
- `matchup_history` - Head-to-head history

**Key Columns in cheatsheet_data:**
```sql
player_name, game_date, stat_type, platform, line,
over_odds, under_odds,
projection, projection_diff, bet_rating,
expected_value, ev_pct, probability, recommended_side,
opp_rank, opp_value,
hit_rate_l5, hit_rate_l15, hit_rate_season,
l5_over, l5_under, l15_over, l15_under,
season_over, season_under,
use_for_betting, fetch_timestamp
```

**Query Example:**
```sql
-- Get today's props with high ratings
SELECT
    player_name,
    stat_type,
    line,
    projection,
    bet_rating,
    ev_pct,
    hit_rate_l5,
    opp_rank
FROM cheatsheet_data
WHERE game_date = CURRENT_DATE
AND bet_rating >= 4
AND ev_pct >= 20
ORDER BY bet_rating DESC, ev_pct DESC;
```

### nba_games (Port 5537)

**Main Tables:**
- `games` - Game metadata
- `box_scores` - Team box scores
- `game_context` - Advanced game stats

### nba_team (Port 5538)

**Main Tables:**
- `teams` - Team info
- `team_stats` - Season team stats
- `team_rolling_stats` - Rolling team performance

---

## PIPELINE OUTPUT LOCATIONS

### Generated Files

| Location | Files | Purpose |
|----------|-------|---------|
| `nba/betting_xl/predictions/` | `pro_picks_YYYY-MM-DD.json` | Pro tier filtered picks (PRODUCTION) |
| `nba/betting_xl/predictions/` | `xl_picks_YYYY-MM-DD.json` | XL model picks |
| `nba/betting_xl/predictions/` | `debug_all_props_YYYY-MM-DD.csv` | All props before filtering (debug) |
| `nba/betting_xl/ab_test/picks/` | `github_YYYY-MM-DD.json` | GitHub filter set picks |
| `nba/betting_xl/ab_test/picks/` | `local_YYYY-MM-DD.json` | LOCAL filter set picks |
| `nba/betting_xl/logs/` | `pipeline_YYYY-MM-DD.log` | Daily pipeline logs |
| `nba/betting_xl/lines/` | `props_*.json` | Raw fetched props (BettingPros) |

### Log Locations

```bash
# Pipeline logs
tail -f ~/Sport-suite/nba/betting_xl/logs/pipeline_$(date +%Y-%m-%d).log

# Training logs (if retraining models)
tail -f /tmp/train_points.log
tail -f /tmp/train_rebounds.log

# Validation outputs
ls ~/Sport-suite/nba/betting_xl/validation_summary_*.md
```

---

## COMMON QUERIES

### Check Data Freshness

```bash
# Cheatsheet props (intelligence DB)
PGPASSWORD=${DB_PASSWORD} psql -h localhost -p 5539 -U ${DB_USER} -d nba_intelligence -c "
SELECT
    game_date,
    COUNT(*) as props,
    COUNT(DISTINCT player_name) as players
FROM cheatsheet_data
WHERE game_date >= CURRENT_DATE - 7
GROUP BY game_date
ORDER BY game_date DESC;
"

# Game logs (players DB)
PGPASSWORD=${DB_PASSWORD} psql -h localhost -p 5536 -U ${DB_USER} -d nba_players -c "
SELECT
    game_date,
    COUNT(*) as logs,
    COUNT(DISTINCT player_id) as players
FROM player_game_logs
WHERE game_date >= CURRENT_DATE - 7
GROUP BY game_date
ORDER BY game_date DESC;
"
```

### Validate Picks Against Actuals

```bash
# Quick validation query
PGPASSWORD=${DB_PASSWORD} psql -h localhost -p 5536 -U ${DB_USER} -d nba_players -c "
SELECT
    pp.full_name,
    pgl.points,
    pgl.rebounds,
    pgl.assists,
    pgl.points + pgl.rebounds as pr
FROM player_game_logs pgl
JOIN player_profile pp ON pgl.player_id = pp.player_id
WHERE pgl.game_date = '2026-01-21'
AND pp.full_name IN ('Ryan Kalkbrenner', 'Scottie Barnes', 'Ausar Thompson')
ORDER BY pp.full_name;
"
```

### Check Filter Performance

```sql
-- Count props by filter criteria
SELECT
    stat_type,
    COUNT(*) FILTER (WHERE bet_rating >= 4) as rating_4_plus,
    COUNT(*) FILTER (WHERE ev_pct >= 20) as ev_20_plus,
    COUNT(*) FILTER (WHERE hit_rate_l5 >= 0.80) as l5_80_plus,
    COUNT(*) FILTER (WHERE opp_rank >= 21) as opp_21_30
FROM cheatsheet_data
WHERE game_date = CURRENT_DATE
GROUP BY stat_type;
```

---

## JUPYTER NOTEBOOK USAGE

### Launch Options

```bash
# Option 1: Jupyter Notebook (classic interface)
cd ~/Sport-suite
jupyter notebook nba_system_review.ipynb

# Option 2: JupyterLab (modern interface)
cd ~/Sport-suite
jupyter lab nba_system_review.ipynb

# Option 3: VS Code (if installed)
code ~/Sport-suite/nba_system_review.ipynb
```

### Notebook Features

1. **Database Inventory** - See all tables and row counts
2. **Pipeline Tracking** - Monitor recent runs and outputs
3. **Filter Performance** - Analyze wins/losses by filter
4. **Data Quality** - Check data freshness and coverage
5. **Custom Queries** - Run ad-hoc SQL queries
6. **Export Data** - Save results to CSV for external analysis

### Tips

- **Run all cells:** Kernel → Restart & Run All
- **Export HTML:** File → Download as → HTML
- **Add custom queries:** Use the blank cells at the bottom
- **Auto-reload:** Add `%load_ext autoreload` and `%autoreload 2` at top

---

## TROUBLESHOOTING

### Databases Not Running

```bash
# Check status
docker ps | grep nba

# Restart all
cd ~/Sport-suite/docker && docker-compose restart

# View logs
docker logs nba_intelligence_db
docker logs nba_players_db
```

### Connection Errors

```bash
# Test connection
PGPASSWORD=${DB_PASSWORD} psql -h localhost -p 5539 -U ${DB_USER} -d nba_intelligence -c "SELECT 1;"

# Check port availability
netstat -tulpn | grep 5539
```

### Missing Data

```bash
# Re-run morning pipeline
cd ~/Sport-suite
./nba/nba-predictions.sh morning

# Check logs
tail -100 ~/Sport-suite/nba/betting_xl/logs/pipeline_$(date +%Y-%m-%d).log
```

---

## PYTHON CONNECTION EXAMPLE

```python
import psycopg2
import pandas as pd

# Connection config
conn = psycopg2.connect(
    host='localhost',
    port=5539,
    database='nba_intelligence',
    user='${DB_USER}',
    password='${DB_PASSWORD}'
)

# Query
query = """
SELECT player_name, stat_type, line, projection, bet_rating
FROM cheatsheet_data
WHERE game_date = CURRENT_DATE
ORDER BY bet_rating DESC
LIMIT 10
"""

df = pd.read_sql(query, conn)
print(df)

conn.close()
```

---

## BACKUP AND MAINTENANCE

### Database Backups

```bash
# Backup intelligence DB (most critical)
pg_dump -h localhost -p 5539 -U ${DB_USER} nba_intelligence > ~/backups/nba_intelligence_$(date +%Y%m%d).sql

# Backup players DB
pg_dump -h localhost -p 5536 -U ${DB_USER} nba_players > ~/backups/nba_players_$(date +%Y%m%d).sql
```

### Space Usage

```bash
# Check database sizes
docker exec nba_intelligence_db psql -U ${DB_USER} -d nba_intelligence -c "
SELECT
    pg_database.datname,
    pg_size_pretty(pg_database_size(pg_database.datname)) AS size
FROM pg_database
ORDER BY pg_database_size(pg_database.datname) DESC;
"
```

---

**Need Help?** Check the Jupyter notebook (`nba_system_review.ipynb`) for interactive exploration.
