# NBA Betting Pipeline - Apache Airflow

Production orchestration for the NBA betting system using Apache Airflow with LocalExecutor and PostgreSQL metadata.

## Production Setup

**Server:** Hetzner VPS at `5.161.239.229`
**Executor:** LocalExecutor (parallel task execution)
**Metadata DB:** PostgreSQL on port 5539 (shared with nba_intelligence)

### Current DAGs

| DAG | Schedule (EST) | Description |
|-----|----------------|-------------|
| `nba_full_pipeline` | 9:00 AM | Complete data collection + predictions |
| `nba_validation_pipeline` | 9:30 AM | Validate yesterday's picks |
| `nba_health_check` | Every 6h | System health monitoring |

---

## Quick Start (Production Server)

### Check Status

```bash
ssh sportsuite@5.161.239.229

# Check Airflow services
sudo systemctl status airflow-scheduler airflow-webserver

# Check DAG status
source venv/bin/activate
airflow dags list-runs -d nba_full_pipeline --limit 5

# View logs
tail -100 /home/sportsuite/sport-suite/airflow/logs/scheduler/latest/*.log
```

### Manual Trigger

```bash
# On server
source venv/bin/activate
airflow dags trigger nba_full_pipeline

# Or via Makefile from local
make server-trigger DAG=nba_full_pipeline
```

### Access Web UI

http://5.161.239.229:8080

- **Username:** admin
- **Password:** (set during installation)

---

## Configuration

### `airflow.cfg` Key Settings

```ini
[core]
executor = LocalExecutor
dags_folder = /home/sportsuite/sport-suite/airflow/dags

[database]
sql_alchemy_conn = postgresql+psycopg2://mlb_user:PASSWORD@localhost:5539/airflow_metadata

[webserver]
web_server_port = 8080
base_url = http://5.161.239.229:8080
```

### Airflow Variables

Set via UI (Admin > Variables):

| Variable | Value | Description |
|----------|-------|-------------|
| `nba_project_root` | `/home/sportsuite/sport-suite` | Project root on server |
| `alert_email` | (your email) | Alert recipients |

### Database Connections

| Connection ID | Database | Port |
|--------------|----------|------|
| `nba_players_db` | nba_players | 5536 |
| `nba_games_db` | nba_games | 5537 |
| `nba_team_db` | nba_team | 5538 |
| `nba_intelligence_db` | nba_intelligence | 5539 |

---

## DAG Details

### `nba_full_pipeline` (9:00 AM EST)

Complete daily workflow:

```
fetch_props → load_to_db → enrich_matchups → fetch_cheatsheet
    ↓
fetch_game_results → populate_actuals → update_injuries
    ↓
load_team_games → update_team_stats → fetch_vegas_lines
    ↓
generate_xl_predictions → output_picks
```

**Tasks:**
1. `fetch_props` - BettingPros + PrizePicks (10 sources)
2. `load_props_to_db` - Store in nba_props_xl
3. `enrich_matchups` - Add opponent/is_home context
4. `fetch_cheatsheet` - BettingPros projections
5. `fetch_game_results` - Yesterday's box scores
6. `populate_actual_values` - Mark prop outcomes
7. `update_injuries` - Injury report sync
8. `load_team_games` - NBA API incremental
9. `update_team_stats` - Pace/ratings
10. `fetch_vegas_lines` - Game spreads/totals
11. `generate_xl_predictions` - XL + V3 models
12. `output_picks` - Write JSON + notify Discord

### `nba_validation_pipeline` (9:30 AM EST)

Validates yesterday's picks:
1. `load_yesterday_picks` - Read prediction JSON
2. `fetch_actual_results` - Get game outcomes
3. `calculate_performance` - Win rate, ROI
4. `update_tracking_sheet` - Log results
5. `send_summary` - Discord notification

### `nba_health_check` (Every 6 Hours)

Monitors system health:
- Database connectivity (4 DBs)
- Model files (24 .pkl files)
- Data freshness (props within 24h)
- Disk space (>10% free)
- Scheduler heartbeat

---

## Systemd Services

### Service Files

```bash
# /etc/systemd/system/airflow-scheduler.service
# /etc/systemd/system/airflow-webserver.service
```

### Management

```bash
# Start/stop/restart
sudo systemctl start airflow-scheduler airflow-webserver
sudo systemctl stop airflow-scheduler airflow-webserver
sudo systemctl restart airflow-scheduler airflow-webserver

# Enable on boot
sudo systemctl enable airflow-scheduler airflow-webserver

# Check status
sudo systemctl status airflow-scheduler airflow-webserver

# View logs
sudo journalctl -u airflow-scheduler -f
sudo journalctl -u airflow-webserver -f
```

---

## Local Development

### Setup (Docker-based)

```bash
cd airflow/

# Initialize
docker-compose -f docker-compose.airflow.yml up airflow-init

# Start services
docker-compose -f docker-compose.airflow.yml up -d

# Access at http://localhost:8080
```

### Mirror Production Config

```bash
# Copy production airflow.cfg
scp sportsuite@5.161.239.229:/home/sportsuite/sport-suite/airflow/airflow.cfg ./

# Update for local paths
sed -i 's|/home/sportsuite/sport-suite|/home/untitled/Sport-suite|g' airflow.cfg
```

---

## Troubleshooting

### Scheduler Not Running

```bash
# Check heartbeat
airflow jobs check --job-type SchedulerJob --hostname $(hostname)

# Restart
sudo systemctl restart airflow-scheduler
```

### Tasks Stuck in Queue

```bash
# Check executor
airflow config get-value core executor
# Should be: LocalExecutor

# Clear stuck tasks
airflow tasks clear nba_full_pipeline --start-date 2026-02-04
```

### Database Connection Issues

```bash
# Test from Airflow
airflow connections test nba_intelligence_db

# Test directly
PGPASSWORD=$DB_PASSWORD psql -h localhost -p 5539 -U $DB_USER -d airflow_metadata -c "SELECT 1;"
```

### DAGs Not Loading

```bash
# Check for syntax errors
python3 -c "import airflow.dags.nba_full_pipeline"

# Force rescan
airflow dags reserialize
```

---

## Directory Structure

```
airflow/
├── dags/
│   ├── nba_full_pipeline.py      # Main pipeline
│   ├── nba_validation_pipeline.py # Daily validation
│   └── nba_health_check.py       # Health monitoring
├── logs/                          # Task execution logs
├── airflow.cfg                    # Configuration
├── docker-compose.airflow.yml     # Local dev setup
└── README.md                      # This file
```

---

## Discord Integration

The bot (`/home/palworld/discord/nba_commands.py`) provides on-demand access:

| Command | Description |
|---------|-------------|
| `/nba` | Show today's picks |
| `/nba-run` | Trigger full pipeline |
| `/nba-refresh` | Quick line refresh |
| `/nba-status` | Check pipeline status |

Auto-DM: Picks sent to owner at 9:15 AM EST daily.

---

## Related

- [Main README](../README.md) - Project overview
- [betting_xl README](../nba/betting_xl/README.md) - Prediction system
- [docker README](../docker/README.md) - Database containers
