# NBA Betting Pipeline - Apache Airflow

Production-grade orchestration for the NBA betting prediction system using Apache Airflow 2.x.

## Overview

This Airflow setup replaces the manual bash script orchestration (`nba/nba-predictions.sh`) with automated, monitored DAGs.

### DAGs

| DAG | Schedule | Description |
|-----|----------|-------------|
| `nba_morning_pipeline` | 10:00 AM EST | Data collection & enrichment |
| `nba_evening_pipeline` | 5:00 PM EST | Prediction generation |
| `nba_health_check` | Every 6 hours | System monitoring |

## Quick Start

### Prerequisites

- Docker and Docker Compose installed
- NBA databases running (ports 5536-5539)
- Environment variables configured

### 1. Set Environment Variables

Create a `.env` file in the `airflow/` directory:

```bash
# Database credentials
DB_USER=mlb_user
DB_PASSWORD=your_db_password_here
NBA_DB_HOST=host.docker.internal  # Use 'localhost' on Linux

# API Keys (if needed)
ODDS_API_KEY=your_api_key_here

# SMTP for alerts (optional)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password
SMTP_MAIL_FROM=your_email@gmail.com

# Airflow UID (Linux only - run: echo $(id -u))
AIRFLOW_UID=1000
```

### 2. Start Airflow

```bash
cd airflow/

# Initialize Airflow (first time only)
docker-compose -f docker-compose.airflow.yml up airflow-init

# Start all services
docker-compose -f docker-compose.airflow.yml up -d

# Check status
docker-compose -f docker-compose.airflow.yml ps
```

### 3. Access Airflow UI

Open http://localhost:8080 in your browser.

- **Username:** `admin`
- **Password:** `admin`

### 4. Enable DAGs

DAGs are paused by default. In the Airflow UI:

1. Navigate to DAGs view
2. Toggle on:
   - `nba_morning_pipeline`
   - `nba_evening_pipeline`
   - `nba_health_check`

## DAG Details

### Morning Pipeline (10:00 AM EST)

Tasks executed in order:

1. **fetch_props** - Multi-source collection from 7 sportsbooks
2. **load_props_to_db** - Store in PostgreSQL
3. **fetch_cheatsheet** - BettingPros recommendations
4. **load_cheatsheet_to_db** - Store cheatsheet data
5. **enrich_matchups** - Add opponent & home/away context
6. **fetch_game_results** - Yesterday's game stats
7. **populate_actual_values** - Update props with results
8. **update_injuries** - Injury report sync
9. **load_team_games** - NBA API incremental load
10. **update_team_stats** - Season pace/ratings
11. **load_team_advanced_stats** - Advanced metrics
12. **fetch_vegas_lines** - Game spreads & totals
13. **update_minutes_projections** - Minutes projections
14. **update_prop_history** - Bayesian hit rates
15. **verify_data_freshness** - Final validation

### Evening Pipeline (5:00 PM EST)

Tasks executed in order:

1. **health_check** - System readiness
2. **validate_data_freshness** - Data quality
3. **check_performance_thresholds** - Stop-loss monitoring
4. **refresh_props** - Capture line movements
5. **load_refreshed_props** - Store refreshed data
6. **refresh_vegas_lines** - Update spreads/totals
7. **refresh_cheatsheet** - Latest projections
8. **enrich_matchups** - Update context
9. **generate_xl_predictions** - XL model predictions
10. **generate_pro_picks** - Cheatsheet-based picks
11. **generate_odds_api_picks** - Pick6 multiplier picks
12. **output_final_picks** - Combine and output

### Health Check (Every 6 Hours)

Monitors:
- Database connectivity (4 databases)
- Model files (24 .pkl files)
- Data freshness (props, coverage)
- Disk space utilization
- API health

## Configuration

### Airflow Variables

Set via UI (Admin > Variables) or CLI:

| Variable | Description | Default |
|----------|-------------|---------|
| `nba_project_root` | Project root path | `/opt/airflow/project` |
| `alert_email` | Alert recipients | `alerts@example.com` |

### Airflow Connections

Pre-configured during initialization:

| Connection ID | Database | Port |
|--------------|----------|------|
| `nba_players_db` | nba_players | 5536 |
| `nba_games_db` | nba_games | 5537 |
| `nba_team_db` | nba_team | 5538 |
| `nba_intelligence_db` | nba_intelligence | 5539 |

To update connections (Admin > Connections):

```bash
# Via CLI
docker-compose -f docker-compose.airflow.yml run --rm airflow-cli \
  airflow connections delete nba_intelligence_db

docker-compose -f docker-compose.airflow.yml run --rm airflow-cli \
  airflow connections add 'nba_intelligence_db' \
    --conn-type 'postgres' \
    --conn-host 'your-host' \
    --conn-schema 'nba_intelligence' \
    --conn-login 'your-user' \
    --conn-password 'your-password' \
    --conn-port '5539'
```

## Manual Trigger

### Via UI

1. Go to DAGs view
2. Click on the DAG name
3. Click "Trigger DAG" (play button)

### Via CLI

```bash
# Trigger morning pipeline
docker-compose -f docker-compose.airflow.yml run --rm airflow-cli \
  airflow dags trigger nba_morning_pipeline

# Trigger evening pipeline
docker-compose -f docker-compose.airflow.yml run --rm airflow-cli \
  airflow dags trigger nba_evening_pipeline

# Trigger health check
docker-compose -f docker-compose.airflow.yml run --rm airflow-cli \
  airflow dags trigger nba_health_check
```

## Monitoring

### Task Logs

Via UI: Click task > Logs

Via CLI:
```bash
docker-compose -f docker-compose.airflow.yml logs -f airflow-scheduler
```

### Health Endpoint

```bash
curl http://localhost:8080/health
```

### Database

```bash
# Access Airflow metadata
docker-compose -f docker-compose.airflow.yml exec postgres \
  psql -U airflow -d airflow
```

## Scaling (CeleryExecutor)

For production with multiple workers, enable CeleryExecutor:

1. Uncomment `redis` and `airflow-worker` in `docker-compose.airflow.yml`
2. Update environment:
   ```yaml
   AIRFLOW__CORE__EXECUTOR: CeleryExecutor
   AIRFLOW__CELERY__BROKER_URL: redis://redis:6379/0
   ```
3. Scale workers:
   ```bash
   docker-compose -f docker-compose.airflow.yml up -d --scale airflow-worker=3
   ```

## Troubleshooting

### DAGs Not Appearing

```bash
# Check scheduler logs
docker-compose -f docker-compose.airflow.yml logs airflow-scheduler

# Force DAG rescan
docker-compose -f docker-compose.airflow.yml exec airflow-scheduler \
  airflow dags reserialize
```

### Database Connection Issues

```bash
# Test connection from inside container
docker-compose -f docker-compose.airflow.yml exec airflow-webserver \
  airflow connections test nba_intelligence_db
```

### Reset Everything

```bash
# Stop and remove all containers
docker-compose -f docker-compose.airflow.yml down -v

# Remove logs
rm -rf logs/*

# Reinitialize
docker-compose -f docker-compose.airflow.yml up airflow-init
docker-compose -f docker-compose.airflow.yml up -d
```

## Directory Structure

```
airflow/
├── dags/
│   ├── __init__.py
│   ├── nba_morning_pipeline.py    # Morning data collection
│   ├── nba_evening_pipeline.py    # Evening predictions
│   └── nba_health_check.py        # Health monitoring
├── plugins/
│   └── __init__.py
├── logs/                          # Task logs (auto-created)
├── docker-compose.airflow.yml     # Airflow services
└── README.md                      # This file
```

## Comparison with Bash Script

| Feature | Bash Script | Airflow |
|---------|-------------|---------|
| Scheduling | Manual cron | Built-in scheduler |
| Monitoring | Log files | Web UI + metrics |
| Retries | None | Configurable (3x default) |
| Alerting | None | Email/Slack on failure |
| Dependencies | Sequential only | DAG graph |
| Parallelism | Limited | Full support |
| History | Log files | Database storage |
| Backfills | Manual | Built-in support |

## Support

- **Logs:** `airflow/logs/` or Airflow UI
- **Predictions:** `nba/betting_xl/predictions/`
- **Health reports:** Sent via email on issues
