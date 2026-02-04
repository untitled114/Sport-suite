# Docker Infrastructure

Database infrastructure for the NBA Props ML system.

## Quick Start

```bash
# Start all databases
cd docker
docker-compose up -d

# Verify containers (8 databases)
docker ps | grep -E "(mlb|nba)_.*_db"

# Check health
docker-compose ps
```

## Databases

### NBA Databases

| Database | Port | Purpose | Tables |
|----------|------|---------|--------|
| nba_players_db | 5536 | Player data | player_profile, player_game_logs, player_rolling_stats |
| nba_games_db | 5537 | Game data | games, box_scores, game_context |
| nba_team_db | 5538 | Team data | teams, team_stats, team_rolling_stats |
| nba_intelligence_db | 5539 | Props + Airflow | nba_props_xl, injury_report, airflow_metadata |

### Additional Services

| Service | Port | Purpose |
|---------|------|---------|
| MongoDB | 27017 | Historical props archive (optional) |

**Note:** `nba_intelligence_db` also hosts the Airflow metadata database (`airflow_metadata`).

---

## Connection Examples

### PostgreSQL

```bash
# Load environment
source .env

# Connect to specific database
PGPASSWORD=$DB_PASSWORD psql -h localhost -p 5536 -U $DB_USER -d nba_players
PGPASSWORD=$DB_PASSWORD psql -h localhost -p 5537 -U $DB_USER -d nba_games
PGPASSWORD=$DB_PASSWORD psql -h localhost -p 5538 -U $DB_USER -d nba_team
PGPASSWORD=$DB_PASSWORD psql -h localhost -p 5539 -U $DB_USER -d nba_intelligence

# Quick query
PGPASSWORD=$DB_PASSWORD psql -h localhost -p 5539 -U $DB_USER -d nba_intelligence \
  -c "SELECT COUNT(*) FROM nba_props_xl WHERE game_date = CURRENT_DATE;"
```

### MongoDB (if using)

```bash
mongosh "mongodb://sports_user:$MONGO_PASSWORD@localhost:27017/nba_betting_xl"
```

---

## Backup & Restore

### Full Backup (All Databases)

```bash
./scripts/backup.sh
```
Creates timestamped, gzip-compressed backups in `backups/`.

### Single Database Backup

```bash
./scripts/backup.sh nba_intelligence
```

### List Available Backups

```bash
./scripts/restore.sh --list
```

### Restore from Backup

```bash
./scripts/restore.sh nba_intelligence_20260129_120000.sql.gz
```

### Verify Backup Integrity

```bash
./scripts/verify-backup.sh --all
./scripts/verify-backup.sh nba_intelligence
```

### Automated Backups (Cron)

```bash
# Add to crontab
0 3 * * * /home/untitled/Sport-suite/docker/scripts/backup.sh >> /var/log/nba-backup.log 2>&1
```

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DB_USER` | Database user | `mlb_user` |
| `DB_PASSWORD` | Database password | (required) |
| `BACKUP_DIR` | Backup directory | `docker/backups/` |
| `RETENTION_DAYS` | Backup retention | 7 days |

### `.env` File

```bash
# docker/.env
DB_USER=mlb_user
DB_PASSWORD=your_secure_password
MONGO_USER=sports_user
MONGO_PASSWORD=your_mongo_password
```

---

## Key Tables

### `nba_props_xl` (port 5539)

Main props table with all book lines:

```sql
-- Table schema
\d nba_props_xl

-- Daily prop count by book
SELECT book_name, COUNT(*)
FROM nba_props_xl
WHERE game_date = CURRENT_DATE
GROUP BY book_name
ORDER BY COUNT(*) DESC;

-- Check data freshness
SELECT MAX(fetch_timestamp) as latest_fetch
FROM nba_props_xl
WHERE game_date = CURRENT_DATE;
```

**Columns:** player_id, player_name, game_date, stat_type, book_name, over_line, consensus_line, line_spread, actual_value, fetch_timestamp, ...

**Indexes:** Optimized for game_date, stat_type, player_name, book_name queries.

### `player_game_logs` (port 5536)

Historical player performance:

```sql
-- Player's last 10 games
SELECT game_date, points, rebounds, assists, minutes
FROM player_game_logs
WHERE player_id = 12345
ORDER BY game_date DESC
LIMIT 10;
```

---

## Maintenance

### View Logs

```bash
docker-compose logs -f nba_intelligence_db
docker-compose logs -f nba_players_db
```

### Restart Single Service

```bash
docker-compose restart nba_intelligence_db
```

### Check Disk Usage

```bash
docker system df
docker volume ls --format "{{.Name}}: {{.Size}}"
```

### Reset Database (DESTRUCTIVE)

```bash
# Stop and remove volumes
docker-compose down -v

# Restart fresh
docker-compose up -d
```

---

## Volume Locations

Docker volumes stored in `/var/lib/docker/volumes/`:

```
docker_nba_players_data/
docker_nba_games_data/
docker_nba_team_data/
docker_nba_intelligence_data/
docker_nba_reference_data/
```

---

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker-compose logs nba_intelligence_db

# Check port conflicts
lsof -i :5539

# Rebuild container
docker-compose up -d --force-recreate nba_intelligence_db
```

### Connection Refused

```bash
# Verify container is running
docker ps | grep nba_intelligence

# Test from inside container
docker exec -it docker-nba_intelligence_db-1 psql -U mlb_user -d nba_intelligence -c "SELECT 1;"
```

### Out of Disk Space

```bash
# Clean unused resources
docker system prune -a

# Remove old backups
find backups/ -mtime +7 -delete
```

---

## Related

- [Main README](../README.md) - Project overview
- [Airflow README](../airflow/README.md) - Pipeline orchestration
- [betting_xl README](../nba/betting_xl/README.md) - Prediction system
