# Docker Infrastructure

Database infrastructure for the NBA Props ML system.

## Quick Start

```bash
# Start all databases
docker-compose up -d

# Verify containers
docker ps | grep nba

# Check health
docker-compose ps
```

## Databases

| Database | Port | Purpose |
|----------|------|---------|
| nba_players_db | 5536 | Player profiles, game logs, rolling stats |
| nba_games_db | 5537 | Game schedules, box scores |
| nba_team_db | 5538 | Team profiles, team stats |
| nba_intelligence_db | 5539 | Prop lines, injuries, picks tracking |
| nba_reference_db | 5540 | Consolidated reference data |
| mongodb | 27017 | Multi-book prop lines (XL system) |

## Connection

```bash
# PostgreSQL
PGPASSWORD=$DB_PASSWORD psql -h localhost -p 5536 -U nba_user -d nba_players

# MongoDB
mongosh "mongodb://sports_user:$MONGO_PASSWORD@localhost:27017/nba_betting_xl"
```

## Backup & Restore

### Full Backup
```bash
./scripts/backup.sh
```
Creates timestamped, gzip-compressed backups in `backups/`.

### Single Database Backup
```bash
./scripts/backup.sh nba_intelligence
```

### List Backups
```bash
./scripts/restore.sh --list
```

### Restore
```bash
./scripts/restore.sh nba_intelligence_20260129_120000.sql.gz
```

### Verify Backup Integrity
```bash
./scripts/verify-backup.sh --all
```

## Backup Configuration

| Setting | Default | Environment Variable |
|---------|---------|---------------------|
| Backup directory | `docker/backups/` | `BACKUP_DIR` |
| Retention period | 7 days | `RETENTION_DAYS` |

## Maintenance

### View Logs
```bash
docker-compose logs -f nba_intelligence_db
```

### Restart Single Service
```bash
docker-compose restart nba_players_db
```

### Reset Database (DESTRUCTIVE)
```bash
docker-compose down -v  # Removes all data
docker-compose up -d
```

## Volume Locations

Docker volumes are stored in `/var/lib/docker/volumes/`:
- `nba_players_data`
- `nba_games_data`
- `nba_team_data`
- `nba_intelligence_data`
- `nba_reference_data`
- `nba_mongodb_data`
