#!/bin/bash
# Deploy to production server
# Usage: ./deploy.sh [--restart] [--dry-run]
#
# Options:
#   --restart   Restart Airflow services after deployment
#   --dry-run   Show what would be synced without making changes

set -e

SERVER="sportsuite@5.161.239.229"
REMOTE_DIR="/home/sportsuite/sport-suite"

# Parse arguments
DRY_RUN=""
RESTART=false

for arg in "$@"; do
  case $arg in
    --dry-run)
      DRY_RUN="--dry-run"
      ;;
    --restart)
      RESTART=true
      ;;
    *)
      echo "Unknown option: $arg"
      echo "Usage: ./deploy.sh [--restart] [--dry-run]"
      exit 1
      ;;
  esac
done

echo "=== Deploying to production ==="
echo "Server: $SERVER"
echo "Remote: $REMOTE_DIR"
[ -n "$DRY_RUN" ] && echo "Mode: DRY RUN (no changes will be made)"
echo ""

# Sync code (excludes data, logs, venv, secrets)
rsync -avz --delete $DRY_RUN \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='venv' \
  --exclude='.venv' \
  --exclude='docker/backups/*' \
  --exclude='airflow/logs/*' \
  --exclude='airflow/airflow.db' \
  --exclude='airflow/airflow.cfg' \
  --exclude='nba/betting_xl/lines/*.json' \
  --exclude='nba/betting_xl/predictions/*.json' \
  --exclude='nba/features/datasets/*.csv' \
  --exclude='.env' \
  --exclude='docker/.env' \
  --exclude='*.log' \
  --exclude='*.pkl' \
  --exclude='catboost_info/' \
  /home/untitled/Sport-suite/ $SERVER:$REMOTE_DIR/

if [ -n "$DRY_RUN" ]; then
  echo ""
  echo "[DRY RUN] No changes made. Remove --dry-run to deploy."
  exit 0
fi

echo ""
echo "[OK] Code synced"

# Restart services if requested
if [ "$RESTART" = true ]; then
  echo ""
  echo "=== Restarting Airflow services ==="
  ssh $SERVER "sudo systemctl restart airflow-scheduler airflow-webserver"
  echo "[OK] Services restarted"

  # Verify services are running
  echo ""
  echo "=== Verifying services ==="
  ssh $SERVER "systemctl is-active airflow-scheduler airflow-webserver" || {
    echo "[ERROR] Services failed to start!"
    exit 1
  }
  echo "[OK] Services running"
fi

echo ""
echo "=== Deployment complete ==="
