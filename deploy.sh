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

# Pre-flight checks
echo "=== Pre-flight checks ==="

# Verify SSH connectivity
ssh -o ConnectTimeout=10 -o BatchMode=yes $SERVER "echo ok" > /dev/null 2>&1 || {
  echo "[ERROR] Cannot connect to $SERVER via SSH"
  exit 1
}
echo "[OK] SSH connection"

# Verify remote .env exists (will not be overwritten since it's excluded)
ssh $SERVER "test -f $REMOTE_DIR/.env" || {
  echo "[ERROR] Remote .env file missing at $REMOTE_DIR/.env"
  exit 1
}
echo "[OK] Remote .env exists"

# Check remote disk space (fail if < 500MB free)
REMOTE_FREE_KB=$(ssh $SERVER "df --output=avail $REMOTE_DIR 2>/dev/null | tail -1" 2>/dev/null || echo "0")
if [ "$REMOTE_FREE_KB" -lt 512000 ] 2>/dev/null; then
  echo "[WARNING] Low disk space on remote: ${REMOTE_FREE_KB}KB free"
fi

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

# Post-deploy verification
echo ""
echo "=== Post-deploy verification ==="
ssh $SERVER "cd $REMOTE_DIR && source venv/bin/activate && python3 -c '
import sys
sys.path.insert(0, \".\")
try:
    from nba.config.database import get_intelligence_db_config
    from nba.betting_xl.xl_predictor import XLPredictor
    print(\"[OK] Core imports successful\")
except ImportError as e:
    print(f\"[ERROR] Import failed: {e}\")
    sys.exit(1)
'" || {
  echo "[WARNING] Post-deploy verification failed - check remote Python environment"
}

echo ""
echo "=== Deployment complete ==="
