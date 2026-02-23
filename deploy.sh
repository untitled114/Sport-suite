#!/bin/bash
# Deploy to production server
# Usage: ./deploy.sh [OPTIONS]
#
# Options:
#   --restart        Restart Airflow services after deployment
#   --restart-bot    Restart Cephalon Axiom (NBA bot) after deployment
#   --restart-fleet  Restart all 3 Cephalon bots after deployment
#   --deploy-fleet   Deploy shared cephalon/ module to server
#   --deploy-axiom   Deploy Axiom bot code to server
#   --deploy-lumen   Deploy Lumen bot code to server
#   --deploy-solace  Deploy Solace bot code to server
#   --deploy-atlas   Deploy Atlas bot code to server
#   --dry-run        Show what would be synced without making changes

set -e

SERVER="${DEPLOY_SERVER:?Set DEPLOY_SERVER env var (e.g. user@your-server-ip)}"
REMOTE_DIR="/home/sportsuite/sport-suite"
FLEET_DIR="/home/cephalons"

# Parse arguments
DRY_RUN=""
RESTART=false
RESTART_BOT=false
RESTART_FLEET=false
DEPLOY_FLEET=false
DEPLOY_AXIOM=false
DEPLOY_LUMEN=false
DEPLOY_SOLACE=false
DEPLOY_ATLAS=false

for arg in "$@"; do
  case $arg in
    --dry-run)
      DRY_RUN="--dry-run"
      ;;
    --restart)
      RESTART=true
      ;;
    --restart-bot)
      RESTART_BOT=true
      ;;
    --restart-fleet)
      RESTART_FLEET=true
      ;;
    --deploy-fleet)
      DEPLOY_FLEET=true
      ;;
    --deploy-axiom)
      DEPLOY_AXIOM=true
      ;;
    --deploy-lumen)
      DEPLOY_LUMEN=true
      ;;
    --deploy-solace)
      DEPLOY_SOLACE=true
      ;;
    --deploy-atlas)
      DEPLOY_ATLAS=true
      ;;
    *)
      echo "Unknown option: $arg"
      echo "Usage: ./deploy.sh [--restart] [--restart-bot] [--restart-fleet] [--deploy-fleet] [--deploy-axiom] [--deploy-lumen] [--deploy-solace] [--deploy-atlas] [--dry-run]"
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

# Sync main code (excludes data, logs, venv, secrets)
echo "=== Syncing main codebase ==="
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
  --exclude='discord/' \
  /home/untitled/Sport-suite/ $SERVER:$REMOTE_DIR/

# Local Cephalons directory
CEPHALONS_DIR="/home/untitled/Cephalons"

# Deploy shared cephalon module (to shared location accessible by all bots)
if [ "$DEPLOY_FLEET" = true ]; then
  echo ""
  echo "=== Deploying Cephalon Fleet (shared module) ==="
  ssh $SERVER "mkdir -p $FLEET_DIR"
  rsync -avz --delete $DRY_RUN \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    $CEPHALONS_DIR/cephalon/ $SERVER:$FLEET_DIR/cephalon/
  echo "[OK] Cephalon module synced to $FLEET_DIR/cephalon/"
fi

# Deploy Axiom bot
if [ "$DEPLOY_AXIOM" = true ]; then
  echo ""
  echo "=== Deploying Cephalon Axiom ==="
  rsync -avz $DRY_RUN \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    $CEPHALONS_DIR/axiom/bot.py \
    $CEPHALONS_DIR/axiom/nba_commands.py \
    $SERVER:$REMOTE_DIR/discord/
  echo "[OK] Axiom bot synced"
fi

# Deploy Lumen bot
if [ "$DEPLOY_LUMEN" = true ]; then
  echo ""
  echo "=== Deploying Cephalon Lumen ==="
  rsync -avz $DRY_RUN \
    $CEPHALONS_DIR/lumen/bot.py \
    $SERVER:/home/palworld/discord/bot.py
  echo "[OK] Lumen bot.py synced"
fi

# Deploy Solace bot (staging via /tmp — /home/trading/solace/ is owned by trading)
if [ "$DEPLOY_SOLACE" = true ]; then
  echo ""
  echo "=== Deploying Cephalon Solace ==="
  rsync -avz $DRY_RUN \
    $CEPHALONS_DIR/solace/bot.py \
    $CEPHALONS_DIR/solace/commands.py \
    $SERVER:/tmp/solace_deploy/
  if [ -z "$DRY_RUN" ]; then
    ssh $SERVER "sudo cp /tmp/solace_deploy/bot.py /tmp/solace_deploy/commands.py /home/trading/solace/ && \
      sudo chown trading:trading /home/trading/solace/bot.py /home/trading/solace/commands.py && \
      rm -rf /tmp/solace_deploy"
  fi
  echo "[OK] Solace bot synced"
fi

# Deploy Atlas bot
if [ "$DEPLOY_ATLAS" = true ]; then
  echo ""
  echo "=== Deploying Cephalon Atlas ==="
  ssh $SERVER "mkdir -p $FLEET_DIR/atlas"
  rsync -avz $DRY_RUN \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='logs/' \
    $CEPHALONS_DIR/atlas/bot.py \
    $CEPHALONS_DIR/atlas/monitors.py \
    $CEPHALONS_DIR/atlas/formatter.py \
    $SERVER:$FLEET_DIR/atlas/
  echo "[OK] Atlas bot synced to $FLEET_DIR/atlas/"
fi

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

# Restart Axiom bot if requested
if [ "$RESTART_BOT" = true ]; then
  echo ""
  echo "=== Restarting Cephalon Axiom (NBA bot) ==="
  ssh $SERVER "sudo systemctl restart cephalon-axiom"
  echo "[OK] Cephalon Axiom restarted"

  # Verify service is running
  ssh $SERVER "systemctl is-active cephalon-axiom" || {
    echo "[ERROR] Cephalon Axiom failed to start!"
    exit 1
  }
  echo "[OK] Cephalon Axiom running"
fi

# Restart all fleet bots if requested
if [ "$RESTART_FLEET" = true ]; then
  echo ""
  echo "=== Restarting Cephalon Fleet ==="

  for service in cephalon-axiom cephalon-lumen cephalon-solace cephalon-atlas; do
    echo "  Restarting $service..."
    ssh $SERVER "sudo systemctl restart $service" && \
      echo "  [OK] $service restarted" || \
      echo "  [WARN] $service restart failed (may not exist)"
  done

  echo ""
  echo "=== Verifying Fleet ==="
  for service in cephalon-axiom cephalon-lumen cephalon-solace cephalon-atlas; do
    STATUS=$(ssh $SERVER "systemctl is-active $service 2>/dev/null" || echo "inactive")
    echo "  $service: $STATUS"
  done
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
