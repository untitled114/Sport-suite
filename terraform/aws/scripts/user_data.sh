#!/bin/bash
# Sport-Suite EC2 Bootstrap
# Runs once on first launch via cloud-init.
# Sets up the base environment — code deployed separately via deploy.sh.
set -euo pipefail
exec > >(tee /var/log/user-data.log | logger -t user-data) 2>&1

echo "[bootstrap] Starting Sport-Suite bootstrap at $(date)"

# ============================================================================
# System packages
# ============================================================================
apt-get update -qq
apt-get install -y -qq \
  curl wget git vim tmux htop \
  build-essential libssl-dev libffi-dev \
  python3.11 python3.11-venv python3.11-dev python3-pip \
  postgresql-client \
  jq unzip \
  ca-certificates gnupg lsb-release \
  awscli

# ============================================================================
# Docker
# ============================================================================
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
  > /etc/apt/sources.list.d/docker.list

apt-get update -qq
apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-compose-plugin

systemctl enable docker
systemctl start docker

# ============================================================================
# sportsuite user
# ============================================================================
if ! id sportsuite &>/dev/null; then
  useradd -m -s /bin/bash -G docker,sudo sportsuite
fi

# Passwordless sudo for sportsuite (matches Hetzner setup)
echo "sportsuite ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/sportsuite
chmod 0440 /etc/sudoers.d/sportsuite

# ============================================================================
# Data volume mount (/dev/xvdf → /data)
# PostgreSQL data, ML models, and predictions live here.
# EBS persists independently of the instance.
# ============================================================================
DATA_DEVICE="/dev/xvdf"
DATA_MOUNT="/data"

# Wait for EBS volume to attach (up to 60s)
for i in $(seq 1 12); do
  if [ -b "$DATA_DEVICE" ]; then break; fi
  echo "[bootstrap] Waiting for $DATA_DEVICE to attach... ($i/12)"
  sleep 5
done

if [ -b "$DATA_DEVICE" ]; then
  # Format only if no filesystem present
  if ! blkid "$DATA_DEVICE" | grep -q "TYPE"; then
    echo "[bootstrap] Formatting $DATA_DEVICE as ext4"
    mkfs.ext4 -F "$DATA_DEVICE"
  fi

  mkdir -p "$DATA_MOUNT"
  mount "$DATA_DEVICE" "$DATA_MOUNT"

  # Persist mount across reboots
  DEVICE_UUID=$(blkid -s UUID -o value "$DATA_DEVICE")
  echo "UUID=$DEVICE_UUID $DATA_MOUNT ext4 defaults,nofail 0 2" >> /etc/fstab

  # Create directory structure on data volume
  mkdir -p \
    "$DATA_MOUNT/postgres" \
    "$DATA_MOUNT/models" \
    "$DATA_MOUNT/predictions" \
    "$DATA_MOUNT/logs" \
    "$DATA_MOUNT/backups"

  chown -R sportsuite:sportsuite "$DATA_MOUNT"
  echo "[bootstrap] Data volume mounted at $DATA_MOUNT"
else
  echo "[bootstrap] WARNING: $DATA_DEVICE not found — data volume may not have attached yet"
fi

# ============================================================================
# Project directories
# ============================================================================
PROJECT_DIR="/home/sportsuite/sport-suite"
mkdir -p "$PROJECT_DIR"
chown -R sportsuite:sportsuite /home/sportsuite

# Symlink data directories into project (matches Hetzner layout)
sudo -u sportsuite ln -sfn "$DATA_MOUNT/models"      "$PROJECT_DIR/nba/models/saved_xl"      2>/dev/null || true
sudo -u sportsuite ln -sfn "$DATA_MOUNT/predictions" "$PROJECT_DIR/nba/betting_xl/predictions" 2>/dev/null || true
sudo -u sportsuite ln -sfn "$DATA_MOUNT/logs"        "$PROJECT_DIR/nba/betting_xl/logs"        2>/dev/null || true

# ============================================================================
# Python tooling
# ============================================================================
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
pip3 install --upgrade pip uv

# ============================================================================
# AWS CLI profile for S3 (uses instance role — no keys needed)
# ============================================================================
mkdir -p /home/sportsuite/.aws
cat > /home/sportsuite/.aws/config <<'EOF'
[default]
region = us-east-1
output = json
EOF
chown -R sportsuite:sportsuite /home/sportsuite/.aws

# ============================================================================
# Backup cron — runs existing docker/scripts/backup.sh daily at 2 AM
# and syncs to S3. Script path set after code is deployed.
# ============================================================================
cat > /etc/cron.d/sport-suite-backup <<'EOF'
0 2 * * * sportsuite /home/sportsuite/sport-suite/docker/scripts/backup.sh >> /data/logs/backup.log 2>&1
EOF

# ============================================================================
# SSH hardening
# ============================================================================
sed -i 's/^#*PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config
sed -i 's/^#*PermitRootLogin.*/PermitRootLogin no/' /etc/ssh/sshd_config
systemctl reload sshd

# ============================================================================
# Done
# ============================================================================
echo "[bootstrap] Sport-Suite bootstrap complete at $(date)"
echo "[bootstrap] Next: run deploy.sh to push code, then start services"
