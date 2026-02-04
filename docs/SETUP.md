# NBA Sports Suite - Setup Guide

This guide covers setting up the NBA prediction system for local development or production deployment (e.g., Hetzner server).

## Quick Start

```bash
# 1. Clone the repository
git clone <repo-url>
cd Sport-suite

# 2. Copy environment template
cp .env.example .env

# 3. Edit .env with your credentials (see below)
nano .env

# 4. Source the environment
source .env
export DB_USER DB_PASSWORD ODDS_API_KEY BETTINGPROS_API_KEY

# 5. Start databases
cd docker && docker-compose up -d && cd ..

# 6. Run predictions
./nba/nba-predictions.sh
```

---

## Environment Variables

### Required Variables

Create a `.env` file in the project root with these **required** variables:

```bash
# =============================================================================
# DATABASE CREDENTIALS (Required)
# =============================================================================
DB_USER=mlb_user
DB_PASSWORD=your_secure_password_here

# =============================================================================
# API KEYS (Required for data fetching)
# =============================================================================
# BettingPros Premium API key (required for /v3/props endpoint)
BETTINGPROS_API_KEY=your_bettingpros_api_key

# The Odds API key (for Pick6 and live odds)
ODDS_API_KEY=your_odds_api_key
```

### Optional Variables

```bash
# =============================================================================
# DATABASE HOSTS (Optional - defaults to localhost)
# =============================================================================
NBA_DB_HOST=localhost
NBA_PLAYERS_DB_PORT=5536
NBA_GAMES_DB_PORT=5537
NBA_TEAM_DB_PORT=5538
NBA_INT_DB_PORT=5539

# =============================================================================
# MONGODB (Optional - only if using MongoDB features)
# =============================================================================
MONGO_USER=sports_user
MONGO_PASSWORD=your_mongo_password

# =============================================================================
# NOTIFICATIONS (Optional)
# =============================================================================
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
```

---

## Loading Environment Variables

### Option 1: Source before running (Development)

```bash
source .env
export DB_USER DB_PASSWORD ODDS_API_KEY BETTINGPROS_API_KEY
./nba/nba-predictions.sh
```

### Option 2: Use direnv (Recommended for development)

```bash
# Install direnv
sudo apt install direnv  # Ubuntu/Debian
brew install direnv      # macOS

# Add to shell config (~/.bashrc or ~/.zshrc)
eval "$(direnv hook bash)"  # or zsh

# Allow the directory
direnv allow

# Now .env is auto-loaded when you cd into the project
```

### Option 3: Systemd service (Production)

Create `/etc/systemd/system/nba-predictions.service`:

```ini
[Unit]
Description=NBA Predictions Pipeline
After=docker.service

[Service]
Type=oneshot
User=nba
WorkingDirectory=/opt/Sport-suite
EnvironmentFile=/opt/Sport-suite/.env
ExecStart=/opt/Sport-suite/nba/nba-predictions.sh
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

---

## Hetzner Production Setup

### 1. Server Provisioning

Recommended specs:
- **CPU**: 4+ vCPUs (for parallel model loading)
- **RAM**: 8GB minimum (16GB recommended for training)
- **Storage**: 50GB SSD (100GB if storing historical data)
- **OS**: Ubuntu 22.04 LTS

### 2. Initial Setup

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y python3.10 python3.10-venv python3-pip \
    docker.io docker-compose postgresql-client git bc jq

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Clone repository
cd /opt
sudo git clone <repo-url> Sport-suite
sudo chown -R $USER:$USER Sport-suite
cd Sport-suite

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Environment Configuration

```bash
# Create production .env
cp .env.example .env
nano .env

# IMPORTANT: Use strong passwords in production!
# Generate a secure password:
openssl rand -base64 32
```

### 4. Database Setup

```bash
# Start PostgreSQL containers
cd docker
cp .env.example .env
# Edit docker/.env with same DB_PASSWORD as root .env
nano .env

docker-compose up -d

# Verify databases are running
docker ps | grep nba

# Test connection
PGPASSWORD=$DB_PASSWORD psql -h localhost -p 5539 -U $DB_USER -d nba_intelligence -c "SELECT 1;"
```

### 5. Cron Schedule (Production)

```bash
# Edit crontab
crontab -e

# Add these entries (adjust timezone as needed):

# Morning data refresh (10 AM EST)
0 10 * * * cd /opt/Sport-suite && source .env && export DB_USER DB_PASSWORD ODDS_API_KEY BETTINGPROS_API_KEY && ./nba/nba-predictions.sh full >> /var/log/nba/morning.log 2>&1

# Evening predictions (5 PM EST)
0 17 * * * cd /opt/Sport-suite && source .env && export DB_USER DB_PASSWORD ODDS_API_KEY BETTINGPROS_API_KEY && ./nba/nba-predictions.sh refresh >> /var/log/nba/evening.log 2>&1

# Hourly line updates (during game days)
0 12-23 * * * cd /opt/Sport-suite && source .env && export DB_USER DB_PASSWORD && ./nba/nba-predictions.sh update_props >> /var/log/nba/lines.log 2>&1

# Create log directory
sudo mkdir -p /var/log/nba
sudo chown $USER:$USER /var/log/nba
```

### 6. Firewall Configuration

```bash
# Only allow SSH and internal services
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw enable

# PostgreSQL should NOT be exposed externally
# Keep ports 5536-5539 internal only
```

---

## Troubleshooting

### "DB_PASSWORD environment variable is required"

The shell script requires `DB_PASSWORD` to be set:

```bash
# Check if it's set
echo $DB_PASSWORD

# If empty, source your .env file
source .env
export DB_PASSWORD
```

### Database connection refused

```bash
# Check if containers are running
docker ps | grep nba

# If not running, start them
cd docker && docker-compose up -d

# Check logs for errors
docker logs nba_intelligence_db
```

### Permission denied on script

```bash
chmod +x nba/nba-predictions.sh
```

### Missing Python packages

```bash
source venv/bin/activate
pip install -r requirements.txt
```

---

## Health Check

Run the health check to verify everything is configured:

```bash
./nba/nba-predictions.sh health
```

This verifies:
- Database connectivity
- Model files present
- Props data available
- Matchup coverage
- Disk space

---

## Security Best Practices

1. **Never commit `.env` files** - They're in `.gitignore` but double-check
2. **Use strong passwords** - Generate with `openssl rand -base64 32`
3. **Rotate API keys** periodically
4. **Keep PostgreSQL internal** - Don't expose ports 5536-5539 externally
5. **Regular backups** - Use `docker/scripts/backup.sh`

---

## Support

- Check logs: `tail -f nba/betting_xl/logs/pipeline_$(date +%Y-%m-%d).log`
- Run health check: `./nba/nba-predictions.sh health`
- Validate picks: `./nba/nba-predictions.sh validate --7d`
