.PHONY: help install test lint fmt run refresh validate picks train walk-forward dbt-run dbt-test dbt-docs db-up db-down backup clean

# Default target
help:
	@echo "NBA Player Props ML System"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Setup:"
	@echo "  install      Install dependencies (dev mode)"
	@echo "  db-up        Start PostgreSQL databases"
	@echo "  db-down      Stop databases"
	@echo ""
	@echo "Development:"
	@echo "  test         Run all tests with coverage"
	@echo "  lint         Run linters (black, isort, flake8)"
	@echo "  fmt          Auto-format code"
	@echo "  typecheck    Run mypy type checking"
	@echo ""
	@echo "Pipeline:"
	@echo "  run          Run full pipeline (data + predictions)"
	@echo "  refresh      Quick refresh (line movements only)"
	@echo "  validate     Validate yesterday's picks"
	@echo "  validate-7d  Validate last 7 days"
	@echo "  picks        Show current picks"
	@echo ""
	@echo "Training:"
	@echo "  train           Retrain all XL models (POINTS, REBOUNDS)"
	@echo "  train-v3        Retrain all V3 models (136 features)"
	@echo "  train-points    Retrain POINTS XL model only"
	@echo "  train-rebounds  Retrain REBOUNDS XL model only"
	@echo "  train-points-v3 Retrain POINTS V3 model only"
	@echo "  train-rebounds-v3 Retrain REBOUNDS V3 model only"
	@echo "  validate-data   Run data quality checks before training"
	@echo "  walk-forward    Run walk-forward validation (all markets)"
	@echo "  walk-forward-points   Walk-forward for POINTS only"
	@echo "  walk-forward-rebounds Walk-forward for REBOUNDS only"
	@echo ""
	@echo "dbt:"
	@echo "  dbt-run      Run dbt models"
	@echo "  dbt-test     Run dbt tests"
	@echo "  dbt-docs     Generate and serve dbt docs"
	@echo ""
	@echo "Maintenance:"
	@echo "  backup       Backup all databases"
	@echo "  clean        Remove build artifacts"

# =============================================================================
# Setup
# =============================================================================

install:
	pip install -e ".[dev]"
	pre-commit install

db-up:
	cd docker && docker-compose up -d
	@echo "Waiting for databases to be ready..."
	@sleep 5
	@echo "Databases started on ports 5536-5539"

db-down:
	cd docker && docker-compose down

# =============================================================================
# Development
# =============================================================================

test:
	python3 -m pytest tests/ -v --cov=nba --cov-report=term-missing --cov-fail-under=70

test-fast:
	python3 -m pytest tests/ -x -q --tb=short

lint:
	black --check nba/ tests/
	isort --check-only nba/ tests/
	flake8 nba/ --max-line-length=100 --extend-ignore=E203,E402,E501,W503

fmt:
	black nba/ tests/
	isort nba/ tests/

typecheck:
	mypy nba/ --ignore-missing-imports || true

# =============================================================================
# Pipeline
# =============================================================================

run:
	./nba/nba-predictions.sh full

refresh:
	./nba/nba-predictions.sh refresh

validate:
	./nba/nba-predictions.sh validate

validate-7d:
	./nba/nba-predictions.sh validate --7d

picks:
	./nba/nba-predictions.sh picks

# =============================================================================
# Training
# =============================================================================

validate-data:
	@echo "Running data quality checks..."
	python3 -m nba.core.data_quality_checks --pre-training
	@echo "Data validation complete"

train: validate-data
	@echo "Retraining POINTS model..."
	python3 nba/models/train_market.py --market POINTS --data nba/features/datasets/xl_training_POINTS_2023_2025.csv
	@echo "Retraining REBOUNDS model..."
	python3 nba/models/train_market.py --market REBOUNDS --data nba/features/datasets/xl_training_REBOUNDS_2023_2025.csv
	@echo "Generating SHAP analysis..."
	python3 -m nba.models.generate_feature_importance --all
	@echo "Training complete"

train-points: validate-data
	python3 nba/models/train_market.py --market POINTS --data nba/features/datasets/xl_training_POINTS_2023_2025.csv
	python3 -m nba.models.generate_feature_importance --market POINTS

train-rebounds: validate-data
	python3 nba/models/train_market.py --market REBOUNDS --data nba/features/datasets/xl_training_REBOUNDS_2023_2025.csv
	python3 -m nba.models.generate_feature_importance --market REBOUNDS

train-v3: validate-data
	@echo "Retraining POINTS V3 model (136 features)..."
	python3 nba/models/train_market.py --market POINTS --model-version v3 --data nba/features/datasets/xl_training_POINTS_2023_2025.csv
	@echo "Retraining REBOUNDS V3 model (136 features)..."
	python3 nba/models/train_market.py --market REBOUNDS --model-version v3 --data nba/features/datasets/xl_training_REBOUNDS_2023_2025.csv
	@echo "V3 training complete"

train-points-v3: validate-data
	python3 nba/models/train_market.py --market POINTS --model-version v3 --data nba/features/datasets/xl_training_POINTS_2023_2025.csv

train-rebounds-v3: validate-data
	python3 nba/models/train_market.py --market REBOUNDS --model-version v3 --data nba/features/datasets/xl_training_REBOUNDS_2023_2025.csv

build-dataset:
	@echo "Building training datasets..."
	python3 nba/features/build_xl_training_dataset.py --output nba/features/datasets/
	@echo "Datasets built"

walk-forward: validate-data
	@echo "Running walk-forward validation for POINTS..."
	python3 -m nba.models.walk_forward_validation --market POINTS --output nba/models/walk_forward_POINTS.txt
	@echo "Running walk-forward validation for REBOUNDS..."
	python3 -m nba.models.walk_forward_validation --market REBOUNDS --output nba/models/walk_forward_REBOUNDS.txt
	@echo "Walk-forward validation complete. See nba/models/walk_forward_*.txt"

walk-forward-points:
	python3 -m nba.models.walk_forward_validation --market POINTS

walk-forward-rebounds:
	python3 -m nba.models.walk_forward_validation --market REBOUNDS

# =============================================================================
# dbt
# =============================================================================

dbt-run:
	cd dbt && dbt run

dbt-test:
	cd dbt && dbt test

dbt-docs:
	cd dbt && dbt docs generate && dbt docs serve

dbt-build:
	cd dbt && dbt build

# =============================================================================
# Maintenance
# =============================================================================

backup:
	./docker/scripts/backup.sh

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache
	rm -rf nba/__pycache__ nba/**/__pycache__
	rm -rf .coverage coverage.xml coverage_html/
	rm -rf dbt/target dbt/logs dbt/dbt_packages
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleaned build artifacts"

# =============================================================================
# Shortcuts
# =============================================================================

t: test-fast
l: lint
f: fmt
r: run
p: picks
v: validate

# =============================================================================
# DEPLOYMENT
# =============================================================================

.PHONY: deploy deploy-restart server-status server-logs

SERVER := sportsuite@5.161.239.229

## Deploy code to production (no restart)
deploy:
	@./deploy.sh

## Deploy and restart Airflow services
deploy-restart:
	@./deploy.sh --restart

## Check server status
server-status:
	@ssh $(SERVER) "systemctl status airflow-scheduler airflow-webserver --no-pager | head -20"

## Tail server logs
server-logs:
	@ssh $(SERVER) "journalctl -u airflow-scheduler -f"

## Run predictions on server
server-predict:
	@ssh $(SERVER) "cd /home/sportsuite/sport-suite && source venv/bin/activate && source .env && export DB_USER DB_PASSWORD TERM=xterm && ./nba/nba-predictions.sh"

## SSH to server
ssh:
	@ssh $(SERVER)
