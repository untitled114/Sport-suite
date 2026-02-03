# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.0.0] - 2026-02-03

### Added
- **V3 Models** (136 features) running in parallel with XL models (102 features)
- 33 new V3 features: temporal decay, volatility metrics, H2H decay, line/book features
- `extract_v3_features()` method in `extract_live_features_xl.py`
- Dual model prediction system - XL and V3 picks generated independently
- `model_version` field in picks output ("xl" or "v3")
- PrizePicks integration (fetcher, loader, backfill scripts)
- Stake sizing configuration with Kelly-inspired logic
- Season phase and conditional threshold configurations
- Walk-forward cross-validation for model training

### Changed
- Predictions now include both XL (102 features) and V3 (136 features) picks
- `nba-predictions.sh` displays XL + V3 breakdown in output
- Health check expects 38+ model files (12 XL + 12 V3 + existing)
- Validation tracks results by model version separately
- Refactored `xl_predictor.py` with helper methods for cleaner code

### Fixed
- Airflow DAG schedules now use explicit EST timezone handling
- Removed hardcoded password fallbacks (use env vars only)

### Models
- **POINTS V3**: AUC 0.740, R² 0.548 (136 features)
- **REBOUNDS V3**: AUC 0.739, R² 0.530 (136 features)
- XL models unchanged (AUC 0.767/0.749 for POINTS/REBOUNDS)

## [1.1.0] - 2026-01-29

### Added
- Pydantic schemas for data validation (`nba/core/schemas.py`)
- Feature drift detection with KS test (`nba/core/drift_detection.py`)
- MLflow experiment tracking integration (`nba/core/experiment_tracking.py`)
- Model cards for all four markets (POINTS, REBOUNDS, ASSISTS, THREES)
- Custom exceptions module (`nba/core/exceptions.py`)
- CONTRIBUTING.md with development guidelines
- Missing `__init__.py` files for proper package structure

### Changed
- Renamed `nba/betting_xl/fetchers/utils.py` to `normalization.py` to avoid import shadowing
- Updated pyproject.toml with pydantic and mlflow dependencies

## [1.0.0] - 2025-11-06

### Added
- Initial release of NBA Props ML System
- Stacked two-head LightGBM architecture (regressor + classifier)
- 102 features (78 player + 20 book disagreement + 4 computed)
- Multi-book line shopping across 7 sportsbooks
- Isotonic calibration for probability reliability
- Ensemble blending (60% classifier, 40% residual-based)

### Models
- **POINTS**: AUC 0.765, deployed (56.7% validation WR)
- **REBOUNDS**: AUC 0.748, deployed (61.2% validation WR)
- **ASSISTS**: AUC 0.588, disabled (14.6% validation WR)
- **THREES**: AUC 0.717, disabled (46.5% validation WR)

### Infrastructure
- PostgreSQL databases for player, game, team, and intelligence data
- BettingPros API integration for prop lines
- Underdog Fantasy integration for DFS lines
- ESPN API fallback for game schedules

### Documentation
- README.md with system overview
- SYSTEM_DESIGN.md with architecture diagrams
- DATABASE_REFERENCE.md with schema documentation
- PREDICTION_SCHEMA.md with output format documentation
- VALIDATION.md with methodology documentation

## [0.9.0] - 2025-11-05 (Pre-release)

### Fixed
- Critical home/away data bug (models were training on 100% home games)
- Enriched training data from `player_game_logs` table
- Achieved 54.5% home / 45.5% away distribution (realistic)

### Changed
- Archived invalid models to `saved_home_only_INVALID/`
- Retrained all models with correct data distribution

## [0.8.0] - 2025-10-15 (Development)

### Added
- Feature extraction pipeline
- Training dataset builder
- Initial model training scripts
- Database schema design

---

## Version History Summary

| Version | Date       | Highlights                                    |
|---------|------------|-----------------------------------------------|
| 2.0.0   | 2026-02-03 | V3 models (136 features), dual XL+V3 system   |
| 1.1.0   | 2026-01-29 | Pydantic schemas, drift detection, MLflow     |
| 1.0.0   | 2025-11-06 | Production release, multi-book line shopping  |
| 0.9.0   | 2025-11-05 | Home/away bug fix, model retraining           |
| 0.8.0   | 2025-10-15 | Initial development version                   |
