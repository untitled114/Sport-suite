# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
| 1.0.0   | 2025-11-06 | Production release, multi-book line shopping  |
| 0.9.0   | 2025-11-05 | Home/away bug fix, model retraining           |
| 0.8.0   | 2025-10-15 | Initial development version                   |
