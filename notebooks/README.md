# Notebooks

Interactive Jupyter notebooks demonstrating the NBA Player Props ML system using **real production data**.

## Notebooks

### 1. Feature Importance Analysis (`01_feature_importance_analysis.ipynb`)
Analyze what drives model predictions:
- LightGBM native feature importance (gain, split)
- SHAP values for interpretable ML
- Feature category breakdown (player stats, book disagreement, H2H, etc.)
- Insights on most predictive features

### 2. Validation Deep Dive (`02_validation_deep_dive.ipynb`)
Comprehensive performance analysis using real results:
- Win rate by edge tier
- Calibration analysis
- ROI over time (cumulative profit)
- Market-by-market breakdown

### 3. Live Prediction Example (`03_live_prediction_example.ipynb`)
End-to-end prediction pipeline walkthrough:
- Load production models
- Examine real production picks
- Validate against actual game outcomes
- Visualize predictions vs results

## Requirements

```bash
pip install jupyter numpy pandas matplotlib seaborn psycopg2-binary
pip install shap  # Optional, for SHAP analysis
```

## Database Setup

Notebooks require PostgreSQL databases running:
```bash
cd docker
docker-compose up -d
```

## Running

```bash
cd notebooks
jupyter notebook
```

Or with JupyterLab:
```bash
jupyter lab
```

## Data Sources

All notebooks use **real data only**:
- Production models from `nba/models/saved_xl/`
- Real prediction files from `nba/betting_xl/predictions/`
- Actual game results from PostgreSQL database (`nba_players`)

**No synthetic or simulated data is used.**
