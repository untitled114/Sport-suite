# Notebooks

Interactive Jupyter notebooks demonstrating the NBA Player Props ML system using **real production data**.

**Important:** No synthetic or simulated data is used. All notebooks use real models, real predictions, and real outcomes.

---

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
- XL vs V3 model comparison

### 3. Live Prediction Example (`03_live_prediction_example.ipynb`)
End-to-end prediction pipeline walkthrough:
- Load production models (XL and V3)
- Examine real production picks
- Validate against actual game outcomes
- Visualize predictions vs results

### 4. Model Retraining Guide (`model_retraining_guide.ipynb`)
Step-by-step retraining workflow:
- Data validation checks
- Feature engineering
- Model training
- SHAP analysis
- Deployment verification

---

## Requirements

```bash
pip install jupyter numpy pandas matplotlib seaborn psycopg2-binary
pip install shap lightgbm  # For SHAP analysis and model loading
```

---

## Running

```bash
cd notebooks
jupyter notebook
# Or with JupyterLab:
jupyter lab
```

**Database requirement:** PostgreSQL databases must be running:
```bash
cd docker && docker-compose up -d
```

---

## Data Sources

All notebooks use **real data only**:
- Production models: `nba/models/saved_xl/*.pkl`
- Real predictions: `nba/betting_xl/predictions/*.json`
- Game results: PostgreSQL `player_game_logs` table
- Props outcomes: PostgreSQL `nba_props_xl` table

---

## Related

- [Main README](../README.md) - Project overview
- [Case Study](../docs/CASE_STUDY_GOBLIN_LINES.md) - Goblin lines analysis
- [betting_xl README](../nba/betting_xl/README.md) - Prediction system
