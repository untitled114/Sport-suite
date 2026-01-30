# ADR-003: LightGBM over XGBoost/CatBoost

## Status

Accepted

## Date

2025-10-15

## Context

We need a gradient boosting framework for both regression and classification
tasks in our two-head model architecture. The main contenders are:

1. **LightGBM** (Microsoft)
2. **XGBoost** (DMLC)
3. **CatBoost** (Yandex)

All three are mature, production-ready frameworks with similar performance
characteristics for most tabular data problems.

## Decision

Use **LightGBM** as the primary gradient boosting framework for:
- Regressor head (predicting stat values)
- Classifier head (predicting P(OVER))

### Key Configuration
```python
LGBMRegressor(
    objective='regression',
    boosting_type='gbdt',
    num_leaves=63,
    learning_rate=0.02,
    n_estimators=2000,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
)
```

## Consequences

### Positive
- **Fast training**: 2-3x faster than XGBoost for our dataset
- **Memory efficient**: Histogram-based algorithm uses less RAM
- **Native categorical handling**: Can handle book_id directly
- **Built-in early stopping**: Prevents overfitting
- **Good default parameters**: Less hyperparameter tuning needed

### Negative
- **Leaf-wise growth**: Can overfit on small datasets (mitigated by early stopping)
- **Less interpretable**: Harder to explain individual trees
- **Version sensitivity**: Model pickles may not load across major versions

### Neutral
- Similar final accuracy to XGBoost (~0.5% difference)
- Similar inference speed for our model sizes

## Alternatives Considered

### 1. XGBoost
- Pros: More mature, level-wise growth is more stable
- Cons: Slower training (3-4 minutes vs 1-2 for LightGBM)
- Rejected because: Training speed matters for iteration

### 2. CatBoost
- Pros: Best categorical handling, ordered boosting
- Cons: Slowest training, largest model files
- Rejected because: Training time too slow for rapid iteration

### 3. sklearn RandomForest
- Pros: Simpler, more interpretable
- Cons: Significantly lower accuracy (0.68 AUC vs 0.767)
- Rejected because: Accuracy difference too large

### 4. Neural Networks (MLP)
- Pros: Can learn complex interactions
- Cons: Needs more data, less interpretable
- Rejected because: GBDT outperforms on tabular data of our size

## Benchmarks

Training on 24,316 samples, 102 features:

| Framework | Training Time | AUC | File Size |
|-----------|--------------|-----|-----------|
| LightGBM  | 67s          | 0.767 | 2.8 MB  |
| XGBoost   | 182s         | 0.764 | 4.1 MB  |
| CatBoost  | 294s         | 0.769 | 12.3 MB |
