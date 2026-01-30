# ADR-001: Two-Head Stacked Model Architecture

## Status

Accepted

## Date

2025-11-06

## Context

We need to predict whether a player will go OVER or UNDER their prop line.
This is fundamentally a binary classification problem, but we also need
to estimate the actual stat value to calculate edge.

Early models used either:
- Pure regression (predict stat value, compare to line)
- Pure classification (predict OVER/UNDER directly)

Neither approach alone provided optimal results:
- Regression ignores the line entirely during training
- Classification doesn't leverage the continuous nature of the problem

## Decision

Implement a **two-head stacked architecture**:

### Head 1: Regressor
- **Input**: 102 features (rolling stats, team context, book features)
- **Output**: Predicted stat value (e.g., 25.3 points)
- **Model**: LightGBM Regressor

### Head 2: Classifier
- **Input**: 103 features (102 base + `expected_diff` from regressor)
- **Output**: P(actual > line) probability
- **Model**: LightGBM Classifier with class weighting

### Stacking
1. Train regressor on features → predict absolute value
2. Calculate `expected_diff = prediction - line`
3. Augment features with `expected_diff`
4. Train classifier on augmented features → predict P(OVER)

### Final Blending
```python
residual_signal = sigmoid(expected_diff / 5.0)
p_over_final = 0.6 * calibrated_classifier + 0.4 * residual_signal
```

## Consequences

### Positive
- **Better AUC**: 0.767 vs 0.72 for regression-only
- **More information**: Classifier sees regressor's conviction
- **Edge calculation**: Regressor provides stat prediction for edge
- **Flexibility**: Can use either head's output depending on context

### Negative
- **Complexity**: Two models to maintain and deploy
- **Training time**: Sequential training increases pipeline time
- **Feature leakage risk**: Must carefully manage `expected_diff` injection

### Neutral
- Model files increased from 3 to 6 per market
- Inference requires both models loaded

## Alternatives Considered

### 1. Single Regression Model
- Pros: Simple, fast inference
- Cons: Line not used during training, suboptimal AUC
- Rejected because: Classification signal from line is valuable

### 2. Single Classification Model
- Pros: Directly optimizes for prediction task
- Cons: No stat value prediction, harder to calculate edge
- Rejected because: Need stat prediction for edge calculation

### 3. Multi-Task Learning
- Pros: Single model with two heads
- Cons: More complex implementation, harder to tune
- Rejected because: Stacking is simpler and achieves similar results

### 4. Three-Head Architecture
- Pros: Could add matchup-specific head
- Cons: Increased complexity
- Status: Explored as optional enhancement (3-head matchup model)
