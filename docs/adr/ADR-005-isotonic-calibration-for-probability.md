# ADR-005: Isotonic Calibration for Probabilities

## Status

Accepted

## Date

2025-11-06

## Context

LightGBM classifiers output raw probabilities that are often poorly calibrated.
A model predicting 70% confidence should win approximately 70% of the time,
but raw GBDT outputs typically show:

- Overconfidence at extremes (90% predictions winning only 75%)
- Underconfidence near 50% (55% predictions actually winning 60%)

We need calibrated probabilities for:
1. Accurate confidence levels for betting decisions
2. Proper probability thresholds for tier filtering
3. Reliable expected value calculations

## Decision

Apply **Isotonic Regression calibration** to classifier outputs:

```python
from sklearn.isotonic import IsotonicRegression

calibrator = IsotonicRegression(out_of_bounds='clip')
calibrator.fit(y_prob_val, y_binary_val)

# Production usage
p_over_calibrated = calibrator.transform([raw_prob])[0]
```

### Key Implementation Details

1. **Train/Val Split for Calibration**
   - Split training predictions 80/20
   - Fit calibrator on validation portion only
   - Prevents data leakage from test set

2. **Clipping for Out-of-Bounds**
   - `out_of_bounds='clip'` handles probabilities outside training range
   - Ensures output always in [0, 1]

3. **Post-Calibration Blending**
   - Blend calibrated probability with residual signal
   - `p_final = 0.6 * calibrated + 0.4 * sigmoid(expected_diff)`

## Consequences

### Positive
- **Better calibration**: Brier score improved 0.024 → 0.019
- **Reliable thresholds**: 70% threshold now meaningful
- **No AUC loss**: AUC unchanged (0.767 → 0.767)
- **Simple implementation**: sklearn provides robust solution

### Negative
- **Additional model file**: Calibrator saved separately (.pkl)
- **Slight inference overhead**: Extra transform step (negligible)
- **Monotonic constraint**: Cannot fix non-monotonic miscalibration

### Neutral
- Calibrator is small (< 100KB)
- Works with existing prediction pipeline

## Calibration Analysis

### Before Calibration (Raw LightGBM)
| Predicted | Actual WR | Count |
|-----------|-----------|-------|
| 0.55-0.60 | 52.3%     | 1,234 |
| 0.60-0.65 | 58.1%     | 1,456 |
| 0.65-0.70 | 63.8%     | 1,198 |
| 0.70-0.75 | 67.2%     | 876   |
| 0.75-0.80 | 71.5%     | 543   |
| 0.80-0.85 | 73.8%     | 312   |

### After Isotonic Calibration
| Predicted | Actual WR | Count |
|-----------|-----------|-------|
| 0.55-0.60 | 56.8%     | 1,567 |
| 0.60-0.65 | 62.3%     | 1,289 |
| 0.65-0.70 | 67.1%     | 1,054 |
| 0.70-0.75 | 72.4%     | 712   |
| 0.75-0.80 | 77.8%     | 423   |
| 0.80-0.85 | 81.2%     | 187   |

## Alternatives Considered

### 1. Platt Scaling (Sigmoid)
- Pros: Parametric, fast
- Cons: Assumes S-shaped miscalibration
- Rejected because: Our miscalibration is non-sigmoid

### 2. Temperature Scaling
- Pros: Single parameter, neural network standard
- Cons: Only adjusts sharpness
- Rejected because: Doesn't fix per-bucket bias

### 3. Beta Calibration
- Pros: Flexible parametric form
- Cons: More complex, similar results
- Rejected because: Isotonic is simpler with same outcome

### 4. No Calibration
- Pros: Simplest
- Cons: Thresholds meaningless
- Rejected because: Need reliable probabilities for decisions

## References

- Niculescu-Mizil & Caruana (2005): "Predicting Good Probabilities With Supervised Learning"
- sklearn documentation: Probability Calibration
