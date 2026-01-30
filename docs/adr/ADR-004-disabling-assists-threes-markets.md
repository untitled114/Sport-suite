# ADR-004: Disabling ASSISTS and THREES Markets

## Status

Accepted

## Date

2025-11-07

## Context

During validation of the XL models (October 23 - November 4, 2024 data),
we observed significantly different performance across markets:

| Market | Win Rate | ROI | Status |
|--------|----------|-----|--------|
| POINTS | 56.7% | +8.27% | Profitable |
| REBOUNDS | 61.2% | +16.96% | Profitable |
| ASSISTS | 14.6% | -72.05% | Severe loss |
| THREES | 46.5% | -11.23% | Loss |

The ASSISTS and THREES models are not just underperforming - they are
actively losing money at rates that could significantly damage overall returns.

## Decision

**Disable ASSISTS and THREES markets** in production:

```python
ACTIVE_MARKETS = frozenset({"POINTS", "REBOUNDS"})

DISABLED_MARKETS = {
    "ASSISTS": "14.6% WR, -72.05% ROI (severe underperformance)",
    "THREES": "46.5% WR, -11.23% ROI (losing strategy)",
}
```

### Implementation
1. `get_market_config()` returns `enabled=False` for disabled markets
2. `LineOptimizer.optimize_line()` returns `None` for disabled markets
3. Prediction pipeline skips disabled markets
4. Models still maintained for future improvement

## Consequences

### Positive
- **Protected capital**: Avoid predictable losses
- **Better ROI**: Portfolio-level returns improve significantly
- **Focused development**: Can concentrate on improving active markets
- **Clear signal**: Data shows these markets need different approach

### Negative
- **Reduced volume**: Fewer betting opportunities per day
- **Missed improvements**: May miss if market becomes predictable
- **Wasted training**: Models trained but not used

### Neutral
- Training pipeline unchanged (can still generate models)
- Can re-enable if performance improves

## Analysis

### Why ASSISTS Fails (14.6% WR)

1. **High variance**: Assists depend heavily on game flow
2. **Teammate factor**: Star player assists depend on teammates making shots
3. **Blowout effect**: Blowouts reduce playing time and assist opportunities
4. **Book accuracy**: Books are better at pricing assists

### Why THREES Fails (46.5% WR)

1. **Extreme variance**: 3PT shooting is streaky
2. **Small sample**: Players take fewer threes than total field goals
3. **Game script**: Trailing teams force more threes
4. **Defensive schemes**: Teams specifically game-plan 3PT defense

## Alternatives Considered

### 1. Increase Probability Thresholds
- Tried: Raised min_p_over from 0.58 to 0.70
- Result: Volume dropped 80%, WR still below 50%
- Rejected: Not enough high-confidence picks

### 2. Player-Specific Models
- Tried: Filter to high-usage players only
- Result: Marginal improvement (49% WR)
- Rejected: Still losing money

### 3. Different Feature Set
- Tried: Added lineup-dependent features
- Result: AUC improved 0.02, WR unchanged
- Status: Requires more research

### 4. Ensemble with Third Model
- Tried: Added matchup-specific head
- Result: Minimal improvement
- Rejected: Complexity not justified

## Future Work

1. Research alternative modeling approaches for ASSISTS/THREES
2. Explore deep learning for sequence modeling (game flow)
3. Investigate player-specific models for high-volume shooters
4. Monitor market efficiency trends for re-evaluation
