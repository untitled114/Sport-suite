# Production Safeguards Configuration

**Status:** Production Ready
**Last Updated:** February 2026

Production policies, monitoring, and data validation for the NBA XL betting system.

---

## Overview

This directory contains safeguards that protect against:
- Poor model performance (stop-loss triggers)
- Stale data (freshness validation)
- Edge anomalies (outlier detection)
- Consecutive losses (automatic pausing)

---

## Files

### `production_policies.py`
Defines all production safety policies:
- **Stop-loss triggers**: WR thresholds per market
- **Warning thresholds**: Early warning before stop-loss
- **Consecutive loss limits**: Max 3 consecutive losing days
- **Edge anomaly detection**: Alert if edge > 15.0

### `data_freshness_validator.py`
Validates data quality before generating predictions:
- Props are from today (minimum 50 required)
- Game results are within 24 hours
- Rolling stats are current
- Injury reports are within 12 hours

### `monitor.py`
Performance monitoring and stop-loss system:
- Calculates rolling 7-day WR and ROI by market
- Detects consecutive losing days
- Compares performance vs validation benchmarks
- Auto-stops betting if triggers hit

---

## Current Thresholds

Based on February 2026 validation (79.5% filtered WR):

| Market | Stop-Loss WR | Warning WR | Validated Benchmark |
|--------|--------------|------------|---------------------|
| POINTS | 55.0% | 60.0% | 74.2% |
| REBOUNDS | 60.0% | 70.0% | 92.3% |
| ASSISTS | DISABLED | - | Poor AUC |
| THREES | DISABLED | - | Poor AUC |

**Logic:**
- If 7-day rolling WR < stop-loss → STOP betting that market
- If 7-day rolling WR < warning → WARNING alert
- If 3+ consecutive losing days → STOP all betting

---

## Escalation Levels

### Level 1: WARNING
**Triggers:**
- WR 10-15 points below benchmark
- Average edge < 1.0
- Picks volume < 5 per day

**Actions:**
- Log warning with metrics
- Continue with caution
- Increase monitoring

### Level 2: CAUTION
**Triggers:**
- WR 15-20 points below benchmark
- 2 consecutive losing days
- Edge anomaly (>15.0)

**Actions:**
- PAUSE affected market
- Analyze recent picks
- Manual review required

### Level 3: STOP
**Triggers:**
- WR >20 points below benchmark
- 3 consecutive losing days
- Negative 7-day ROI

**Actions:**
- STOP ALL BETTING
- Full investigation
- Manual approval to resume

---

## Data Freshness Requirements

| Component | Max Age | Validation |
|-----------|---------|------------|
| Props | Today's date | COUNT > 50 |
| Game Results | 24 hours | Games resolved |
| Rolling Stats | 7 days | Recent season |
| Injury Reports | 12 hours | Active status |
| Models | 30 days | File timestamp |

---

## Integration

### Pre-Flight Checks (Run Before Predictions)

```bash
# Check data freshness
python3 config/data_freshness_validator.py

# Check performance thresholds
python3 monitor.py
```

### Automated in Airflow

The `nba_full_pipeline` DAG includes:
1. `check_data_freshness` task
2. `check_performance_thresholds` task
3. Both must pass before `generate_xl_predictions`

### Manual Override

If safeguards block predictions incorrectly:
```bash
# Skip checks (use with caution)
python3 generate_xl_predictions.py --skip-validation
```

---

## Usage Examples

### Check if market should be stopped

```python
from config.production_policies import should_stop_market

if should_stop_market('POINTS', rolling_wr=50.5):
    print("STOP betting POINTS - WR below threshold")
```

### Run data freshness check

```bash
python3 config/data_freshness_validator.py
# Exit code 0 = PASSED, 1 = FAILED
```

### Run performance monitor

```bash
python3 monitor.py
# Exit code 0 = SAFE, 1 = STOP-LOSS TRIGGERED
```

---

## Troubleshooting

### Safeguards blocking when performance is good
- Check `picks_log.json` has recent data
- Verify rolling window (7 days)
- Ensure results are updated daily

### Data freshness always failing
- Check database connectivity (ports 5536-5539)
- Verify props fetched today
- Check injury report timestamp

### Stop-loss not triggering
- Verify `picks_log.json` has WIN/LOSS results
- Ensure 20+ picks in 7-day window
- Check threshold values in `production_policies.py`

---

## Related

- [betting_xl README](../README.md) - Prediction system
- [Case Study](../../../docs/CASE_STUDY_GOBLIN_LINES.md) - Performance analysis
- [Airflow README](../../../airflow/README.md) - Pipeline orchestration
