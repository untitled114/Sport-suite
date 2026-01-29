# Production Safeguards Configuration

This directory contains production policies, monitoring, and data validation for the NBA XL betting system.

## Files

### `production_policies.py`
Defines all production safety policies:
- **Stop-loss triggers**: WR thresholds per market (POINTS: 52%, REBOUNDS: 55%)
- **Warning thresholds**: Early warning before stop-loss
- **Consecutive loss limits**: Max 3 consecutive losing days
- **Edge anomaly detection**: Alert if edge > 15.0 or avg > 8.0
- **Escalation levels**: WARNING → CAUTION → STOP

**Usage:**
```python
from betting_xl.config.production_policies import (
    should_stop_market,
    should_warn_market,
    get_escalation_level
)

# Check if POINTS market should be stopped
if should_stop_market('POINTS', rolling_wr=50.5):
    print("STOP betting POINTS - WR below threshold")
```

### `data_freshness_validator.py`
Validates data quality before generating predictions:
- Props are from today (minimum 50 props required)
- Game results are within 24 hours
- Rolling stats are current
- Injury reports are within 12 hours
- Models are not older than 30 days

**Usage:**
```bash
# Run validation check
python3 data_freshness_validator.py

# Exit code 0 = PASSED, 1 = FAILED
```

**Integrated into automation workflow:**
- Called during evening workflow pre-flight checks
- Blocks prediction generation if critical failures detected

### `monitor.py`
Performance monitoring and stop-loss system:
- Calculates rolling 7-day WR and ROI by market
- Detects consecutive losing days
- Checks WR vs stop-loss thresholds
- Compares performance vs validation benchmarks
- Auto-stops betting if triggers hit

**Usage:**
```bash
# Run performance monitoring
python3 ../monitor.py

# Exit code 0 = SAFE, 1 = STOP-LOSS TRIGGERED
```

**Integrated into automation workflow:**
- Called during evening workflow pre-flight checks
- Blocks prediction generation if stop-loss triggered

## Stop-Loss Thresholds

| Market | Stop-Loss WR | Warning WR | Validation Benchmark |
|--------|--------------|------------|---------------------|
| POINTS | 52.0% | 54.0% | 59.2% |
| REBOUNDS | 55.0% | 58.0% | 63.5% |
| ASSISTS | DISABLED | DISABLED | 14.6% (losing) |
| THREES | DISABLED | DISABLED | 46.5% (losing) |

**Logic:**
- If 7-day rolling WR < stop-loss threshold → STOP betting that market
- If 7-day rolling WR < warning threshold → WARNING alert
- If 3+ consecutive losing days → STOP all betting

## Escalation Procedures

### Level 1: WARNING
**Triggers:**
- WR drop of 5-10 percentage points vs validation
- Average edge < 1.0 (possible line efficiency)
- Picks volume < 5 per day (limited opportunities)

**Actions:**
- Log detailed warning with metrics
- Increase monitoring frequency
- Continue betting with caution
- Review edge distribution

### Level 2: CAUTION
**Triggers:**
- WR drop of 10-15 percentage points vs validation
- 2 consecutive losing days
- Edge anomaly detected (>15.0 on any pick)
- Data freshness failure (props > 24 hours old)

**Actions:**
- PAUSE affected market immediately
- Analyze recent picks for patterns
- Check data quality (injuries, rolling stats)
- Review model performance on recent data
- Manual review before resuming

### Level 3: STOP
**Triggers:**
- WR drop >15 percentage points vs validation
- 3 consecutive losing days
- Negative ROI over 7-day rolling window
- Critical data freshness failure
- Model age > 30 days

**Actions:**
- STOP ALL BETTING immediately
- Full system investigation required
- Retrain models if needed
- Validate data pipeline integrity
- Manual approval required to resume

## Data Freshness Requirements

| Component | Requirement | Max Age |
|-----------|-------------|---------|
| Props | Must be from today | CURRENT_DATE |
| Game Results | Recent games processed | 24 hours |
| Rolling Stats | Current season stats | 7 days |
| Injury Reports | Active injury status | 12 hours |
| Models | Retrain if too old | 30 days (recommended: 14 days) |

## Integration with Automation

### Pre-Flight Checks (Evening Workflow)
```bash
./nba-predictions.sh evening
```

Executes in order:
1. **Data Freshness Validator** - Ensures all data is current
2. **Performance Monitor** - Checks stop-loss triggers
3. If both pass → Generate predictions
4. If either fails → ABORT with error message

### Health Check
```bash
./nba-predictions.sh health
```

Tests all safeguard components:
- Database connectivity
- XL model files
- Recent props availability
- Data freshness validator executable
- Performance monitor executable

## Performance Tracking

### picks_log.json Format
Location: `betting_xl/logs/picks_log.json`

```json
{
  "picks": [
    {
      "player_name": "Giannis Antetokounmpo",
      "stat_type": "POINTS",
      "side": "OVER",
      "prediction": 32.5,
      "line": 30.5,
      "edge": 2.0,
      "edge_pct": 6.56,
      "best_book": "draftkings",
      "game_date": "2025-11-07",
      "generated_at": "2025-11-07T17:00:00-05:00",
      "result": "WIN"  // or "LOSS" or "PENDING"
    }
  ]
}
```

**Updating Results:**
Results should be updated the next morning after games complete.
This can be automated via a results tracking script.

## Testing the Safeguards

### Test Data Freshness Validator
```bash
cd $(pwd)/nba
python3 betting_xl/config/data_freshness_validator.py
```

Expected output: Validation report with status for each check

### Test Performance Monitor
```bash
cd $(pwd)/nba
python3 betting_xl/monitor.py
```

Expected output: Performance report with rolling WR/ROI by market

### Test Full Workflow
```bash
cd $(pwd)/nba
./nba-predictions.sh health    # Verify all systems operational
./nba-predictions.sh evening   # Run full evening workflow with safeguards
```

If stop-loss triggered, workflow will abort with clear error message.

## Modifying Policies

To adjust stop-loss thresholds or other policies:

1. Edit `production_policies.py`
2. Update threshold values (e.g., `STOP_LOSS_WR_THRESHOLDS`)
3. No restart required - changes take effect immediately

Example:
```python
# Make POINTS more conservative
STOP_LOSS_WR_THRESHOLDS = {
    'POINTS': 54.0,  # Changed from 52.0% → 54.0%
    'REBOUNDS': 55.0
}
```

## Troubleshooting

### Safeguards blocking picks when performance is good
- Check picks_log.json has recent data
- Verify rolling window calculation (7 days)
- Ensure results are being updated daily

### Data freshness validator always failing
- Check database connectivity (ports 5536, 5539)
- Verify props were fetched today
- Check injury report update timestamp

### Performance monitor not detecting issues
- Verify picks_log.json is being updated
- Check that results (WIN/LOSS) are being populated
- Ensure enough picks in 7-day window (minimum 20)

## Production Checklist

Before deploying to production:
- [ ] All safeguard scripts tested and working
- [ ] Stop-loss thresholds reviewed and confirmed
- [ ] Data freshness requirements validated
- [ ] picks_log.json initialized
- [ ] Automation script tested end-to-end
- [ ] Escalation procedures documented
- [ ] Rollback plan in place

---

**Last Updated:** November 7, 2025
**Status:** Production Safeguards Complete ✅
