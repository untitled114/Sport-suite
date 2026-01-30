---
name: Data Pipeline Issue
about: Report issues with data fetching, loading, or processing
title: '[DATA] '
labels: data-pipeline
assignees: ''
---

## Pipeline Stage

Select the affected stage:

- [ ] **Fetching** - API calls, data retrieval
- [ ] **Loading** - Database insertion, deduplication
- [ ] **Feature Extraction** - Rolling stats, book features, computed features
- [ ] **Prediction** - Model inference, probability calibration

## Data Source

Select the affected source(s):

- [ ] BettingPros API (`/v3/events`, `/v3/props`)
- [ ] ESPN API (fallback schedule)
- [ ] Underdog Fantasy
- [ ] Database (specify below)

### Database Affected (if applicable)

| Database | Table | Port |
|----------|-------|------|
| nba_intelligence | nba_prop_lines | 5539 |
| nba_players | player_game_logs | 5536 |
| nba_players | player_rolling_stats | 5536 |
| nba_games | games | 5537 |
| nba_team | team_stats | 5538 |

## Date Range

- **Start Date:** YYYY-MM-DD
- **End Date:** YYYY-MM-DD
- **Total Records Affected:** (estimated)

## Issue Description

Describe what went wrong with the data pipeline.

## Sample Data (Sanitized)

<details>
<summary>Example of Problematic Data</summary>

```json
{
  "player_name": "REDACTED",
  "game_date": "2025-01-15",
  "stat_type": "POINTS",
  "line": 24.5,
  "issue": "describe the issue with this record"
}
```

</details>

## Expected Data Format

What the data should look like:

```json
{
  "example": "expected format"
}
```

## Steps to Reproduce

```bash
# Commands to reproduce the issue
cd /home/untitled/Sport-suite

```

## Error Messages

<details>
<summary>API Response (if applicable)</summary>

```json
{
  "status_code": 000,
  "response": "..."
}
```

</details>

<details>
<summary>Database Error (if applicable)</summary>

```sql
-- Query that failed or returned unexpected results
```

</details>

## Data Validation

- [ ] Checked for NULL values in required fields
- [ ] Verified date format consistency
- [ ] Confirmed team/player name mappings
- [ ] Validated numeric ranges (lines, stats)

## Impact

- **Severity:** Critical / High / Medium / Low
- **Blocks Predictions:** Yes / No
- **Affects Historical Data:** Yes / No
- **Affects Live Data:** Yes / No

## Temporary Workaround (if any)

Describe any temporary fix or workaround you've found.

## Additional Context

Add any other context about the problem here.
