---
name: Bug Report
about: Report a bug or unexpected behavior
title: '[BUG] '
labels: bug
assignees: ''
---

## Description

A clear and concise description of the bug.

## Environment

| Component | Version/Status |
|-----------|----------------|
| Python | `python --version` |
| OS | e.g., Ubuntu 22.04, macOS 14 |
| Docker | Running / Not Running |
| Branch | e.g., main, feature/xyz |
| Commit | `git rev-parse --short HEAD` |

## Database Status

```bash
# Run: docker ps | grep -E "(mlb|nba)_.*_db"
```

| Database | Port | Status |
|----------|------|--------|
| nba_players | 5536 | Up / Down |
| nba_games | 5537 | Up / Down |
| nba_team | 5538 | Up / Down |
| nba_intelligence | 5539 | Up / Down |

## Steps to Reproduce

1.
2.
3.

## Expected Behavior

What you expected to happen.

## Actual Behavior

What actually happened.

## Relevant Logs

<details>
<summary>Error Output</summary>

```
Paste error messages or stack traces here
```

</details>

<details>
<summary>Pipeline Logs</summary>

```bash
# Check: tail -100 nba/betting_xl/logs/pipeline_$(date +%Y-%m-%d).log
```

</details>

## Model Information (if applicable)

- Market: POINTS / REBOUNDS / ASSISTS / THREES
- Model directory: `nba/models/saved_xl/`
- Feature count: 102 (expected)

## Additional Context

Add any other context about the problem here.

## Possible Fix

If you have suggestions on how to fix the bug, describe them here.
