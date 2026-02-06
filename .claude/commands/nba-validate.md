Run the NBA pick validation report.

Execute the validation script from the betting_xl directory:

```
cd /home/untitled/Sport-suite/nba/betting_xl && python3 validate_predictions.py $ARGUMENTS
```

- If the user provides arguments (e.g., `--start-date 2026-01-15 --end-date 2026-02-05`), pass them through as $ARGUMENTS.
- If no arguments are provided, run with NO arguments (the script defaults to the last 7 days).
- The script requires database access. Make sure to source env vars first: `source /home/untitled/Sport-suite/.env`
- Run from the `nba/betting_xl/` directory so the `predictions/` path resolves correctly.

Show the full output to the user. Do not summarize or truncate it.
