Diagnose what's wrong with the system. Run these checks in order and report findings:

1. **Database connectivity**:
   ```
   source .env && python3 -c "
   from nba.config.database import get_connection
   for s in ['players','games','teams','intelligence','axiom','features']:
       try:
           c = get_connection(s); c.cursor().execute('SELECT 1'); c.close(); print(f'  {s}: OK')
       except Exception as e: print(f'  {s}: FAIL — {e}')
   "
   ```

2. **Model registry status**:
   ```
   source .env && python3 -c "
   from nba.models.model_registry import ModelRegistry
   r = ModelRegistry()
   for m in r.list_models():
       print(f'  {m[\"market\"]:10} {m[\"status\"]:12} {m[\"version\"]}  AUC={m[\"auc\"]}')
   "
   ```

3. **Recent pipeline runs** (check for anomalies):
   ```
   PGPASSWORD=$DB_PASSWORD psql -h localhost -p 5500 -U $DB_USER -d sportsuite -c "
   SELECT run_date, status, summary->>'picks_generated' as picks,
          jsonb_array_length(COALESCE(anomalies, '[]'::jsonb)) as anomalies
   FROM axiom.pipeline_runs ORDER BY started_at DESC LIMIT 5;"
   ```

4. **Test health**: `python3 -m pytest tests/unit/ -x -q --tb=line`

5. **Recent prediction performance**:
   ```
   PGPASSWORD=$DB_PASSWORD psql -h localhost -p 5500 -U $DB_USER -d sportsuite -c "
   SELECT run_date, COUNT(*) as picks,
          SUM(CASE WHEN is_hit THEN 1 ELSE 0 END) as wins,
          ROUND(AVG(CASE WHEN is_hit THEN 1.0 ELSE 0.0 END) * 100, 1) as wr_pct
   FROM axiom.nba_prediction_history
   WHERE is_hit IS NOT NULL
   GROUP BY run_date ORDER BY run_date DESC LIMIT 7;"
   ```

6. **Docker containers**: `docker ps --format '{{.Names}}\t{{.Status}}' | grep -E 'sportsuite|nba_|cephalon'`

Report what's healthy, what's degraded, and what's down.
