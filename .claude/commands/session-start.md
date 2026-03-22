Before writing any code, run this checklist:

1. **Read the diff since last session**: `git log --oneline -10` and `git diff HEAD~1 --stat` — understand what changed
2. **Check test health**: `python3 -m pytest tests/unit/ -x -q --tb=line` — if anything is broken, fix it first
3. **Check pipeline health**: `source .env && python3 -c "from nba.config.database import get_connection; c=get_connection('intelligence'); cur=c.cursor(); cur.execute('SELECT COUNT(*) FROM nba_props_xl'); print(f'{cur.fetchone()[0]:,} props'); c.close()"` — verify DB is up
4. **Check model registry**: `source .env && python3 -c "from nba.models.model_registry import ModelRegistry; r=ModelRegistry(); [print(f'{m[\"market\"]} {m[\"status\"]}: {m[\"version\"]}') for m in r.list_models()]"` — know what's deployed
5. **Read open anomalies**: check `axiom.pipeline_runs` for recent failures
6. **State your plan** before starting work — what you're changing and why

After completing work:
- Run `python3 -m pytest tests/unit/ -q` — all tests must pass
- Run `/commit` if the user asks
- Update CHANGELOG.md if the change is user-facing
