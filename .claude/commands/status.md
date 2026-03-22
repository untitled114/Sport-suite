Quick system status. Show:

1. Current version from `nba/__init__.py`
2. Git status: `git log --oneline -5` and `git status --short`
3. Test count: `python3 -m pytest tests/unit/ -q --co 2>&1 | tail -1`
4. Coverage: last known from `coverage.xml` if it exists
5. Model versions deployed (query model_registry)
6. Database row counts for key tables

Keep output concise — one screen max.
