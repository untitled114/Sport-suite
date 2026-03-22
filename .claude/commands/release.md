Release a new version. Takes a version bump type as argument: `major`, `minor`, or `patch`.

Steps:
1. Run all tests: `python3 -m pytest tests/unit/ -q` — must pass
2. Read current version from `nba/__init__.py`
3. Bump version based on argument:
   - `patch`: 3.0.0 → 3.0.1 (bug fixes)
   - `minor`: 3.0.0 → 3.1.0 (new features, backward compatible)
   - `major`: 3.0.0 → 4.0.0 (breaking changes)
4. Update `nba/__init__.py` with new version
5. Move CHANGELOG.md `[Unreleased]` section to `[X.Y.Z] - YYYY-MM-DD`
6. Add empty `[Unreleased]` section
7. Commit: `chore(release): vX.Y.Z`
8. Tag: `git tag vX.Y.Z`
9. Print the release summary

Do NOT push — let the user review and push manually.
