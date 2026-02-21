---
name: proceed-with-qa-audit
description: >
  Quality-assess lopezdp_utils modules against López de Prado theory and apply fixes.
  Fully autonomous — execute end-to-end without stopping.
  Use when the user says "proceed with qa-audit" or "continue qa-audit".
---

# qa-audit Workflow

Quality-assess lopezdp_utils modules against López de Prado theory and apply fixes.

## Execution Mode

**FULLY AUTONOMOUS** — Execute the complete workflow end-to-end without stopping to ask for permission.

DO NOT ask before:
- Creating/merging/deleting feature branches
- Creating directories/files
- Running quality checks
- Updating TODO.md or other documentation
- Committing and pushing

Proceed directly through all steps. Only stop if encountering an unrecoverable error.

## Scope Per Session

One module quality assessment per run. If the assessed module requires NO fixes (zero P0/P1 issues), immediately proceed to assess the next unchecked module in the same session. This allows a single 2-hour window to cover multiple clean modules.

## Full Lifecycle

### 1. Setup

```
git checkout main
git pull origin main
```

Read TODO.md to determine the next unit of work:
- If user specified a module → use that
- If user said "proceed"/"continue" → find next unchecked item
- If ALL items are checked → exit with message: "All work complete. Nothing to do."

### 2. Branch

```
git checkout -b feat/qa-audit-<module-name>
```

### 3. Work

Invoke the `quality-assessment` skill for the target module. Follow ALL steps 1-7 exactly:

1. **Read all source files** in `src/tradelab/lopezdp_utils/<module>/`
2. **Read all test files** in `tests/<module>/`
3. **Query NotebookLM** via the `notebooklm-research` skill for the relevant theory:
   - First query: comprehensive overview of ALL concepts from the relevant chapters
   - Second query (same session): specific algorithms, parameter conventions, edge cases
4. **Run existing tests**: `uv run pytest tests/<module>/ -v --tb=short`
5. **Compare** implementation vs theory — produce structured assessment (correctness, missing functionality, edge cases, API consistency, test quality)
6. **Output the structured report** to `docs/quality_assessment/<module>.md`
7. **Decide and apply fixes**:
   - **If zero P0 and zero P1 issues**: mark module as PASS, save report, skip to next module (go back to step 1 of Setup with next unchecked TODO item)
   - **If P0/P1 issues exist**: apply fixes following the criteria in the quality-assessment skill (be critical — only fix what is clearly wrong per the book, no refactoring, no large new features)

### 4. Finalize

Run quality gate:
```
uvx ruff check --fix . && uvx ruff format .
uv run pytest -v --tb=short
```

Update TODO.md — mark completed items with `[x]`.

### 5. Commit

Stage only the files you created or modified (never use `git add -A`):
```
git add <specific files changed>
git commit -m "feat(qa-audit): assess <module> — <PASS|FIXED N issues>"
```

### 6. Complete

```
git checkout main
git merge --no-ff feat/qa-audit-<module-name> -m "Merge feat/qa-audit-<module>: quality assessment <PASS|FIXED>"
git branch -d feat/qa-audit-<module-name>
```

### 7. Update Progress

Update TODO.md with final status. If all items complete, note completion.

## Error Handling

| Error | Action |
|-------|--------|
| NotebookLM unavailable | Retry once after 30 seconds. If still fails, exit with error — do NOT assess without NotebookLM |
| Quality gate fails | Attempt auto-fix. If still fails, fix manually and retry |
| Merge conflict | Abort merge, leave feature branch intact, exit with error message |
| Tests fail after applying fixes | Debug and correct the fix. If unable to resolve in 3 attempts, revert the fix and document as "needs manual review" |
| Nothing to do | Exit with message: "All work complete. Nothing to do." |

## Completion Condition

All 6 modules assessed and validated (all TODO.md items checked): data, labeling, features, modeling, evaluation, allocation.
