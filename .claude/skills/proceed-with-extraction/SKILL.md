---
name: proceed-with-extraction
description: >
  Orchestrates López de Prado extraction workflow for tradelab-lib-lopezdp-utils. Use when
  starting a work session, when user asks to proceed/continue/resume extraction work, or when
  user says "let's extract Chapter N". Handles both starting new chapters and resuming
  interrupted sessions. Supports headless/cron execution with full branch lifecycle.
---

# López de Prado Extraction Workflow

Orchestrates chapter-by-chapter extraction following the methodology defined in WORKFLOW.md.

## Execution Mode

**FULLY AUTONOMOUS** — Execute the complete workflow end-to-end without stopping to ask for permission.

DO NOT ask before:
- Creating/merging/deleting feature branches
- Creating directories/files for new submodules
- Implementing functions
- Running quality checks (ruff)
- Updating TODO.md, README.md, or other documentation
- Committing and merging to main

Proceed directly through all steps. Only stop if encountering an unrecoverable error.

## Scope Per Session

**Extract exactly ONE chapter per session.** This keeps context usage manageable and ensures
clean git history. If the user specifies a chapter, extract that one. If resuming, pick up
the current in-progress chapter or start the next unchecked one.

## Full Lifecycle

### 1. Setup

```
git checkout main
git pull origin main
```

Read TODO.md to determine the next chapter:
- If user specified "Chapter N" → use that
- If user said "proceed"/"continue" → find next unchecked chapter in Global Tasks
- If ALL chapters are checked → exit with message: "All chapters extracted. Nothing to do."

### 2. Branch

```
git checkout -b feat/chapter<N>-<topic>
```

### 3. Plan (NotebookLM)

Query `notebooklm-researcher` agent for chapter contents:
- "What are the main functionalities, algorithms, and utilities in Chapter N of AFML?"
- "Does ML for Asset Managers have complementary content for this topic?"

Update TODO.md with the chapter's function checklist. Do NOT stop for user approval — proceed
directly to implementation.

### 4. Implement

For each function in the chapter checklist:
1. Query `notebooklm-researcher` for implementation details
2. Implement v1 following WORKFLOW.md § "Phase 1: Pre-Production" rules
3. Mark `[x]` in TODO.md
4. Run quality gate: `uvx ruff check --fix . && uvx ruff format .`

### 5. Finalize

- Update README.md with the new submodule
- Update CLAUDE.md if project-level guidance changes
- Run final quality gate on entire codebase
- Verify all imports work: `uv run python -c "from tradelab.lopezdp_utils.<topic> import *"`

### 6. Commit

```
git add src/tradelab/lopezdp_utils/<topic>/ TODO.md README.md
git commit -m "feat(ch<N>): extract <topic> from AFML Ch.<N>"
```

Include any additional changed files (CLAUDE.md, pyproject.toml) if applicable.

### 7. Merge to Main

```
git checkout main
git merge --no-ff feat/chapter<N>-<topic> -m "Merge feat/chapter<N>-<topic>: Complete Chapter <N> (<Topic>)"
git branch -d feat/chapter<N>-<topic>
```

## Error Handling

| Error | Action |
|-------|--------|
| NotebookLM query fails | Retry once. If still fails, skip that function, add `[ ] ⚠️ SKIPPED (NotebookLM unavailable)` in TODO.md, continue with next |
| Ruff check fails | Attempt auto-fix with `--fix`. If still fails, fix manually and retry |
| Import verification fails | Debug the import error, fix, and retry |
| Merge conflict | Abort merge (`git merge --abort`), leave feature branch intact, exit with error: "Merge conflict on feat/chapter<N>-<topic> — manual resolution needed" |
| No pending chapters | Exit with message: "All chapters extracted. Nothing to do." |

## Key References

- **WORKFLOW.md** — Complete methodology, v1 rules, quality gates, submodule naming
- **TODO.md** — Progress tracking (read first, update as work progresses)
- **CLAUDE.md** — Project instructions

## v1 Implementation Rules (Quick Reference)

- Use book's libraries: pandas, numpy, scipy (no Polars — that's Phase 2)
- snake_case everywhere (convert book's CamelCase)
- Type hints on all function signatures
- Google-style docstrings with theory context and book reference
- No tests, no extra error handling beyond what the book provides

## Important Rules

- **Never** use own training knowledge for López de Prado theory
- **Always** query `notebooklm-researcher` agent before implementing
- **Always** update TODO.md progress as you work
- **One chapter per session** — do not start the next chapter after finishing one
