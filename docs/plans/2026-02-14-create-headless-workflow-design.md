# Design: create-headless-workflow Skill

**Date:** 2026-02-14
**Status:** Approved

## Summary

A single Claude skill (`create-headless-workflow`) that interviews the user about a repetitive
workflow, then generates 4 artifacts: an autonomous Claude skill, a shell script runner, a launchd
plist for scheduling, and a TODO.md for progress tracking.

## Approach

Monolithic skill (Approach A) — all templates and interview logic in one skill file.

## Trigger

Phrases like: "create headless claude schedule", "create automated claude workflow",
"setup headless workflow"

## Interview Flow

### Base questions (1-7):
1. **Workflow name** — kebab-case identifier (filenames, launchd label, branches)
2. **Workflow description** — one sentence, used for skill description field
3. **Git strategy** — merge-to-main or PR-to-main
4. **Schedule interval** — hours between runs (default 2)
5. **Permission mode** — acceptEdits (default) or other
6. **Allowed tools** — comma-separated, with sensible defaults
7. **Completion condition** — how one "unit of work" ends

### Workflow-specific questions (8-12):
8. **What does one unit of work look like?** → "Scope Per Session" section
9. **How do you track progress?** → TODO.md structure and completion detection
10. **What are the main steps in one unit?** → Lifecycle steps
11. **Quality gate commands** → Finalize step
12. **External dependencies** → Error handling table

## Generated Artifacts

### 1. `.claude/skills/<name>/skill.md`
- Frontmatter: name + description (derived from interview)
- FULLY AUTONOMOUS execution mode
- Lifecycle: setup → branch → work → finalize → commit → merge/PR
- Error handling table
- Git strategy baked in:
  - **merge-to-main:** feature branch → merge --no-ff → delete branch
  - **PR-to-main:** feature branch → merge to develop → push → gh pr create/update

### 2. `scripts/<name>.sh`
- Lives in repo root `scripts/` (not ~/scripts)
- Relative path resolution via `$(cd "$(dirname "$0")/.." && pwd)`
- Runs `claude -p "Proceed with <skill-name>"` with configured permission mode and tools
- Post-run: commit log, push

### 3. `scripts/<name>.launchd.plist`
- Label: `com.user.<name>`
- StartInterval: configurable (default 7200 = 2h)
- PATH: ~/.local/bin, /opt/homebrew/bin, /usr/local/bin, /usr/bin, /bin
- Logs: ~/logs/<name>-{stdout,stderr}.log

### 4. `TODO.md`
- Created or appended if exists
- Global checklist of work items (if enumerable) or template for dynamic items
- Updated by the generated skill as work progresses

## Install Instructions (printed after generation)
```bash
cp scripts/<name>.launchd.plist ~/Library/LaunchAgents/com.user.<name>.plist
launchctl load ~/Library/LaunchAgents/com.user.<name>.plist
```
