---
name: proceed-with-extraction
description: >
  Orchestrates López de Prado extraction workflow for tradelab-lib-lopezdp-utils. Use when
  starting a work session, when user asks to proceed/continue/resume extraction work, or when
  user says "let's extract Chapter N". Handles both starting new chapters and resuming
  interrupted sessions.
---

# López de Prado Extraction Workflow

Orchestrates chapter-by-chapter extraction following the methodology defined in WORKFLOW.md.

## Execution Mode

**FULLY AUTONOMOUS** — Execute the complete workflow end-to-end without stopping to ask for permission.

DO NOT ask before:
- Creating feature branches
- Creating directories/files for new submodules
- Implementing functions
- Running quality checks (ruff)
- Updating TODO.md, README.md, or other documentation

Proceed directly through all steps. Only stop if encountering an error that blocks progress.

## Process

1. **Read TODO.md** to determine current state:
   - If user specified "Chapter N" → start new chapter workflow
   - If user said "proceed"/"continue" → resume from next unchecked item

2. **Follow the workflow** defined in WORKFLOW.md:
   - Read the "Session Workflow" section for complete step-by-step instructions
   - Use `notebooklm-researcher` agent for all theory queries (mandatory - never use own knowledge)
   - Follow v1 implementation rules and quality gates

3. **Execute autonomously**:
   - Create feature branch: `git checkout -b feat/chapter<N>-<topic>`
   - Create submodule structure: `src/tradelab/lopezdp_utils/<topic>/`
   - Implement all functions listed in TODO.md chapter section
   - Run quality gates after each function
   - Update TODO.md marking `[x]` as complete
   - Update README.md when chapter complete

4. **Key workflow files**:
   - **WORKFLOW.md** — Complete methodology (session workflow, v1 rules, quality gates)
   - **TODO.md** — Progress tracking (read first, update as work progresses)
   - **CLAUDE.md** — Project instructions (update if project-level guidance changes)

## Quick Reference

**Starting new chapter:**
- Follow WORKFLOW.md § "Starting a New Chapter"
- Query NotebookLM for chapter plan → update TODO.md → implement → update README.md

**Resuming work:**
- Follow WORKFLOW.md § "Resuming an Interrupted Session"
- Read TODO.md → find next `[ ]` item → query NotebookLM → implement

**v1 Implementation rules:**
- See WORKFLOW.md § "Phase 1: Pre-Production" for complete standards
- Key: pandas (not Polars), snake_case, type hints, Google docstrings, book references

**Quality gates:**
- See WORKFLOW.md § "Quality Gates (Pre-Production Phase)"
- Run before marking `[x]`: imports work, ruff passes, docstrings complete

## Important Rules

- **Never** use own training knowledge for López de Prado theory
- **Always** query `notebooklm-researcher` agent before implementing
- **Always** follow WORKFLOW.md methodology exactly
- **Always** update TODO.md progress as you work
