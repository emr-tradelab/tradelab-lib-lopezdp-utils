---
name: update-docs-on-completion
description: Use when a plan, workflow, feature branch, extraction session, or significant implementation task has been completed — triggers mandatory documentation updates via a cheap subagent
---

# Update Docs on Completion

## Overview

**Mandatory post-completion documentation sync.** Dispatches the `docs-updater` haiku subagent to update all relevant .md files (CLAUDE.md, README.md, AGENTS.md, copilot-instructions.md, GEMINI.md, TODO.md, etc.) so the next session starts with accurate context.

**Core principle:** Stale documentation is worse than no documentation — it actively misleads future sessions.

## When to Trigger

**Mandatory after:**
- Completing an implementation plan (all steps done)
- Finishing a feature branch before merge/PR
- Completing a workflow or extraction session
- Any task that changed project structure, conventions, or phase

**Not needed for:**
- Single-file bug fixes with no structural impact
- Exploratory research sessions that produced no code changes
- Work-in-progress that will continue in the same session

## Process

```
1. SUMMARIZE what was completed (2-5 bullet points):
   - What was built/changed/removed
   - New modules, files, or patterns introduced
   - Any conventions established or changed
   - Phase/status changes

2. DISPATCH docs-updater subagent:
   Use Task tool with subagent_type="docs-updater", model="haiku"
   Pass the summary as the prompt, including:
   - The completion summary
   - Which directories/modules were affected
   - Any specific documentation concerns

3. REVIEW the subagent's report:
   - Verify edits are accurate
   - Ensure nothing critical was missed
   - Ensure CLAUDE.md changes are concise (token budget matters)
```

## Prompt Template for Subagent

```
The following work was just completed in this repository:

**What was done:**
{COMPLETION_SUMMARY}

**Affected areas:**
{LIST_OF_DIRECTORIES_OR_MODULES}

**Specific documentation concerns:**
{ANY_KNOWN_GAPS_OR_STALE_INFO}

Find and update all relevant .md documentation files to reflect these changes.
Focus especially on CLAUDE.md accuracy — it drives all future AI sessions.
```

## Red Flags

- About to merge/PR without running this skill
- "Documentation can wait" — it can't, you'll forget the context
- "The changes are obvious" — obvious to you now, not to the next session
- Skipping because "it's just a small change" — small changes accumulate into big drift

## The Bottom Line

Every completed task that changes project state MUST end with a docs-updater dispatch. It costs almost nothing (haiku model) and prevents the #1 cause of wasted time: starting a session with wrong context.
