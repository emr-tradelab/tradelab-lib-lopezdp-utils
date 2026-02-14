---
name: create-headless-workflow
description: Creates automated headless Claude workflow schedules for macOS. Use when the user says "create headless claude schedule", "create automated claude workflow", "setup headless workflow", or "create headless automated workflow". Interviews the user about their repetitive workflow, then generates 4 artifacts for fully autonomous scheduled execution via launchd.
---

# Create Headless Workflow

Interviews the user about a repetitive workflow, then generates 4 artifacts for fully autonomous scheduled execution via Claude headless mode and macOS launchd.

**Platform:** macOS only (uses launchd for scheduling).

## Interview Flow

Conduct the interview using AskUserQuestion for multiple-choice and free-text follow-ups. Collect all answers before generating artifacts.

### Base Questions (1-8)

**Q1 — Workflow name** (free text)
Ask: "What should we call this workflow? Use kebab-case (e.g., `daily-report-gen`)."
→ Maps to: filenames, launchd label, branch names, skill name

**Q2 — Workflow description** (free text)
Ask: "Describe what this workflow does in one sentence."
→ Maps to: skill description, script comment, TODO.md header

**Q3 — Git strategy** (AskUserQuestion)
Options: "merge-to-main" / "PR-to-main"
- merge-to-main: feature branch → `git merge --no-ff` to main → delete branch
- PR-to-main: feature branch → merge to develop → push → `gh pr create` or update
→ Maps to: step 6 of generated skill lifecycle

**Q4 — Schedule interval** (AskUserQuestion)
Options: 1h / 2h (default) / 3h / 4h / 6h
→ Maps to: StartInterval in launchd plist (value × 3600)

**Q5 — Permission mode** (AskUserQuestion)
Options: acceptEdits (default) / plan / bypassPermissions
→ Maps to: `--permission-mode` flag in shell script

**Q6 — Model** (AskUserQuestion)
Options: sonnet (default) / opus / haiku
→ Maps to: `--model` flag in shell script

**Q7 — Allowed tools** (free text)
Default: "Read,Write,Edit,Bash,Glob,Grep"
Ask: "Which tools should Claude have access to? (comma-separated)"
→ Maps to: `--allowedTools` flag in shell script

**Q8 — Completion condition** (free text)
Ask: "When should the workflow stop running? (e.g., 'all chapters extracted', 'queue empty')"
→ Maps to: completion condition in generated skill

### Workflow-Specific Questions (9-13)

**Q9 — Unit of work** (free text)
Ask: "What does one unit of work look like? (e.g., 'extract one chapter', 'process one file')"
→ Maps to: "Scope Per Session" section of generated skill

**Q10 — Main steps** (free text)
Ask: "What are the main steps in one unit of work? (numbered list)"
→ Maps to: lifecycle step 3 (Work) in generated skill

**Q11 — Quality gate commands** (free text)
Ask: "What commands should run as a quality gate? (e.g., `ruff check --fix && ruff format .`)"
→ Maps to: step 4 (Finalize) in generated skill

**Q12 — External dependencies** (free text)
Ask: "Does this workflow depend on external services? (e.g., 'NotebookLM', 'an API', or 'none')"
→ Maps to: error handling table rows in generated skill

**Q13 — Existing skills to preload** (free text, optional)
Ask: "Should the generated skill reference any existing skills? (e.g., 'notebooklm-research', or 'none')"
→ Maps to: instructions in the generated skill to invoke those skills during work

## Templates

All templates live in the `templates/` folder alongside this file. Each uses `{{placeholder}}` syntax for variable substitution.

| Template | Output Path | Description |
|----------|-------------|-------------|
| `SKILL.md.template` | `.claude/skills/{{name}}/SKILL.md` | The autonomous Claude skill |
| `runner.sh.template` | `scripts/{{name}}.sh` | Shell script that invokes Claude headless |
| `launchd.plist.template` | `scripts/{{name}}.launchd.plist` | macOS launchd schedule |
| `TODO.md.template` | `TODO.md` (create or append) | Progress tracking checklist |

### Git Strategy Blocks

The `{{GIT_STRATEGY_BLOCK}}` placeholder in the skill template is replaced based on Q3:

**If merge-to-main:**
```
git checkout main
git merge --no-ff feat/{{name}}-<unit-identifier> -m "Merge feat/{{name}}-<unit>: <description>"
git branch -d feat/{{name}}-<unit-identifier>
```

**If PR-to-main:**
```
git checkout develop
git merge --no-ff feat/{{name}}-<unit-identifier> -m "Merge feat/{{name}}-<unit>: <description>"
git branch -d feat/{{name}}-<unit-identifier>
git push origin develop

# Create or update PR
if gh pr list --head develop --base main --json number -q '.[0].number' | grep -q '.'; then
    echo "PR already exists — push updated develop branch"
else
    gh pr create --base main --head develop --title "feat: {{name}} progress" --body "Automated PR from {{name}} workflow"
fi
```

## Generation Logic

After the interview is complete:

1. **Collect variables** from all 13 answers:

   | Variable | Source | Notes |
   |----------|--------|-------|
   | `{{name}}` | Q1 | kebab-case identifier |
   | `{{description}}` | Q2 | one-sentence summary |
   | `{{GIT_STRATEGY_BLOCK}}` | Q3 | embed matching block above |
   | `{{interval_seconds}}` | Q4 | value × 3600 |
   | `{{permission_mode}}` | Q5 | CLI flag value |
   | `{{model}}` | Q6 | sonnet, opus, or haiku |
   | `{{allowed_tools}}` | Q7 | comma-separated tool list |
   | `{{completion_condition}}` | Q8 | free text |
   | `{{unit_of_work}}` | Q9 | free text |
   | `{{main_steps}}` | Q10 | format as numbered markdown steps |
   | `{{quality_gate}}` | Q11 | shell commands |
   | `{{ERROR_TABLE_ROWS}}` | Q12 | one row per dependency: `\| <dep> unavailable \| Retry once. If still fails, exit with error \|` |
   | `{{preload_skills}}` | Q13 | skill names to reference in generated skill, or empty |
   | `{{absolute_script_path}}` | derived | project path + `scripts/{{name}}.sh` |
   | `{{home}}` | derived | `$HOME` |

   Note: `<unit-identifier>` is NOT a generation-time placeholder — it's runtime text that Claude fills during skill execution.

2. **Read each template** from `templates/` and substitute all `{{placeholders}}`.

3. **Write each artifact** using the Write tool to the output paths listed in the Templates table.

4. **Post-generation steps:**
   ```bash
   chmod +x scripts/{{name}}.sh
   mkdir -p ~/logs
   ```

5. **Print install instructions:**
   ```
   To install the schedule:
     cp scripts/{{name}}.launchd.plist ~/Library/LaunchAgents/com.user.{{name}}.plist
     launchctl load ~/Library/LaunchAgents/com.user.{{name}}.plist

   To verify:
     launchctl list | grep {{name}}

   To run manually:
     bash scripts/{{name}}.sh
   ```
