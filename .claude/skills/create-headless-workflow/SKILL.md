---
name: create-headless-workflow
description: Creates automated headless Claude workflow schedules. Use when the user says "create headless claude schedule", "create automated claude workflow", "setup headless workflow", or "create headless automated workflow". Interviews the user about their repetitive workflow, then generates 4 artifacts: an autonomous Claude skill, a shell script runner, a launchd plist for scheduling, and a TODO.md for progress tracking.
---

# Create Headless Workflow

Interviews the user about a repetitive workflow, then generates 4 artifacts for fully autonomous scheduled execution via Claude headless mode and macOS launchd.

## Interview Flow

Conduct the interview using AskUserQuestion for multiple-choice and free-text follow-ups. Collect all answers before generating artifacts.

### Base Questions (1-7)

**Q1 — Workflow name** (free text)
Ask: "What should we call this workflow? Use kebab-case (e.g., `daily-report-gen`)."
→ Maps to: filenames, launchd label, branch names, skill name

**Q2 — Workflow description** (free text)
Ask: "Describe what this workflow does in one sentence."
→ Maps to: skill description field, script comment, TODO.md header

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

**Q6 — Allowed tools** (free text)
Default: "Read,Write,Edit,Bash,Glob,Grep"
Ask: "Which tools should Claude have access to? (comma-separated)"
→ Maps to: `--allowedTools` flag in shell script

**Q7 — Completion condition** (free text)
Ask: "When should the workflow stop running? (e.g., 'all chapters extracted', 'queue empty')"
→ Maps to: completion condition in generated skill

### Workflow-Specific Questions (8-12)

**Q8 — Unit of work** (free text)
Ask: "What does one unit of work look like? (e.g., 'extract one chapter', 'process one file')"
→ Maps to: "Scope Per Session" section of generated skill

**Q9 — Progress tracking** (AskUserQuestion)
Options: "TODO.md with checkboxes" / "a queue file" / "other"
→ Maps to: how the generated skill reads/updates progress

**Q10 — Main steps** (free text)
Ask: "What are the main steps in one unit of work? (numbered list)"
→ Maps to: lifecycle steps 3 (Work) in generated skill

**Q11 — Quality gate commands** (free text)
Ask: "What commands should run as a quality gate? (e.g., `ruff check --fix && ruff format .`)"
→ Maps to: step 4 (Finalize) in generated skill

**Q12 — External dependencies** (free text)
Ask: "Does this workflow depend on external services? (e.g., 'NotebookLM', 'an API', or 'none')"
→ Maps to: error handling table rows in generated skill

## Skill Template

Generate `.claude/skills/{{name}}/SKILL.md` with these contents (replace all `{{placeholders}}`):

````markdown
---
name: proceed-with-{{name}}
description: >
  {{description}} Fully autonomous — execute end-to-end without stopping.
  Use when the user says "proceed with {{name}}" or "continue {{name}}".
---

# {{name}} Workflow

{{description}}

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

{{unit_of_work}}

## Full Lifecycle

### 1. Setup

```
git checkout main
git pull origin main
```

Read TODO.md to determine the next unit of work:
- If user specified a unit → use that
- If user said "proceed"/"continue" → find next unchecked item
- If ALL items are checked → exit with message: "All work complete. Nothing to do."

### 2. Branch

```
git checkout -b feat/{{name}}-<unit-identifier>
```

### 3. Work

{{main_steps}}

### 4. Finalize

Run quality gate:
```
{{quality_gate}}
```

Update TODO.md — mark completed items with `[x]`.

### 5. Commit

```
git add -A
git commit -m "feat({{name}}): <description of unit completed>"
```

### 6. Complete

{{GIT_STRATEGY_BLOCK}}

### 7. Update Progress

Update TODO.md with final status. If all items complete, note completion.

## Error Handling

| Error | Action |
|-------|--------|
{{ERROR_TABLE_ROWS}}
| Quality gate fails | Attempt auto-fix. If still fails, fix manually and retry |
| Merge conflict | Abort merge, leave feature branch intact, exit with error message |
| Nothing to do | Exit with message: "All work complete. Nothing to do." |

## Completion Condition

{{completion_condition}}
````

### Git Strategy Blocks

**If merge-to-main:**
```markdown
git checkout main
git merge --no-ff feat/{{name}}-<unit-identifier> -m "Merge feat/{{name}}-<unit>: <description>"
git branch -d feat/{{name}}-<unit-identifier>
```

**If PR-to-main:**
```markdown
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

## Shell Script Template

Generate `scripts/{{name}}.sh` with these contents:

```bash
#!/bin/bash
set -euo pipefail

# {{description}}
# Generated by create-headless-workflow skill

PROJECT_PATH="$(cd "$(dirname "$0")/.." && pwd)"
LOG_FILE="{{name}}-log.md"

cd "$PROJECT_PATH" || exit 1

# Log run start
echo "" >> "$LOG_FILE"
echo "---" >> "$LOG_FILE"
echo "## Run: $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# Run Claude headless
claude -p "Proceed with {{name}}" \
  --permission-mode {{permission_mode}} \
  --allowedTools "{{allowed_tools}}" 2>&1 | tee -a "$LOG_FILE"

# Post-run git operations
git add -A
if ! git diff --cached --quiet; then
    git commit -m "chore: automated {{name}} run $(date '+%Y-%m-%d %H:%M')"
    git push
    echo "✅ Changes committed and pushed" >> "$LOG_FILE"
else
    echo "ℹ️ No changes to commit" >> "$LOG_FILE"
fi

echo "**Completed at:** $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"
```

## Launchd Plist Template

Generate `scripts/{{name}}.launchd.plist` with these contents:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.user.{{name}}</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>-l</string>
        <string>{{absolute_script_path}}</string>
    </array>
    <key>StartInterval</key>
    <integer>{{interval_seconds}}</integer>
    <key>RunAtLoad</key>
    <false/>
    <key>StandardOutPath</key>
    <string>{{home}}/logs/{{name}}-stdout.log</string>
    <key>StandardErrorPath</key>
    <string>{{home}}/logs/{{name}}-stderr.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>{{home}}/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    </dict>
</dict>
</plist>
```

## TODO.md Template

Generate or append to `TODO.md` with this structure:

```markdown
# {{name}} — Progress Tracking

{{description}}

## Global Tasks

- [ ] Item 1
- [ ] Item 2
- [ ] ...

*(Populate from interview answers if units are enumerable, or leave a template row for dynamic items.)*
```

## Generation Logic

After the interview is complete:

1. **Collect variables** from all 12 answers:
   - `{{name}}` — Q1
   - `{{description}}` — Q2
   - `{{git_strategy}}` — Q3 (determines which git block to embed)
   - `{{interval_seconds}}` — Q4 × 3600
   - `{{permission_mode}}` — Q5
   - `{{allowed_tools}}` — Q6
   - `{{completion_condition}}` — Q7
   - `{{unit_of_work}}` — Q8
   - `{{progress_tracking}}` — Q9
   - `{{main_steps}}` — Q10 (format as numbered markdown steps)
   - `{{quality_gate}}` — Q11
   - `{{external_dependencies}}` — Q12
   - `{{ERROR_TABLE_ROWS}}` — build from Q12: one row per dependency in format `| <dep> unavailable | Retry once. If still fails, exit with error message |`
   - `{{absolute_script_path}}` — resolve from project path + `scripts/{{name}}.sh`
   - `{{home}}` — resolve from `$HOME`

   Note: `<unit-identifier>` in the skill template is NOT a generation-time placeholder. It is runtime-dynamic text that Claude fills during skill execution based on the current unit of work.

2. **Generate each artifact** using Write tool:
   - `.claude/skills/{{name}}/SKILL.md` — from Skill Template, embedding the correct git strategy block and `{{ERROR_TABLE_ROWS}}`
   - `scripts/{{name}}.sh` — from Shell Script Template
   - `scripts/{{name}}.launchd.plist` — from Launchd Plist Template
   - `TODO.md` — from TODO.md Template (create or append)

3. **Make script executable:**
   ```bash
   chmod +x scripts/{{name}}.sh
   ```

4. **Create logs directory:**
   ```bash
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
