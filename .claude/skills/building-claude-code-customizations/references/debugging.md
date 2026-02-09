# Debugging Claude Code Customizations

Common issues and solutions for skills, subagents, commands, hooks, and plugins.

## Skill Not Activating

### Description too vague
The #1 cause. Claude selects from 100+ skills based on description matching.

**Fix**: Make description specific with trigger terms:
```yaml
# Bad
description: Helps with data

# Good
description: >
  Analyzes Excel spreadsheets, generates pivot tables, and creates charts.
  Use when working with Excel files, spreadsheets, or .xlsx files.
```

### YAML syntax invalid
Invalid YAML silently prevents skill loading.

**Check**:
```bash
cat .claude/skills/my-skill/SKILL.md | head -n 15
```

Common issues:
- Missing opening or closing `---`
- Tabs instead of spaces
- Unquoted strings with special characters (`:`, `#`, `[`, `{`)
- Description containing unescaped quotes

### Wrong file path
Skills must be at exact paths with exact filenames.

**Verify**:
```bash
# Personal skills
ls ~/.claude/skills/*/SKILL.md

# Project skills
ls .claude/skills/*/SKILL.md
```

The file must be named `SKILL.md` (not `Skill.md`, `skill.md`, `SKILLS.md`).

### Context budget exceeded
Too many skills can overflow the description budget.

**Check**: Run `/context` in Claude Code to see warnings about excluded skills.

**Fix**:
- Shorten descriptions
- Set `SLASH_COMMAND_TOOL_CHAR_BUDGET` env var to increase budget
- Consolidate related skills
- Remove unused skills

### `disable-model-invocation: true`
If set, Claude won't auto-activate the skill. User must type `/skill-name`.

---

## Subagent Not Working

### Not appearing in `/agents`
- Verify file is in `.claude/agents/` (project) or `~/.claude/agents/` (user)
- Check YAML frontmatter is valid
- Check for name conflicts (project agents override user agents)
- Restart Claude Code after adding new agents

### Not being auto-delegated
The `description` field drives delegation. Make it specific and action-oriented.

**Add urgency cues**:
```yaml
description: >
  Expert code reviewer. Use PROACTIVELY after any code changes.
  MUST BE USED when reviewing PRs or checking code quality.
```

### Agent not using expected tools
- If `tools:` is specified, only listed tools are available
- If `tools:` is omitted, agent inherits ALL tools from main thread
- MCP tools are included in inheritance
- Unknown tool names are silently ignored

### Subagent trying to spawn subagent
Not supported. Subagents cannot spawn other subagents (no nesting).
Design workflows that don't require recursive delegation.

---

## Slash Command Not Working

### Command not appearing
- Skills in `.claude/skills/` are visible in slash menu by default
- Legacy commands in `.claude/commands/` also work
- Check file extension is `.md`
- Restart Claude Code to load new commands

### `$ARGUMENTS` not working
- Must be uppercase: `$ARGUMENTS`
- Only works when user provides text after the command name
- Test with: `/my-command test input here`

### `!` preprocessing not executing
- Shell commands in `!` backticks are preprocessing — run before Claude sees content
- Verify the command works in your shell directly
- Check for proper backtick syntax: `` !`command here` ``

---

## Hooks Not Firing

### Check event name
Valid events: `PreToolUse`, `PostToolUse`, `Notification`, `Stop`
Event names are case-sensitive.

### Check matcher pattern
The `matcher` field is a regex pattern matching tool names:
```json
{
  "matcher": "Write|Edit",
  "hooks": [...]
}
```

### Check script permissions
```bash
chmod +x .claude/hooks/my-script.sh
```

### Check settings location
Hooks go in `.claude/settings.json` (project) or `~/.claude/settings.json` (user).
Not in CLAUDE.md.

---

## General Debugging

### `/context` command
Shows what's loaded into Claude's context, including:
- CLAUDE.md files
- Active skills and their descriptions
- Warnings about budget limits or loading errors

### `claude --debug`
Starts Claude Code in debug mode showing skill loading errors and other diagnostics.

### Check file encoding
- Files must be UTF-8
- Use forward slashes in paths (even on Windows)
- No BOM (byte order mark)

### Restart after changes
- Skills from `.claude/skills/` support live reload in some versions
- Agents require restart to load
- CLAUDE.md changes are picked up on next session start
- Use `/clear` to start fresh context

### Name conflicts
When the same name exists at multiple levels:
- **Skills**: managed > user > project (project wins for same-level)
- **Agents**: managed > CLI flag > project > user > plugin
- **MCP servers**: local > project > user
- **Plugin skills**: namespaced (`/plugin-name:skill-name`) to avoid conflicts
