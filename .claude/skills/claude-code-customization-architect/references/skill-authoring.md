# Skill Authoring Best Practices

Comprehensive guide for writing effective Claude Code skills. Based on official Anthropic
documentation and community best practices.

## The Cardinal Rule

**Skills are agnostic. Skills have no persona. Skills have no identity.**

A skill is a reference manual, not a team member. It describes WHAT to do and HOW to do it.
Any agent — or the main Claude thread — should be able to use it.

❌ Wrong: `You are an expert code reviewer who...`
✅ Right: `When reviewing code, check for: ...`

If you need a persona, create a subagent that preloads the skill via the `skills:` field.

## Naming Conventions

- Use **gerund form** (verb + -ing): `generating-commit-messages`, `reviewing-code`
- Lowercase letters, numbers, hyphens only
- Maximum 64 characters
- The `name` field becomes the `/slash-command`
- Cannot contain: XML tags, reserved words ("anthropic", "claude")

## Writing Descriptions

The description is the **most critical field**. It drives skill discovery from potentially
100+ available skills. Claude reads all descriptions at startup to know what's available.

Rules:
- Write in **third person** ("Generates..." not "Generate..." or "I generate...")
- Include WHAT the skill does AND WHEN to use it
- Maximum 1024 characters
- Include specific trigger terms users would naturally mention
- Be specific enough to avoid false activation on unrelated prompts

Good:
```yaml
description: >
  Generates descriptive commit messages by analyzing git diffs. Use when the user
  asks for help writing commit messages, reviewing staged changes, or preparing
  commits for pull requests.
```

Bad:
```yaml
description: Helps with git stuff
```

## SKILL.md Structure

### Minimal Skill (single file)
```
my-skill/
└── SKILL.md
```

### Skill with Supporting Files
```
my-skill/
├── SKILL.md              # Required — overview and navigation (<500 lines)
├── reference.md          # Detailed docs (loaded on demand)
├── examples.md           # Usage examples (loaded on demand)
└── scripts/
    └── helper.py         # Utility script (executed, not loaded into context)
```

### Body Structure

```markdown
# Skill Title

## Instructions
[Clear, step-by-step guidance. No persona.]

## When to use
[Specific triggers and contexts]

## Process
[Detailed workflow steps]

## Examples
[Concrete input/output examples]

## Additional resources
- For detailed API docs, see [reference.md](reference.md)
- For templates, see [templates/](templates/)
```

## Progressive Disclosure

This is the key architecture that makes skills context-efficient:

1. **Startup**: Only `name` + `description` loaded (minimal token cost)
2. **When relevant**: Claude reads full SKILL.md body
3. **As needed**: Claude follows links to read reference files
4. **Scripts**: Executed via bash; only output consumes tokens (not source code)

Design your skill to exploit this:
- Keep SKILL.md under **500 lines**
- Put detailed reference material in separate files
- Link to files with clear descriptions of what they contain and when to read them
- Use scripts for heavy computation — their code never enters context

## Frontmatter Fields (Claude Code)

```yaml
---
name: doing-something              # Required
description: Does X for Y          # Required
disable-model-invocation: true     # Only user can invoke (/name)
user-invocable: false              # Hide from slash menu
allowed-tools: Read, Grep, Glob   # Restrict available tools
context: fork                      # Run in isolated subagent context
agent: Explore                     # Subagent type for context: fork
skills: other-skill-a, other-b    # Preload other skills
---
```

### `allowed-tools`
Restricts which tools Claude can use when the skill is active. Only supported in Claude Code.
Omit to use normal permission model.

### `context: fork`
Runs the skill in an isolated subagent. The skill content becomes the subagent's prompt.
It won't have access to conversation history. Best for self-contained tasks.

Use `agent:` to specify which subagent type: `Explore` (read-only), `Plan`, `general-purpose`,
or any custom agent name from `.claude/agents/`.

### `disable-model-invocation: true`
Prevents Claude from auto-loading the skill. User must type `/skill-name` explicitly.
Best for action-oriented commands (deploy, release, destructive operations).

### Dynamic Content with `!` Preprocessing

Skills support shell command preprocessing with `!` backticks:
```markdown
## Current context
- Branch: !`git branch --show-current`
- Recent changes: !`git log --oneline -5`
- PR diff: !`gh pr diff`
```

Commands are executed BEFORE the skill content reaches Claude. Claude only sees the output.

### `$ARGUMENTS` Placeholder

Captures user input after the slash command:
```markdown
Research $ARGUMENTS thoroughly using web search and codebase analysis.
```

When user types `/research authentication flow`, Claude sees:
"Research authentication flow thoroughly using web search and codebase analysis."

## Skill vs Slash Command vs CLAUDE.md

| Need | Use | Why |
|------|-----|-----|
| Always-on conventions | CLAUDE.md | Zero activation cost, always in context |
| Reference knowledge loaded when relevant | Skill (default) | Auto-activates, progressive disclosure |
| Explicit user-triggered workflow | Skill + `disable-model-invocation: true` | User controls when it runs |
| Isolated task execution | Skill + `context: fork` | Own context window, no history |
| Domain persona with tool restrictions | Subagent | Has identity, isolated context, specific tools |

## Iterative Development Process

The recommended approach (from Anthropic's skill-creator):

1. **Capture intent**: What should the skill do? What's the workflow?
2. **Write a draft**: Create SKILL.md with frontmatter and instructions
3. **Test with real prompts**: Try prompts that should and shouldn't trigger it
4. **Evaluate**: Did it activate correctly? Were instructions followed?
5. **Iterate**: Refine description, instructions, examples based on observed behavior
6. **Expand**: Add more test cases, edge cases, supporting files

### Two-Claude Method

Work with one Claude instance ("Claude A") to create the skill, then test it with a fresh
instance ("Claude B") that has the skill installed. Claude A designs; Claude B reveals gaps.

### What to Watch For

- Does Claude read files in the order you expected?
- Does Claude miss references to important files?
- Does Claude repeatedly read the same section? (move it to SKILL.md)
- Does Claude ignore bundled files? (improve signaling in instructions)

## Context Budget

Skill descriptions consume context budget. The budget scales at 2% of context window with
a 16,000 character fallback. If you have many skills and some are excluded:

- Run `/context` to check for warnings
- Set `SLASH_COMMAND_TOOL_CHAR_BUDGET` env var to override
- Shorten descriptions of less-used skills
- Consider consolidating related skills

## Anti-Patterns

1. **Persona in skills**: Skills should never say "You are..." — that's for agents
2. **Tool names in skills**: Skills can't control tools directly. Use `allowed-tools` in frontmatter
3. **Huge SKILL.md**: Keep under 500 lines. Split to reference files.
4. **Vague descriptions**: "Helps with code" won't trigger. Be specific.
5. **Hardcoded absolute paths**: Use relative paths within the skill directory
6. **Duplicating CLAUDE.md content**: If it's always needed, put it in CLAUDE.md, not a skill
