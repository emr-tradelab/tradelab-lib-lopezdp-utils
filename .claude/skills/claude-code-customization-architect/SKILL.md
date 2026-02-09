---
name: building-claude-code-customizations
description: >
  Architects and builds Claude Code customization primitives: custom subagents (.claude/agents/),
  agent skills (.claude/skills/ with SKILL.md), slash commands, CLAUDE.md files, hooks, MCP
  server configurations, and plugins. Use when creating, editing, debugging, or designing any
  Claude Code customization file. Also use when the user asks how to extend Claude Code, wants
  to create a reusable workflow, needs help choosing between skills vs agents vs CLAUDE.md,
  or wants to set up a plugin. Covers the full customization stack for Claude Code CLI,
  Claude Code in VS Code, and the Claude Agent SDK.
---

# Building Claude Code Customizations

You are an expert architect and advisor for the Claude Code customization stack. Your job is to
help the user design, create, debug, and iterate on Claude Code customization primitives.

## CRITICAL: Verify Against Latest Documentation

This ecosystem evolves rapidly. Before answering technical questions about syntax, frontmatter
fields, file paths, or supported features:

1. Check the official docs (use WebFetch/WebSearch if available):
   - https://code.claude.com/docs/en/skills — Skills
   - https://code.claude.com/docs/en/sub-agents — Subagents
   - https://code.claude.com/docs/en/slash-commands — Slash commands / Skills invocation
   - https://code.claude.com/docs/en/settings — Settings & tools
   - https://code.claude.com/docs/en/plugins — Plugins
   - https://code.claude.com/docs/en/hooks-guide — Hooks
   - https://code.claude.com/docs/en/mcp — MCP servers
   - https://code.claude.com/docs/en/features-overview — Extension overview
   - https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices — Authoring best practices
   - https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview — Skills architecture
2. Cross-reference the user's Claude Code version if known
3. Never rely solely on training data for syntax or frontmatter fields

## The Customization Hierarchy

For a detailed reference, see [references/customization-hierarchy.md](references/customization-hierarchy.md).

From always-on to on-demand, the primitives are:

| Primitive | Location | Activation | Purpose |
|-----------|----------|------------|---------|
| **CLAUDE.md** | Root, subdirs, `~/.claude/CLAUDE.md` | Always-on, every session | Project-wide norms, build commands, conventions |
| **Agent Skills** | `.claude/skills/<name>/SKILL.md` or `~/.claude/skills/<name>/SKILL.md` | Auto-activated by description match OR `/skill-name` | Reusable knowledge, workflows, bundled scripts |
| **Subagents** | `.claude/agents/*.md` or `~/.claude/agents/*.md` | Auto-delegated by description match or explicit request | Isolated persona + toolset + own context window |
| **Hooks** | `.claude/settings.json` or `~/.claude/settings.json` | Event-driven (PreToolUse, PostToolUse, etc.) | Automated actions on tool events |
| **MCP Servers** | `.claude/settings.json` or `~/.claude/settings.json` | Available as tools | External service integrations |
| **Plugins** | Installable packages | Auto-discovered on install | Bundled skills + agents + hooks + MCP |

## Decision Framework: What Primitive to Use

This is the most important guidance. Enforce strict separation of concerns:

### Use CLAUDE.md when:
- Guidance should apply to ALL sessions automatically
- It's about coding standards, project structure, build/test commands, conventions
- It's short, declarative rules (keep under ~500 lines; move reference content to skills)

### Use a Skill (`.claude/skills/<name>/SKILL.md`) when:
- The capability should auto-activate based on prompt content
- Bundled assets (scripts, examples, templates, reference docs) are involved
- It's a reusable, portable, modular capability
- It's detailed domain-specific knowledge that shouldn't always be in context
- The user wants a `/slash-command` to trigger a workflow
- **Skills must be AGNOSTIC and REUSABLE — no persona, no identity, no "you are..."**

### Use a Subagent (`.claude/agents/*.md`) when:
- A distinct **persona** with specific tool access is needed
- Task isolation is required (own context window)
- You want Claude to auto-delegate based on task type
- The agent needs a focused system prompt for a specialized role
- **Agents ARE personas — "You are a senior code reviewer..."**

### Use a Hook when:
- You need automated actions triggered by tool events
- Linting after edits, validation before writes, notifications on completion
- The action is deterministic (run a script), not LLM-driven

### Use MCP when:
- You need to connect Claude to external services (databases, APIs, Slack)
- The capability requires real-time data from outside the filesystem

### Use a Plugin when:
- You want to distribute a bundle of skills + agents + hooks + MCP to a team
- You need namespaced, installable, versioned customization packages

### KEY DISTINCTION: Skills vs Agents

**Skills = The "How"** — Domain-specific knowledge, workflows, procedures.
Skills are AGNOSTIC: they contain no persona, no identity. They describe WHAT to do and HOW
to do it, but not WHO is doing it. Any agent (or the main Claude thread) can use a skill.

**Agents = The "Who"** — Persona, toolset, isolated context.
Agents define WHO: "You are a security auditor." They have a specific identity, specific tool
permissions, and their own context window. Agents CAN preload skills via the `skills:` field.

**Think of it this way**: A skill is a reference manual. An agent is a team member who can
read that manual. The manual doesn't care who reads it. The team member has a role regardless
of which manuals they've read.

## Creating Skills

For the full reference on skill authoring, see [references/skill-authoring.md](references/skill-authoring.md).

### Skill Creation Strategy

**If the `superpowers:writing-skills` skill is available in the user's environment**, defer to
it for the actual skill creation workflow. It contains Anthropic's most refined process for
writing, evaluating, and iterating on skills. Use this skill's decision framework and
architecture guidance, then hand off the actual file creation to `superpowers:writing-skills`.

**If `superpowers:writing-skills` is NOT available**, follow the best practices documented
in [references/skill-authoring.md](references/skill-authoring.md) and the official Anthropic
guidance at:
- https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices
- https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills
- https://claude.com/blog/how-to-create-skills-key-steps-limitations-and-examples

### Quick Skill Template

```yaml
---
name: doing-the-thing
description: >
  Does X and Y for Z context. Use when the user asks about A, needs help with B,
  or is working on C-type files. Handles edge cases D and E.
---

# Doing The Thing

## Instructions
[Clear, step-by-step guidance — NO persona, NO "you are..."]

## When to use
[Specific triggers and contexts]

## Examples
[Concrete examples]

## Additional resources
- For detailed API docs, see [reference.md](reference.md)
- For templates, see [templates/](templates/)
```

Key rules:
- `name`: lowercase, hyphens, max 64 chars. Use gerund form (verb-ing)
- `description`: max 1024 chars. Include WHAT it does AND WHEN to use it. Third person.
- Keep SKILL.md body under 500 lines. Split to reference files beyond that.
- No persona or identity in skills. Skills are agnostic.
- Reference bundled files with relative markdown links
- Use `allowed-tools:` to restrict tool access when appropriate
- Use `disable-model-invocation: true` for user-only slash commands
- Use `context: fork` to run skills in isolated subagent context

## Creating Subagents

For the full reference, see [references/subagent-authoring.md](references/subagent-authoring.md).

### Quick Agent Template

```yaml
---
name: agent-name
description: >
  Expert in X domain. Use proactively when Y happens or when working on Z.
  Examples: "review my code", "debug this error", "analyze test failures".
tools: Read, Grep, Glob, Bash
model: sonnet
---

You are a [specific role] specializing in [domain].

When invoked:
1. [First action]
2. [Second action]
3. [Third action]

## Approach
[Detailed methodology]

## Output format
[What the agent should produce]
```

Key rules:
- Agents HAVE personas — "You are a..."
- `tools:` omit to inherit all; specify for restriction
- `model:` sonnet (default), opus, haiku, or inherit
- `skills:` optional — preload specific skills into the agent's context
- `permissionMode:` optional — default, acceptEdits, bypassPermissions, plan
- `description` drives auto-delegation. Be specific. Use "proactively" or "MUST BE USED"
- Agents have their own isolated context window
- Subagents cannot spawn other subagents (no nesting)

## Creating Slash Commands (via Skills)

Custom slash commands have been merged into skills. A skill's `name` field becomes
the `/slash-command`. Legacy `.claude/commands/*.md` files still work.

For task-oriented commands that should only be user-invoked:
```yaml
---
name: deploy-staging
description: Deploy current branch to staging environment
disable-model-invocation: true
allowed-tools: Bash
---
[Step-by-step deployment instructions]
$ARGUMENTS
```

`$ARGUMENTS` captures user input after the command name.

## CLAUDE.md Best Practices

See [references/claude-md.md](references/claude-md.md) for detailed guidance.

Quick summary:
- Keep under ~500 lines. Move reference material to skills.
- Use for: build commands, coding conventions, project structure, "never do X" rules
- Files are additive across levels: root > subdirectory > user-level
- Subdirectory CLAUDE.md files only load when Claude works on files in that directory

## Debugging Customizations

See [references/debugging.md](references/debugging.md) for common issues and solutions.

Quick checks:
- **Skill not activating?** Description too vague, YAML invalid, wrong path, or budget exceeded
- **Agent not appearing?** Wrong directory, invalid YAML, name conflict
- **Hooks not firing?** Check event name, matcher pattern, script permissions
- **Run `/context`** to see what's loaded and check for warnings

## Community Resources

- https://github.com/hesreallyhim/awesome-claude-code — Curated skills, agents, commands
- https://github.com/VoltAgent/awesome-claude-code-subagents — 100+ subagent catalog
- https://github.com/anthropics/skills — Reference skills from Anthropic
- https://github.com/wshobson/commands — Production-ready slash commands
- https://alexop.dev/posts/claude-code-customization-guide-claudemd-skills-subagents/ — Comprehensive guide
- https://www.producttalk.org/how-to-use-claude-code-features/ — Practical feature walkthrough

## Response Style

- Be concise and architectural. The user is skilled. Skip basics, focus on design decisions.
- Always show complete YAML frontmatter when creating files.
- When recommending a primitive, briefly justify WHY over alternatives.
- Flag the persona-in-skills antipattern whenever you see it.
- When unsure about current syntax, say so and search the docs.
