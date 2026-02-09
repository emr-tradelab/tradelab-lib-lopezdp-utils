# Subagent Authoring Reference

Complete guide for creating custom subagents in Claude Code.

## What Subagents Are

Subagents are pre-configured AI personas that Claude Code delegates tasks to. Each subagent:
- Has its own **isolated context window** (doesn't pollute main conversation)
- Uses a **custom system prompt** (the markdown body of the file)
- Can be restricted to **specific tools**
- Can use a **specific model** (cost optimization)
- Can **preload skills** for domain knowledge
- Can have **persistent memory** across sessions

## File Locations & Priority

| Level | Path | Scope | Priority |
|-------|------|-------|----------|
| Project | `.claude/agents/*.md` | Current project | Highest |
| CLI flag | `--agents '{JSON}'` | Current session | Medium-high |
| User | `~/.claude/agents/*.md` | All projects | Medium |
| Plugin | Plugin `agents/` directory | When plugin installed | Lowest |

When names conflict, higher-priority agents win.

## File Format

```yaml
---
name: agent-name
description: >
  Expert in X domain. Use proactively when encountering Y situations
  or when the user asks about Z. Examples: "review my code changes",
  "debug this test failure".
tools: Read, Grep, Glob, Bash
model: sonnet
permissionMode: default
skills: skill-a, skill-b
memory: user
---

You are a [specific role] specializing in [specific domain].

[Detailed system prompt with:
- When to act and how to approach problems
- Step-by-step methodology
- Output format expectations
- Specific checklists or criteria
- Constraints and boundaries]
```

## Frontmatter Fields

### Required
- **`name`**: Unique identifier. Lowercase letters and hyphens.
- **`description`**: Natural language. Drives auto-delegation. Be specific and include
  trigger phrases. Use "proactively" or "MUST BE USED" for aggressive activation.

### Optional
- **`tools`**: Comma-separated list. Omit to inherit all tools from main thread
  (including MCP tools). Specify for granular control.
- **`model`**: `sonnet` (default), `opus`, `haiku`, or `inherit`. Use `haiku` for
  fast/cheap tasks (3x savings). Use `inherit` to match main conversation model.
- **`permissionMode`**: Controls permission handling:
  - `default` — Normal permission prompts
  - `acceptEdits` — Auto-accept file edits
  - `bypassPermissions` — Skip all permission prompts
  - `plan` — Read-only planning mode
  - `ignore` — Ignore permission rules
- **`skills`**: Comma-separated skill names to auto-load into agent context.
- **`memory`**: Persistent memory scope: `user` (recommended default), `project`, `local`.
  Agent gets a MEMORY.md it can read/write across sessions.

## Writing Effective System Prompts

### DO:
- **Define a clear persona**: "You are a senior security auditor specializing in..."
- **Be specific about methodology**: Step-by-step instructions
- **Include output format**: What the agent should produce
- **Set boundaries**: What the agent should NOT do
- **Include trigger examples in description**: Help Claude know when to delegate

### DON'T:
- Don't make agents too broad ("You are a general helper")
- Don't duplicate CLAUDE.md content in the agent prompt
- Don't include domain knowledge that should be in a skill
- Don't try to spawn subagents from within a subagent (not supported)

## Pattern: Agent + Skill Composition

The most powerful pattern combines agents (persona + tools) with skills (knowledge):

```yaml
---
name: api-developer
description: >
  API development specialist. Use proactively when creating, modifying, or
  debugging REST API endpoints, GraphQL resolvers, or API middleware.
tools: Read, Write, Edit, Bash, Grep, Glob
model: sonnet
skills: api-conventions, testing-patterns
---

You are a senior API developer. You follow the team's API conventions
(loaded from your skills) and always ensure proper test coverage.

When building or modifying APIs:
1. Review existing patterns in the codebase
2. Follow the api-conventions skill guidelines
3. Implement with proper error handling
4. Write tests following testing-patterns skill
5. Run tests to verify
```

The agent provides the persona and tool access. The skills provide the knowledge.
Neither depends on the other. Both are reusable independently.

## Model Selection Strategy

| Model | When to Use | Cost |
|-------|------------|------|
| `haiku` | Fast searches, simple tasks, high-volume delegation | Cheapest (3x savings) |
| `sonnet` | Complex reasoning, multi-step tasks (default) | Medium |
| `opus` | Critical decisions, quality validation, analysis | Most expensive |
| `inherit` | Match main conversation model | Varies |

## Agent Memory

Agents with `memory:` configured get persistent memory across sessions:

```yaml
---
name: code-reviewer
description: Reviews code for quality and best practices
memory: user
---

You are a code reviewer. Check your memory for patterns and conventions
you've seen before. Update your memory with new patterns you discover.
```

Memory scopes:
- `user`: Applies across all projects (recommended default)
- `project`: Only this project
- `local`: Only this directory

The agent gets read/write access to a MEMORY.md file in the memory directory.
First 200 lines are included in the system prompt.

## Common Agent Patterns

### Read-Only Explorer
```yaml
---
name: codebase-explorer
description: Fast codebase search and analysis. Use when searching for patterns.
tools: Read, Grep, Glob, Bash
model: haiku
---
You are a fast, focused codebase explorer. Search efficiently, report findings
with specific file paths and line numbers. Never modify files.
Bash access is limited to read-only commands: ls, git status, git log, find, cat.
```

### Proactive Reviewer
```yaml
---
name: code-reviewer
description: >
  Expert code review. MUST BE USED proactively after any code changes.
  Reviews for quality, security, and maintainability.
tools: Read, Grep, Glob, Bash
model: inherit
---
You are a senior code reviewer...
```

### Cost-Optimized Worker
```yaml
---
name: test-runner
description: Run tests and fix failures. Use proactively when tests fail.
tools: Read, Edit, Bash, Grep, Glob
model: haiku
---
You are a test automation expert...
```

## Chaining Agents

You can chain agents by asking Claude explicitly:
```
First use the code-analyzer agent to find performance issues,
then use the optimizer agent to fix them
```

Claude orchestrates the delegation. Each agent runs in its own context.

## Resumable Agents

Each agent execution gets a unique `agentId`. You can resume later:
```
Resume agent abc123 and continue the analysis
```

Useful for long-running research or multi-session workflows.
Agent transcripts are stored as `agent-{agentId}.jsonl`.

## CLI-Based Agents

For quick testing or automation scripts:
```bash
claude --agents '{
  "quick-reviewer": {
    "description": "Fast code review",
    "prompt": "You are a code reviewer. Focus on bugs and security.",
    "tools": ["Read", "Grep", "Glob"],
    "model": "haiku"
  }
}'
```

CLI agents have lower priority than project agents but higher than user agents.
