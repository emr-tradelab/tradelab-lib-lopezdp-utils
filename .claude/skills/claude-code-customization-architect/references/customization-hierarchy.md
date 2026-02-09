# Claude Code Customization Hierarchy — Detailed Reference

## Layer 1: CLAUDE.md (Always-On Context)

**What**: Markdown files that Claude reads at the start of every session.
**Where**:
- `CLAUDE.md` or `.claude/CLAUDE.md` at project root (highest priority)
- `CLAUDE.md` in subdirectories (loaded when Claude works on files there)
- `~/.claude/CLAUDE.md` (user-level, applies everywhere)

**How they layer**: All levels are additive. Content from all matching CLAUDE.md files is
merged into Claude's context. More specific files add to broader ones.

**Best for**:
- Build/test/lint commands
- Coding standards and conventions
- Project architecture overview
- "Never do X" rules
- Short, declarative guidance

**Limitations**:
- Every token is always in context (costs budget every session)
- Keep under ~500 lines total
- Not for detailed reference material (use skills instead)

**Syntax**: Plain markdown. No frontmatter required.

```markdown
# Project Conventions

## Build
- `npm run build` — production build
- `npm test` — run all tests
- `npm run test:unit -- path/to/file` — run single test

## Coding Standards
- Use TypeScript strict mode
- Prefer functional patterns over classes
- All API responses must use the Result<T, E> type
- Never commit console.log statements
```

---

## Layer 2: Agent Skills (On-Demand Knowledge & Workflows)

**What**: Directories containing a `SKILL.md` file with YAML frontmatter, optional reference
files, scripts, and templates.
**Where**:
- `.claude/skills/<skill-name>/SKILL.md` (project-level, sharable via git)
- `~/.claude/skills/<skill-name>/SKILL.md` (user-level, personal)
- Plugin skills (from installed plugins, namespaced)

**Activation**:
- **Auto-activated**: Claude reads skill descriptions at startup and loads the full skill
  when it matches the current task
- **User-invoked**: Type `/skill-name` to invoke directly
- Both by default; control with `disable-model-invocation` and `user-invocable` frontmatter

**Best for**:
- Domain-specific knowledge (API conventions, style guides)
- Repeatable workflows (deploy, review, release)
- Bundled scripts and templates
- Anything that should load only when relevant (context-efficient)

**Key architecture**: Progressive disclosure
1. At startup: only `name` + `description` loaded into system prompt
2. When relevant: Claude reads full SKILL.md
3. As needed: Claude reads referenced files (reference.md, scripts/, etc.)
4. Scripts: executed via bash; only output consumes tokens

**Frontmatter fields** (Claude Code specific):
```yaml
---
name: doing-something          # Required. Lowercase, hyphens, max 64 chars
description: >                 # Required. Max 1024 chars. What + when.
  Does X for Y. Use when Z.
disable-model-invocation: true # Optional. Only user can invoke (/name)
user-invocable: false          # Optional. Hide from slash command menu
allowed-tools: Read, Grep      # Optional. Restrict tools (Claude Code only)
context: fork                  # Optional. Run in isolated subagent context
agent: Explore                 # Optional. Which subagent to use with context: fork
skills: other-skill            # Optional. Preload other skills
---
```

**CRITICAL RULE: Skills must be agnostic. No persona. No "You are..."**

---

## Layer 3: Subagents (Isolated Personas)

**What**: Markdown files with YAML frontmatter defining specialized AI assistants with their
own context window, system prompt, and tool permissions.
**Where**:
- `.claude/agents/*.md` (project-level)
- `~/.claude/agents/*.md` (user-level)
- Plugin agents (from installed plugins)
- CLI flag: `--agents '{JSON}'` (session-specific)

**Priority**: managed > CLI flag > project > user > plugin

**Activation**:
- **Auto-delegated**: Claude reads agent descriptions and delegates tasks that match
- **Explicit**: User says "use the X agent to..."

**Best for**:
- Specialized roles (code reviewer, debugger, data analyst)
- Task isolation (keep exploration out of main context)
- Tool restriction (read-only agents, limited-scope agents)
- Cost optimization (route to haiku for simple tasks)

**Built-in subagents**:
- **Explore**: Fast (haiku), read-only. For codebase search and analysis.
- **Plan**: Research agent for plan mode. Uses sonnet.
- **General-purpose**: Full capability. Uses sonnet. Can read and write.

**Frontmatter fields**:
```yaml
---
name: agent-name               # Required. Lowercase, hyphens
description: >                 # Required. Drives auto-delegation
  Expert in X. Use proactively when Y.
tools: Read, Grep, Glob, Bash  # Optional. Omit = inherit all
model: sonnet                  # Optional. sonnet/opus/haiku/inherit
permissionMode: default        # Optional. default/acceptEdits/bypassPermissions/plan
skills: skill-a, skill-b       # Optional. Auto-load skills into agent context
memory: user                   # Optional. user/project/local — persistent memory
---

You are a [persona with specific role and expertise]...
```

**Key constraint**: Subagents cannot spawn other subagents (no nesting).

---

## Layer 4: Hooks (Event-Driven Automation)

**What**: Scripts that run automatically in response to tool events.
**Where**: Configured in `.claude/settings.json` or `~/.claude/settings.json`

**Events**: PreToolUse, PostToolUse, Notification, Stop, and others.

**Best for**: Linting after edits, validation before writes, desktop notifications.

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Write|Edit",
        "hooks": [
          {
            "type": "command",
            "command": "prettier --write $CLAUDE_FILE_PATH"
          }
        ]
      }
    ]
  }
}
```

---

## Layer 5: MCP Servers (External Integrations)

**What**: Model Context Protocol servers that give Claude access to external tools and data.
**Where**: Configured in `.claude/settings.json` or `~/.claude/settings.json`

**Best for**: Database access, Slack, GitHub API, custom internal APIs.

**Note**: Skills provide the knowledge of HOW to use MCP tools effectively. MCP provides
the connection; skills provide the expertise.

---

## Layer 6: Plugins (Distributable Packages)

**What**: Installable packages bundling skills, agents, hooks, and MCP configurations.
**Where**: Installed via `/plugin` command from marketplaces or local directories.

**Best for**: Team-wide distribution, versioned customization bundles.

**Structure**:
```
my-plugin/
├── .claude-plugin/
│   └── plugin.json          # Manifest (name required, rest optional)
├── skills/
│   └── my-skill/
│       └── SKILL.md
├── agents/
│   └── my-agent.md
└── hooks.json               # Optional
```

Plugin skills are namespaced: `/plugin-name:skill-name`
