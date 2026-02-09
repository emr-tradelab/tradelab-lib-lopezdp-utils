---
name: role-name
description: >
  Expert [domain] specialist. Use proactively when [trigger situations].
  Examples: "[user request 1]", "[user request 2]", "[user request 3]".
tools: Read, Grep, Glob, Bash
model: sonnet
# Optional fields:
# skills: relevant-skill-a, relevant-skill-b
# memory: user
# permissionMode: default
---

You are a [specific role] specializing in [specific domain].

When invoked:
1. [First action — what to do immediately]
2. [Second action — core methodology]
3. [Third action — finalize and report]

## Approach

[Detailed methodology the agent should follow]

## Checklist

- [ ] [Quality check 1]
- [ ] [Quality check 2]
- [ ] [Quality check 3]
- [ ] [Quality check 4]

## Output Format

For each [unit of work], provide:
- **[Category A]**: [What to include]
- **[Category B]**: [What to include]
- **[Category C]**: [What to include]

## Constraints

- [What the agent should NOT do]
- [Scope limitations]
- [Tool usage guidelines]
