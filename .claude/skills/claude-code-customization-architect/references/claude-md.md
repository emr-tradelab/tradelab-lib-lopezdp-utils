# CLAUDE.md Best Practices

## What CLAUDE.md Is

CLAUDE.md is a markdown file that Claude reads at the start of every session. It provides
persistent project context — think of it as a "memory card" for your project.

## File Locations

| Location | Scope | When Loaded |
|----------|-------|-------------|
| `CLAUDE.md` at project root | Project-wide | Every session in this project |
| `.claude/CLAUDE.md` at project root | Project-wide | Every session in this project |
| `CLAUDE.md` in subdirectories | Directory-specific | When Claude works on files there |
| `~/.claude/CLAUDE.md` | User-wide | Every session, every project |

All levels are **additive** — content from all matching files is merged.

## What to Include

### Always Include
- **Build commands**: `npm run build`, `npm test`, `cargo test`, etc.
- **Coding standards**: TypeScript strict mode, functional patterns, naming conventions
- **Project architecture**: Brief overview of directory structure and key abstractions
- **Test commands**: How to run specific test files vs full suite
- **"Never do X" rules**: Hard constraints the AI must always follow

### Consider Including
- **Git workflow**: Branch naming, commit message format, PR process
- **Environment setup**: Required env vars, how to start dev server
- **Key dependencies**: Frameworks, ORMs, testing libraries in use
- **Error handling patterns**: Your project's conventions for errors/exceptions

## What NOT to Include

- **Detailed API documentation** → Use a skill instead
- **Long style guides** → Use a skill instead
- **Multi-step workflows** → Use a skill (with `/slash-command`)
- **Persona instructions** → Use a subagent instead
- **Anything over ~500 lines total** → Split excess into skills

## Example

```markdown
# Project: Acme Dashboard

## Tech Stack
- Next.js 15 with App Router
- TypeScript strict mode
- Prisma ORM with PostgreSQL
- Tailwind CSS
- Jest + React Testing Library

## Build & Test
- `npm run dev` — start dev server (port 3000)
- `npm run build` — production build
- `npm test` — run all tests
- `npm test -- --testPathPattern=path/to/file` — single test file
- `npm run lint` — ESLint + Prettier check

## Conventions
- Use Server Components by default; Client Components only when needed
- All database queries go through Prisma service layer in `src/services/`
- Use zod for runtime validation at API boundaries
- Error responses follow RFC 7807 (Problem Details)
- Never use `any` type — use `unknown` and narrow

## Directory Structure
- `src/app/` — Next.js App Router pages and layouts
- `src/components/` — Shared React components
- `src/services/` — Business logic and database access
- `src/lib/` — Utility functions and shared types
- `prisma/` — Schema and migrations

## Git
- Branch naming: `feature/TICKET-123-short-description`
- Commit messages follow Conventional Commits
- Always run `npm test` before committing
```

## Subdirectory CLAUDE.md

Add CLAUDE.md files in subdirectories for context-specific guidance:

```
src/
├── CLAUDE.md              # General source conventions
├── services/
│   └── CLAUDE.md          # Service layer patterns, Prisma conventions
├── components/
│   └── CLAUDE.md          # Component patterns, Tailwind usage
└── tests/
    └── CLAUDE.md          # Test conventions, fixture patterns
```

These only load when Claude works on files in that directory.

## Referencing Other Files

You can reference other files from CLAUDE.md:
```markdown
@AGENTS.md
See also: docs/architecture.md
```

This tells Claude to check these files for additional context.

## CLAUDE.md vs Skills vs Agents

| Question | Answer |
|----------|--------|
| Should Claude always know this? | → CLAUDE.md |
| Is it reference material for specific tasks? | → Skill |
| Does it require a persona or tool restrictions? | → Subagent |
| Is it a repeatable workflow? | → Skill (with `/slash-command`) |
| Is it growing past 500 lines? | → Move overflow to skills |
