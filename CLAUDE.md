# CLAUDE.md — tradelab-lib-lopezdp-utils

## Project Overview

This library extracts and productionizes all relevant financial ML utilities from:
- **"Advances in Financial Machine Learning"** (AFML) by Marcos López de Prado — primary source
- **"Machine Learning for Asset Managers"** by Marcos López de Prado — complementary source

The goal is a standalone Python library (`tradelab.lopezdp_utils`) containing every useful algorithm,
data structure, and utility from these books, organized by chapter/topic as submodules.

**Development is split into two phases:**
1. **Pre-Production (current):** Extract v1 implementations faithful to the book's code, using the same libraries (pandas, numpy, etc.), with clean Python style and docstrings.
2. **Production:** Optimize for algo trading (pandas → Polars, tests, performance, validation).

See `WORKFLOW.md` for the full methodology and `TODO.md` for progress tracking.

### NotebookLM — Single Source of Truth

- **Notebook**: AFML - López de Prado
- **URL**: https://notebooklm.google.com/notebook/334b6110-699f-4e34-acfc-05e138b65062
- **Library ID**: `afml-l-pez-de-prado`

---

## Mandatory Rules

### 1. ALWAYS consult NotebookLM for López de Prado theory

When the user references **any concept, method, or algorithm** from "Advances in Financial Machine Learning" (e.g., triple-barrier method, meta-labeling, fractional differentiation, purged cross-validation, feature importance, HRP, VPIN, entropy, structural breaks, bet sizing, etc.):

- **NEVER** rely on your own training knowledge for the theoretical or algorithmic details.
- **ALWAYS** delegate the question to NotebookLM first using the `ask_question` tool with notebook URL `https://notebooklm.google.com/notebook/334b6110-699f-4e34-acfc-05e138b65062`.
- Use the NotebookLM answer as the **ground truth** for implementation.
- You may use your own knowledge only for general Python/software engineering concerns (syntax, libraries, testing, etc.), never for the financial ML theory itself.

### 2. Use the NotebookLM research agent for queries

To save tokens, **ALWAYS** use the Task tool with `subagent_type: "general-purpose"` and `model: "haiku"` when querying NotebookLM. The agent prompt should:

1. Call `mcp__notebooklm__ask_question` with the question and the notebook URL above.
2. Return the answer verbatim.

Example usage pattern:
```
Task(
  description="Ask NotebookLM about X",
  subagent_type="general-purpose",
  model="haiku",
  prompt="Use the mcp__notebooklm__ask_question tool to ask the following question to notebook URL https://notebooklm.google.com/notebook/334b6110-699f-4e34-acfc-05e138b65062: '<your question here>'. Return the full answer."
)
```

This ensures NotebookLM queries are handled by a cheaper model while the main conversation uses the full model for implementation work.

### 3. Follow the extraction workflow

Before implementing anything, always:
1. Read `TODO.md` to understand current progress
2. Follow the session workflow described in `WORKFLOW.md`
3. Update `TODO.md`, `README.md`, and this file as work progresses