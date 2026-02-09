---
name: notebooklm-research
description: >
  Querying NotebookLM for López de Prado financial ML theory from AFML and ML for Asset Managers.
  Use when implementing any concept from these books, starting a new chapter extraction,
  or needing algorithm details, code snippets, or theoretical context from the books.
---

# NotebookLM Research for López de Prado

## Core Rule

**NEVER** rely on training knowledge for financial ML theory from López de Prado's books.
**ALWAYS** query NotebookLM first. Own knowledge is only acceptable for general Python/engineering.

## How to Query

Use the `mcp__notebooklm__ask_question` tool. Always reuse `session_id` within a session for contextual follow-ups.

### Session Flow

1. **Start broad** (no session_id — creates one):
   ```
   ask_question({ question: "What are the main functionalities in Chapter N of AFML?" })
   ```
   Save the returned `session_id`.

2. **Go specific** (same session):
   ```
   ask_question({ question: "Show the Python implementation for X", session_id })
   ```

3. **Check complementary source** (same session):
   ```
   ask_question({ question: "Does ML for Asset Managers cover this topic? If so, what's complementary?", session_id })
   ```

4. **Get implementation details** (same session):
   ```
   ask_question({ question: "What are the exact parameters and edge cases for this algorithm?", session_id })
   ```

## Important

- Keep queries focused — one concept per question gets better answers
- Always save and reuse `session_id` within the same work session
- The notebook contains both AFML and ML for Asset Managers content
