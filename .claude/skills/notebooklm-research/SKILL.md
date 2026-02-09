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

## Notebook

- **Name**: AFML - López de Prado
- **Library ID**: `afml-l-pez-de-prado`

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

## Query Templates

### Starting a new chapter
- "What are the main functionalities, algorithms, and utilities presented in Chapter N of AFML? For each, provide: name, brief description, and whether Python code is given in the book."
- "Are there any related functionalities in 'ML for Asset Managers' that cover the same topic as AFML Chapter N? List them and note whether they are redundant or complementary."

### Getting implementation details
- "Show the complete Python code snippet for [function name] from Chapter N."
- "What are the input parameters, return values, and mathematical formula for [algorithm]?"

### Clarifying theory
- "Explain the intuition behind [concept] and when it should be used in practice."

## Important

- Keep queries focused — one concept per question gets better answers
- Always save and reuse `session_id` within the same work session
- The notebook contains both AFML and ML for Asset Managers content
