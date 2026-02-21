---
name: quality-assessment
description: >
  Comprehensive quality assessment of a lopezdp_utils module against
  López de Prado theory. Use when evaluating implementation correctness
  after completing a Phase 2 migration session.
user-invocable: true
argument-hint: "<module_name> (e.g., evaluation, labeling, features)"
---

# Quality Assessment — López de Prado Module Audit

## Purpose

Systematically compare a module's implementation against the book's theory, checking for formula correctness, missing functionality, edge cases, API consistency, and test coverage gaps.

## Process

### Step 1: Read all source files

Read every `.py` file in the target module:
```
src/tradelab/lopezdp_utils/<module>/
```
Note every public function, its signature, docstring, and implementation details.

### Step 2: Read all test files

Read every test file:
```
tests/<module>/
```
Including `conftest.py` fixtures. Note what is tested and what is NOT tested.

### Step 3: Query NotebookLM for the relevant theory

Query `notebooklm-researcher` agent for:

**First query — broad overview:**
> "Give me a comprehensive overview of ALL concepts López de Prado covers
> in [relevant chapters]. List every function, formula, algorithm, and
> parameter with full detail."

**Second query — specific details** (same session):
> Fill gaps from the first answer. Ask about specific algorithms,
> parameter conventions, edge cases the book warns about.

**NotebookLM URL:** `https://notebooklm.google.com/notebook/334b6110-699f-4e34-acfc-05e138b65062`

### Step 4: Run existing tests

```bash
uv run pytest tests/<module>/ -v --tb=short
```

Record pass/fail count.

### Step 5: Compare and produce assessment

For each function in the module, check:

1. **Formula correctness** — Does the implementation match the book's math?
   Pay attention to: sign conventions, parameter definitions (raw vs excess
   kurtosis, annualized vs non-annualized), denominator terms.

2. **Missing functionality** — Are there algorithms from the book chapters
   that are NOT implemented? List them with chapter references.

3. **Parameter naming** — Does the API use the book's conventions or
   Python-idiomatic names? Flag confusing mismatches (e.g., `seed` used
   for initial price instead of RNG seed).

4. **Edge cases** — Division by zero, empty inputs, degenerate cases
   (all positive returns, single element, etc.). Check guards exist.

5. **API consistency** — Polars I/O at boundaries? Snake_case? Docstrings
   with type info? Consistent with other modules in the library?

6. **Test quality** — Are tests smoke tests (type/bounds checks) or
   analytical correctness tests (hand-computed reference values)?
   Flag missing correctness tests.

7. **Performance concerns** — O(n^2) loops in Python where vectorized
   alternatives exist? Note but don't block on these.

### Step 6: Output structured report

Format the assessment as:

```markdown
# <Module> — Quality Assessment

## Overall Verdict: <one-line summary>

## 1. Correctness Issues
### 1.1 <issue title> (Bug/Warning)
<file:line> — <description, expected vs actual>

## 2. Missing Functionality (vs. the books)
### 2.1 <feature> — Not implemented
<chapter reference, what it does, impact>

## 3. Edge Cases & Robustness Issues
### 3.1 <issue>
<description, suggested fix>

## 4. Test Quality Assessment
<coverage summary, gaps>

## 5. Summary: Priority Fixes
| Priority | Issue | Impact |
|----------|-------|--------|
| P0 | ... | ... |
| P1 | ... | ... |
| P2 | ... | ... |
```

### Step 7: Decide and apply fixes

After producing the assessment, evaluate whether fixes are needed:

**No fixes needed** — If there are zero P0 and zero P1 issues:
- Save the assessment report to `docs/quality_assessment/<module>.md`
- Return verdict: `PASS — no fixes required`

**Fixes needed** — If there are P0 or P1 issues:
- Apply fixes in priority order (P0 first, then P1)
- For each fix:
  1. Edit the source file with the correction
  2. If the fix changes behavior, update or add tests to validate the correction
  3. Re-query NotebookLM if uncertain about the correct formula or parameter
- Do NOT apply P2 fixes unless they are trivial (< 5 lines, no behavior change)
- After all fixes, run the full quality gate:
  ```bash
  uvx ruff check --fix . && uvx ruff format .
  uv run pytest tests/<module>/ -v --tb=short
  ```
- If tests fail after fixes, debug and correct — do not leave broken tests
- Save the assessment report (including applied fixes) to `docs/quality_assessment/<module>.md`
- Return verdict: `FIXED — <N> issues resolved`

**Fix application criteria (be critical):**
- Only fix what is clearly wrong or missing according to the book's theory (confirmed via NotebookLM)
- Do NOT refactor working code for style preferences
- When in doubt, skip the fix and document it as "needs manual review"

## Rules

- NEVER rely on training knowledge for López de Prado theory — ALWAYS query NotebookLM
- Python/software engineering knowledge is fine to use directly
- Be specific: cite file paths, line numbers, and exact formulas
- Distinguish between bugs (wrong output) and gaps (missing features)
- P0 = incorrect results silently, P1 = crashes or missing critical features, P2 = nice-to-have
