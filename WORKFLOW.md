# WORKFLOW.md — tradelab-lib-lopezdp-utils

## Purpose

This document defines the work methodology for extracting, implementing, and productionizing
financial ML utilities from López de Prado's books into this library.

---

## Sources of Truth

| Source | Role |
|--------|------|
| **NotebookLM** (AFML + ML for Asset Managers) | **Only** source for theory, algorithms, and implementation details |
| **CLAUDE.md** | Project-specific instructions for Claude Code |
| **docs/phase1_extraction/TODO.md** | Phase 1 progress log (archived) |
| **This file** | Workflow rules and methodology |

**Rule:** Claude must NEVER use its own training knowledge for López de Prado theory.
Always query NotebookLM first. Own knowledge is only used for Python/engineering concerns.

---

## Two-Phase Development

### Phase 1: Pre-Production (Completed)

Extract all functionalities chapter by chapter, creating v1 implementations.

**Goal:** Complete coverage of all book functionalities with working, readable code.

**What v1 code looks like:**
- Faithful replication of the book's Python snippets
- Uses the same libraries as the book (pandas, numpy, scipy, etc.) — no Polars conversion yet
- Clean Python style: snake_case variables (no CamelCase from book), type hints, Google-style docstrings
- Docstrings include theoretical context: what the function does, when/why to use it, and book reference
- No tests, no performance optimization, no error handling beyond what the book provides
- Each chapter = one submodule under `tradelab.lopezdp_utils.<chapter_topic>`

### Phase 2: Production Optimization (Completed)

Convert all v1 code into production-grade, high-performance library code.

**Completed Scope:**
- Migrated pandas → Polars where applicable
- Added comprehensive test suites (unit + property-based): 285 tests passing
- Implemented robust error handling and input validation
- Public API design and `__init__.py` exports cleanup
- Documentation polish

**Session progress:**
- Session 1 (`_hpc.py`): ✅ Complete
- Session 2 (`data/`): ✅ Complete — 61 tests
- Session 3 (`labeling/`): ✅ Complete — 96 tests
- Session 4 (`features/`): ✅ Complete — 150 tests
- Session 5 (`modeling/`): ✅ Complete — 176 tests
- Session 6 (`evaluation/`): ✅ Complete
- Session 7 (`allocation/`): ✅ Complete — 285 total tests
- All sessions merged to `main`

---

## Session Workflow

### Starting a New Chapter

1. **User says:** "Let's extract all functionalities from Chapter N"
2. **Claude queries NotebookLM:**
   - "What are the main functionalities, algorithms, and utilities presented in Chapter N of AFML? For each, provide: name, brief description, and whether Python code is given in the book."
   - "Are there any related functionalities in 'ML for Asset Managers' that cover the same topic as AFML Chapter N? If so, list them and note whether they are redundant or complementary to the AFML versions."
3. **Claude updates TODO.md:**
   - Adds chapter section with all functionalities listed as `[ ]` (to-do)
   - Marks complementary items from ML for Asset Managers if any
4. **Claude implements** each functionality one by one:
   - Query NotebookLM for detailed algorithm/snippet for each function
   - Implement v1 following the rules above
   - Mark as `[x]` in TODO.md when done
5. **Claude updates README.md** with the new submodule and its functionalities
6. **Claude updates CLAUDE.md** if any project-level guidance changes

### Resuming an Interrupted Session

1. **Claude reads TODO.md** to see what's done and what's pending
2. **Claude picks up** from the first unchecked item in the current chapter
3. No need to re-query NotebookLM for the chapter plan — it's already in TODO.md

### Adding a New Dependency

If a book snippet requires a library not yet in `pyproject.toml`:
1. Add it to `project.dependencies` (runtime) or `project.optional-dependencies` (optional)
2. Run `uv sync --all-extras --dev`
3. Note the addition in the commit message

---

## Submodule Naming Convention

Each chapter maps to a submodule. Naming follows the chapter's primary topic:

```
src/tradelab/lopezdp_utils/
├── __init__.py
├── _hpc.py                  # Ch. 20 - Multiprocessing (Phase 2 complete)
# Phase 2 complete packages
├── data/                    # Ch. 2 + 19 - Bars, Sampling, Futures, ETF, Microstructure
├── labeling/                # Ch. 3 + 4 - Labeling + Sample Weights
├── features/                # Ch. 5, 8, 17, 18 + MLAM - Features
├── modeling/                # Ch. 6 + 7 + 9 - Ensemble, Cross-Validation, Hyperparameter Tuning
├── evaluation/              # Ch. 10-15 - Bet Sizing, Backtesting, Statistics, Strategy Risk
└── allocation/              # Ch. 16 - Machine Learning Asset Allocation
```

> **Note:** Chapters 1 (intro), 21-22 (quantum/specialized HPC) may not have extractable utilities.
> Structure confirmed via NotebookLM query of complete AFML table of contents.

---

## File Structure Per Submodule

```
data_structures/
├── __init__.py             # Public exports for this submodule
├── bars.py                 # Standard bars (time, tick, volume, dollar)
├── imbalance_bars.py       # Imbalance bars (tick, volume, dollar)
├── run_bars.py             # Run bars
└── ...                     # One file per logical grouping
```

---

## Commit Conventions

- **Extraction commits:** `feat(ch<N>): extract <functionality> from AFML Ch.<N>`
- **Bug fixes during extraction:** `fix(ch<N>): correct <issue> in <functionality>`
- **Doc updates:** `docs: update README/TODO/CLAUDE.md for Ch.<N>`
- **Production phase:** `refactor(ch<N>): optimize <functionality> for production`

---

## Quality Gates (Pre-Production Phase)

Before marking a chapter as complete in TODO.md:
- [ ] All functionalities from NotebookLM query are implemented
- [ ] All functions have docstrings with theory context
- [ ] Code uses snake_case (no book-style CamelCase)
- [ ] Code runs without import errors
- [ ] `uv run ruff check . --fix && uv run ruff format .` passes
- [ ] README.md updated with new submodule
- [ ] TODO.md fully updated
