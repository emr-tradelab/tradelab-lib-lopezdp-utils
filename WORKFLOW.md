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
| **TODO.md** | Progress tracking across all chapters and sessions |
| **This file** | Workflow rules and methodology |

**Rule:** Claude must NEVER use its own training knowledge for López de Prado theory.
Always query NotebookLM first. Own knowledge is only used for Python/engineering concerns.

---

## Two-Phase Development

### Phase 1: Pre-Production (Current)

Extract all functionalities chapter by chapter, creating v1 implementations.

**Goal:** Complete coverage of all book functionalities with working, readable code.

**What v1 code looks like:**
- Faithful replication of the book's Python snippets
- Uses the same libraries as the book (pandas, numpy, scipy, etc.) — no Polars conversion yet
- Clean Python style: snake_case variables (no CamelCase from book), type hints, Google-style docstrings
- Docstrings include theoretical context: what the function does, when/why to use it, and book reference
- No tests, no performance optimization, no error handling beyond what the book provides
- Each chapter = one submodule under `tradelab.lopezdp_utils.<chapter_topic>`

### Phase 2: Production Optimization (Later)

Convert all v1 code into production-grade, high-performance library code.

**Scope (notes for later):**
- Migrate pandas → Polars where applicable
- Add comprehensive test suites (unit + property-based)
- Performance optimization (pure Python + Polars, evaluate Numba for hot paths)
- Robust error handling and input validation (Pydantic)
- Public API design and `__init__.py` exports cleanup
- Documentation polish

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
4. **Claude presents the plan** to user for approval/adjustment
5. **Claude implements** each functionality one by one:
   - Query NotebookLM for detailed algorithm/snippet for each function
   - Implement v1 following the rules above
   - Mark as `[x]` in TODO.md when done
6. **Claude updates README.md** with the new submodule and its functionalities
7. **Claude updates CLAUDE.md** if any project-level guidance changes

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
# Part 1: Data Analysis
├── data_structures/         # Ch. 2 - Financial Data Structures
├── labeling/                # Ch. 3 - Labeling (Triple-Barrier, Meta-Labeling)
├── sample_weights/          # Ch. 4 - Sample Weights
├── fractional_diff/         # Ch. 5 - Fractionally Differentiated Features
# Part 2: Modelling
├── ensemble_methods/        # Ch. 6 - Ensemble Methods
├── cross_validation/        # Ch. 7 - Cross-Validation in Finance
├── feature_importance/      # Ch. 8 - Feature Importance
├── hyperparameter_tuning/   # Ch. 9 - Hyper-Parameter Tuning
# Part 3: Backtesting
├── bet_sizing/              # Ch. 10 - Bet Sizing
├── backtest_dangers/        # Ch. 11 - The Dangers of Backtesting
├── backtest_cv/             # Ch. 12 - Backtesting through Cross-Validation
├── backtest_synthetic/      # Ch. 13 - Backtesting on Synthetic Data
├── backtest_statistics/     # Ch. 14 - Backtest Statistics
├── strategy_risk/           # Ch. 15 - Understanding Strategy Risk
├── ml_asset_allocation/     # Ch. 16 - Machine Learning Asset Allocation
# Part 4: Useful Financial Features
├── structural_breaks/       # Ch. 17 - Structural Breaks
├── entropy/                 # Ch. 18 - Entropy Features
├── microstructure/          # Ch. 19 - Microstructural Features
# Part 5: High-Performance Computing
└── hpc/                     # Ch. 20 - Multiprocessing and Vectorization
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
