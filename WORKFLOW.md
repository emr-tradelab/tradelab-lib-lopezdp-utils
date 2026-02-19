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

### Phase 2: Production Optimization (Current)

Convert all v1 code into production-grade, high-performance library code.

**Scope (notes for later):**
- Migrate pandas → Polars where applicable
- Add comprehensive test suites (unit + property-based)
- Performance optimization (pure Python + Polars, evaluate Numba for hot paths)
- Robust error handling and input validation (Pydantic)
- Public API design and `__init__.py` exports cleanup
- Documentation polish

**Session progress:**
- Session 1 (`_hpc.py`): ✅ Complete — merged to `main`
- Session 2 (`data/`): ✅ Complete — merged to `main`, 61 tests passing
- Session 3 (`labeling/`): ✅ Complete — merged to `main`, 96 tests passing
- Session 4 (`features/`): ✅ Complete — merged to `main`, 150 tests passing
- Sessions 5-7: pending
- See `docs/plans/phase2_migration/` for detailed session plans

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
├── _hpc.py                  # Ch. 20 - Multiprocessing (Phase 2 complete, merged from hpc/)
# Phase 2 packages (Polars, tests, production-grade)
├── data/                    # Ch. 2 + 19 - Bars, Sampling, Futures, ETF, Microstructure (Phase 2 complete)
│   ├── __init__.py          # Re-exports bars + sampling public API
│   ├── bars.py              # Standard + information-driven bars
│   ├── sampling.py          # CUSUM filter + linspace/uniform sampling
│   ├── futures.py           # Roll gaps, roll-and-rebase, HDF5 loader stub
│   ├── etf.py               # ETF trick
│   └── microstructure.py    # Spreads, price impact, VPIN
├── labeling/                # Ch. 3 + 4 - Labeling + Sample Weights (Phase 2 complete)
│   ├── __init__.py          # Public exports
│   ├── triple_barrier.py    # Triple-Barrier, Meta-Labeling, Fixed-Horizon, Trend-Scanning
│   ├── meta_labeling.py     # Meta-Labeling asymmetric barriers
│   ├── sample_weights.py    # Concurrency, uniqueness, sequential bootstrap, return attribution
│   └── class_balance.py     # Class weights and imbalance handling
├── features/                # Ch. 5, 8, 17, 18 + MLAM - Features (Phase 2 complete)
│   ├── __init__.py          # Public exports (32 symbols)
│   ├── fractional_diff.py   # FFD, expanding fracdiff, min-FFD (Polars I/O, NumPy core)
│   ├── entropy.py           # Shannon/LZ estimators, encoding, market efficiency, MI, VI
│   ├── structural_breaks.py # SADF, CUSUM (BDE + CSW), Chow-DF, QADF, CADF
│   ├── importance.py        # MDI, MDA, SFI, ONC clustering, clustered MDI/MDA, synthetic data
│   └── orthogonal.py        # PCA weights, orthogonal features, weighted Kendall tau
# Phase 1 submodules (pandas/numpy, v1 only — pending Phase 2 migration)
├── ensemble_methods/        # Ch. 6 - Ensemble Methods
├── cross_validation/        # Ch. 7 - Cross-Validation in Finance
├── hyperparameter_tuning/   # Ch. 9 - Hyper-Parameter Tuning
├── bet_sizing/              # Ch. 10 - Bet Sizing
├── backtest_dangers/        # Ch. 11 - The Dangers of Backtesting
├── backtest_cv/             # Ch. 12 - Backtesting through Cross-Validation
├── backtest_synthetic/      # Ch. 13 - Backtesting on Synthetic Data
├── backtest_statistics/     # Ch. 14 - Backtest Statistics
├── strategy_risk/           # Ch. 15 - Understanding Strategy Risk
└── ml_asset_allocation/     # Ch. 16 - Machine Learning Asset Allocation
# Deleted (Phase 2 migration):
# data_structures/           → data/ (merged in session 2)
# microstructure/            → data/microstructure.py (merged in session 2)
# hpc/                       → _hpc.py (merged in session 1)
# labeling/ (old v1 files)   → labeling/ (merged in session 3)
# sample_weights/            → labeling/sample_weights.py + labeling/class_balance.py (merged in session 3)
# fractional_diff/           → features/fractional_diff.py (merged in session 4)
# entropy_features/          → features/entropy.py (merged in session 4)
# structural_breaks/         → features/structural_breaks.py (merged in session 4)
# feature_importance/        → features/importance.py + features/orthogonal.py (merged in session 4)
# data_structures/discretization.py and pca.py → features/entropy.py + features/orthogonal.py (merged in session 4)
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
