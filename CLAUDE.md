# CLAUDE.md — tradelab-lib-lopezdp-utils

## Project Overview

This library extracts and productionizes all relevant financial ML utilities from:
- **"Advances in Financial Machine Learning"** (AFML) by Marcos López de Prado — primary source
- **"Machine Learning for Asset Managers"** by Marcos López de Prado — complementary source

The goal is a standalone Python library (`tradelab.lopezdp_utils`) containing every useful algorithm, data structure, and utility from these books, converted into efficient polars/numpy algorithms and organized by functionality.

Both pre-production extraction (Phase 1) and production optimization (Phase 2) are complete. Currently in **Phase 3: Quality Assessment**. Phase 1 history is archived at `docs/phase1_extraction/TODO.md`.

---

## Mandatory Rules

### 1. ALWAYS consult NotebookLM for López de Prado theory

- **NEVER** rely on your own training knowledge for financial ML theory from these books.
- **ALWAYS** delegate to the `notebooklm-researcher` agent before implementing any concept.
- You may use your own knowledge only for general Python/software engineering concerns.
- **NotebookLM URL:** `https://notebooklm.google.com/notebook/334b6110-699f-4e34-acfc-05e138b65062`

### 2. Document as you go

After any big or complex implementation, call `Docs-Updater` subagent to update docs.

---

## Phase 2 Status

**Phase 2 (Production Optimization) is COMPLETE.** All 7 sessions done, 285 total tests passing.

**Final Module Layout:**
- `_hpc.py` — HPC engine (internal)
- `data/` — bars, sampling, futures, etf, microstructure (Polars I/O)
- `labeling/` — triple-barrier, meta-labeling, sample weights, class balance (Polars I/O)
- `features/` — fractional diff, entropy, structural breaks, feature importance, orthogonal (Polars I/O)
- `modeling/` — cross-validation, ensemble, hyperparameter tuning (sklearn-native)
- `evaluation/` — bet sizing, combinatorial backtests, overfitting detection, strategy risk, synthetic backtests (pandas/numpy)
- `allocation/` — HRP, RMT denoising, NCO, Monte Carlo simulation (numpy/pandas/scipy)

**Chapter-to-package mapping:**
```
src/tradelab/lopezdp_utils/
├── _hpc.py         # Ch. 20 — Multiprocessing
├── data/           # Ch. 2 + 19 — Bars, Sampling, Futures, ETF, Microstructure
├── labeling/       # Ch. 3 + 4 — Labeling + Sample Weights
├── features/       # Ch. 5, 8, 17, 18 + MLAM — Features
├── modeling/       # Ch. 6, 7, 9 — Ensemble, Cross-Validation, Hyperparameter Tuning
├── evaluation/     # Ch. 10-15 — Bet Sizing, Backtesting, Statistics, Strategy Risk
└── allocation/     # Ch. 16 + MLAM — HRP, Denoising, NCO
```

See `LIBRARY_STANDARDS.md` for verified Polars patterns.

---

## Phase 3: Quality Assessment (Current)

Validate each module's implementation against López de Prado's theory using the quality-assessment skill.

**Process:**
1. Run quality-assessment skill on one module per session
2. Apply corrections for theory mismatches, API improvements, or missing edge cases
3. Mark module as validated in documentation

**Modules to assess (in order):**
- `data/` (bars, sampling, futures, etf, microstructure)
- `labeling/` (triple-barrier, meta-labeling, sample weights, class balance)
- `features/` (fractional diff, entropy, structural breaks, feature importance, orthogonal)
- `modeling/` (cross-validation, ensemble, hyperparameter tuning)
- `evaluation/` (bet sizing, combinatorial backtests, overfitting, strategy risk, synthetic)
- `allocation/` (HRP, RMT denoising, NCO, simulation)