# CLAUDE.md — tradelab-lib-lopezdp-utils

## Project Overview

This library extracts and productionizes all relevant financial ML utilities from:
- **"Advances in Financial Machine Learning"** (AFML) by Marcos López de Prado — primary source
- **"Machine Learning for Asset Managers"** by Marcos López de Prado — complementary source

The goal is a standalone Python library (`tradelab.lopezdp_utils`) containing every useful algorithm, data structure, and utility from these books, converted into efficient polars/numpy algorithms and organized by functionality.

**Development is split into two phases:**
1. **Pre-Production (completed):** Extract v1 implementations faithful to the book's code, using the same libraries (pandas, numpy, etc.), with clean Python style and docstrings.
2. **Production (current):** Optimize for algo trading (pandas → Polars, tests, performance, validation).

See `WORKFLOW.md` for the full methodology. Phase 1 TODO is archived at `docs/phase1_extraction/TODO.md`.

See `docs/plans/phase2_migration` for phase 2 context.

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

- **Session 1** (`_hpc.py`): complete
- **Session 2** (`data/`): complete — `data_structures/` and `microstructure/` are **deleted**; use `tradelab.lopezdp_utils.data` instead
- **Session 3** (`labeling/`): complete — old `labeling/` (barriers, bet_sizing, fixed_horizon, thresholds, trend_scanning) and `sample_weights/` are **deleted**; use `tradelab.lopezdp_utils.labeling` instead; `bet_sizing.py` and `strategy_redundancy.py` deferred to session 6
- **Session 4** (`features/`): complete — `fractional_diff/`, `entropy_features/`, `structural_breaks/`, `feature_importance/`, and remaining `data_structures/` files are **deleted**; use `tradelab.lopezdp_utils.features` instead; 150 total tests passing
- **Session 5** (`modeling/`): next
- See `docs/plans/phase2_migration/` for session plans and `LIBRARY_STANDARDS.md` for verified Polars patterns