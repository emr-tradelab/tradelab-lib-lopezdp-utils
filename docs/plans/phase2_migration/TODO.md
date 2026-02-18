# Phase 2 Migration — TODO

## Sessions Overview

| Session | Module | Status |
|---------|--------|--------|
| Session 1 | `_hpc.py` | ✅ Complete |
| Session 2 | `data/` | ✅ Complete |
| Session 3 | `labeling/` | 🔄 In Progress |
| Session 4 | `features/` | ⏳ Pending |
| Session 5 | `cross_validation/` | ⏳ Pending |
| Session 6 | `evaluation/` | ⏳ Pending |
| Session 7 | `backtest/` | ⏳ Pending |

---

## Session 3: `labeling/` — Labeling + Sample Weights

**Branch:** `phase2/labeling`
**Source directories:** `labeling/` (8 files) + `sample_weights/` (5 files)
**Target:** merged `labeling/` (4 files + `__init__.py`)

### Tasks

- [x] Task 1: Create branch, package structure, test fixtures
- [x] Task 2: Migrate `triple_barrier.py` (Polars)
- [x] Task 3: Migrate `meta_labeling.py` (Polars)
- [x] Task 4: Migrate `sample_weights.py` (Polars I/O + NumPy loops)
- [x] Task 5: Migrate `class_balance.py` (Polars)
- [x] Task 6: Create `labeling/__init__.py` with public exports
- [x] Task 7: Integration tests — labeling → weights pipeline
- [x] Task 8: Delete old directories, verify, lint
- [x] Task 9: Merge to main

### File Mapping

| Old | New | Action |
|-----|-----|--------|
| `labeling/triple_barrier.py` | `labeling/triple_barrier.py` | Migrate + merge |
| `labeling/meta_labeling.py` | `labeling/meta_labeling.py` | Migrate |
| `labeling/thresholds.py` | `labeling/triple_barrier.py` | Merge into |
| `labeling/barriers.py` | `labeling/triple_barrier.py` | Merge into |
| `labeling/fixed_horizon.py` | `labeling/triple_barrier.py` | Merge into |
| `labeling/trend_scanning.py` | `labeling/triple_barrier.py` | Merge into |
| `labeling/class_balance.py` | `labeling/class_balance.py` | Migrate |
| `labeling/bet_sizing.py` | `evaluation/bet_sizing.py` | Deferred to session 6 |
| `sample_weights/concurrency.py` | `labeling/sample_weights.py` | Merge |
| `sample_weights/sequential_bootstrap.py` | `labeling/sample_weights.py` | Merge |
| `sample_weights/return_attribution.py` | `labeling/sample_weights.py` | Merge |
| `sample_weights/class_weights.py` | `labeling/class_balance.py` | Merge |
| `sample_weights/strategy_redundancy.py` | `evaluation/overfitting.py` | Deferred to session 6 |
