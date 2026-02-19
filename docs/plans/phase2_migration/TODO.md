# Phase 2 Migration — TODO

## Sessions Overview

| Session | Module | Status |
|---------|--------|--------|
| Session 1 | `_hpc.py` | ✅ Complete |
| Session 2 | `data/` | ✅ Complete |
| Session 3 | `labeling/` | ✅ Complete |
| Session 4 | `features/` | ✅ Complete |
| Session 5 | `modeling/` | 📋 Planned |
| Session 6 | `evaluation/` | ⏳ Pending |
| Session 7 | `backtest/` | ⏳ Pending |

---

## Session 4: `features/` — Fractional Diff, Entropy, Structural Breaks, Feature Importance, Orthogonal Features

**Branch:** `phase2/features`
**Source directories:** `fractional_diff/`, `entropy_features/`, `structural_breaks/`, `feature_importance/`, `data_structures/discretization.py`, `data_structures/pca.py`
**Target:** new `features/` package (5 submodules + `__init__.py`)

### Tasks

- [x] Task 1: Create branch, package structure, test fixtures
- [x] Task 2: Migrate `fractional_diff.py` (FFD, expanding fracdiff, min-FFD — Polars I/O, NumPy core)
- [x] Task 3: Migrate `entropy.py` (Shannon/LZ estimators, binary/quantile/sigma encoding, market efficiency, MI, VI — Polars I/O for encoding)
- [x] Task 4: Migrate `structural_breaks.py` (SADF, CUSUM BDE+CSW, Chow-DF, QADF, CADF — Polars I/O, pandas/NumPy internals)
- [x] Task 5: Migrate `importance.py` (MDI, MDA, SFI, ONC clustering, clustered MDI/MDA, synthetic test data — pandas/sklearn)
- [x] Task 6: Migrate `orthogonal.py` (PCA weights, orthogonal features, weighted Kendall tau — NumPy/scipy)
- [x] Task 7: Create `features/__init__.py` with 32 public exports
- [x] Task 8: Add 54 tests across 6 test files (including 3 integration tests); 150 total passing
- [x] Task 9: Delete old directories, verify, lint, merge to main

### File Mapping

| Old | New | Action |
|-----|-----|--------|
| `fractional_diff/` | `features/fractional_diff.py` | Migrate + merge |
| `entropy_features/` | `features/entropy.py` | Migrate + merge |
| `structural_breaks/` | `features/structural_breaks.py` | Migrate + merge |
| `feature_importance/` | `features/importance.py` + `features/orthogonal.py` | Migrate + split |
| `data_structures/discretization.py` | `features/entropy.py` | Merge into |
| `data_structures/pca.py` | `features/orthogonal.py` | Merge into |

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
