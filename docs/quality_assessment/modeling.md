# Quality Assessment: modeling

**Module:** `src/tradelab/lopezdp_utils/modeling/`
**Chapters:** AFML Ch. 6 (Ensemble Methods), Ch. 7 (Cross-Validation), Ch. 9 (Hyperparameter Tuning) + MLAM Section 6.4
**Date:** 2026-02-21
**Result:** PASS

## Source Files Assessed

- `cross_validation.py` — `get_train_times`, `get_embargo_times`, `PurgedKFold`, `cv_score`, `probability_weighted_accuracy`
- `ensemble.py` — `bagging_accuracy`, `build_random_forest`, `bagging_classifier_factory`
- `hyperparameter_tuning.py` — `_LogUniformGen`, `log_uniform`, `MyPipeline`, `clf_hyper_fit`

## Tests

26 tests, all passing.

## Theory Comparison

### cross_validation.py

| Function | Snippet | Verdict | Notes |
|----------|---------|---------|-------|
| `get_train_times` | 7.1 | Correct | Three overlap conditions match book exactly |
| `get_embargo_times` | 7.2 | Correct | Step computation and concat logic match |
| `PurgedKFold` | 7.3 | Correct | Uses searchsorted as book does; purge + embargo logic faithful |
| `cv_score` | 7.4 | Correct | Fixes both sklearn bugs: `labels=clf.classes_` for log_loss, sample_weight routing |
| `probability_weighted_accuracy` | MLAM 6.4 | Correct | Formula: `sum((p-1/k) * correct) / sum(p-1/k)` matches book |

### ensemble.py

| Function | Snippet | Verdict | Notes |
|----------|---------|---------|-------|
| `bagging_accuracy` | 6.1 | Correct | Majority voting formula matches |
| `build_random_forest` method 0 | 6.2 | Correct | `balanced_subsample`, `entropy`, 1000 estimators |
| `build_random_forest` method 1 | 6.2 | Correct | Book uses `max_features='auto'`; we use `'sqrt'` — correct sklearn modernization (`auto` deprecated in favor of `sqrt`) |
| `build_random_forest` method 2 | 6.2 | Correct | Single-tree RF wrapped in BaggingClassifier with `max_features=1.0` |
| `bagging_classifier_factory` | — | N/A | Utility not from a specific snippet; reasonable factory |

### hyperparameter_tuning.py

| Function | Snippet | Verdict | Notes |
|----------|---------|---------|-------|
| `log_uniform` | 9.4 | Correct | CDF and PDF match book's `logUniform_gen` |
| `MyPipeline` | 9.2 | Correct | Routes `sample_weight` via `steps[-1][0] + '__sample_weight'` |
| `clf_hyper_fit` | 9.1/9.3 | Correct | Scoring: f1 for {0,1} labels, neg_log_loss otherwise; grid/random search with PurgedKFold; optional bagging wrapper refits on full data |

## Issues Found

**P0 (Critical):** None
**P1 (Important):** None
**P2 (Minor):** None

## Conclusion

All implementations faithfully follow López de Prado's book. The only deviation (`max_features='auto'` → `'sqrt'`) is a correct sklearn API modernization. Module validated as correct.
