## Sessions 2-7: High-Level Outlines

Each session gets its own detailed plan document when we start it. Below is the scope and key decisions for each.

### Session 2: `data/` (branch: `phase2/data`)
**Source:** `data_structures/` (9 files, 1888 lines) + `microstructure/` (5 files, 568 lines)
**Target:** `data/bars.py`, `data/sampling.py`, `data/futures.py`, `data/etf.py`, `data/microstructure.py`
**Key work:**
- Merge 3 bar files (standard, imbalance, runs) into `bars.py`
- Polars migration for bar construction (heavy aggregation/rolling — big Polars win)
- Polars migration for CUSUM filter
- Microstructure stays NumPy-heavy (regressions, covariance)
- `discretization.py` and `pca.py` move to `features/` (session 4)
- `# TODO(numba): evaluate JIT` on imbalance bar inner loops
- Tests: bar correctness with known inputs, CUSUM event detection, spread estimators

### Session 3: `labeling/` (branch: `phase2/labeling`)
**Source:** `labeling/` (8 files, 1088 lines) + `sample_weights/` (5 files, 922 lines)
**Target:** `labeling/triple_barrier.py`, `labeling/meta_labeling.py`, `labeling/sample_weights.py`, `labeling/class_balance.py`
**Key work:**
- Merge `thresholds.py`, `barriers.py`, `fixed_horizon.py`, `trend_scanning.py` into `triple_barrier.py`
- Merge 4 sample_weights files into `labeling/sample_weights.py`
- `strategy_redundancy.py` moves to `evaluation/overfitting.py` (session 6)
- `labeling/bet_sizing.py` moves to `evaluation/bet_sizing.py` (session 6)
- Polars migration for triple-barrier event loop, concurrency counting
- Pydantic: `TripleBarrierConfig`, `SampleWeightConfig`
- Enforce t1 non-null validation at boundaries
- `# TODO(numba): evaluate JIT` on sequential bootstrap probability update
- Tests: triple-barrier with known price paths, uniqueness scores, sequential bootstrap IID improvement

### Session 4: `features/` (branch: `phase2/features`)
**Source:** `fractional_diff/` (2 files, 417 lines) + `entropy_features/` (4 files, 501 lines) + `structural_breaks/` (3 files, 605 lines) + `feature_importance/` (4 files, 888 lines)
**Target:** `features/fractional_diff.py`, `features/entropy.py`, `features/structural_breaks.py`, `features/importance.py`, `features/orthogonal.py`
**Key work:**
- Absorb `data_structures/discretization.py` into `features/entropy.py`
- Absorb `data_structures/pca.py` into `features/orthogonal.py`
- Merge clustering + synthetic + importance into `features/importance.py`
- Polars for tabular parts of fracdiff, entropy encoding
- NumPy stays for ADF tests, eigendecomposition, OLS in SADF
- `# TODO(numba): evaluate JIT` on FFD convolution loop, LZ decomposition
- Tests: fracdiff known values, entropy estimators, SADF on synthetic explosive series, MDI/MDA on synthetic data

### Session 5: `modeling/` (branch: `phase2/modeling`)
**Source:** `cross_validation/` (2 files, 326 lines) + `ensemble_methods/` (1 file, 183 lines) + `hyperparameter_tuning/` (2 files, 223 lines)
**Target:** `modeling/cross_validation.py`, `modeling/ensemble.py`, `modeling/hyperparameter_tuning.py`
**Key work:**
- PurgedKFold stays sklearn-compatible (inherits KFold) — no Polars here
- cv_score stays NumPy/sklearn
- Validate t1 presence at PurgedKFold boundaries (raise if missing)
- Ensemble configs remain sklearn wrappers
- Tests: purging correctness (no overlap between train/test), embargo gap verification, cv_score with known classifier

### Session 6: `evaluation/` (branch: `phase2/evaluation`)
**Source:** `backtest_statistics/` + `backtest_cv/` + `backtesting_dangers/` + `backtest_synthetic/` + `strategy_risk/` + `bet_sizing/` + `sample_weights/strategy_redundancy.py` + `labeling/bet_sizing.py`
**Target:** 6 files under `evaluation/`
**Key work:**
- Polars for backtest statistics (return series aggregation, drawdown computation)
- CPCV stays sklearn-compatible (inherits from PurgedKFold in modeling/)
- DSR requires explicit `num_trials` parameter (no default)
- Strategy redundancy merges with CSCV into `overfitting.py`
- Both bet_sizing sources merge into `evaluation/bet_sizing.py`
- Tests: SR/PSR/DSR with known values, CPCV path count formulas, OTR convergence

### Session 7: `allocation/` (branch: `phase2/allocation`)
**Source:** `ml_asset_allocation/` (5 files, 793 lines)
**Target:** `allocation/hrp.py`, `allocation/denoising.py`, `allocation/nco.py`
**Key work:**
- Mostly NumPy/SciPy (matrix operations, optimization) — minimal Polars opportunity
- Merge `simulation.py` into `hrp.py`
- NCO depends on `features/importance.py` clustering (already migrated in session 4)
- Tests: HRP on known correlation matrix, Marcenko-Pastur fit, NCO allocation sums to 1

---

## Final Session: Cleanup
- Update top-level `__init__.py` with package exports
- Rewrite `README.md` with new structure
- Update `CLAUDE.md` with new import paths
- Update `WORKFLOW.md` for Phase 2 completion
- Version bump `pyproject.toml` to 0.2.0
- Tag release
