# Phase 2 Production Optimization — Design Document

**Date:** 2026-02-17
**Status:** Approved
**Scope:** Full production optimization of tradelab-lib-lopezdp-utils

---

## Context

Phase 1 is complete: 20 AFML chapters extracted into 19 flat submodules (~10,500 lines, 84 files).
All code is pandas/numpy v1 with no tests, no Polars, no Pydantic.

The user works exclusively with **1-minute OHLCV time bars** (no tick data). López de Prado recommends
the CUSUM filter as the best event-driven sampling method for time-bar-only practitioners.

---

## Theoretical Foundation (from NotebookLM)

### López de Prado's Pipeline Order
1. Bar formation → 2. Labeling (triple-barrier, generates t1) → 3. Sample weighting (requires t1) →
4. Feature engineering (fracdiff for stationarity) → 5. Cross-validation (purged, requires t1) →
6. Backtesting (CPCV + DSR, final sanity check only)

### Critical Safety Guardrails
- **t1 timestamps are first-class citizens** — never discard; required by weighting, purging, embargo
- **Purging + embargoing mandatory** for any cross-validation
- **Deflated Sharpe Ratio** must account for number of trials (no default that hides multiple testing)
- **Sequential bootstrap** over standard bootstrap (overlapping labels are not IID)
- **Triple-barrier** over fixed-horizon labeling (path-dependent, respects risk management)
- **Backtesting is a final sanity check**, never a research tool (Marcos' Laws)

### Bar Types with Time Bars Only
- Imbalance/run bars cannot be built from OHLCV (require tick sequences)
- Volume/dollar bars can be approximated by aggregating 1-min bars (resolution capped at 1 min)
- **CUSUM filter** is the recommended approach for event-driven sampling from time bars

---

## Decisions

### Polars Strategy: Hybrid
- **Polars** for tabular operations: bar construction, labeling, CUSUM, sample weights, backtest stats
- **NumPy/SciPy** for matrix math, statistical tests (ADF, OLS), sklearn interfaces, clustering
- **Numba** only if profiling justifies it — add `# TODO(numba): evaluate JIT for this loop` comments

### Migration Approach: Module-by-Module
- Each package fully restructured, migrated, tested before moving to the next
- Each session on its own branch: `phase2/<package>`

### Testing: Unit + Integration
- Unit tests for every public function with deterministic inputs
- Integration tests verifying outputs feed correctly into downstream modules
- pytest fixtures with small realistic datasets

### Backward Compatibility: Clean Break
- Old import paths will break (major version bump 0.1 → 0.2)
- No compatibility shims
- Plotting functions isolated in `_plots.py` (optional, matplotlib stays optional dep)

---

## New Module Structure

```
tradelab.lopezdp_utils/
├── data/                          # Data layer
│   ├── bars.py                    # standard, imbalance, runs bars (merged)
│   ├── sampling.py                # CUSUM filter, event detection
│   ├── futures.py                 # roll adjustment
│   ├── etf.py                     # ETF trick
│   └── microstructure.py          # spreads, price impact, VPIN
├── features/                      # Feature engineering
│   ├── fractional_diff.py         # fracdiff, FFD
│   ├── entropy.py                 # all entropy estimators + applications
│   ├── structural_breaks.py       # CUSUM tests, SADF, explosiveness
│   ├── importance.py              # MDI, MDA, SFI, clustered
│   └── orthogonal.py              # PCA decorrelation
├── labeling/                      # Labeling + weighting (tightly coupled)
│   ├── triple_barrier.py          # barriers, events, bins
│   ├── meta_labeling.py           # secondary model labels
│   ├── sample_weights.py          # concurrency, uniqueness, sequential bootstrap, decay
│   └── class_balance.py           # class weights, drop_labels
├── modeling/                      # Model training + tuning
│   ├── cross_validation.py        # PurgedKFold, cv_score, embargo
│   ├── ensemble.py                # RF configs, bagging factory
│   └── hyperparameter_tuning.py   # grid/random search with purged CV
├── evaluation/                    # Backtest + risk assessment
│   ├── statistics.py              # SR, PSR, DSR, HHI, drawdowns, bet timing
│   ├── cpcv.py                    # combinatorial purged CV paths
│   ├── overfitting.py             # CSCV/PBO, strategy redundancy
│   ├── synthetic.py               # OTR, O-U process
│   ├── strategy_risk.py           # binomial model
│   └── bet_sizing.py              # signal -> position size
├── allocation/                    # Portfolio construction (standalone)
│   ├── hrp.py                     # HRP pipeline
│   ├── denoising.py               # Marcenko-Pastur, shrinkage
│   └── nco.py                     # nested clustered optimization
└── _hpc.py                        # Internal parallel dispatch (private)
```

### Mapping: Old → New

| Old Module | New Location |
|------------|-------------|
| `data_structures/standard_bars.py` | `data/bars.py` |
| `data_structures/imbalance_bars.py` | `data/bars.py` |
| `data_structures/runs_bars.py` | `data/bars.py` |
| `data_structures/sampling.py` | `data/sampling.py` |
| `data_structures/futures.py` | `data/futures.py` |
| `data_structures/etf.py` | `data/etf.py` |
| `data_structures/pca.py` | `features/orthogonal.py` (merge with existing) |
| `data_structures/discretization.py` | `features/entropy.py` (merge) |
| `microstructure/*` | `data/microstructure.py` |
| `labeling/triple_barrier.py` | `labeling/triple_barrier.py` |
| `labeling/meta_labeling.py` | `labeling/meta_labeling.py` |
| `labeling/thresholds.py` | `labeling/triple_barrier.py` (merge) |
| `labeling/barriers.py` | `labeling/triple_barrier.py` (merge) |
| `labeling/trend_scanning.py` | `labeling/triple_barrier.py` (merge) |
| `labeling/bet_sizing.py` | `evaluation/bet_sizing.py` |
| `labeling/fixed_horizon.py` | `labeling/triple_barrier.py` (merge) |
| `labeling/class_balance.py` | `labeling/class_balance.py` |
| `sample_weights/concurrency.py` | `labeling/sample_weights.py` |
| `sample_weights/sequential_bootstrap.py` | `labeling/sample_weights.py` |
| `sample_weights/return_attribution.py` | `labeling/sample_weights.py` |
| `sample_weights/class_weights.py` | `labeling/class_balance.py` (merge) |
| `sample_weights/strategy_redundancy.py` | `evaluation/overfitting.py` |
| `fractional_diff/*` | `features/fractional_diff.py` |
| `entropy_features/*` | `features/entropy.py` |
| `structural_breaks/*` | `features/structural_breaks.py` |
| `feature_importance/importance.py` | `features/importance.py` |
| `feature_importance/clustering.py` | `features/importance.py` (merge) |
| `feature_importance/synthetic.py` | `features/importance.py` (merge) |
| `feature_importance/orthogonal.py` | `features/orthogonal.py` |
| `cross_validation/*` | `modeling/cross_validation.py` |
| `ensemble_methods/*` | `modeling/ensemble.py` |
| `hyperparameter_tuning/*` | `modeling/hyperparameter_tuning.py` |
| `backtest_statistics/*` | `evaluation/statistics.py` |
| `backtest_cv/*` | `evaluation/cpcv.py` |
| `backtesting_dangers/*` | `evaluation/overfitting.py` |
| `backtest_synthetic/*` | `evaluation/synthetic.py` |
| `strategy_risk/*` | `evaluation/strategy_risk.py` |
| `bet_sizing/*` | `evaluation/bet_sizing.py` |
| `ml_asset_allocation/hrp.py` | `allocation/hrp.py` |
| `ml_asset_allocation/denoising.py` | `allocation/denoising.py` |
| `ml_asset_allocation/nco.py` | `allocation/nco.py` |
| `ml_asset_allocation/simulation.py` | `allocation/hrp.py` (merge) |
| `hpc/*` | `_hpc.py` |

---

## Data Contracts

Packages communicate via Polars DataFrames with enforced schemas:

| Column | Type | Description | Produced by | Consumed by |
|--------|------|-------------|-------------|-------------|
| `timestamp` | `Datetime` | Bar/event time | `data/` | All |
| `t1` | `Datetime` | Label expiry | `labeling/` | `labeling/sample_weights`, `modeling/`, `evaluation/` |
| `label` | `Int8` | {-1, 0, 1} or {0, 1} | `labeling/` | `modeling/` |
| `weight` | `Float64` | Combined sample weight | `labeling/` | `modeling/` |
| `signal` | `Float64` | Bet size / position | `evaluation/` | downstream trading |

Pydantic validators at package entry points enforce schemas.

---

## Migration Order

| Session | Package | Source modules | Branch | Dependencies |
|---------|---------|---------------|--------|-------------|
| 1 | `_hpc` | `hpc/` | `phase2/hpc` | None |
| 2 | `data/` | `data_structures/`, `microstructure/` | `phase2/data` | `_hpc` |
| 3 | `labeling/` | `labeling/`, `sample_weights/` | `phase2/labeling` | `data/` |
| 4 | `features/` | `fractional_diff/`, `entropy_features/`, `structural_breaks/`, `feature_importance/` | `phase2/features` | `data/` |
| 5 | `modeling/` | `cross_validation/`, `ensemble_methods/`, `hyperparameter_tuning/` | `phase2/modeling` | `labeling/`, `features/` |
| 6 | `evaluation/` | `backtest_*`, `backtesting_dangers/`, `strategy_risk/`, `bet_sizing/` | `phase2/evaluation` | `modeling/` |
| 7 | `allocation/` | `ml_asset_allocation/` | `phase2/allocation` | `features/` |

### Per-Session Recipe
1. **Restructure** — move files, merge, update imports, verify imports work
2. **Pydantic validation** — input models for public functions at package boundaries
3. **Polars migration** — tabular ops to Polars, keep NumPy for matrix/stats
4. **Tests** — unit + integration, pytest fixtures with deterministic data
5. **Lint & verify** — `uvx ruff check --fix . && uvx ruff format . && uv run pytest -v`

---

## Version Bump

- 0.1.0 → 0.2.0 on completion
- Old import paths break (no shims)
- README.md, CLAUDE.md, WORKFLOW.md all updated
