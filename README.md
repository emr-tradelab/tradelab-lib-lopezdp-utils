# tradelab-lib-lopezdp-utils

A Python library of financial machine learning utilities extracted from:
- **"Advances in Financial Machine Learning"** by Marcos López de Prado (primary)
- **"Machine Learning for Asset Managers"** by Marcos López de Prado (complementary)

Part of the Tradelab algorithmic trading ecosystem.

---

## 🚀 Starting a Work Session

**Always begin with:**
```
/resume-extraction
```

This reads `TODO.md` and shows you:
- ✅ What's complete
- ⏭️ Next task to work on
- 📚 Reminder to query `notebooklm-researcher` agent for theory

---

## Usage

```python
from tradelab.lopezdp_utils.data_structures import ...
from tradelab.lopezdp_utils.labeling import ...
```

## Modules

Each submodule corresponds to a chapter/topic from the books:

### Part 1: Data Analysis
| Module | Chapter | Topic | Status |
|--------|---------|-------|--------|
| `data_structures` | Ch 2 | Financial Data Structures (bars, imbalance bars) | ✅ v1 Complete |
| `labeling` | Ch 3 | Triple-Barrier Method, Meta-Labeling, Trend-Scanning | ✅ v1 Complete |
| `sample_weights` | Ch 4 | Sample Weights, Uniqueness, Sequential Bootstrap | ✅ v1 Complete |
| `fractional_diff` | Ch 5 | Fractionally Differentiated Features | Planned |

### Part 2: Modelling
| Module | Chapter | Topic | Status |
|--------|---------|-------|--------|
| `ensemble_methods` | Ch 6 | Ensemble Methods | Planned |
| `cross_validation` | Ch 7 | Purged K-Fold Cross-Validation | Planned |
| `feature_importance` | Ch 8 | Feature Importance (MDA, MDI, SFI) | Planned |
| `hyperparameter_tuning` | Ch 9 | Hyper-Parameter Tuning | Planned |

### Part 3: Backtesting
| Module | Chapter | Topic | Status |
|--------|---------|-------|--------|
| `bet_sizing` | Ch 10 | Bet Sizing | Planned |
| `backtest_dangers` | Ch 11 | Backtesting Pitfalls | Planned |
| `backtest_cv` | Ch 12 | Backtesting through Cross-Validation | Planned |
| `backtest_synthetic` | Ch 13 | Backtesting on Synthetic Data | Planned |
| `backtest_statistics` | Ch 14 | Backtest Statistics | Planned |
| `strategy_risk` | Ch 15 | Understanding Strategy Risk | Planned |
| `ml_asset_allocation` | Ch 16 | Machine Learning Asset Allocation | Planned |

### Part 4: Useful Financial Features
| Module | Chapter | Topic | Status |
|--------|---------|-------|--------|
| `structural_breaks` | Ch 17 | Structural Breaks (CUSUM, etc.) | Planned |
| `entropy` | Ch 18 | Entropy Features | Planned |
| `microstructure` | Ch 19 | Market Microstructure Features | Planned |

### Part 5: High-Performance Computing
| Module | Chapter | Topic | Status |
|--------|---------|-------|--------|
| `hpc` | Ch 20 | Multiprocessing and Vectorization | Planned |

---

## Module Details

### `labeling` — Chapter 3: Labeling

Path-dependent labeling methods that account for volatility and price paths during holding periods.

**Triple-Barrier Method** (AFML 3.4-3.5):
- `triple_barrier_labels()` — Complete workflow wrapper
- `daily_volatility()` — Dynamic thresholds based on volatility
- `add_vertical_barrier()` — Time-based expiration barriers
- `get_events()` — Find first barrier touch (profit/stop/time)
- `get_bins()` — Generate labels from barrier touches

**Meta-Labeling** (AFML 3.6):
- `get_events_meta()` — Asymmetric barriers for known position sides
- `get_bins_meta()` — Binary labels for bet sizing (act/pass)
- Separates side (primary model) from size (meta-model)

**Alternative Methods**:
- `fixed_time_horizon()` — Standard approach (has known flaws)
- `trend_scanning_labels()` — Statistical significance-based (MLAM 5.4)

**Utilities**:
- `drop_labels()` — Handle class imbalance
- `bet_size_from_probability()` — Single classifier bet sizing (MLAM 5.5)
- `bet_size_from_ensemble()` — Multi-classifier bet sizing (MLAM 5.5)

### `sample_weights` — Chapter 4: Sample Weights

Correcting for non-IID violations in financial data where labels overlap in time, causing informational redundancy.

**Core Problem**: Financial labels often depend on the same price returns (concurrent labels), violating the IID assumption required by standard ML algorithms.

**Concurrency and Uniqueness** (AFML 4.1-4.2):
- `mp_num_co_events()` — Count overlapping labels at each time point
- `mp_sample_tw()` — Compute average uniqueness scores for each label
- Labels sharing returns get lower uniqueness weights

**Sequential Bootstrap** (AFML 4.3-4.5):
- `get_ind_matrix()` — Build indicator matrix of label lifespans
- `get_avg_uniqueness()` — Calculate uniqueness from indicator matrix
- `seq_bootstrap()` — Resample with probability proportional to uniqueness
- Creates training sets closer to IID by avoiding redundant samples

**Return Attribution** (AFML 4.10-4.11):
- `mp_sample_w()` — Weight by attributed absolute log-returns
- `get_time_decay()` — Apply piecewise-linear decay based on cumulative uniqueness
- Emphasizes both magnitude (large moves) and recency (adaptive markets)

**Class Imbalance**:
- `get_class_weights()` — Correct for underrepresented classes

**Strategy-Level Redundancy** (MLAM Ch 8):
- `false_strategy_theorem()` — Expected max Sharpe under multiple testing
- `familywise_error_rate()` — FWER (α_K) with Šidàk's correction
- `type_ii_error_prob()` — Power analysis for detecting true strategies
- `min_variance_cluster_weights()` — Aggregate correlated backtests
- `estimate_independent_trials()` — Placeholder for ONC clustering

**Key Insight**: AFML ensures individual observations are weighted properly; MLAM extends this to strategy selection under multiple testing.

> See `TODO.md` for detailed progress tracking.