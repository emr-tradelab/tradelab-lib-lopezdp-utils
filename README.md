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
from tradelab.lopezdp_utils.sample_weights import ...
from tradelab.lopezdp_utils.fractional_diff import ...
from tradelab.lopezdp_utils.ensemble_methods import ...
from tradelab.lopezdp_utils.cross_validation import ...
from tradelab.lopezdp_utils.feature_importance import ...
from tradelab.lopezdp_utils.hyperparameter_tuning import ...
```

## Modules

Each submodule corresponds to a chapter/topic from the books:

### Part 1: Data Analysis
| Module | Chapter | Topic | Status |
|--------|---------|-------|--------|
| `data_structures` | Ch 2 | Financial Data Structures (bars, imbalance bars) | ✅ v1 Complete |
| `labeling` | Ch 3 | Triple-Barrier Method, Meta-Labeling, Trend-Scanning | ✅ v1 Complete |
| `sample_weights` | Ch 4 | Sample Weights, Uniqueness, Sequential Bootstrap | ✅ v1 Complete |
| `fractional_diff` | Ch 5 | Fractionally Differentiated Features | ✅ v1 Complete |

### Part 2: Modelling
| Module | Chapter | Topic | Status |
|--------|---------|-------|--------|
| `ensemble_methods` | Ch 6 | Ensemble Methods (Bagging, Random Forest) | ✅ v1 Complete |
| `cross_validation` | Ch 7 | Purged K-Fold Cross-Validation | ✅ v1 Complete |
| `feature_importance` | Ch 8 | Feature Importance (MDI, MDA, SFI, Clustered) | ✅ v1 Complete |
| `hyperparameter_tuning` | Ch 9 | Hyper-Parameter Tuning with Purged CV | ✅ v1 Complete |

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

### `fractional_diff` — Chapter 5: Fractionally Differentiated Features

Solving the stationarity vs. memory trade-off: standard differentiation (d=1, i.e., log-returns) achieves stationarity but discards most of the predictive signal. Fractional differentiation with d < 1 preserves memory while achieving stationarity.

**Weight Generation** (AFML 5.1):
- `get_weights()` — Binomial series expansion weights for (1-B)^d operator
- `get_weights_ffd()` — Threshold-based weights for FFD method
- `plot_weights()` — Visualize weight decay across d values

**Fractional Differentiation** (AFML 5.2-5.3):
- `frac_diff()` — Expanding window method with weight-loss threshold (has negative drift)
- `frac_diff_ffd()` — Fixed-Width Window method (recommended, driftless)

**Optimal d Selection** (AFML 5.4):
- `plot_min_ffd()` — Find minimum d* for ADF stationarity while maximizing memory

**Key Insight**: For liquid instruments, d* ≈ 0.35 achieves stationarity with correlation > 0.99 to the original series, far superior to log-returns (d=1, correlation ≈ 0).

### `ensemble_methods` — Chapter 6: Ensemble Methods

Ensemble classifiers adapted for non-IID financial data, where overlapping labels and low signal-to-noise ratios make standard implementations suboptimal.

**Bagging Theory** (AFML 6.1):
- `bagging_accuracy()` — Theoretical accuracy of majority-voting ensemble given N learners, accuracy p, and k classes

**Random Forest Configurations** (AFML 6.2):
- `build_random_forest()` — Three RF setups for financial data:
  - Method 0: Standard RF with balanced subsamples
  - Method 1: BaggingClassifier + DecisionTree with `max_samples=avgU`
  - Method 2: BaggingClassifier + single-tree RF with `max_samples=avgU`

**Scalability** (AFML 6.6):
- `bagging_classifier_factory()` — Wrap any base estimator (SVM, etc.) in BaggingClassifier for parallelized training

**Key Insight**: Methods 1 and 2 are recommended for financial data because they set `max_samples` to average uniqueness (from Ch.4), preventing trees from oversampling redundant overlapping observations.

### `cross_validation` — Chapter 7: Cross-Validation in Finance

Time-aware cross-validation that prevents information leakage from overlapping financial labels. Standard k-fold CV assumes IID observations, which is violated when labels span time intervals.

**Purging** (AFML 7.1):
- `get_train_times()` — Remove training observations overlapping with test set labels
- Checks three overlap conditions: start-within, end-within, and complete envelope

**Embargoing** (AFML 7.2):
- `get_embargo_times()` — Define wait period after test sets to guard against serial correlation (ARMA effects)

**PurgedKFold** (AFML 7.3):
- `PurgedKFold` — scikit-learn KFold extension that enforces purging and embargoing
- Shuffle is forced to False (shuffling defeats the purpose with serial data)
- Test sets are always contiguous blocks

**Scoring** (AFML 7.4, MLAM 6.4):
- `cv_score()` — Robust `cross_val_score` replacement fixing scikit-learn bugs with sample weights and `classes_` attribute
- `probability_weighted_accuracy()` — Penalizes high-confidence wrong predictions more than standard accuracy (MLAM)

**Key Insight**: Standard CV leaks information through overlapping labels. Purging removes concurrent training samples; embargoing adds a buffer for serial correlation. Together they prevent the most common source of backtest overfitting.

### `feature_importance` — Chapter 8: Feature Importance

Framework for understanding which variables drive model performance. López de Prado's "first law of backtesting": feature importance is the research tool, not backtesting.

**Core Methods** (AFML 8.2-8.4):
- `feat_imp_mdi()` — Mean Decrease Impurity: in-sample importance for tree ensembles (normalized, CLT-scaled)
- `feat_imp_mda()` — Mean Decrease Accuracy: OOS permutation importance with purged CV
- `feat_imp_sfi()` — Single Feature Importance: evaluates each feature in isolation (immune to substitution)

**Orthogonalization and Validation** (AFML 8.5-8.6):
- `get_ortho_feats()` — PCA-based decorrelation to alleviate linear multicollinearity
- `get_e_vec()` — Eigenvector computation with variance threshold
- `weighted_kendall_tau()` — Consistency check between supervised importance and PCA rankings

**Synthetic Testing Suite** (AFML 8.7-8.10):
- `get_test_data()` — Generate datasets with Informative, Redundant, and Noise features
- `feat_importance()` — Unified wrapper for MDI/MDA/SFI with BaggingClassifier
- `test_func()` — End-to-end pipeline: generate → analyze → plot
- `plot_feat_importance()` — Horizontal bar chart with error bars

**Clustered Feature Importance** (MLAM 4.1-4.2, 6.4-6.5):
- `cluster_kmeans_base()` — ONC base clustering (silhouette t-stat optimization)
- `cluster_kmeans_top()` — Recursive refinement of below-average clusters
- `feat_imp_mdi_clustered()` — Clustered MDI: sums importance within ONC clusters
- `feat_imp_mda_clustered()` — Clustered MDA: shuffles entire feature clusters

**Key Insight**: MDI is biased and in-sample; MDA and SFI are OOS but susceptible to substitution effects. Clustered Feature Importance (MLAM) solves this by grouping correlated features via ONC and measuring importance at the cluster level.

### `hyperparameter_tuning` — Chapter 9: Hyper-Parameter Tuning with Cross-Validation

Financial-aware hyperparameter optimization using purged k-fold CV to prevent leakage during model selection.

**Search** (AFML 9.1, 9.3):
- `clf_hyper_fit()` — Grid or randomized search with PurgedKFold CV and optional bagging
- Auto-selects F1 scoring for meta-labeling ({0,1}) and neg-log-loss otherwise

**Pipeline Fix** (AFML 9.2):
- `MyPipeline` — Pipeline subclass routing `sample_weight` correctly to the final estimator

**Distributions** (AFML 9.4):
- `log_uniform()` — Log-uniform distribution for parameters spanning orders of magnitude (SVM C, gamma, etc.)

**Key Insight**: Standard `GridSearchCV` with k-fold leaks information through overlapping labels. Using `PurgedKFold` as the inner CV eliminates this bias. Log-uniform distributions make randomized search far more efficient for non-linear parameters.

> See `TODO.md` for detailed progress tracking.