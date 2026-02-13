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
from tradelab.lopezdp_utils.bet_sizing import ...
from tradelab.lopezdp_utils.backtesting_dangers import ...
from tradelab.lopezdp_utils.backtest_cv import ...
from tradelab.lopezdp_utils.backtest_synthetic import ...
from tradelab.lopezdp_utils.backtest_statistics import ...
from tradelab.lopezdp_utils.strategy_risk import ...
from tradelab.lopezdp_utils.ml_asset_allocation import ...
from tradelab.lopezdp_utils.structural_breaks import ...
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
| `bet_sizing` | Ch 10 | Bet Sizing (Signal Generation, Dynamic Position Sizing) | ✅ v1 Complete |
| `backtesting_dangers` | Ch 11 | Backtesting Pitfalls (CSCV, PBO) | ✅ v1 Complete |
| `backtest_cv` | Ch 12 | Combinatorial Purged Cross-Validation (CPCV) | ✅ v1 Complete |
| `backtest_synthetic` | Ch 13 | Backtesting on Synthetic Data (OTR, O-U Process) | ✅ v1 Complete |
| `backtest_statistics` | Ch 14 | Backtest Statistics (SR, PSR, DSR, HHI, DD) | ✅ v1 Complete |
| `strategy_risk` | Ch 15 | Understanding Strategy Risk (Binomial Model) | ✅ v1 Complete |
| `ml_asset_allocation` | Ch 16 | ML Asset Allocation (HRP, Denoising, NCO) | ✅ v1 Complete |

### Part 4: Useful Financial Features
| Module | Chapter | Topic | Status |
|--------|---------|-------|--------|
| `structural_breaks` | Ch 17 | Structural Breaks (CUSUM, SADF) | ✅ v1 Complete |
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

### `bet_sizing` — Chapter 10: Bet Sizing

Translates ML predictions into actionable position sizes using two complementary approaches.

**Signal-Based Sizing** (AFML 10.1-10.3):
- `get_signal()` — Convert classifier probabilities to bet sizes via z-statistic → Normal CDF
- `avg_active_signals()` — Average signals among concurrently active bets to prevent concentration
- `discrete_signal()` — Round to step increments to prevent jitter/overtrading

**Dynamic Position Sizing** (AFML 10.4):
- `bet_size()` — Width-regulated sigmoid: m(x) = x / sqrt(w + x²)
- `get_target_pos()` — Target position from forecast-market divergence
- `inv_price()` — Inverse sizing to find breakeven price for given size
- `limit_price()` — Average limit price for multi-unit orders
- `get_w()` — Calibrate sigmoid width for desired divergence-to-size mapping

**Key Insight**: Signal-based sizing maps statistical confidence to bet size (strategy-independent). Dynamic sizing adjusts position as market converges to forecast, with limit prices ensuring breakeven or better execution.

**Note**: MLAM bet sizing (single classifier and ensemble methods) is in `labeling.bet_sizing`.

### `backtesting_dangers` — Chapter 11: The Dangers of Backtesting

Tools for detecting backtest overfitting. Chapter 11 is primarily conceptual, introducing the "Seven Sins of Quantitative Investing" and heuristic laws. The main extractable algorithm is CSCV.

**Combinatorially Symmetric Cross-Validation** (AFML 11.5):
- `probability_of_backtest_overfitting()` — Estimate PBO by evaluating all combinatorial train/test splits of strategy trials, computing rank logits, and measuring the fraction with negative logits (below-median OOS performance)

**Key Insight**: If the best in-sample strategy consistently performs below median out-of-sample across combinatorial splits, the backtest is overfit. PBO > 0.5 indicates likely overfitting. For multiple testing corrections (Deflated Sharpe, FWER), see `sample_weights.strategy_redundancy`.

### `backtest_cv` — Chapter 12: Backtesting through Cross-Validation

Combinatorial Purged Cross-Validation (CPCV) overcomes the "single path" limitation of Walk-Forward and standard CV by generating multiple out-of-sample backtest paths.

**Combinatorial Splitter** (AFML 12.3):
- `CombinatorialPurgedKFold` — scikit-learn-compatible CV splitter generating all C(N,k) train/test splits with purging and embargoing
- `get_test_group_map()` — Returns which groups were tested in each split (for path assembly)

**Combinatorial Formulas** (AFML 12.3):
- `get_num_splits()` — Number of unique splits: C(N, k)
- `get_num_backtest_paths()` — Number of complete paths: φ[N,k] = k/N * C(N,k)

**Path Assembly** (AFML 12.3):
- `assemble_backtest_paths()` — Combine OOS forecasts into φ complete backtest paths, each covering all T observations

**Key Insight**: Standard CV and Walk-Forward produce a single backtest path, making it easy to overfit. CPCV generates φ paths (e.g., N=6, k=2 → 5 paths from 15 splits), enabling an empirical Sharpe ratio distribution. A strategy must remain profitable across many "what-if" scenarios, making overfitting significantly harder.

### `backtest_synthetic` — Chapter 13: Backtesting on Synthetic Data

Calibrating optimal trading rules (profit-taking and stop-loss thresholds) using Monte Carlo simulation on synthetic Ornstein-Uhlenbeck price paths.

**O-U Process Estimation** (AFML 13.5.1):
- `ou_fit()` — Estimate O-U parameters (phi, sigma) from price series via OLS on linearized specification
- `ou_half_life()` — Half-life of mean reversion: tau = -log(2)/log(phi)

**Optimal Trading Rules** (AFML 13.1-13.2):
- `otr_batch()` — Monte Carlo engine: simulate O-U paths and compute Sharpe ratios across (profit-taking, stop-loss) mesh
- `otr_main()` — Run OTR experiment across multiple market regimes (forecast levels x half-lives)

**Key Insight**: Instead of backtesting on historical data (single path, overfitting risk), generate thousands of synthetic paths under estimated O-U parameters. The optimal trading rule is the (profit-taking, stop-loss) pair maximizing Sharpe ratio across regimes. This reveals whether a strategy's edge is structural or spurious.

### `backtest_statistics` — Chapter 14: Backtest Statistics

Comprehensive backtest evaluation metrics covering bet characterization, risk measurement, and risk-adjusted performance testing.

**Bet Characterization** (AFML 14.1-14.2):
- `get_bet_timing()` — Identify independent bets from position flattening/flipping (not raw trades)
- `get_holding_period()` — Average holding period via weighted average entry time algorithm

**Risk Metrics** (AFML 14.3-14.4):
- `get_hhi()` — Herfindahl-Hirschman Index for return concentration (positive, negative, temporal)
- `compute_dd_tuw()` — Drawdown series and time-under-water between high-watermarks

**Risk-Adjusted Performance** (AFML 14.5-14.7):
- `sharpe_ratio()` — Annualized Sharpe ratio from excess returns
- `probabilistic_sharpe_ratio()` — PSR adjusting for skewness, kurtosis, and track record length
- `deflated_sharpe_ratio()` — DSR correcting for selection bias under multiple testing

**Strategy Discovery Metrics** (MLAM Section 8):
- `strategy_precision()` — Precision based on universe odds ratio theta
- `strategy_recall()` — Recall (statistical power) of discovery process
- `multi_test_precision_recall()` — Precision/recall under K trials with Sidak correction

**Key Insight**: Standard Sharpe ratio overstates performance when returns are non-Normal or when multiple strategies are tested. PSR adjusts for skewness/kurtosis; DSR further deflates for the number of trials. Strategy precision reveals that even low p-values can yield high false discovery rates when the ratio of true to false strategies (theta) is small.

### `strategy_risk` — Chapter 15: Understanding Strategy Risk

Binomial framework for modeling strategy risk — the probability that a strategy fails to achieve its target Sharpe ratio given its precision, payout structure, and betting frequency.

**Sharpe Ratio Formulas** (AFML 15.4):
- `sharpe_ratio_symmetric()` — Annualized SR for symmetric payouts (+π/-π)
- `sharpe_ratio_asymmetric()` — Annualized SR for asymmetric payouts (different PT/SL)
- `implied_precision_symmetric()` — Minimum precision for symmetric payouts given target SR

**Implied Parameters** (AFML Snippets 15.3-15.4):
- `bin_hr()` — Minimum precision (hit rate) required for target SR given asymmetric payoffs
- `bin_freq()` — Minimum betting frequency required for target SR given precision and payoffs

**Strategy Failure** (AFML Snippet 15.5):
- `mix_gaussians()` — Simulate returns from mixture of two Gaussians (regime model)
- `prob_failure()` — Probability that strategy's precision falls below required threshold

**Key Insight**: Strategy risk is distinct from portfolio risk. A strategy can have low volatility but high failure probability if its precision (p) is close to the required minimum (p*). Strategies with P[p < p*] > 5% should be discarded. Use mixture-of-Gaussians (not simple averages) for realistic payout estimation.

### `ml_asset_allocation` — Chapter 16: Machine Learning Asset Allocation

Portfolio construction methods that bypass the instability of traditional mean-variance optimization (Markowitz's Curse).

**Hierarchical Risk Parity** (AFML 16.1-16.4):
- `correl_dist()` — Correlation-based distance metric: d = sqrt(0.5 * (1 - rho))
- `tree_clustering()` — Hierarchical clustering on distance matrix via scipy linkage
- `get_quasi_diag()` — Reorder covariance matrix so similar assets cluster along diagonal
- `get_rec_bipart()` — Top-down recursive bisection allocating by inverse cluster variance
- `get_ivp()` — Inverse-variance portfolio baseline weights
- `get_cluster_var()` — Cluster variance via IVP weights on sub-covariance
- `hrp_alloc()` — Full HRP pipeline: cluster -> quasi-diag -> recursive bisection

**Covariance Matrix Denoising** (MLAM 2.1-2.9):
- `mp_pdf()` — Marcenko-Pastur PDF for random matrix eigenvalue distribution
- `find_max_eval()` — Fit MP to find noise/signal cutoff eigenvalue (lambda+)
- `denoised_corr()` — Constant Residual Eigenvalue method (replace noise with average)
- `denoised_corr_shrinkage()` — Targeted Shrinkage (blend noise diagonal vs full)
- `denoise_cov()` — High-level wrapper: covariance -> correlation -> PCA -> denoise -> covariance

**Detoning** (MLAM 2.6):
- `detone_corr()` — Remove market component (first eigenvector) to amplify sector signals

**Nested Clustered Optimization** (MLAM 7.3-7.6):
- `opt_port_nco()` — Intracluster + intercluster optimization wrapper (can use any optimizer)
- Uses ONC clustering from `feature_importance.clustering`

**Simulation** (AFML 16.4-16.5):
- `generate_data()` — Synthetic correlated time series with common/specific shocks
- `hrp_mc()` — Monte Carlo comparison of HRP vs IVP out-of-sample performance

**Key Insight**: Traditional optimizers treat every asset as a substitute for every other (complete graph), producing unstable, concentrated portfolios. HRP uses tree clustering to allocate within hierarchical groups, achieving lower OOS variance than CLA (72% less error) and IVP (38% less). Denoising via Marcenko-Pastur further stabilizes the covariance matrix. NCO generalizes by wrapping any optimizer within a cluster-aware framework.

---

### `structural_breaks` — Chapter 17: Structural Breaks

Tests for detecting structural breaks and explosive behavior (bubbles) in financial time series.

**CUSUM Tests** (AFML 17.3):
- `brown_durbin_evans_cusum()` — CUSUM on recursive residuals to detect parameter instability
- `chu_stinchcombe_white_cusum()` — Simplified CUSUM on levels assuming martingale null

**SADF Core** (AFML 17.4, Snippets 17.1-17.4):
- `lag_df()` — Apply lags to a DataFrame for time-series regression
- `get_y_x()` — Prepare numpy arrays for recursive ADF regressions
- `get_betas()` — Fit ADF regression specification via OLS
- `get_bsadf()` — Backward-shifting SADF inner loop (supremum ADF at each endpoint)
- `sadf_test()` — Full SADF test over expanding windows

**Explosiveness Variants** (AFML 17.4):
- `chow_type_dickey_fuller()` — Chow-type DF test for regime switch from random walk to explosive
- `qadf_test()` — Quantile ADF: q-quantile instead of supremum for outlier robustness
- `cadf_test()` — Conditional ADF: conditional expectation of right tail (Expected Shortfall logic)

**Key Insight**: Standard unit-root tests (ADF) have low power against explosive alternatives because they are designed to detect stationarity. The SADF family addresses this by computing right-tail statistics over expanding windows, detecting transitions to explosive regimes (bubbles). QADF and CADF improve robustness over the pure supremum.

> See `TODO.md` for detailed progress tracking.