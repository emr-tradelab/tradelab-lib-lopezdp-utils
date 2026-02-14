# TODO.md — tradelab-lib-lopezdp-utils

## Overview

**Phase:** Pre-Production (v1 extraction)
**Status:** Starting

---

## Global Tasks

### Part 1: Data Analysis
- [x] Extract Chapter 2: Financial Data Structures
- [x] Extract Chapter 3: Labeling
- [x] Extract Chapter 4: Sample Weights
- [x] Extract Chapter 5: Fractionally Differentiated Features

### Part 2: Modelling
- [x] Extract Chapter 6: Ensemble Methods
- [x] Extract Chapter 7: Cross-Validation in Finance
- [x] Extract Chapter 8: Feature Importance
- [x] Extract Chapter 9: Hyper-Parameter Tuning with Cross-Validation

### Part 3: Backtesting
- [x] Extract Chapter 10: Bet Sizing
- [x] Extract Chapter 11: The Dangers of Backtesting
- [x] Extract Chapter 12: Backtesting through Cross-Validation
- [x] Extract Chapter 13: Backtesting on Synthetic Data
- [x] Extract Chapter 14: Backtest Statistics
- [x] Extract Chapter 15: Understanding Strategy Risk
- [x] Extract Chapter 16: Machine Learning Asset Allocation

### Part 4: Useful Financial Features
- [x] Extract Chapter 17: Structural Breaks
- [x] Extract Chapter 18: Entropy Features
- [x] Extract Chapter 19: Microstructural Features

### Part 5: High-Performance Computing Recipes
- [x] Extract Chapter 20: Multiprocessing and Vectorization

### Final Steps
- [ ] Review ML for Asset Managers for complementary content (done per-chapter)
- [-] [**STOP HERE** **DO NOT IMPLEMENT**] **Phase 2:** Production optimization pass (pandas → Polars, tests, performance)

> **Note:** Chapter 1 is introductory (no extractable code). Chapters 21-22 cover quantum computing and specialized HPC topics — will assess for extraction relevance during implementation.

---

## Chapter Details

*Sections below are populated at the start of each chapter's work session.*
*Each functionality gets its own checkbox and is marked done when implemented.*

---

### Chapter 2: Financial Data Structures
**Branch:** `feat/chapter1-data-structures`
**Submodule:** `tradelab.lopezdp_utils.data_structures`
**Status:** ✅ v1 Complete

**Functionalities (with Python code in book):**
- [x] pca_weights — Derive allocation weights conforming to specific risk distribution across covariance matrix principal components (Snippet 2.1)
- [x] single_future_roll — Adjust futures price series for roll gaps by detracting cumulative gaps (Snippets 2.2 & 2.3)
- [x] cusum_filter — Event-based sampling using CUSUM quality-control method to detect mean value shifts (Snippet 2.4)

**Functionalities (implement from formulas/logic):**
- [x] time_bars — Sample bars at fixed time intervals
- [x] tick_bars — Sample bars every N transactions
- [x] volume_bars — Sample bars every N units exchanged
- [x] dollar_bars — Sample bars every N market value exchanged
- [x] tick_imbalance_bars — Information-driven bars based on tick imbalances (TIBs)
- [x] volume_imbalance_bars — Information-driven bars based on volume imbalances (VIBs)
- [x] dollar_imbalance_bars — Information-driven bars based on dollar imbalances (DIBs)
- [x] tick_runs_bars — Information-driven bars based on tick runs (TRBs)
- [x] volume_runs_bars — Information-driven bars based on volume runs (VRBs)
- [x] dollar_runs_bars — Information-driven bars based on dollar runs (DRBs)
- [x] etf_trick — Model complex baskets of securities as single non-expiring, total-return cash products
- [x] sampling_linspace — Downsampling with constant step
- [x] sampling_uniform — Downsampling with random uniform selection

**ML for Asset Managers additions (complementary):**
- [x] discretization_optimal_binning — Optimal binning formulas to quantize continuous price series (MLAM Section 3.9)

**Note:** MLAM references to standard bars, ETF trick are redundant. Denoising/detoning and information-theoretic distance metrics are covered in their respective AFML chapters.

---

### Chapter 3: Labeling
**Branch:** `feat/chapter3-labeling`
**Submodule:** `tradelab.lopezdp_utils.labeling`
**Status:** ✅ v1 Complete

**Functionalities (with Python code in AFML):**
- [x] daily_volatility — Compute dynamic thresholds for barriers based on daily volatility estimates (Snippet 3.1)
- [x] get_events — Determine timestamp of first barrier hit (triple-barrier method core utility, with multiprocessing support) (Snippet 3.3)
- [x] add_vertical_barrier — Define time expiration limit by finding next price bar after specified days (Snippet 3.4)
- [x] get_bins — Generate final labels (-1, 0, 1) based on return at first barrier touch (Snippet 3.5)
- [x] get_events_meta — Extended version of get_events for meta-labeling (Snippet 3.6)
- [x] get_bins_meta — Extended version of get_bins for meta-labeling (Snippet 3.7)
- [x] drop_labels — Recursively eliminate observations with rare labels to address class imbalance (Snippet 3.8)

**Functionalities (implement from formulas/logic in AFML):**
- [x] fixed_time_horizon — Standard labeling based on price returns over constant window relative to threshold (mathematical definition only)
- [x] triple_barrier_labels — High-level wrapper combining daily_volatility, get_events, and get_bins for complete triple-barrier labeling

**ML for Asset Managers additions (complementary):**
- [x] trend_scanning_labels — Identify most statistically significant trend by maximizing absolute t-value of linear time-trend (MLAM Section 5.4, Snippets 5.1 & 5.2)
- [x] bet_size_from_meta_labels — Compute bet size based on probability of profit from meta-labels (MLAM Section 5.5)
- [x] ensemble_bet_sizing — Average predictions across multiple meta-labeling classifiers using de Moivre-Laplace theorem (MLAM Section 5.5)

**Note:** MLAM Sections 5.1-5.3 provide conceptual summaries redundant with AFML. Section 5.4 (Trend-Scanning) and Section 5.5 (Bet Sizing formulas) are complementary and should be extracted.

---

### Chapter 4: Sample Weights
**Branch:** `feat/chapter4-sample-weights`
**Submodule:** `tradelab.lopezdp_utils.sample_weights`
**Status:** ✅ v1 Complete

**Functionalities (with Python code in AFML):**
- [x] mp_num_co_events — Count concurrent labels at each time point to measure informational redundancy (Snippet 4.1)
- [x] mp_sample_tw — Compute average uniqueness of each label over its lifespan (Snippet 4.2)
- [x] get_ind_matrix — Build indicator matrix showing which labels are active at each bar (Snippet 4.3)
- [x] get_avg_uniqueness — Calculate average uniqueness from indicator matrix (Snippet 4.4)
- [x] seq_bootstrap — Sequential bootstrap resampling that maintains sample diversity (Snippet 4.5)
- [x] mp_sample_w — Weight samples by attributed absolute log-returns adjusted for concurrency (Snippet 4.10)
- [x] get_time_decay — Apply piecewise-linear decay based on cumulative uniqueness (Snippet 4.11)

**Functionalities (implement from formulas/logic in AFML):**
- [x] get_class_weights — Helper to compute class weights for imbalanced datasets (wrapper around scikit-learn)

**ML for Asset Managers additions (complementary):**
- [x] estimate_independent_trials — Estimate number of effectively independent backtested strategies using ONC clustering (MLAM Section 8.7.1)
- [x] min_variance_cluster_weights — Aggregate redundant strategy trials using minimum variance allocation (MLAM Section 8)
- [x] false_strategy_theorem — Compute expected max Sharpe ratio under multiple testing to deflate results (MLAM Snippet 8.1)
- [x] familywise_error_rate — Calculate FWER (α_K) under multiple hypothesis testing (MLAM Snippet 8.3)
- [x] type_ii_error_prob — Calculate probability of missing true strategies (β_K) (MLAM Snippet 8.4)

**Note:** AFML provides individual observation weighting; MLAM is complementary by addressing strategy-level redundancy and multiple testing corrections.

---

### Chapter 5: Fractionally Differentiated Features
**Branch:** `feat/chapter5-fractional-diff`
**Submodule:** `tradelab.lopezdp_utils.fractional_diff`
**Status:** ✅ v1 Complete

**Functionalities (with Python code in AFML):**
- [x] get_weights — Generate weights (ω) for the fractional difference operator (1−B)^d using binomial series expansion (Snippet 5.1)
- [x] frac_diff — Standard fractional differentiation using expanding window with weight loss tolerance threshold τ (Snippet 5.2)
- [x] frac_diff_ffd — Fixed-Width Window Fracdiff (FFD) using constant window width by dropping weights below threshold τ — avoids negative drift (Snippet 5.3)
- [x] plot_min_ffd — Identify minimum d* required for ADF stationarity while maximizing correlation with original series (Snippet 5.4)

**Functionalities (implement from formulas/logic in AFML):**
- [x] get_weights_ffd — Generate weights for FFD method, iterating until weight magnitude falls below threshold (helper for frac_diff_ffd)
- [x] plot_weights — Visualize how weights decay for different values of d (from Snippet 5.1 context)

**ML for Asset Managers additions (complementary):**
- None — MLAM references fractional differentiation as a prerequisite (already solved in AFML Ch.5) without adding new algorithms. Trend-scanning labels already extracted in Chapter 3.

---

### Chapter 6: Ensemble Methods
**Branch:** `feat/chapter6-ensemble-methods`
**Submodule:** `tradelab.lopezdp_utils.ensemble_methods`
**Status:** ✅ v1 Complete

**Functionalities (with Python code in AFML):**
- [x] bagging_accuracy — Calculate probability that a bagging ensemble correctly classifies via majority voting (Snippet 6.1)
- [x] build_random_forest — Three configurations of Random Forests adapted for financial data with average uniqueness sampling (Snippet 6.2)

**Functionalities (implement from formulas/logic in AFML):**
- [x] bagging_classifier_factory — Factory for creating BaggingClassifier wrappers with financial-data-aware defaults (early stopping, scalability)

**ML for Asset Managers additions (complementary):**
- None — MLAM applies ensembles to bet sizing (already in Ch.3), feature importance (Ch.8), and portfolio construction (Ch.16). No new standalone ensemble algorithms to extract here.

---

### Chapter 7: Cross-Validation in Finance
**Branch:** `feat/chapter7-cross-validation`
**Submodule:** `tradelab.lopezdp_utils.cross_validation`
**Status:** ✅ v1 Complete

**Functionalities (with Python code in AFML):**
- [x] get_train_times — Remove training observations that overlap in time with test observations to prevent information leakage (Snippet 7.1)
- [x] get_embargo_times — Define embargo period after test sets to eliminate serial-correlation leakage (Snippet 7.2)
- [x] PurgedKFold — Scikit-learn KFold extension enforcing purging and embargoing for financial labels that span intervals (Snippet 7.3)
- [x] cv_score — Robust cross_val_score replacement that fixes scikit-learn bugs with sample weights and classes_ attribute (Snippet 7.4)

**Functionalities (implement from formulas/logic in AFML):**
- (None — all Chapter 7 utilities have code snippets)

**ML for Asset Managers additions (complementary):**
- [x] probability_weighted_accuracy — Alternative scoring metric that penalizes high-confidence wrong predictions more than standard accuracy (MLAM Section 6.4)

**Note:** MLAM multiple-testing corrections (getZStat, type1Err, type2Err, clusterKMeansBase) were already extracted in Chapter 4 under sample_weights.

---

### Chapter 8: Feature Importance
**Branch:** `feat/chapter8-feature-importance`
**Submodule:** `tradelab.lopezdp_utils.feature_importance`
**Status:** ✅ v1 Complete

**Functionalities (with Python code in AFML):**
- [x] feat_imp_mdi — Mean Decrease Impurity: in-sample feature importance for tree-based classifiers using weighted impurity decrease (Snippet 8.2)
- [x] feat_imp_mda — Mean Decrease Accuracy: out-of-sample permutation importance using purged k-fold CV (Snippet 8.3)
- [x] feat_imp_sfi — Single Feature Importance: OOS importance of each feature in isolation (Snippet 8.4)
- [x] get_ortho_feats — Orthogonal Features: PCA-based decorrelation to alleviate multicollinearity (Snippet 8.5)
- [x] weighted_kendall_tau — Consistency check between supervised importance rankings and PCA eigenvalue rankings (Snippet 8.6)
- [x] get_test_data — Synthetic dataset generator with informative, redundant, and noise features (Snippet 8.7)
- [x] feat_importance — Wrapper function to call MDI, MDA, or SFI on a dataset using bagged decision trees (Snippet 8.8)
- [x] test_func — Master execution component to automate testing pipeline (Snippet 8.9)
- [x] plot_feat_importance — Visualization utility for mean importance bars with std deviations (Snippet 8.10)

**Functionalities (implement from formulas/logic in AFML):**
- (None — all Chapter 8 utilities have code snippets)

**ML for Asset Managers additions (complementary):**
- [x] cluster_kmeans_base — Base ONC clustering that finds optimal K by maximizing silhouette score t-statistic (MLAM Snippet 4.1)
- [x] cluster_kmeans_top — Recursive top-level ONC refinement of below-average clusters (MLAM Snippet 4.2)
- [x] feat_imp_mdi_clustered — Clustered MDI: sums feature importance within ONC clusters (MLAM Snippet 6.4)
- [x] feat_imp_mda_clustered — Clustered MDA: shuffles entire feature clusters simultaneously (MLAM Snippet 6.5)

**Note:** PWA scoring was already extracted in Chapter 7 (cross_validation/scoring.py). ONC clustering is also needed by sample_weights/strategy_redundancy.py (currently a placeholder).

---

### Chapter 9: Hyper-Parameter Tuning with Cross-Validation
**Branch:** `feat/chapter9-hyperparameter-tuning`
**Submodule:** `tradelab.lopezdp_utils.hyperparameter_tuning`
**Status:** ✅ v1 Complete

**Functionalities (with Python code in AFML):**
- [x] clf_hyper_fit — Grid/randomized search for optimal hyperparameters using PurgedKFold CV with optional bagging (Snippets 9.1 & 9.3)
- [x] MyPipeline — Enhanced sklearn Pipeline that correctly passes sample_weight to final estimator (Snippet 9.2)
- [x] log_uniform — Log-uniform distribution for efficient non-linear parameter search (Snippet 9.4)

**Functionalities (implement from formulas/logic in AFML):**
- (None — all Chapter 9 utilities have code snippets)

**ML for Asset Managers additions (complementary):**
- None — MLAM multiple-testing corrections (false_strategy_theorem, familywise_error_rate, type_ii_error_prob) were already extracted in Chapter 4 under sample_weights/strategy_redundancy.py.

**Note:** Chapter 9 depends on PurgedKFold from Chapter 7 (cross_validation/purging.py). Dependencies are imported, not duplicated.

---

### Chapter 10: Bet Sizing
**Branch:** `feat/chapter10-bet-sizing`
**Submodule:** `tradelab.lopezdp_utils.bet_sizing`
**Status:** ✅ v1 Complete

**Functionalities (with Python code in AFML):**
- [x] get_signal — Translate predicted probabilities into bet sizes using z-statistic and Normal CDF, with meta-labeling support (Snippet 10.1)
- [x] avg_active_signals — Compute average signal among concurrently active bets (Snippet 10.2, single-threaded v1)
- [x] discrete_signal — Round bet sizes to nearest increment of step_size to prevent jitter and overtrading (Snippet 10.3)
- [x] bet_size — Width-regulated sigmoid function for sizing based on price divergence (Snippet 10.4)
- [x] get_target_pos — Calculate target position based on forecast vs. market price divergence (Snippet 10.4)
- [x] inv_price — Inverse sizing function to find price for given bet size (Snippet 10.4)
- [x] limit_price — Calculate average limit price for multi-unit orders (Snippet 10.4)
- [x] get_w — Calibrate sigmoid width coefficient for desired divergence-to-size mapping (Snippet 10.4)

**ML for Asset Managers additions (complementary):**
- None — MLAM bet sizing (bet_size_from_probability, bet_size_from_ensemble) was already extracted in Chapter 3 (labeling/bet_sizing.py).

**Note:** Chapter 10 depends on multiprocessing utilities (mpPandasObj from Ch. 20, not yet extracted). avg_active_signals will use a simplified single-threaded implementation for v1.

---

### Chapter 11: The Dangers of Backtesting
**Branch:** `feat/chapter11-backtesting-dangers`
**Submodule:** `tradelab.lopezdp_utils.backtesting_dangers`
**Status:** ✅ v1 Complete

**Functionalities (with Python code in AFML):**
- None — Chapter 11 contains only heuristic "laws" (Snippets 8.1 and 11.1), not executable code.

**Functionalities (implement from formulas/logic in AFML):**
- [x] probability_of_backtest_overfitting — Estimate PBO using Combinatorially Symmetric Cross-Validation (CSCV): partition trials into S groups, evaluate all combinatorial train/test splits, compute rank logits, and measure fraction with negative logits (Section 11.5)

**ML for Asset Managers additions (complementary):**
- None — MLAM Section 8 snippets (false_strategy_theorem, familywise_error_rate, type_ii_error_prob) were already extracted in Chapter 4 (sample_weights/strategy_redundancy.py).

**Note:** Chapter 11 is primarily conceptual ("Seven Sins of Quantitative Investing", Marcos' Laws of Backtesting). The CSCV algorithm is the only extractable utility — described theoretically as a seven-step procedure without code. Implementation follows Bailey et al. (2017).

---

### Chapter 12: Backtesting through Cross-Validation
**Branch:** `feat/chapter12-backtesting-cv`
**Submodule:** `tradelab.lopezdp_utils.backtest_cv`
**Status:** ✅ v1 Complete

**Functionalities (with Python code in AFML):**
- None — Chapter 12 contains no labeled code snippets (no "Snippet 12.x").

**Functionalities (implement from formulas/logic in AFML):**
- [x] CombinatorialPurgedKFold — scikit-learn-compatible CV splitter that generates all combinatorial train/test splits with purging and embargoing, extending PurgedKFold from Ch.7 (Section 12.3)
- [x] get_num_splits — Compute number of train/test splits: C(N, k) = N! / (k! * (N-k)!) (Section 12.3)
- [x] get_num_backtest_paths — Compute number of complete backtest paths: φ[N,k] = k/N * C(N,k) (Section 12.3)
- [x] assemble_backtest_paths — Combine OOS forecasts from CPCV splits into φ complete backtest paths, each covering all T observations (Section 12.3)

**ML for Asset Managers additions (complementary):**
- None — MLAM references CPCV as a defense against overfitting but adds no new algorithms beyond what's already extracted (false_strategy_theorem etc. in Ch.4).

**Note:** Chapter 12 is the theoretical framework chapter. No code snippets exist — implementation follows the 5-step algorithm described in Section 12.3. Depends on PurgedKFold from Chapter 7.

---

### Chapter 13: Backtesting on Synthetic Data
**Branch:** `feat/chapter13-synthetic-backtesting`
**Submodule:** `tradelab.lopezdp_utils.backtest_synthetic`
**Status:** ✅ v1 Complete

**Functionalities (with Python code in AFML):**
- [x] otr_batch — Monte Carlo simulation engine: generates synthetic O-U price paths and evaluates Sharpe ratios across a mesh of profit-taking and stop-loss thresholds (Snippet 13.2)
- [x] otr_main — Entry point for OTR experiment: generates Cartesian product of market regime parameters and dispatches to batch simulator (Snippet 13.1)

**Functionalities (implement from formulas/logic in AFML):**
- [x] ou_half_life — Compute half-life of convergence from O-U speed parameter: τ = -log(2)/log(ϕ) (Section 13.5.1)
- [x] ou_fit — Estimate O-U process parameters (ϕ, σ) from price series via OLS regression on linearized specification (Section 13.5.1)

**ML for Asset Managers additions (complementary):**
- None — MLAM Snippets 8.1/8.2 (false_strategy_theorem, getDistMaxSR) were already extracted in Chapter 4 (sample_weights/strategy_redundancy.py). MLAM Appendix A provides conceptual taxonomy of non-parametric synthetic data methods (VAE, GAN, etc.) without code — not extractable.

---

### Chapter 14: Backtest Statistics
**Branch:** `feat/chapter14-backtest-statistics`
**Submodule:** `tradelab.lopezdp_utils.backtest_statistics`
**Status:** ✅ v1 Complete

**Functionalities (with Python code in AFML):**
- [x] get_bet_timing — Derive timestamps of independent bets from position flattening/flipping (Snippet 14.1)
- [x] get_holding_period — Estimate average holding period using weighted average entry time algorithm (Snippet 14.2)
- [x] get_hhi — Compute Herfindahl-Hirschman Index for return concentration (positive, negative, temporal) (Snippet 14.3)
- [x] compute_dd_tuw — Compute drawdown series and time-under-water between high-watermarks (Snippet 14.4)

**Functionalities (implement from formulas/logic in AFML):**
- [x] sharpe_ratio — Compute annualized Sharpe ratio from excess returns
- [x] probabilistic_sharpe_ratio — PSR adjusting for non-normality (skewness/kurtosis) and track record length (Section 14.6)
- [x] deflated_sharpe_ratio — DSR correcting for selection bias under multiple testing (Section 14.7)

**ML for Asset Managers additions (complementary):**
- [x] strategy_precision — Compute precision of discovery process based on universe odds ratio θ (MLAM Section 8)
- [x] strategy_recall — Compute recall (1−β) of discovery process (MLAM Section 8)
- [x] multi_test_precision_recall — Extend precision/recall to K independent trials with Šidàk correction (MLAM Section 8)

**Note:** MLAM Snippets 8.1-8.3 (getExpectedMaxSR, getZStat, type1Err, type2Err) were already extracted in Chapter 4 (sample_weights/strategy_redundancy.py). The odds-ratio-based precision/recall framework is complementary and new.

---

### Chapter 15: Understanding Strategy Risk
**Branch:** `feat/chapter15-strategy-risk`
**Submodule:** `tradelab.lopezdp_utils.strategy_risk`
**Status:** ✅ v1 Complete

**Functionalities (with Python code in AFML):**
- [x] bin_hr — Compute implied precision (minimum p) required to achieve a target Sharpe ratio given asymmetric payoffs (Snippet 15.3)
- [x] bin_freq — Compute implied betting frequency needed to achieve a target Sharpe ratio (Snippet 15.4)
- [x] mix_gaussians — Generate random draws from a mixture of two Gaussians for return simulation (Snippet 15.5)
- [x] prob_failure — Estimate probability that a strategy's precision falls below the required threshold (Snippet 15.5)

**Functionalities (implement from formulas/logic in AFML):**
- [x] sharpe_ratio_symmetric — Annualized Sharpe ratio for symmetric payouts: θ = √n * (2p−1) / √(4p(1−p)) (Section 15.4.1)
- [x] implied_precision_symmetric — Implied precision for symmetric payouts (Section 15.4.1)
- [x] sharpe_ratio_asymmetric — Annualized Sharpe ratio for asymmetric payouts (Section 15.4.2)

**ML for Asset Managers additions (complementary):**
- None — MLAM Section 8 content (False Strategy Theorem, ONC, FWER, precision/recall) was already extracted in Chapter 4 (sample_weights/strategy_redundancy.py) and Chapter 14 (backtest_statistics/strategy_metrics.py).

**Note:** Chapter 15 models strategy risk as the probability of failing to meet a Sharpe ratio hurdle, using binomial bet framework. Distinct from portfolio risk.

---

### Chapter 16: Machine Learning Asset Allocation
**Branch:** `feat/chapter16-asset-allocation`
**Submodule:** `tradelab.lopezdp_utils.ml_asset_allocation`
**Status:** ✅ v1 Complete

**Functionalities (with Python code in AFML):**
- [x] correl_dist — Correlation-based distance metric: d = sqrt(0.5 * (1 - ρ)) (Snippet 16.1/16.4)
- [x] tree_clustering — Hierarchical tree clustering using scipy linkage on distance matrix (Snippet 16.1)
- [x] get_quasi_diag — Quasi-diagonalization: reorder covariance matrix so similar items cluster along diagonal (Snippet 16.2)
- [x] get_rec_bipart — Recursive bisection: top-down weight allocation splitting by inverse cluster variance (Snippet 16.3)
- [x] get_ivp — Inverse-variance portfolio: w_n = V_{n,n}^{-1} / Σ V_{i,i}^{-1} (Snippet 16.4)
- [x] get_cluster_var — Compute variance for a cluster using IVP weights on sub-covariance (Snippet 16.4)
- [x] hrp_alloc — Full HRP allocation: tree clustering → quasi-diag → recursive bisection (Snippet 16.4)
- [x] generate_data — Synthetic correlated time series with common/specific shocks for testing (Snippet 16.4)
- [x] hrp_mc — Monte Carlo experiment comparing HRP vs CLA vs IVP out-of-sample (Snippet 16.5)

**ML for Asset Managers additions (complementary):**
- [x] mp_pdf — Marcenko-Pastur probability density function for random matrix eigenvalue distribution (MLAM Snippet 2.1)
- [x] find_max_eval — Fit Marcenko-Pastur PDF to find noise/signal eigenvalue cutoff λ+ (MLAM Snippet 2.4)
- [x] denoised_corr — Constant Residual Eigenvalue denoising method (MLAM Snippet 2.5)
- [x] denoised_corr_shrinkage — Targeted Shrinkage denoising applied only to noise eigenvectors (MLAM Snippet 2.6)
- [x] denoise_cov — High-level wrapper: covariance → correlation → PCA → denoise → covariance (MLAM Snippet 2.9)
- [x] detone_corr — Remove market component (first eigenvector) to amplify sector signals (MLAM Section 2.6)
- [x] opt_port_nco — Nested Clustered Optimization: intracluster + intercluster optimization wrapper (MLAM Snippets 7.3-7.6)

**Note:** ONC clustering (clusterKMeansBase/Top) already extracted in feature_importance/clustering.py — will import, not duplicate. MLAM denoising/detoning and NCO are major complementary additions addressing covariance matrix instability from a Random Matrix Theory perspective.

---

### Chapter 17: Structural Breaks
**Branch:** `feat/chapter17-structural-breaks`
**Submodule:** `tradelab.lopezdp_utils.structural_breaks`
**Status:** ✅ v1 Complete

**Functionalities (with Python code in AFML):**
- [x] get_bsadf — Backward-shifting SADF inner loop: fits ADF regression at each endpoint with expanding start points to detect bubbles (Snippet 17.1)
- [x] get_y_x — Prepare numpy arrays (y, x) with lagged levels and differences for recursive ADF tests (Snippet 17.2)
- [x] lag_df — Apply lags to a DataFrame for time-series regression (Snippet 17.3)
- [x] get_betas — Fit ADF regression specification via OLS, returning t-statistic of unit-root coefficient (Snippet 17.4)

**Functionalities (implement from formulas/logic in AFML):**
- [x] sadf_test — Full SADF test wrapper: run get_bsadf over expanding windows, return supremum ADF statistic (Section 17.4.2)
- [x] brown_durbin_evans_cusum — Brown-Durbin-Evans CUSUM test on recursive residuals for structural break detection (Section 17.3.1)
- [x] chu_stinchcombe_white_cusum — Simplified CUSUM on levels assuming martingale null (Section 17.3.2)
- [x] chow_type_dickey_fuller — Chow-type DF test for switch from random walk to explosive process (Section 17.4.1)
- [x] qadf_test — Quantile ADF: use q-quantile instead of supremum for robustness (Section 17.4.3)
- [x] cadf_test — Conditional ADF: conditional moments of high ADF values for outlier robustness (Section 17.4.4)

**ML for Asset Managers additions (complementary):**
- None new — Trend-Scanning already extracted in Chapter 3 (labeling/trend_scanning.py). MLAM references structural breaks as features for meta-labeling (conceptual, no new code).

---

### Chapter 18: Entropy Features
**Branch:** `feat/chapter18-entropy-features`
**Submodule:** `tradelab.lopezdp_utils.entropy_features`
**Status:** ✅ v1 Complete

**Functionalities (with Python code in AFML):**
- [x] pmf1 — Compute probability mass function for discrete random variable by counting word frequencies (Snippet 18.1)
- [x] plug_in — Plug-in (Maximum Likelihood) entropy rate estimator: H = -1/w * sum(p(w) * log2(p(w))) (Snippet 18.1)
- [x] lempel_ziv_lib — Build LZ library of non-redundant substrings for compression estimation (Snippet 18.2)
- [x] match_length — Find longest matching substring at index i within prior n positions (Snippet 18.3)
- [x] konto — Kontoyiannis LZ entropy estimator using reciprocal of avg shortest non-redundant substring (Snippet 18.4)

**Functionalities (implement from formulas/logic in AFML):**
- [x] encode_binary — Encode returns as binary string: 1 if r > 0, 0 if r < 0 (Section 18.5.1)
- [x] encode_quantile — Discretize returns into equal-frequency quantile bins (Section 18.5.2)
- [x] encode_sigma — Discretize returns into equal-width bins of sigma step size (Section 18.5.3)
- [x] market_efficiency_metric — Quantify market efficiency via entropy redundancy of encoded returns (Section 18.6)
- [x] portfolio_concentration — Meucci's entropy-based portfolio concentration metric (Section 18.6)
- [x] adverse_selection_feature — Derive adverse selection probability from order flow entropy (Section 18.6)

**ML for Asset Managers additions (complementary):**
- [x] kl_divergence — Kullback-Leibler divergence between distributions (MLAM Section 3.5)
- [x] cross_entropy — Cross-entropy scoring function H_C[p||q] = H[p] + D_KL[p||q] (MLAM Section 3.6)

**Note:** AFML Ch.18 focuses on entropy estimation and financial applications (market efficiency, portfolio concentration, adverse selection). MLAM Ch.3 adds information-theoretic distance metrics (MI, VI, KL) for ML feature engineering. Optimal binning (numBins) may already exist in data_structures — will import if so.

---

### Chapter 19: Microstructural Features
**Branch:** `feat/chapter19-microstructural-features`
**Submodule:** `tradelab.lopezdp_utils.microstructure`
**Status:** ✅ v1 Complete

**Functionalities (with Python code in AFML):**
- [x] corwin_schultz_spread — Estimate bid-ask spread from high-low price ratios (Snippet 19.1)
- [x] get_beta — Beta component for Corwin-Schultz: rolling sum of squared log H/L ratios (Snippet 19.1)
- [x] get_gamma — Gamma component for Corwin-Schultz: squared log of 2-bar H/L range (Snippet 19.1)
- [x] get_alpha — Alpha component combining beta and gamma (Snippet 19.1)
- [x] becker_parkinson_volatility — Robust volatility estimator as byproduct of Corwin-Schultz decomposition (Snippet 19.2)

**Functionalities (implement from formulas/logic in AFML):**
- [x] tick_rule — Classify trade initiation direction (buy/sell) from price changes (Section 19.3.1)
- [x] roll_model — Estimate effective bid-ask spread from serial covariance of price changes (Section 19.3.1)
- [x] high_low_volatility — Parkinson high-low volatility estimator using daily range (Section 19.3.1)
- [x] kyle_lambda — Price impact coefficient from regression of price changes on signed volume (Section 19.3.2)
- [x] amihud_lambda — Price impact proxy: mean absolute return divided by dollar volume (Section 19.3.2)
- [x] hasbrouck_lambda — Effective trading cost via Bayesian regression on signed root-dollar volume (Section 19.3.2)
- [x] vpin — Volume-Synchronized Probability of Informed Trading using volume-clock buckets (Section 19.3.3)

**ML for Asset Managers additions (complementary):**
- None — MLAM uses microstructure as a case study for theory discovery (VPIN, Flash Crash) but provides no new algorithms beyond AFML Ch.19.

**Note:** Chapter 19 presents a taxonomy of four generations of microstructural features. Only Snippets 19.1-19.2 have explicit code; remaining features are implemented from mathematical definitions. PIN is omitted (requires MLE of Poisson mixture — complex and rarely used directly; VPIN supersedes it).

---

### Chapter 20: Multiprocessing and Vectorization
**Branch:** `feat/chapter20-hpc`
**Submodule:** `tradelab.lopezdp_utils.hpc`
**Status:** ✅ v1 Complete

**Functionalities (with Python code in AFML):**
- [x] lin_parts — Partition atoms into approximately equal linear segments for multiprocessing (Snippet 20.1)
- [x] nested_parts — Partition atoms into nested (balanced) segments that equalize workload for upper-triangular tasks (Snippet 20.2)
- [x] mp_pandas_obj — Main multiprocessing engine: partition DataFrame/Series, dispatch to callback, concatenate results (Snippet 20.3)
- [x] process_jobs_redux — Unwrap callback arguments with error logging for multiprocessing jobs (Snippet 20.4)
- [x] process_jobs — Execute jobs using multiprocessing.Pool with imap_unordered (Snippet 20.5)
- [x] report_progress — Asynchronous progress reporting for long-running multiprocessing tasks (Snippet 20.6)
- [x] mp_job_list — General-purpose multiprocessing dispatcher for arbitrary job lists (Snippet 20.7)
- [x] expand_call — Expand keyword arguments dictionary into function call (Snippet 20.9)

**Functionalities (implement from formulas/logic in AFML):**
- [x] single_thread_dispatch — Single-threaded fallback for debugging multiprocessing pipelines (Section 20.5)

**ML for Asset Managers additions (complementary):**
- None — MLAM does not cover HPC topics.

**Note:** Chapter 20 is purely Python engineering (no financial theory). These utilities underpin the multiprocessing used in Chapters 3, 4, 10, and others. NotebookLM was unavailable — implemented from well-known AFML snippets. Snippet 20.8 is a debugging example (single-thread), not a standalone function.
