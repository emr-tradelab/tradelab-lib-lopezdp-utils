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
- [ ] Extract Chapter 5: Fractionally Differentiated Features

### Part 2: Modelling
- [ ] Extract Chapter 6: Ensemble Methods
- [ ] Extract Chapter 7: Cross-Validation in Finance
- [ ] Extract Chapter 8: Feature Importance
- [ ] Extract Chapter 9: Hyper-Parameter Tuning with Cross-Validation

### Part 3: Backtesting
- [ ] Extract Chapter 10: Bet Sizing
- [ ] Extract Chapter 11: The Dangers of Backtesting
- [ ] Extract Chapter 12: Backtesting through Cross-Validation
- [ ] Extract Chapter 13: Backtesting on Synthetic Data
- [ ] Extract Chapter 14: Backtest Statistics
- [ ] Extract Chapter 15: Understanding Strategy Risk
- [ ] Extract Chapter 16: Machine Learning Asset Allocation

### Part 4: Useful Financial Features
- [ ] Extract Chapter 17: Structural Breaks
- [ ] Extract Chapter 18: Entropy Features
- [ ] Extract Chapter 19: Microstructural Features

### Part 5: High-Performance Computing Recipes
- [ ] Extract Chapter 20: Multiprocessing and Vectorization

### Final Steps
- [ ] Review ML for Asset Managers for complementary content (done per-chapter)
- [ ] **Phase 2:** Production optimization pass (pandas → Polars, tests, performance)

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
