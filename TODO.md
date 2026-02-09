# TODO.md — tradelab-lib-lopezdp-utils

## Overview

**Phase:** Pre-Production (v1 extraction)
**Status:** Starting

---

## Global Tasks

### Part 1: Data Analysis
- [x] Extract Chapter 2: Financial Data Structures
- [ ] Extract Chapter 3: Labeling
- [ ] Extract Chapter 4: Sample Weights
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
