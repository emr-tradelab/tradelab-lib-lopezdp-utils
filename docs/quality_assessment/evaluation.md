# Quality Assessment: evaluation

**Module:** `src/tradelab/lopezdp_utils/evaluation/`
**Chapters:** AFML Ch. 10 (Bet Sizing), Ch. 11 (PBO), Ch. 12 (CPCV), Ch. 13 (Synthetic), Ch. 14 (Statistics), Ch. 15 (Strategy Risk) + MLAM Section 8
**Date:** 2026-02-21
**Result:** PASS

## Source Files Assessed

- `bet_sizing.py` — `get_signal`, `avg_active_signals`, `discrete_signal`, `bet_size`, `get_target_pos`, `inv_price`, `limit_price`, `get_w`, `bet_size_mixture`, `bet_size_ensemble`
- `cpcv.py` — `CombinatorialPurgedKFold`, `get_num_splits`, `get_num_backtest_paths`, `assemble_backtest_paths`
- `overfitting.py` — `probability_of_backtest_overfitting`
- `redundancy.py` — `get_effective_trials`, `deflated_sharpe_ratio_clustered`
- `statistics.py` — `sharpe_ratio`, `sharpe_ratio_non_annualized`, `probabilistic_sharpe_ratio`, `deflated_sharpe_ratio`, `compute_dd_tuw`, `get_hhi`, `get_hhi_decomposed`, `get_bet_timing`, `get_holding_period`, `strategy_precision`, `strategy_recall`, `multi_test_precision_recall`
- `strategy_risk.py` — `sharpe_ratio_symmetric`, `implied_precision_symmetric`, `sharpe_ratio_asymmetric`, `bin_hr`, `bin_freq`, `mix_gaussians`, `prob_failure`
- `synthetic.py` — `ou_half_life`, `ou_fit`, `otr_batch`, `otr_main`

## Tests

86 tests, all passing.

## Theory Comparison

### bet_sizing.py (Ch. 10)

| Function | Snippet | Verdict | Notes |
|----------|---------|---------|-------|
| `get_signal` | 10.1-10.3 | Correct | z-stat = (prob - 1/K) / sqrt(prob*(1-prob)), then 2*Phi(z)-1, pred multiplied after CDF, side applied last |
| `avg_active_signals` | 10.2 | Correct | Averages concurrent active bets to reduce turnover |
| `discrete_signal` | 10.3 | Correct | Round to step_size, clip to [-1,1] |
| `bet_size` | 10.4 | Correct | Sigmoid m(x) = x*(w+x²)^(-0.5) |
| `get_target_pos` | 10.4 | Correct | int(m * max_pos) |
| `inv_price` | 10.4 | Correct | Inverse of sizing function |
| `limit_price` | 10.4 | Correct | Average limit price for multi-unit order |
| `get_w` | 10.4 | Correct | Calibrates omega from divergence/size pair |
| `bet_size_mixture` | — | Correct | Mixture CDF approach |
| `bet_size_ensemble` | — | Correct | Student-t sizing for ensemble classifiers |

### cpcv.py (Ch. 12)

| Function | Snippet | Verdict | Notes |
|----------|---------|---------|-------|
| `get_num_splits` | 12.1 | Correct | C(N, k) |
| `get_num_backtest_paths` | 12.1 | Correct | phi = k/N * C(N,k), verified N=6,k=2 → 5 |
| `CombinatorialPurgedKFold` | 12.2 | Correct | Generates all combinatorial splits with purging via get_train_times and embargo via get_embargo_times |
| `assemble_backtest_paths` | 12.3 | Correct | Partition-based assembly of OOS predictions |

### overfitting.py (Ch. 11)

| Function | Snippet | Verdict | Notes |
|----------|---------|---------|-------|
| `probability_of_backtest_overfitting` | 11.1 | Correct | CSCV: even S partitions, C(S, S/2) combos, logit of OOS rank, PBO = fraction of logits ≤ 0 |

### redundancy.py (Ch. 14 / MLAM)

| Function | Snippet | Verdict | Notes |
|----------|---------|---------|-------|
| `get_effective_trials` | — | Correct | ONC clustering + min-variance aggregate per cluster |
| `deflated_sharpe_ratio_clustered` | — | Correct | DSR with ONC-based effective trial count |

### statistics.py (Ch. 14 + MLAM)

| Function | Snippet | Verdict | Notes |
|----------|---------|---------|-------|
| `sharpe_ratio` | — | Correct | (mu/sigma) * sqrt(periods_per_year) |
| `probabilistic_sharpe_ratio` | 14.6 | Correct | z = (SR-SR*)*sqrt(n-1) / sqrt(1 - skew*SR + (kurtosis-1)/4 * SR²), raw kurtosis (3 for Gaussian) |
| `deflated_sharpe_ratio` | 14.7 | Correct | E[max SR] = sigma * ((1-gamma)*Phi^{-1}(1-1/N) + gamma*Phi^{-1}(1-1/(N*e))), gamma = Euler-Mascheroni |
| `compute_dd_tuw` | 14.4 | Correct | HWM drawdown + time-under-water episodes |
| `get_hhi` | 14.3 | Correct | Normalized HHI: (sum(w²) - 1/n) / (1 - 1/n) |
| `get_bet_timing` | 14.1 | Correct | Detects flattenings and flips |
| `get_holding_period` | 14.2 | Correct | Weighted average holding period |
| `strategy_precision` | MLAM 8 | Correct | (1-beta)*theta / ((1-beta)*theta + alpha) |
| `strategy_recall` | MLAM 8 | Correct | 1 - beta |
| `multi_test_precision_recall` | MLAM 8 | Correct | Sidak correction: alpha_k = 1-(1-alpha)^k, beta_k = beta^k |

### strategy_risk.py (Ch. 15)

| Function | Snippet | Verdict | Notes |
|----------|---------|---------|-------|
| `sharpe_ratio_symmetric` | 15.1 | Correct | sqrt(n)*(2p-1)/sqrt(4p(1-p)) |
| `sharpe_ratio_asymmetric` | 15.2 | Correct | mean=(pt-sl)*p+sl, var=(pt-sl)²*p*(1-p), SR=sqrt(n)*mean/sqrt(var) |
| `bin_hr` | 15.3 | Correct | Quadratic formula for implied precision |
| `bin_freq` | 15.4 | Correct | Implied frequency formula |
| `mix_gaussians` | 15.5 | Correct | Mixture generator with reproducible seed |
| `prob_failure` | 15.6 | Correct | P[p < p*] via normal CDF approximation |

### synthetic.py (Ch. 13)

| Function | Snippet | Verdict | Notes |
|----------|---------|---------|-------|
| `ou_half_life` | 13.1 | Correct | -log(2)/log(phi) |
| `ou_fit` | 13.2 | Correct | OLS regression on (p_{t-1} - forecast) |
| `otr_batch` | 13.3 | Correct | Monte Carlo OTR with pt/sl mesh |
| `otr_main` | 13.4 | Correct | Multi-regime OTR experiment |

## Issues Found

**P0 (Critical):** None
**P1 (Important):** None
**P2 (Minor):** None

## Conclusion

All implementations across 7 submodules faithfully follow López de Prado's theory from AFML Chapters 10-15 and MLAM Section 8. Module validated as correct.
