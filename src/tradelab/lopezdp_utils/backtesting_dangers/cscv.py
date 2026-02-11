"""Combinatorially Symmetric Cross-Validation (CSCV) and Probability of Backtest Overfitting.

Implements the CSCV procedure described in AFML Chapter 11 for estimating the
Probability of Backtest Overfitting (PBO). CSCV evaluates whether a strategy's
in-sample performance persists out-of-sample across all combinatorial train/test
splits.

The key idea: partition T observations into S groups, form all possible S/2-group
training sets (and their complements as test sets), find the best in-sample strategy,
and check how it ranks out-of-sample. If the best IS strategy consistently degrades
OOS, the backtest is likely overfit.

Key metrics:
- **Relative rank (omega_c):** Where the best IS strategy ranks among all strategies
  OOS (0 = worst, 1 = best).
- **Logit (lambda_c):** log[(1 - omega_c) / omega_c]. Zero means median OOS
  performance; negative means below-median (overfitting evidence).
- **PBO:** Fraction of combinatorial splits where the logit is <= 0 (the best IS
  strategy performed at or below median OOS).

References:
    - AFML Chapter 11, Section 11.5
    - Bailey, Borwein, López de Prado, Zhu (2017): "The Probability of Backtest
      Overfitting", Journal of Computational Finance
"""

import itertools

import numpy as np
import pandas as pd


def probability_of_backtest_overfitting(
    trial_returns: pd.DataFrame,
    n_partitions: int = 16,
    metric: str = "sharpe_ratio",
) -> dict:
    """Estimate the Probability of Backtest Overfitting (PBO) using CSCV.

    Args:
        trial_returns: T x N DataFrame where each column is a PnL/return series
            from a backtested strategy trial, and the index represents time.
        n_partitions: Number of equal-sized groups (S) to split time axis into.
            Must be even. More partitions = more combinations but each split has
            fewer observations. Typical values: 8, 10, 16.
        metric: Performance metric to evaluate strategies. One of:
            - ``"sharpe_ratio"``: Annualized Sharpe ratio (default)
            - ``"total_return"``: Cumulative return
            - ``"mean_return"``: Mean period return

    Returns:
        Dictionary with:
            - ``"pbo"``: Probability of Backtest Overfitting (0 to 1).
              Values > 0.5 indicate likely overfitting.
            - ``"logits"``: Array of logit values for each combinatorial split.
            - ``"ranks"``: Array of relative ranks (omega_c) for each split.
            - ``"n_combinations"``: Total number of train/test splits evaluated.

    Raises:
        ValueError: If n_partitions is odd, too large for the data, or trial_returns
            has fewer than 2 columns.
    """
    if n_partitions % 2 != 0:
        raise ValueError(f"n_partitions must be even, got {n_partitions}")
    if trial_returns.shape[1] < 2:
        raise ValueError("Need at least 2 strategy trials")

    t, n = trial_returns.shape
    rows_per_partition = t // n_partitions
    if rows_per_partition < 2:
        raise ValueError(f"Not enough observations ({t}) for {n_partitions} partitions")

    # Truncate to make partitions equal-sized
    trial_returns = trial_returns.iloc[: rows_per_partition * n_partitions]

    # Split into S groups by time
    groups = []
    for i in range(n_partitions):
        start = i * rows_per_partition
        end = start + rows_per_partition
        groups.append(trial_returns.iloc[start:end])

    # Generate all combinations of S/2 groups for training
    half = n_partitions // 2
    combo_indices = list(itertools.combinations(range(n_partitions), half))

    logits = []
    ranks = []

    for train_idx in combo_indices:
        test_idx = tuple(i for i in range(n_partitions) if i not in train_idx)

        # Concatenate groups for train and test
        train_data = pd.concat([groups[i] for i in train_idx])
        test_data = pd.concat([groups[i] for i in test_idx])

        # Compute performance metric for each trial
        train_perf = _compute_metric(train_data, metric)
        test_perf = _compute_metric(test_data, metric)

        # Find best in-sample strategy
        n_star = train_perf.idxmax()

        # Relative rank of n_star's OOS performance among all OOS performances
        oos_perf_n_star = test_perf[n_star]
        omega_c = (test_perf < oos_perf_n_star).sum() / (n - 1)

        # Clamp to avoid log(0) or division by zero
        omega_c = np.clip(omega_c, 1e-10, 1 - 1e-10)

        logit_c = np.log((1 - omega_c) / omega_c)

        logits.append(logit_c)
        ranks.append(omega_c)

    logits = np.array(logits)
    ranks = np.array(ranks)
    pbo = (logits <= 0).sum() / len(logits)

    return {
        "pbo": pbo,
        "logits": logits,
        "ranks": ranks,
        "n_combinations": len(combo_indices),
    }


def _compute_metric(data: pd.DataFrame, metric: str) -> pd.Series:
    """Compute performance metric for each column (strategy trial).

    Args:
        data: Subset of trial returns (T' x N).
        metric: One of "sharpe_ratio", "total_return", "mean_return".

    Returns:
        Series of performance values indexed by column name.
    """
    if metric == "sharpe_ratio":
        means = data.mean()
        stds = data.std()
        # Avoid division by zero for constant strategies
        stds = stds.replace(0, np.nan)
        return means / stds
    elif metric == "total_return":
        return data.sum()
    elif metric == "mean_return":
        return data.mean()
    else:
        raise ValueError(f"Unknown metric: {metric}")
