"""CSCV and Probability of Backtest Overfitting — AFML Chapter 11.

References:
    López de Prado, "Advances in Financial Machine Learning", Chapter 11
    Bailey, Borwein, López de Prado, Zhu (2017)
"""

import itertools

import numpy as np
import pandas as pd
import polars as pl


def probability_of_backtest_overfitting(
    trial_returns: pl.DataFrame,
    n_partitions: int = 16,
    metric: str = "sharpe",
) -> dict:
    """Estimate Probability of Backtest Overfitting (PBO) using CSCV.

    Args:
        trial_returns: DataFrame with 'timestamp' column and one column per
            strategy trial containing returns.
        n_partitions: Number of equal-sized groups (S). Must be even.
        metric: Performance metric: 'sharpe', 'total_return', or 'mean_return'.

    Returns:
        Dictionary with 'pbo', 'logits', 'ranks', 'n_combinations'.
    """
    if n_partitions % 2 != 0:
        raise ValueError(f"n_partitions must be even, got {n_partitions}")

    # Convert to pandas for internal combinatorial loop (via dict, no pyarrow)
    cols = [c for c in trial_returns.columns if c != "timestamp"]
    pdf = pd.DataFrame({c: trial_returns[c].to_numpy() for c in cols})

    if pdf.shape[1] < 2:
        raise ValueError("Need at least 2 strategy trials")

    t, n = pdf.shape
    rows_per_partition = t // n_partitions
    if rows_per_partition < 2:
        raise ValueError(f"Not enough observations ({t}) for {n_partitions} partitions")

    pdf = pdf.iloc[: rows_per_partition * n_partitions]

    groups = []
    for i in range(n_partitions):
        start = i * rows_per_partition
        end = start + rows_per_partition
        groups.append(pdf.iloc[start:end])

    half = n_partitions // 2
    combo_indices = list(itertools.combinations(range(n_partitions), half))

    logits = []
    ranks = []

    for train_idx in combo_indices:
        test_idx = tuple(i for i in range(n_partitions) if i not in train_idx)

        train_data = pd.concat([groups[i] for i in train_idx])
        test_data = pd.concat([groups[i] for i in test_idx])

        train_perf = _compute_metric(train_data, metric)
        test_perf = _compute_metric(test_data, metric)

        n_star = train_perf.idxmax()
        oos_perf_n_star = test_perf[n_star]
        omega_c = (test_perf < oos_perf_n_star).sum() / (n - 1)
        omega_c = np.clip(omega_c, 1e-10, 1 - 1e-10)

        logit_c = np.log((1 - omega_c) / omega_c)
        logits.append(logit_c)
        ranks.append(omega_c)

    logits = np.array(logits)
    ranks = np.array(ranks)
    pbo = float((logits <= 0).sum() / len(logits))

    return {
        "pbo": pbo,
        "logits": logits,
        "ranks": ranks,
        "n_combinations": len(combo_indices),
    }


def _compute_metric(data: pd.DataFrame, metric: str) -> pd.Series:
    """Compute performance metric for each column."""
    if metric == "sharpe":
        means = data.mean()
        stds = data.std()
        stds = stds.replace(0, np.nan)
        return means / stds
    elif metric == "total_return":
        return data.sum()
    elif metric == "mean_return":
        return data.mean()
    else:
        raise ValueError(f"Unknown metric: {metric}")
