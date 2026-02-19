"""Fractional Differentiation — AFML Chapter 5.

Implements two approaches to fractional differentiation:

1. **Expanding window** (frac_diff): Uses all available history. Introduces
   negative drift — not recommended for production.

2. **Fixed-Width Window** (frac_diff_ffd): Uses constant-width window determined
   by a weight threshold. No drift. Recommended method.

The core insight: integer differentiation (d=1) removes far more memory than
necessary to achieve stationarity. Fractional d (e.g., d~0.35) preserves memory
while still rendering the series stationary.

References:
    López de Prado, "Advances in Financial Machine Learning", Chapter 5
"""

from __future__ import annotations

import numpy as np
import polars as pl
from statsmodels.tsa.stattools import adfuller


# ---------------------------------------------------------------------------
# Weight generation
# ---------------------------------------------------------------------------


def get_weights(d: float, size: int) -> np.ndarray:
    """Generate weights for the fractional difference operator (1 - B)^d.

    Args:
        d: Differentiation order. Typically 0 < d < 1.
        size: Number of weights to generate (number of lags).

    Returns:
        Array of shape (size, 1) with weights in reverse chronological order.

    Reference:
        AFML Snippet 5.1
    """
    w = [1.0]
    for k in range(1, size):
        w_ = -w[-1] / k * (d - k + 1)
        w.append(w_)
    return np.array(w[::-1]).reshape(-1, 1)


def get_weights_ffd(d: float, thres: float = 1e-5) -> np.ndarray:
    """Generate weights for Fixed-Width Window fractional differentiation (FFD).

    Weights are generated until |omega_k| < thres. This determines the constant
    window width used by frac_diff_ffd.

    Args:
        d: Differentiation order. Typically 0 < d < 1.
        thres: Cut-off threshold for weight magnitude. Default 1e-5.

    Returns:
        Array of shape (n_weights, 1) with weights in reverse chronological order.

    Reference:
        AFML Snippet 5.3 (helper)
    """
    w = [1.0]
    k = 1
    while True:
        w_ = -w[-1] / k * (d - k + 1)
        if abs(w_) < thres:
            break
        w.append(w_)
        k += 1
    return np.array(w[::-1]).reshape(-1, 1)


# ---------------------------------------------------------------------------
# Fractional differentiation
# ---------------------------------------------------------------------------


def frac_diff(
    series: pl.DataFrame,
    column: str,
    d: float,
    thres: float = 0.01,
) -> pl.DataFrame:
    """Fractionally differentiate using an expanding window.

    Warning: Introduces negative drift. Prefer frac_diff_ffd for production.

    Args:
        series: DataFrame with 'timestamp' and value column.
        column: Name of the column to differentiate.
        d: Differentiation order.
        thres: Weight-loss tolerance threshold. Default 0.01.

    Returns:
        DataFrame with 'timestamp' and '{column}_ffd' columns.

    Reference:
        AFML Snippet 5.2
    """
    values = series[column].to_numpy().astype(float)
    timestamps = series["timestamp"]
    n = len(values)

    # Compute weights for the longest series
    w = get_weights(d, n)

    # Determine initial observations to skip based on weight-loss threshold
    w_cum = np.cumsum(np.abs(w))
    w_cum /= w_cum[-1]
    skip = int(np.sum(w_cum > thres))

    # Apply weights
    # TODO(numba): evaluate JIT for fracdiff convolution loop
    output = np.full(n, np.nan)
    for iloc in range(skip, n):
        if not np.isfinite(values[iloc]):
            continue
        output[iloc] = np.dot(w[-(iloc + 1) :, 0], values[: iloc + 1])

    # Filter out NaN rows
    mask = np.isfinite(output)
    return pl.DataFrame(
        {
            "timestamp": timestamps.filter(pl.Series(mask)),
            f"{column}_ffd": output[mask],
        }
    )


def frac_diff_ffd(
    series: pl.DataFrame,
    column: str,
    d: float,
    thres: float = 1e-5,
) -> pl.DataFrame:
    """Fractionally differentiate using Fixed-Width Window (FFD).

    Recommended method: constant window produces driftless, stationary output
    while preserving maximum memory.

    Args:
        series: DataFrame with 'timestamp' and value column.
        column: Name of the column to differentiate.
        d: Differentiation order. Typically 0 < d < 1.
        thres: Cut-off threshold for weight magnitude. Default 1e-5.

    Returns:
        DataFrame with 'timestamp' and '{column}_ffd' columns.

    Reference:
        AFML Snippet 5.3
    """
    values = series[column].to_numpy().astype(float)
    timestamps = series["timestamp"]

    # Compute fixed weight vector
    w = get_weights_ffd(d, thres)
    width = len(w) - 1

    # Apply the constant weight vector
    # TODO(numba): evaluate JIT for FFD convolution loop
    n = len(values)
    output = np.full(n, np.nan)
    for iloc in range(width, n):
        if not np.isfinite(values[iloc]):
            continue
        output[iloc] = np.dot(w[:, 0], values[iloc - width : iloc + 1])

    # Filter out NaN rows
    mask = np.isfinite(output)
    return pl.DataFrame(
        {
            "timestamp": timestamps.filter(pl.Series(mask)),
            f"{column}_ffd": output[mask],
        }
    )


# ---------------------------------------------------------------------------
# Minimum FFD analysis
# ---------------------------------------------------------------------------


def plot_min_ffd(
    series: pl.DataFrame,
    column: str = "close",
    d_values: np.ndarray | None = None,
    thres: float = 1e-2,
) -> pl.DataFrame:
    """Find the minimum fractional differentiation order d* for stationarity.

    Iterates through d values, applying FFD and running the ADF test.
    The minimum d* is where ADF crosses the 95% critical value.

    Args:
        series: DataFrame with 'timestamp' and value column.
        column: Column to analyze. Default "close".
        d_values: Array of d values to test. Default linspace(0, 1, 11).
        thres: Weight threshold for FFD. Default 1e-2.

    Returns:
        DataFrame with columns: d, adf_stat, p_value, lags, n_obs,
        confidence_95pct, corr_with_original.

    Reference:
        AFML Snippet 5.4
    """
    if d_values is None:
        d_values = np.linspace(0, 1, 11)

    original = series[column].to_numpy().astype(float)
    results: list[dict] = []

    for d in d_values:
        ffd = frac_diff_ffd(series, column=column, d=d, thres=thres)
        ffd_values = ffd[f"{column}_ffd"].to_numpy()

        if len(ffd_values) < 20:
            continue

        # Correlation with original (aligned at the end)
        orig_aligned = original[-len(ffd_values) :]
        corr = float(np.corrcoef(orig_aligned, ffd_values)[0, 1])

        # ADF test
        adf_result = adfuller(ffd_values, maxlag=1, regression="c", autolag=None)

        results.append(
            {
                "d": d,
                "adf_stat": adf_result[0],
                "p_value": adf_result[1],
                "lags": adf_result[2],
                "n_obs": adf_result[3],
                "confidence_95pct": adf_result[4]["5%"],
                "corr_with_original": corr,
            }
        )

    return pl.DataFrame(results)


# ---------------------------------------------------------------------------
# Visualization (optional matplotlib)
# ---------------------------------------------------------------------------


def plot_weights(
    d_range: list[float],
    n_plots: int,
    size: int,
):
    """Visualize weight decay curves for different d values.

    Args:
        d_range: [d_min, d_max] range of differentiation orders.
        n_plots: Number of curves to plot.
        size: Number of lags to display.

    Returns:
        Matplotlib Axes object.

    Reference:
        AFML Snippet 5.1 (plotWeights)
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    w = pd.DataFrame()
    for d in np.linspace(d_range[0], d_range[1], n_plots):
        w_ = get_weights(d, size=size)
        w_ = pd.DataFrame(w_, index=range(w_.shape[0])[::-1], columns=[d])
        w = w.join(w_, how="outer")
    ax = w.plot()
    ax.legend(loc="upper left")
    return ax
