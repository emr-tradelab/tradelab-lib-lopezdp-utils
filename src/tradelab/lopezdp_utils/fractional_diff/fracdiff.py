"""
Fractional Differentiation Methods

This module implements two approaches to fractional differentiation of financial
time series:

1. **Expanding window** (frac_diff): Uses all available history for each observation,
   with a weight-loss threshold to skip initial unstable observations. Introduces
   negative drift because the window grows over time.

2. **Fixed-Width Window** (frac_diff_ffd): Uses a constant-width window determined
   by a weight threshold. Avoids negative drift and produces a stationary, driftless
   blend of the original series. This is the author's recommended method.

The core problem solved: standard integer differentiation (d=1, i.e., log-returns)
achieves stationarity but discards most of the memory (signal) in the series.
Fractional differentiation with d < 1 (typically d ≈ 0.35) achieves stationarity
while preserving significantly more memory, as measured by correlation with the
original series.

References:
    - AFML Chapter 5, Snippets 5.2-5.4
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from .weights import get_weights, get_weights_ffd


def frac_diff(
    series: pd.DataFrame,
    d: float,
    thres: float = 0.01,
) -> pd.DataFrame:
    """
    Fractionally differentiate a time series using an expanding window.

    Applies the fractional difference operator (1 - B)^d with an expanding window.
    Initial observations are skipped where the cumulative weight loss exceeds the
    threshold τ, ensuring consistent memory coverage across the output series.

    Warning: The expanding window introduces negative drift because each new
    observation adds a new negative weight as the window grows. For production
    use, prefer frac_diff_ffd which avoids this problem.

    Args:
        series: Input time series (e.g., log-prices). Each column is
            differentiated independently.
        d: Differentiation order. Any positive fractional value. Typically
            0 < d < 1 for fractional differentiation.
        thres: Weight-loss tolerance threshold (τ). Observations where cumulative
            weight loss exceeds this fraction are skipped. thres=1 skips nothing.
            Default 0.01.

    Returns:
        Fractionally differentiated series. Initial observations are dropped
        based on the weight-loss threshold.

    Mathematical Logic:
        For each observation at time t:
            X̃_t = Σ(k=0 to t) ωₖ x X_{t-k}

        Weight-loss mechanism:
            λₗ = Σ(k=0 to l) |ωₖ| / Σ(k=0 to T) |ωₖ|
        Skip observations where λₗ > τ (insufficient weight coverage).

    Example:
        >>> import pandas as pd
        >>> prices = pd.DataFrame({'close': [100, 101, 102, 103, 104]},
        ...     index=pd.date_range('2020-01-01', periods=5))
        >>> log_prices = np.log(prices)
        >>> result = frac_diff(log_prices, d=0.5)

    Reference:
        AFML Snippet 5.2
    """
    # 1) Compute weights for the longest series
    w = get_weights(d, series.shape[0])

    # 2) Determine initial calcs to be skipped based on weight-loss threshold
    w_ = np.cumsum(abs(w))
    w_ /= w_[-1]
    skip = w_[w_ > thres].shape[0]

    # 3) Apply weights to values
    df = {}
    for name in series.columns:
        series_f = series[[name]].ffill().dropna()
        df_ = pd.Series(dtype=float)
        for iloc in range(skip, series_f.shape[0]):
            loc = series_f.index[iloc]
            if not np.isfinite(series.loc[loc, name]):
                continue  # exclude NAs
            df_[loc] = np.dot(w[-(iloc + 1) :, :].T, series_f.loc[:loc])[0, 0]
        df[name] = df_.copy(deep=True)
    df = pd.concat(df, axis=1)
    return df


def frac_diff_ffd(
    series: pd.DataFrame,
    d: float,
    thres: float = 1e-5,
) -> pd.DataFrame:
    """
    Fractionally differentiate using Fixed-Width Window (FFD).

    Applies the fractional difference operator (1 - B)^d with a constant-width
    window. The window width is determined by the weight threshold τ: weights
    are generated until their absolute value falls below τ, then this fixed-size
    weight vector is applied uniformly to all observations.

    This is the author's recommended method because:
    - Constant window → no negative drift (unlike expanding window)
    - Same weight vector for all observations → consistent transformation
    - Achieves stationarity while preserving maximum memory

    Args:
        series: Input time series (e.g., log-prices). Each column is
            differentiated independently.
        d: Differentiation order. Typically 0 < d < 1.
        thres: Cut-off threshold for weight magnitude. Determines window
            width: weights with |ωₖ| < τ are dropped. Smaller τ means
            wider window and more memory. Default 1e-5.

    Returns:
        Fractionally differentiated series. Initial observations are dropped
        to accommodate the window width.

    Mathematical Logic:
        Window width l* is determined by: |ω_{l*+1}| < τ
        For each observation at time t:
            X̃_t = Σ(k=0 to l*) ωₖ x X_{t-k}

        The key difference from frac_diff: the weight vector [ω₀, ..., ω_{l*}]
        is identical for every observation, producing a driftless output.

    Example:
        >>> import pandas as pd
        >>> prices = pd.DataFrame({'close': [100, 101, 102, 103, 104]},
        ...     index=pd.date_range('2020-01-01', periods=5))
        >>> log_prices = np.log(prices)
        >>> result = frac_diff_ffd(log_prices, d=0.5, thres=1e-5)

    Reference:
        AFML Snippet 5.3
    """
    # 1) Compute weights until the modulus falls below the threshold
    w = get_weights_ffd(d, thres)
    width = len(w) - 1

    # 2) Apply the constant vector of weights to the series
    df = {}
    for name in series.columns:
        series_f = series[[name]].ffill().dropna()
        df_ = pd.Series(dtype=float)
        for iloc1 in range(width, series_f.shape[0]):
            loc0 = series_f.index[iloc1 - width]
            loc1 = series_f.index[iloc1]
            if not np.isfinite(series.loc[loc1, name]):
                continue  # exclude remaining NAs
            df_[loc1] = np.dot(w.T, series_f.loc[loc0:loc1])[0, 0]
        df[name] = df_.copy(deep=True)
    df = pd.concat(df, axis=1)
    return df


def plot_min_ffd(
    series: pd.DataFrame,
    column: str = "close",
    d_values: np.ndarray | None = None,
    thres: float = 1e-2,
) -> pd.DataFrame:
    """
    Find the minimum fractional differentiation order d* for stationarity.

    Iterates through differentiation orders from 0 to 1, applying FFD to
    the input series and running the Augmented Dickey-Fuller (ADF) test.
    The minimum d* is the smallest d where the ADF statistic crosses below
    the 95% confidence critical value, indicating stationarity.

    This function helps identify the optimal d that achieves stationarity
    while preserving maximum memory (correlation with original series).
    Typical result for liquid instruments: d* ≈ 0.35 with correlation > 0.99,
    far superior to standard returns (d=1, correlation ≈ 0).

    Args:
        series: Input time series. Should contain price data (will be
            log-transformed internally).
        column: Column name to analyze. Default "close".
        d_values: Array of d values to test. Default np.linspace(0, 1, 11).
        thres: Weight threshold for FFD method. Default 1e-2.

    Returns:
        DataFrame with columns ['adf_stat', 'p_val', 'lags', 'n_obs',
        '95_conf', 'corr'], indexed by d value.

    Example:
        >>> prices = pd.DataFrame({'close': price_series},
        ...     index=pd.date_range('2020-01-01', periods=1000))
        >>> results = plot_min_ffd(prices, column='close')
        >>> # Find minimum d for stationarity
        >>> d_star = results[results['p_val'] < 0.05].index.min()
        >>> print(f"Minimum d for stationarity: {d_star}")

    Reference:
        AFML Snippet 5.4
    """
    if d_values is None:
        d_values = np.linspace(0, 1, 11)

    out = pd.DataFrame(columns=["adf_stat", "p_val", "lags", "n_obs", "95_conf", "corr"])

    # Log-transform the series
    df0 = np.log(series[[column]]).resample("1D").last()

    for d in d_values:
        # Apply FFD
        df1 = frac_diff_ffd(df0, d, thres=thres)

        # Correlation with original log-prices
        corr = np.corrcoef(df0.loc[df1.index, column], df1[column])[0, 1]

        # ADF test (maxlag=1 for daily data as per the author)
        adf_result = adfuller(df1[column], maxlag=1, regression="c", autolag=None)

        out.loc[d] = [*list(adf_result[:4]), adf_result[4]["5%"], corr]

    # Plot results
    out[["adf_stat", "corr"]].plot(secondary_y="adf_stat")
    plt.axhline(out["95_conf"].mean(), linewidth=1, color="r", linestyle="dotted")
    plt.title("Minimum FFD: ADF Statistic vs Correlation")
    plt.xlabel("d (differentiation order)")

    return out
