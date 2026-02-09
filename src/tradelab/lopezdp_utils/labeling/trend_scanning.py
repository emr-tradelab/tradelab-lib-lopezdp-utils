"""Trend-scanning labeling method.

This module implements the trend-scanning method from ML for Asset Managers,
which identifies labels based on the most statistically significant trend
rather than predefined barriers.

Unlike the triple-barrier method, trend-scanning:
- Avoids fixed barriers (no predefined profit-taking/stop-loss levels)
- Uses dynamic horizons (searches for strongest trend across multiple periods)
- Relies on statistical significance (t-values) rather than discrete barrier touches

Reference: MLAM Section 5.4
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm


def t_value_linear_trend(close: np.ndarray) -> float:
    """Calculate t-value of the slope coefficient in a linear time-trend model.

    Fits a simple linear regression model: price[t] = β₀ + β₁·t + ε[t]
    and returns the t-statistic for the slope coefficient β₁.

    A large positive t-value indicates a statistically significant uptrend.
    A large negative t-value indicates a statistically significant downtrend.

    Args:
        close: Array of prices over which to fit the linear trend.

    Returns:
        t-value associated with the slope coefficient β₁.

    Reference:
        Snippet 5.1 in MLAM Section 5.4
    """
    # Create design matrix: intercept and time index
    x = np.ones((close.shape[0], 2))
    x[:, 1] = np.arange(close.shape[0])

    # Fit OLS regression
    ols = sm.OLS(close, x).fit()

    # Return t-value for slope coefficient (index 1)
    return ols.tvalues[1]


def trend_scanning_labels(
    close: pd.Series, t_events: pd.DatetimeIndex, span: tuple[int, int] | range
) -> pd.DataFrame:
    """Generate labels using the trend-scanning method.

    This method identifies the most statistically significant trend by:
    1. Fitting linear time-trend models over various look-forward periods
    2. Computing t-values for each horizon's slope coefficient
    3. Selecting the horizon that maximizes the absolute t-value
    4. Labeling based on the sign of that maximum t-value

    The output label represents the direction of the strongest observable trend,
    and the t-value magnitude can be used for sample weighting or confidence scoring.

    Advantages over triple-barrier:
    - No need to specify profit-taking/stop-loss levels
    - Automatically finds the optimal horizon for each event
    - Provides continuous measure of trend strength (t-value)

    Args:
        close: Series of closing prices indexed by timestamp.
        t_events: DatetimeIndex of observation timestamps to label.
            Typically output of event-based sampling (e.g., CUSUM filter).
        span: Tuple (start, end) or range object defining look-forward periods
            to evaluate (e.g., (3, 11) checks horizons from 3 to 10 bars).

    Returns:
        DataFrame with index matching t_events, containing:
            - 't1': End timestamp for the identified trend
            - 'tVal': t-value associated with the trend coefficient
            - 'bin': Label (-1, 0, or 1) based on sign of t-value

    Reference:
        Snippet 5.2 in MLAM Section 5.4

    Example:
        >>> from tradelab.lopezdp_utils.data_structures import get_t_events
        >>> events = get_t_events(close, threshold=0.01)  # CUSUM filter
        >>> labels = trend_scanning_labels(
        ...     close=close,
        ...     t_events=events,
        ...     span=(3, 11)  # Test horizons from 3 to 10 bars
        ... )
    """
    out = pd.DataFrame(index=t_events, columns=["t1", "tVal", "bin"])

    # Convert span to range if it's a tuple
    if isinstance(span, tuple):
        hrzns = range(*span)
    else:
        hrzns = span

    for dt0 in t_events:
        df0 = pd.Series()
        iloc0 = close.index.get_loc(dt0)

        # Skip if insufficient data for maximum horizon
        if iloc0 + max(hrzns) > close.shape[0]:
            continue

        # Test each horizon to find strongest trend
        for hrzn in hrzns:
            dt1 = close.index[iloc0 + hrzn - 1]
            df1 = close.loc[dt0:dt1]
            df0.loc[dt1] = t_value_linear_trend(df1.values)

        # Identify horizon that maximizes absolute t-value
        # Replace invalid values with 0 before finding max
        dt1 = df0.replace([-np.inf, np.inf, np.nan], 0).abs().idxmax()

        # Store results
        out.loc[dt0, ["t1", "tVal", "bin"]] = dt1, df0[dt1], np.sign(df0[dt1])

    # Convert types
    out["t1"] = pd.to_datetime(out["t1"])
    out["tVal"] = pd.to_numeric(out["tVal"])
    out["bin"] = pd.to_numeric(out["bin"], downcast="signed")

    return out.dropna(subset=["bin"])
