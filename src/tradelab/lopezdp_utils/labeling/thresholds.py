"""Dynamic threshold computation for labeling methods.

This module provides utilities for computing dynamic thresholds based on market
volatility, essential for the triple-barrier method and other labeling approaches.

Reference: AFML Chapter 3, Section 3.3
"""

import pandas as pd


def daily_volatility(close: pd.Series, span: int = 100) -> pd.Series:
    """Estimate daily volatility using exponentially weighted moving standard deviation.

    This function computes volatility by:
    1. Identifying the price bar from exactly 24 hours prior to each observation
    2. Computing daily returns relative to the prior day's price
    3. Applying EWMA to the standard deviation of those returns

    The resulting volatility estimates are used to set dynamic profit-taking and
    stop-loss thresholds in the triple-barrier method. By scaling barriers based
    on volatility, the algorithm ensures targets are adjusted for risk: wider
    barriers when volatility is high, narrower when volatility is low.

    This prevents labeling with fixed thresholds (e.g., 1% move), which ignores
    heteroscedasticity—the fact that market volatility changes over time.

    Args:
        close: Series of closing prices indexed by timestamp.
        span: Number of days for exponentially weighted moving standard deviation.
            Default is 100 days.

    Returns:
        Series of daily volatility estimates, reindexed to match the input close
        price index.

    Reference:
        Snippet 3.1 in AFML Chapter 3, Section 3.3

    Example:
        >>> prices = pd.Series([100, 101, 99, 102],
        ...                    index=pd.date_range('2020-01-01', periods=4))
        >>> vol = daily_volatility(prices, span=50)
        >>> # Use vol to scale triple-barrier thresholds
    """
    # Find index of price bar from 24 hours prior
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df0 = df0[df0 > 0]

    # Map current timestamps to prior day timestamps
    df0 = pd.Series(close.index[df0 - 1], index=close.index[close.shape[0] - df0.shape[0] :])

    # Compute daily returns
    df0 = close.loc[df0.index] / close.loc[df0.values].values - 1

    # Apply exponentially weighted moving standard deviation
    df0 = df0.ewm(span=span).std()

    return df0
