"""Drawdown and time-under-water computation from AFML Chapter 14.

Computes sequences of maximum drawdowns between high-watermarks and the
associated recovery times. López de Prado recommends reporting the
95th percentile of both DD and TuW rather than maximums.

Reference: AFML Snippet 14.4
"""

import numpy as np
import pandas as pd


def compute_dd_tuw(series: pd.Series, dollars: bool = False) -> tuple[pd.Series, pd.Series]:
    """Compute drawdown series and time-under-water between high-watermarks.

    For each drawdown episode (period between consecutive high-watermarks),
    computes the maximum loss and the duration until recovery.

    Args:
        series: Performance series (cumulative returns or dollar PnL).
        dollars: If True, DD in dollar terms. If False, in percentage terms.

    Returns:
        Tuple of (dd, tuw) where:
            dd: Series of drawdowns indexed by high-watermark timestamps.
            tuw: Series of time-under-water in years.

    Reference:
        AFML Snippet 14.4
    """
    df0 = series.to_frame("pnl")
    df0["hwm"] = series.expanding().max()
    df1 = df0.groupby("hwm").min().reset_index()
    df1.columns = ["hwm", "min"]

    # Time of each high-watermark
    df1.index = df0["hwm"].drop_duplicates(keep="first").index
    # Filter for HWMs followed by actual drawdown
    df1 = df1[df1["hwm"] > df1["min"]]

    if dollars:
        dd = df1["hwm"] - df1["min"]
    else:
        dd = 1 - df1["min"] / df1["hwm"]

    # Time under water in years
    tuw = (df1.index[1:] - df1.index[:-1]) / np.timedelta64(1, "Y")
    tuw = pd.Series(tuw, index=df1.index[:-1])

    return dd, tuw
