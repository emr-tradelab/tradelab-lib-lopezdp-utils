"""Fixed-time horizon labeling method.

This module implements the standard fixed-time horizon labeling approach,
which is prevalent in financial ML literature but has known shortcomings.

WARNING: The book recommends AGAINST using this method due to:
- Ignoring heteroscedasticity (changing market volatility)
- Discarding path information (intermediate price movements)
- Information loss when threshold is misaligned with realized volatility

Consider using the triple-barrier method instead, which addresses these issues.

Reference: AFML Chapter 3, Section 3.2
"""

import numpy as np
import pandas as pd


def fixed_time_horizon(close: pd.Series, horizon: int, threshold: float) -> pd.DataFrame:
    """Label observations using the fixed-time horizon method.

    This method assigns labels based on price returns over a constant number
    of bars (horizon) relative to a fixed threshold. While prevalent in
    financial ML literature, this approach has significant flaws:

    1. Heteroscedasticity: Uses fixed threshold despite changing market volatility
    2. Path neglect: Ignores intermediate price movements (e.g., stop-outs)
    3. Information loss: Many labels become 0 if threshold is misaligned

    The labeling formula:
        r = (price[t+h] / price[t]) - 1

    Labels:
        -1 if r < -threshold (significant downward move)
         0 if |r| <= threshold (insignificant move)
         1 if r > threshold (significant upward move)

    Args:
        close: Series of closing prices indexed by timestamp.
        horizon: Number of bars ahead to evaluate return (fixed window).
        threshold: Constant threshold (e.g., 0.01 for 1%) to determine
            significance of return. Applied symmetrically.

    Returns:
        DataFrame with index matching close (excluding last `horizon` bars),
        containing:
            - 'ret': Return from current bar to bar at horizon
            - 'bin': Label (-1, 0, or 1) based on threshold comparison

    Reference:
        AFML Chapter 3, Section 3.2 (mathematical definition)

    Example:
        >>> prices = pd.Series([100, 101, 99, 102, 104],
        ...                    index=pd.date_range('2020-01-01', periods=5))
        >>> labels = fixed_time_horizon(prices, horizon=2, threshold=0.02)
        >>> # Labels based on 2-bar return vs 2% threshold
    """
    # Calculate returns over fixed horizon
    ret = close.pct_change(periods=horizon)

    # Create output dataframe (excluding NaN values from the end)
    out = pd.DataFrame(index=close.index[:-horizon])
    out["ret"] = ret.iloc[horizon:]

    # Apply threshold to generate labels
    out["bin"] = np.where(
        out["ret"] > threshold,
        1,  # Significant upward move
        np.where(
            out["ret"] < -threshold,
            -1,  # Significant downward move
            0,  # Insignificant move
        ),
    )

    return out
