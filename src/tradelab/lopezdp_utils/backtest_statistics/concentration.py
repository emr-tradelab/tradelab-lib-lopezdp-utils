"""Return concentration metrics from AFML Chapter 14.

Computes the Herfindahl-Hirschman Index (HHI) for assessing whether
strategy performance is driven by a few outlier bets or distributed
evenly. Applied separately to positive returns, negative returns,
and temporal distribution of bets.

Reference: AFML Snippet 14.3
"""

import numpy as np
import pandas as pd


def get_hhi(bet_ret: pd.Series) -> float:
    """Compute normalized Herfindahl-Hirschman Index for return concentration.

    HHI measures whether returns are concentrated in a few bets (close to 1)
    or distributed evenly (close to 0). Should be computed separately for
    positive returns, negative returns, and temporal bet counts.

    Args:
        bet_ret: Series of returns from bets (or bet counts for temporal HHI).

    Returns:
        Normalized HHI between 0 (diversified) and 1 (concentrated).
        NaN if fewer than 3 observations.

    Reference:
        AFML Snippet 14.3
    """
    if bet_ret.shape[0] <= 2:
        return np.nan

    wght = bet_ret / bet_ret.sum()
    hhi = (wght**2).sum()
    hhi = (hhi - bet_ret.shape[0] ** -1) / (1.0 - bet_ret.shape[0] ** -1)
    return hhi
