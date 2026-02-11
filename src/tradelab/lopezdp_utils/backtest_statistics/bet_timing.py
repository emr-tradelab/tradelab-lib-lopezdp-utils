"""Bet timing and holding period estimation from AFML Chapter 14.

Derives independent bet boundaries and average holding periods from
a series of target positions, accounting for scaling in/out, flattening,
and position flips.

Reference: AFML Snippets 14.1, 14.2
"""

import numpy as np
import pandas as pd


def get_bet_timing(t_pos: pd.Series) -> pd.DatetimeIndex:
    """Derive timestamps of independent bets from a target position series.

    Identifies when positions flatten (go to 0) or flip (reverse direction),
    marking the end of each independent bet. This prevents overestimation of
    bet frequency through raw trade counts.

    Args:
        t_pos: Target positions indexed by timestamps.

    Returns:
        Sorted DatetimeIndex of bet-ending timestamps.

    Reference:
        AFML Snippet 14.1
    """
    # Flattening: current position is 0, previous was non-zero
    df0 = t_pos[t_pos == 0].index
    df1 = t_pos.shift(1)
    df1 = df1[df1 != 0].index
    bets = df0.intersection(df1)

    # Flips: product of current and previous position is negative
    df0 = t_pos.iloc[1:] * t_pos.iloc[:-1].values
    bets = bets.union(df0[df0 < 0].index).sort_values()

    # Ensure the last timestamp is included
    if t_pos.index[-1] not in bets:
        bets = bets.append(t_pos.index[-1:])

    return bets


def get_holding_period(t_pos: pd.Series) -> float:
    """Estimate average holding period using weighted average entry time.

    Accounts for complex scaling in/out scenarios by tracking a
    weighted average entry time that updates as positions increase,
    decrease, or flip.

    Args:
        t_pos: Target positions with DatetimeIndex.

    Returns:
        Average holding period in days. NaN if no completed trades.

    Reference:
        AFML Snippet 14.2
    """
    hp = pd.DataFrame(columns=["dT", "w"])
    t_entry = 0.0
    p_diff = t_pos.diff()
    t_diff = (t_pos.index - t_pos.index[0]) / np.timedelta64(1, "D")

    for i in range(1, t_pos.shape[0]):
        if p_diff.iloc[i] * t_pos.iloc[i - 1] >= 0:  # Increased or unchanged
            if t_pos.iloc[i] != 0:
                t_entry = (t_entry * t_pos.iloc[i - 1] + t_diff[i] * p_diff.iloc[i]) / t_pos.iloc[i]
        else:  # Decreased position
            if t_pos.iloc[i] * t_pos.iloc[i - 1] < 0:  # Flip
                hp.loc[t_pos.index[i], ["dT", "w"]] = (
                    t_diff[i] - t_entry,
                    abs(t_pos.iloc[i - 1]),
                )
                t_entry = t_diff[i]
            else:  # Scaled out
                hp.loc[t_pos.index[i], ["dT", "w"]] = (
                    t_diff[i] - t_entry,
                    abs(p_diff.iloc[i]),
                )

    if hp["w"].sum() > 0:
        hp = (hp["dT"] * hp["w"]).sum() / hp["w"].sum()
    else:
        hp = np.nan

    return hp
