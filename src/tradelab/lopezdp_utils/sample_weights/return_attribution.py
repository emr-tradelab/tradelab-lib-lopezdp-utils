"""
Return Attribution and Time Decay Functions

This module implements sample weighting based on return attribution and time decay.
These methods ensure that:
1. Labels associated with large price moves receive more importance
2. Recent observations are prioritized over older ones
3. Weights are adjusted for concurrency (overlapping labels)

The combination of return attribution and time decay creates a weighting scheme that
respects both the magnitude of market events and the adaptive nature of financial markets.

References:
    - AFML Chapter 4, Snippets 4.10-4.11
    - Section 4.6: Time Decay
"""

import numpy as np
import pandas as pd


def mp_sample_w(
    t1: pd.Series,
    num_co_events: pd.Series,
    close: pd.Series,
    molecule: list | np.ndarray | pd.Index,
) -> pd.Series:
    """
    Weight samples by attributed absolute log-returns adjusted for concurrency.

    Observations associated with large price movements should receive more weight than
    observations with negligible returns. This function computes weights based on the
    sum of attributed log-returns over each label's lifespan, correcting for both
    magnitude and uniqueness.

    Args:
        t1: Series with index=event start times, values=event end times
        num_co_events: Concurrency count per bar (output from mp_num_co_events)
        close: Price series for computing log-returns
        molecule: Subset of t1.index (event start times) to process

    Returns:
        Series of absolute attributed return weights indexed by event start times

    Mathematical Logic:
        Attributed return for label i:
            w̃_i = Σ(t=t_in to t_out) [r_{t-1,t} / c_t]

        Where:
            r_{t-1,t} = log-return at time t
            c_t = number of concurrent labels at time t

        Final weight (absolute value):
            w_i = |w̃_i|

    Interpretation:
        - Higher weight → Label experienced larger attributed price movements
        - Lower weight → Label experienced smaller attributed price movements
        - Division by concurrency ensures highly overlapping outcomes don't
          receive disproportionate total weight

    Example:
        >>> t1 = pd.Series(...)  # event start -> end times
        >>> num_co_events = mp_num_co_events(close.index, t1, molecule)
        >>> weights = mp_sample_w(t1, num_co_events, close, molecule)
        >>> print(weights)  # Absolute attributed return weights

    Best Practice:
        When using return-based weighting, consider dropping "neutral" labels
        (returns below a threshold) to avoid giving importance to low-signal events.

    Reference:
        AFML Snippet 4.10
    """
    # Compute log-returns (additive property)
    ret = np.log(close).diff()

    # Derive sample weight by return attribution
    wght = pd.Series(index=molecule, dtype=float)
    for t_in, t_out in t1.loc[wght.index].items():
        wght.loc[t_in] = (ret.loc[t_in:t_out] / num_co_events.loc[t_in:t_out]).sum()

    return wght.abs()


def get_time_decay(tw: pd.Series, clf_last_w: float = 1.0) -> pd.Series:
    """
    Apply piecewise-linear decay to observations based on cumulative uniqueness.

    Markets are adaptive systems where older data becomes less relevant as regimes shift.
    Time decay ensures the ML algorithm prioritizes recent information while accounting
    for the non-IID nature of overlapping observations.

    Unlike traditional exponential decay based on chronological time, this method uses
    cumulative uniqueness, which prevents overly aggressive weight reduction during
    high-redundancy periods.

    Args:
        tw: Series of average uniqueness values (ū_i) indexed by observation start time
        clf_last_w: Decay parameter c ∈ (-1, 1] controlling decay behavior.
            - c = 1.0: No decay (all weights = 1)
            - 0 < c < 1: Linear decay (oldest obs gets weight = c)
            - c = 0: Linear decay to zero
            - c < 0: Oldest observations get weight = 0 (erased)

    Returns:
        Series of time-decay factors indexed by observation start time

    Mathematical Logic:
        Piecewise-linear decay function:
            d = max{0, a + b·x}

        Where:
            x = cumulative uniqueness: x ∈ [0, Σ ū_i]
            a (const) = 1 - b · Σ ū_i
            b (slope) = depends on clf_last_w

        Slope calculation:
            If clf_last_w ≥ 0:
                b = (1 - clf_last_w) / Σ ū_i
            If clf_last_w < 0:
                b = 1 / [(clf_last_w + 1) · Σ ū_i]

    Boundary Conditions:
        - Newest observation always receives weight = 1
        - Weights are clipped at 0 (no negative weights)

    Example:
        >>> uniqueness = mp_sample_tw(t1, num_co_events, molecule)
        >>> decay_weights = get_time_decay(uniqueness, clf_last_w=0.5)
        >>> final_weights = uniqueness * decay_weights  # Combined weighting

    Reference:
        AFML Snippet 4.11
    """
    # Compute cumulative uniqueness (sorted by time)
    clf_w = tw.sort_index().cumsum()

    # Calculate slope based on clf_last_w parameter
    if clf_last_w >= 0:
        slope = (1.0 - clf_last_w) / clf_w.iloc[-1]
    else:
        slope = 1.0 / ((clf_last_w + 1) * clf_w.iloc[-1])

    # Calculate intercept (newest observation gets weight = 1)
    const = 1.0 - slope * clf_w.iloc[-1]

    # Apply linear decay: d = const + slope * cumulative_uniqueness
    clf_w = const + slope * clf_w

    # Floor at zero (no negative weights)
    clf_w[clf_w < 0] = 0

    return clf_w
