"""
Concurrency and Uniqueness Functions

This module implements functions to measure and correct for overlapping (concurrent) labels
in financial data. Standard ML assumes IID observations, but financial labels often overlap
in time, causing informational redundancy.

The key insight: if multiple labels depend on the same price returns, each label should be
weighted by how "unique" its information is — treating each return as a finite resource
to be shared among concurrent labels.

References:
    - AFML Chapter 4, Snippets 4.1-4.2
    - Mathematical foundation: Section 4.2 (Estimating the Uniqueness of a Label)
"""

import numpy as np
import pandas as pd


def mp_num_co_events(
    close_idx: pd.DatetimeIndex,
    t1: pd.Series,
    molecule: list | np.ndarray | pd.Index,
) -> pd.Series:
    """
    Compute the number of concurrent events per bar.

    Two labels are concurrent if they both depend on at least one common return.
    This function counts how many label events are "active" (overlapping) at each
    price bar within the specified range.

    This is the foundational step for identifying informational redundancy in
    financial ML, where overlapping labels violate the IID assumption.

    Args:
        close_idx: DatetimeIndex of price bars (typically from OHLC data)
        t1: Series with index=event start times, values=event end times
            (first barrier touch). NaT values represent unclosed events.
        molecule: Subset of t1.index representing event start times to process
            (for multiprocessing support)

    Returns:
        Series indexed by price bar timestamps, where each value is the count
        of concurrent (overlapping) events at that bar

    Mathematical Logic:
        For each time point t, the concurrency count is:
            c_t = Σ(i=1 to I) I_{t,i}
        where I_{t,i} = 1 if label i's lifespan overlaps with [t-1, t]

    Example:
        >>> close_idx = pd.date_range('2020-01-01', periods=10, freq='D')
        >>> t1 = pd.Series(
        ...     index=pd.date_range('2020-01-01', periods=5, freq='D'),
        ...     data=pd.date_range('2020-01-03', periods=5, freq='D')
        ... )
        >>> molecule = t1.index[:3]
        >>> counts = mp_num_co_events(close_idx, t1, molecule)
        >>> print(counts)  # Shows how many events overlap each bar

    Reference:
        AFML Snippet 4.1
    """
    # 1) Find events that span the period [molecule, molecule[-1]]
    t1 = t1.fillna(close_idx[-1])  # unclosed events still must impact other weights
    t1 = t1[t1 >= molecule[0]]  # events that end at or after molecule start
    t1 = t1.loc[: t1[molecule].max()]  # events that start at or before last molecule end

    # 2) Count events spanning a bar
    iloc = close_idx.searchsorted(np.array([t1.index[0], t1.max()]))
    count = pd.Series(0, index=close_idx[iloc[0] : iloc[1] + 1])
    for t_in, t_out in t1.items():
        count.loc[t_in:t_out] += 1.0
    return count.loc[molecule[0] : t1[molecule].max()]


def mp_sample_tw(
    t1: pd.Series,
    num_co_events: pd.Series,
    molecule: list | np.ndarray | pd.Index,
) -> pd.Series:
    """
    Compute average uniqueness of each label over its lifespan.

    Uniqueness measures how "independent" a label is from other concurrent labels.
    If a label's lifespan overlaps with many other labels, its uniqueness is low
    because the underlying price returns are shared among many observations.

    The weight assigned to each label is the average of its point-in-time uniqueness
    scores across its entire lifespan.

    Args:
        t1: Series with index=event start times, values=event end times
        num_co_events: Output from mp_num_co_events (concurrency count per bar)
        molecule: Subset of t1.index (event start times) to process

    Returns:
        Series indexed by event start times (molecule), where each value is
        a uniqueness score between 0 and 1

    Mathematical Logic:
        Point-in-time uniqueness:
            u(t,i) = 1 / c_t
        where c_t is the number of concurrent events at time t

        Average uniqueness for label i:
            ū_i = mean(1 / c_t) over the event's lifespan [t_in, t_out]

    Interpretation:
        - Score = 1.0: Label is completely independent (no overlaps)
        - Score = 0.5: On average, label shares returns with one other label
        - Score = 0.1: Label is highly redundant (overlaps with ~9 other labels)

    The uniqueness weight corrects for non-IID violations caused by overlapping
    labels, ensuring more independent observations receive higher influence.

    Example:
        >>> t1 = pd.Series(...)  # event start -> end times
        >>> num_co_events = mp_num_co_events(close_idx, t1, molecule)
        >>> uniqueness = mp_sample_tw(t1, num_co_events, molecule)
        >>> print(uniqueness)  # Uniqueness scores for each event

    Reference:
        AFML Snippet 4.2
    """
    # Derive average uniqueness over the event's lifespan
    wght = pd.Series(index=molecule, dtype=float)
    for t_in, t_out in t1.loc[wght.index].items():
        wght.loc[t_in] = (1.0 / num_co_events.loc[t_in:t_out]).mean()
    return wght
