"""Barrier utilities for the triple-barrier labeling method.

This module provides helper functions for defining barriers in the triple-barrier
method, including vertical (time-based) expiration limits.

Reference: AFML Chapter 3, Section 3.4
"""

import pandas as pd


def add_vertical_barrier(t_events: pd.DatetimeIndex, close: pd.Series, num_days: int) -> pd.Series:
    """Add a vertical (time-based) barrier for the triple-barrier method.

    The vertical barrier represents a maximum holding period—an expiration limit
    for each seeded event. This function finds the timestamp of the price bar
    that occurs after a specified number of days from each event timestamp.

    In the triple-barrier method, if neither the profit-taking nor stop-loss
    horizontal barriers are touched within the specified time period, the position
    is forced to close when the vertical barrier is reached.

    Args:
        t_events: DatetimeIndex of timestamps that seed each triple-barrier event.
            Typically the output of a CUSUM filter or other event-based sampling method.
        close: Series of closing prices used to provide the index for valid price bars.
        num_days: Number of days until the expiration limit is reached.

    Returns:
        Series where index is the start time of each event (from t_events) and
        values are the timestamps of the corresponding vertical barrier bars.
        Events beyond the available data will have NaN values.

    Reference:
        Snippet 3.4 in AFML Chapter 3, Section 3.4

    Example:
        >>> events = pd.DatetimeIndex(['2020-01-01', '2020-01-05', '2020-01-10'])
        >>> prices = pd.Series([100, 101, 99, 102],
        ...                    index=pd.date_range('2020-01-01', periods=4))
        >>> t1 = add_vertical_barrier(events, prices, num_days=2)
        >>> # t1 contains the expiration timestamps for each event
    """
    # Find the index position of the bar occurring after t_events + num_days
    t1 = close.index.searchsorted(t_events + pd.Timedelta(days=num_days))

    # Filter out indices beyond the available data
    t1 = t1[t1 < close.shape[0]]

    # Create series mapping event timestamps to vertical barrier timestamps
    # NaNs will appear at the end for events beyond available data
    t1 = pd.Series(close.index[t1], index=t_events[: t1.shape[0]])

    return t1
