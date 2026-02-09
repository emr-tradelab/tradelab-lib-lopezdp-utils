"""Triple-barrier labeling method for financial machine learning.

This module implements the core triple-barrier method, which labels observations
based on the first of three barriers touched: upper horizontal (profit-taking),
lower horizontal (stop-loss), or vertical (time expiration).

Reference: AFML Chapter 3, Sections 3.4-3.5
"""

import numpy as np
import pandas as pd


def apply_pt_sl_on_t1(
    close: pd.Series, events: pd.DataFrame, pt_sl: list[float], molecule: pd.DatetimeIndex
) -> pd.DataFrame:
    """Apply stop-loss/profit-taking barriers on a subset of events.

    This function evaluates the price path for each event to determine which
    barrier—profit-taking (pt), stop-loss (sl), or vertical (t1)—is touched
    first along the path.

    For each event, it:
    1. Calculates target profit-taking and stop-loss price levels
    2. Slices the price path from event start to vertical barrier
    3. Computes returns along the path (adjusted for position side)
    4. Identifies earliest timestamps where barriers are breached

    Args:
        close: Series of prices used to evaluate the security's path.
        events: DataFrame with columns:
            - 't1': Timestamp of vertical barrier (expiration)
            - 'trgt': Width of horizontal barriers (typically volatility estimate)
            - 'side': Position side (1 for long, -1 for short)
        pt_sl: List of two non-negative floats:
            - pt_sl[0]: Multiplier for profit-taking barrier
            - pt_sl[1]: Multiplier for stop-loss barrier
        molecule: Subset of event indices to process (allows parallel execution).

    Returns:
        DataFrame with columns 't1', 'sl', 'pt' containing timestamps of
        first touch for each barrier type.

    Reference:
        Snippet 3.2 in AFML Chapter 3, Section 3.4
    """
    # Filter events to process only the specified molecule
    events_ = events.loc[molecule]
    out = events_[["t1"]].copy(deep=True)

    # Set up profit-taking barrier
    if pt_sl[0] > 0:
        pt = pt_sl[0] * events_["trgt"]
    else:
        pt = pd.Series(index=events_.index)  # NaNs (disabled)

    # Set up stop-loss barrier
    if pt_sl[1] > 0:
        sl = -pt_sl[1] * events_["trgt"]
    else:
        sl = pd.Series(index=events_.index)  # NaNs (disabled)

    # Evaluate path for each event
    for loc, t1 in events_["t1"].fillna(close.index[-1]).items():
        # Get price path from event start to vertical barrier
        df0 = close[loc:t1]

        # Calculate path returns, adjusted for position side
        df0 = (df0 / close[loc] - 1) * events_.at[loc, "side"]

        # Find earliest stop-loss touch
        out.loc[loc, "sl"] = df0[df0 < sl[loc]].index.min()

        # Find earliest profit-taking touch
        out.loc[loc, "pt"] = df0[df0 > pt[loc]].index.min()

    return out


def get_events(
    close: pd.Series,
    t_events: pd.DatetimeIndex,
    pt_sl: float,
    trgt: pd.Series,
    min_ret: float,
    num_threads: int = 1,
    t1: pd.Series | bool = False,
) -> pd.DataFrame:
    """Identify the time of first barrier touch for triple-barrier method.

    This is the core engine of the triple-barrier labeling method. It determines
    which of three barriers—upper horizontal (profit-taking), lower horizontal
    (stop-loss), or vertical (time expiration)—is touched first for each event.

    The function:
    1. Filters events by minimum target return threshold
    2. Evaluates price paths to find first barrier touches
    3. Returns timestamps and types of barriers hit

    This enables path-dependent labeling that accounts for the security's
    trajectory during the holding period, not just the final price.

    Args:
        close: Series of prices used to track the security's path.
        t_events: DatetimeIndex of timestamps that seed each triple-barrier event.
            Typically output of CUSUM filter or other event-based sampling.
        pt_sl: Non-negative float setting width of horizontal barriers as a
            multiple of the target volatility. Applied symmetrically to both
            profit-taking and stop-loss.
        trgt: Series of targets (absolute returns), typically from volatility
            estimator like daily_volatility().
        min_ret: Minimum target return required to initiate a triple-barrier
            search. Filters out noise and insignificant events.
        num_threads: Number of threads for parallel processing. Default 1 (serial).
            Note: Full multiprocessing support via mpPandasObj (Chapter 20) will
            be integrated in production phase.
        t1: Series of vertical barrier timestamps (expiration limits), or False
            to disable vertical barriers.

    Returns:
        DataFrame with columns:
            - 't1': Timestamp of first barrier touched
            - 'trgt': Target return (volatility) for the event

    Reference:
        Snippet 3.3 in AFML Chapter 3, Section 3.5

    Example:
        >>> vol = daily_volatility(close, span=100)
        >>> events_idx = cusum_filter(close, threshold=vol.mean())
        >>> t1 = add_vertical_barrier(events_idx, close, num_days=10)
        >>> events = get_events(close, events_idx, pt_sl=1.0, trgt=vol,
        ...                     min_ret=0.01, t1=t1)
    """
    # Align target with event timestamps
    trgt = trgt.loc[t_events]

    # Filter for minimum return threshold
    trgt = trgt[trgt > min_ret]

    # Set up vertical barriers
    if t1 is False:
        t1 = pd.Series(pd.NaT, index=t_events)

    # Create events object with side (1.0 for symmetric barriers)
    side_ = pd.Series(1.0, index=trgt.index)
    events = pd.concat({"t1": t1, "trgt": trgt, "side": side_}, axis=1).dropna(subset=["trgt"])

    # Apply barriers to find first touches
    # Note: v1 uses serial processing. Full multiprocessing via mpPandasObj
    # (from Chapter 20) will be integrated in production phase.
    df0 = apply_pt_sl_on_t1(
        close=close,
        events=events,
        pt_sl=[pt_sl, pt_sl],  # Symmetric barriers
        molecule=events.index,
    )

    # Get earliest barrier touch for each event
    events["t1"] = df0.dropna(how="all").min(axis=1)

    # Clean up and return
    events = events.drop("side", axis=1)

    return events


def get_bins(events: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    """Generate labels (-1, 0, 1) from triple-barrier events.

    This function takes the output from get_events() and generates the final
    labels by evaluating the realized return at the time the first barrier
    was touched.

    The function:
    1. Aligns prices with event start and barrier touch timestamps
    2. Calculates realized returns from event start to first barrier touch
    3. Applies sign function to generate labels:
       - 1 if return is positive (profit-taking hit or positive at expiration)
       - -1 if return is negative (stop-loss hit or negative at expiration)
       - 0 if return is exactly zero (rare)

    Note: The function can be modified to return 0 specifically when the
    vertical barrier is touched first, to filter out neutral price paths.

    Args:
        events: DataFrame output from get_events(), with index as event start
            times and 't1' column containing timestamps of first barrier touches.
        close: Series of prices used to calculate returns between event start
            and barrier touch.

    Returns:
        DataFrame with index matching events, containing:
            - 'ret': Realized return from event start to first touch
            - 'bin': Label (-1, 0, or 1) based on sign of return

    Reference:
        Snippet 3.5 in AFML Chapter 3, Section 3.5

    Example:
        >>> events = get_events(close, t_events, pt_sl=1.0, trgt=vol,
        ...                     min_ret=0.01, t1=t1)
        >>> labels = get_bins(events, close)
        >>> # labels['bin'] contains -1, 0, or 1 for each event
    """
    # Drop events with no barrier touch
    events_ = events.dropna(subset=["t1"])

    # Get all unique timestamps (event starts and barrier touches)
    px = events_.index.union(events_["t1"].values).drop_duplicates()

    # Align prices to these timestamps
    px = close.reindex(px, method="bfill")

    # Calculate returns and labels
    out = pd.DataFrame(index=events_.index)
    out["ret"] = px.loc[events_["t1"].values].values / px.loc[events_.index] - 1
    out["bin"] = np.sign(out["ret"])

    return out


def triple_barrier_labels(
    close: pd.Series,
    t_events: pd.DatetimeIndex,
    pt_sl: float,
    num_days: int,
    vol_span: int = 100,
    min_ret: float = 0.0,
    num_threads: int = 1,
) -> pd.DataFrame:
    """Complete triple-barrier labeling workflow (high-level wrapper).

    This convenience function combines all the core triple-barrier utilities
    into a single workflow:
    1. Compute dynamic volatility thresholds (daily_volatility)
    2. Add vertical time barriers (add_vertical_barrier)
    3. Find first barrier touches (get_events)
    4. Generate final labels (get_bins)

    This implements the full triple-barrier method, which addresses the
    shortcomings of fixed-time horizon labeling by:
    - Using dynamic thresholds based on volatility (heteroscedasticity)
    - Accounting for the price path during holding period (path-dependency)
    - Combining profit-taking, stop-loss, and time expiration barriers

    Args:
        close: Series of closing prices indexed by timestamp.
        t_events: DatetimeIndex of timestamps that seed each triple-barrier event.
            Typically output of CUSUM filter or other event-based sampling.
        pt_sl: Non-negative float setting width of horizontal barriers as a
            multiple of target volatility. Applied symmetrically.
        num_days: Number of days for vertical barrier (time expiration).
        vol_span: Number of days for volatility EWMA calculation. Default 100.
        min_ret: Minimum target return to initiate barrier search. Default 0.0.
        num_threads: Number of threads for parallel processing. Default 1.

    Returns:
        DataFrame with index matching successful events, containing:
            - 'ret': Realized return at first barrier touch
            - 'bin': Label (-1, 0, or 1) based on sign of return
            - 't1': Timestamp of first barrier touched
            - 'trgt': Target volatility for the event

    Example:
        >>> from tradelab.lopezdp_utils.data_structures import get_t_events
        >>> events_idx = get_t_events(close, threshold=0.01)  # CUSUM filter
        >>> labels = triple_barrier_labels(
        ...     close=close,
        ...     t_events=events_idx,
        ...     pt_sl=1.0,
        ...     num_days=10,
        ...     vol_span=100,
        ...     min_ret=0.005
        ... )
    """
    from .barriers import add_vertical_barrier
    from .thresholds import daily_volatility

    # Step 1: Compute dynamic volatility thresholds
    trgt = daily_volatility(close, span=vol_span)

    # Step 2: Add vertical barriers
    t1 = add_vertical_barrier(t_events, close, num_days)

    # Step 3: Find first barrier touches
    events = get_events(
        close=close,
        t_events=t_events,
        pt_sl=pt_sl,
        trgt=trgt,
        min_ret=min_ret,
        num_threads=num_threads,
        t1=t1,
    )

    # Step 4: Generate final labels
    labels = get_bins(events, close)

    # Merge events info with labels
    labels = labels.join(events[["t1", "trgt"]], how="left")

    return labels
