"""Sample weighting utilities for financial ML.

Merges concurrency.py, sequential_bootstrap.py, and return_attribution.py
from the original sample_weights/ package into a single Polars-I/O module.

The core insight: overlapping labels violate IID assumptions. Sample weights
correct for this by downweighting observations that share information with
many other concurrent labels.

Reference: AFML Chapter 4
"""

from __future__ import annotations

import numpy as np
import polars as pl


def _validate_t1(events: pl.DataFrame) -> None:
    """Validate t1 column exists and has no nulls."""
    if "t1" not in events.columns:
        raise ValueError("events DataFrame must contain 't1' column")
    if events["t1"].null_count() > 0:
        raise ValueError("t1 column must not contain null values — use fill_null first")


def mp_num_co_events(
    close_idx: pl.Series,
    t1: pl.DataFrame,
) -> pl.DataFrame:
    """Compute the number of concurrent events per bar.

    Two labels are concurrent if they both depend on at least one common return.
    Counts how many label events are "active" (overlapping) at each price bar.

    This is the foundational step for identifying informational redundancy in
    financial ML, where overlapping labels violate the IID assumption.

    Args:
        close_idx: Series of price bar timestamps (Datetime).
        t1: DataFrame with 'timestamp' (event start) and 't1' (event end) columns.

    Returns:
        DataFrame with 'timestamp' and 'num_co_events' columns covering the
        range from the first to last active event.

    Reference:
        Snippet 4.1, AFML Chapter 4
    """
    _validate_t1(t1)

    close_ts = close_idx.cast(pl.Int64).to_numpy()
    t0_arr = t1["timestamp"].cast(pl.Int64).to_numpy()
    t1_arr = t1["t1"].cast(pl.Int64).to_numpy()

    # Fill unclosed events with last bar
    last_bar = int(close_ts[-1])
    t1_arr = np.where(t1_arr == 0, last_bar, t1_arr)

    # Determine range to compute over
    range_start = int(t0_arr.min())
    range_end = int(t1_arr.max())

    start_idx = int(np.searchsorted(close_ts, range_start, side="left"))
    end_idx = int(np.searchsorted(close_ts, range_end, side="right"))
    end_idx = min(end_idx, len(close_ts) - 1)

    bar_ts_range = close_ts[start_idx : end_idx + 1]
    count = np.zeros(len(bar_ts_range), dtype=np.float64)

    for i in range(len(t0_arr)):
        t_in = int(t0_arr[i])
        t_out = int(t1_arr[i])
        # Find bar indices that fall within [t_in, t_out]
        lo = int(np.searchsorted(bar_ts_range, t_in, side="left"))
        hi = int(np.searchsorted(bar_ts_range, t_out, side="right"))
        count[lo:hi] += 1.0

    return pl.DataFrame(
        {
            "timestamp": pl.Series(bar_ts_range, dtype=pl.Int64).cast(pl.Datetime("us")),
            "num_co_events": count,
        }
    )


def mp_sample_tw(
    t1: pl.DataFrame,
    num_co_events: pl.DataFrame,
) -> pl.DataFrame:
    """Compute average uniqueness of each label over its lifespan.

    Uniqueness measures how independent a label is from other concurrent labels.
    High overlap → low uniqueness → low weight.

    Args:
        t1: DataFrame with 'timestamp' and 't1' columns.
        num_co_events: Output of mp_num_co_events with 'timestamp' and
            'num_co_events' columns.

    Returns:
        DataFrame with 'timestamp' and 'uniqueness' columns (values in (0, 1]).

    Reference:
        Snippet 4.2, AFML Chapter 4
    """
    _validate_t1(t1)

    co_ts = num_co_events["timestamp"].cast(pl.Int64).to_numpy()
    co_counts = num_co_events["num_co_events"].to_numpy()

    t0_arr = t1["timestamp"].cast(pl.Int64).to_numpy()
    t1_arr = t1["t1"].cast(pl.Int64).to_numpy()

    uniqueness = np.zeros(len(t1), dtype=np.float64)

    for i in range(len(t1)):
        t_in = int(t0_arr[i])
        t_out = int(t1_arr[i])

        lo = int(np.searchsorted(co_ts, t_in, side="left"))
        hi = int(np.searchsorted(co_ts, t_out, side="right"))
        window_counts = co_counts[lo:hi]

        if len(window_counts) == 0 or window_counts.sum() == 0:
            uniqueness[i] = 0.0
        else:
            uniqueness[i] = float(np.mean(1.0 / window_counts[window_counts > 0]))

    return pl.DataFrame(
        {
            "timestamp": t1["timestamp"],
            "uniqueness": uniqueness,
        }
    )


def get_ind_matrix(bar_idx: pl.Series, t1: pl.DataFrame) -> np.ndarray:
    """Build binary indicator matrix showing which bars each event spans.

    Args:
        bar_idx: Series of all price bar timestamps (Datetime).
        t1: DataFrame with 'timestamp' and 't1' columns.

    Returns:
        NumPy array of shape (num_bars, num_events) where entry [t, i] = 1
        if event i is active at bar t.

    Reference:
        Snippet 4.3, AFML Chapter 4
    """
    bar_ts = bar_idx.cast(pl.Int64).to_numpy()
    t0_arr = t1["timestamp"].cast(pl.Int64).to_numpy()
    t1_arr = t1["t1"].cast(pl.Int64).to_numpy()

    n_bars = len(bar_ts)
    n_events = len(t1)
    ind_m = np.zeros((n_bars, n_events), dtype=np.float64)

    for i in range(n_events):
        lo = int(np.searchsorted(bar_ts, int(t0_arr[i]), side="left"))
        hi = int(np.searchsorted(bar_ts, int(t1_arr[i]), side="right"))
        ind_m[lo:hi, i] = 1.0

    return ind_m


def get_avg_uniqueness(ind_m: np.ndarray) -> np.ndarray:
    """Compute average uniqueness of each observation from an indicator matrix.

    Args:
        ind_m: Indicator matrix of shape (num_bars, num_observations).

    Returns:
        1D NumPy array of average uniqueness scores, one per observation.
        Values in (0, 1]: 1.0 = completely unique, lower = more redundant.

    Reference:
        Snippet 4.4, AFML Chapter 4
    """
    # Concurrency: number of active events per bar
    c = ind_m.sum(axis=1)  # shape (num_bars,)
    # Avoid division by zero
    c_safe = np.where(c > 0, c, 1.0)
    # Uniqueness per bar per event
    u = ind_m / c_safe[:, np.newaxis]
    # Average uniqueness per event (only where event is active)
    avg_u = np.where(ind_m.sum(axis=0) > 0, u.sum(axis=0) / ind_m.sum(axis=0), 0.0)
    return avg_u


def seq_bootstrap(ind_m: np.ndarray, s_length: int | None = None) -> list:
    """Generate a sample via sequential bootstrap.

    Unlike standard bootstrap, adjusts selection probabilities after each draw
    based on uniqueness of candidates, producing a more IID-like training set.

    # TODO(numba): evaluate JIT for sequential bootstrap probability update

    Args:
        ind_m: Indicator matrix from get_ind_matrix.
        s_length: Number of draws. Defaults to ind_m.shape[1].

    Returns:
        List of sampled observation indices (can have duplicates).

    Reference:
        Snippet 4.5, AFML Chapter 4
    """
    if s_length is None:
        s_length = ind_m.shape[1]

    phi: list[int] = []
    n_events = ind_m.shape[1]

    while len(phi) < s_length:
        avg_u = np.zeros(n_events, dtype=np.float64)

        for i in range(n_events):
            # Compute uniqueness of candidate i given current selection
            cols = [*phi, i]
            sub = ind_m[:, cols]
            avg_u[i] = get_avg_uniqueness(sub)[-1]

        # Convert to probability (normalize)
        total = avg_u.sum()
        if total == 0:
            prob = np.ones(n_events) / n_events
        else:
            prob = avg_u / total

        phi.append(int(np.random.choice(n_events, p=prob)))

    return phi


def mp_sample_w(
    t1: pl.DataFrame,
    num_co_events: pl.DataFrame,
    close: pl.DataFrame,
) -> pl.DataFrame:
    """Weight samples by attributed absolute log-returns adjusted for concurrency.

    Observations associated with large price movements receive higher weight.

    Args:
        t1: DataFrame with 'timestamp' and 't1' columns.
        num_co_events: Output of mp_num_co_events.
        close: DataFrame with 'timestamp' and 'close' columns.

    Returns:
        DataFrame with 'timestamp' and 'weight' columns (absolute attributed returns).

    Reference:
        Snippet 4.10, AFML Chapter 4
    """
    _validate_t1(t1)

    # Compute log-returns
    log_ret = close.with_columns(pl.col("close").log(base=np.e).diff().alias("log_ret")).select(
        ["timestamp", "log_ret"]
    )

    ret_ts = log_ret["timestamp"].cast(pl.Int64).to_numpy()
    ret_arr = log_ret["log_ret"].to_numpy()
    co_ts = num_co_events["timestamp"].cast(pl.Int64).to_numpy()
    co_arr = num_co_events["num_co_events"].to_numpy()

    t0_arr = t1["timestamp"].cast(pl.Int64).to_numpy()
    t1_arr = t1["t1"].cast(pl.Int64).to_numpy()

    weights = np.zeros(len(t1), dtype=np.float64)

    for i in range(len(t1)):
        t_in = int(t0_arr[i])
        t_out = int(t1_arr[i])

        # Get returns over window
        lo_r = int(np.searchsorted(ret_ts, t_in, side="left"))
        hi_r = int(np.searchsorted(ret_ts, t_out, side="right"))
        window_ret = ret_arr[lo_r:hi_r]

        # Get co-events over same window
        lo_c = int(np.searchsorted(co_ts, t_in, side="left"))
        hi_c = int(np.searchsorted(co_ts, t_out, side="right"))
        window_co = co_arr[lo_c:hi_c]

        n = min(len(window_ret), len(window_co))
        if n > 0 and window_co[:n].sum() > 0:
            safe_co = np.where(window_co[:n] > 0, window_co[:n], 1.0)
            weights[i] = abs((window_ret[:n] / safe_co).sum())

    return pl.DataFrame(
        {
            "timestamp": t1["timestamp"],
            "weight": weights,
        }
    )


def get_time_decay(
    tw: pl.DataFrame,
    clf_last_w: float = 1.0,
) -> pl.DataFrame:
    """Apply piecewise-linear decay based on cumulative uniqueness.

    Older observations receive lower weight. Uses cumulative uniqueness rather
    than chronological time to avoid over-penalizing high-redundancy periods.

    Args:
        tw: DataFrame with 'timestamp' and 'uniqueness' columns
            (output of mp_sample_tw).
        clf_last_w: Decay parameter c ∈ (-1, 1]:
            1.0 → no decay (all weights = 1)
            0 < c < 1 → linear decay (oldest gets weight = c)
            c ≤ 0 → oldest observations zeroed out

    Returns:
        DataFrame with 'timestamp' and 'weight' columns.

    Reference:
        Snippet 4.11, AFML Chapter 4
    """
    # Sort by timestamp to ensure cumsum is chronological
    tw_sorted = tw.sort("timestamp")

    cum_u = tw_sorted["uniqueness"].cum_sum()
    total = float(cum_u[-1])

    if total == 0:
        return tw_sorted.with_columns(pl.lit(1.0).alias("weight")).select(["timestamp", "weight"])

    # Compute slope
    if clf_last_w >= 0:
        slope = (1.0 - clf_last_w) / total
    else:
        slope = 1.0 / ((clf_last_w + 1) * total)

    const = 1.0 - slope * total

    weights = (const + slope * cum_u).clip(lower_bound=0.0)

    return tw_sorted.with_columns(weights.alias("weight")).select(["timestamp", "weight"])
