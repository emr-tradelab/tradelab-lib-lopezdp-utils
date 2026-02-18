"""Triple-barrier labeling method for financial machine learning.

Implements the triple-barrier method (AFML Ch. 3), fixed-time horizon labeling
(AFML Ch. 3.2), and trend-scanning labeling (MLAM Ch. 5.4) with Polars I/O.

The triple-barrier method labels observations based on the first of three barriers
touched: upper horizontal (profit-taking), lower horizontal (stop-loss), or
vertical (time expiration). This approach addresses the shortcomings of fixed-time
horizon labeling by using dynamic thresholds and accounting for the price path.

Reference: AFML Chapters 3-4; MLAM Section 5.4
"""

from __future__ import annotations

import numpy as np
import polars as pl
import statsmodels.api as sm


def _validate_close(close: pl.DataFrame) -> None:
    """Validate that close DataFrame has required columns."""
    if "timestamp" not in close.columns or "close" not in close.columns:
        raise ValueError("close DataFrame must have 'timestamp' and 'close' columns")


def _validate_t1(events: pl.DataFrame) -> None:
    """Validate t1 column exists and has no nulls."""
    if "t1" not in events.columns:
        raise ValueError("events DataFrame must contain 't1' column")
    if events["t1"].null_count() > 0:
        raise ValueError("t1 column must not contain null values")


def daily_volatility(close: pl.DataFrame, span: int = 100) -> pl.DataFrame:
    """Estimate daily volatility using EWMA of daily returns.

    Computes daily returns by finding the price bar from approximately 1 day
    (num_bars = span proxy) prior, then applies an exponentially weighted
    moving standard deviation. Used to set dynamic barrier widths in the
    triple-barrier method.

    For intrabar data (minute bars), "daily" means looking back 1440 bars;
    for daily data, it means 1 bar back. Here we use a simpler approach:
    compute 1-bar pct_change EWMA std, which approximates bar-to-bar vol.
    For the day-prior lookup the book uses searchsorted on index - 1 day;
    since we accept generic Polars DataFrames we use pct_change + ewm_std
    directly (appropriate for any bar frequency).

    Args:
        close: DataFrame with 'timestamp' (Datetime) and 'close' (Float) columns,
            sorted ascending by timestamp.
        span: EWMA span for the rolling standard deviation. Default 100.

    Returns:
        DataFrame with 'timestamp' and 'volatility' columns.

    Reference:
        Snippet 3.1, AFML Chapter 3
    """
    _validate_close(close)
    result = close.with_columns(
        pl.col("close").pct_change().ewm_std(span=span).alias("volatility")
    ).select(["timestamp", "volatility"])
    return result


def add_vertical_barrier(
    t_events: pl.Series,
    close: pl.DataFrame,
    num_bars: int,
) -> pl.DataFrame:
    """Add a vertical (time-based) barrier for the triple-barrier method.

    For each event timestamp, finds the price bar that is `num_bars` ahead,
    which serves as the maximum holding period (vertical barrier / expiration).

    Args:
        t_events: Series of event timestamps (Datetime) that seed each event.
        close: DataFrame with 'timestamp' column (sorted ascending).
        num_bars: Number of bars ahead to set the vertical barrier.

    Returns:
        DataFrame with 'timestamp' (event start) and 't1' (vertical barrier) columns.
        For events near the end of the series, t1 is capped at the last available bar.

    Reference:
        Snippet 3.4, AFML Chapter 3
    """
    _validate_close(close)
    all_ts = close["timestamp"].to_numpy()
    event_ts = t_events.to_numpy()

    # Find index of each event in close
    start_idxs = np.searchsorted(all_ts, event_ts, side="left")
    # Vertical barrier = num_bars ahead, capped at last bar
    end_idxs = np.minimum(start_idxs + num_bars, len(all_ts) - 1)

    t1_values = all_ts[end_idxs]

    return pl.DataFrame(
        {
            "timestamp": t_events,
            "t1": pl.Series(t1_values).cast(pl.Datetime("us")),
        }
    )


def fixed_time_horizon(
    close: pl.DataFrame,
    horizon: int,
    threshold: float,
) -> pl.DataFrame:
    """Label observations using the fixed-time horizon method.

    WARNING: The book recommends against this method. Ignores heteroscedasticity
    and path information. Use triple_barrier_labels() instead.

    Args:
        close: DataFrame with 'timestamp' and 'close' columns.
        horizon: Number of bars ahead to evaluate return.
        threshold: Symmetric threshold for label assignment.

    Returns:
        DataFrame with 'timestamp', 'ret', and 'label' columns.
        Labels: -1 (ret < -threshold), 0 (|ret| <= threshold), 1 (ret > threshold).

    Reference:
        AFML Chapter 3, Section 3.2
    """
    _validate_close(close)
    result = (
        close.with_columns((pl.col("close").shift(-horizon) / pl.col("close") - 1).alias("ret"))
        .head(len(close) - horizon)
        .with_columns(
            pl.when(pl.col("ret") > threshold)
            .then(pl.lit(1))
            .when(pl.col("ret") < -threshold)
            .then(pl.lit(-1))
            .otherwise(pl.lit(0))
            .alias("label")
        )
        .select(["timestamp", "ret", "label"])
    )
    return result


def apply_pt_sl_on_t1(
    close_np: np.ndarray,
    ts_np: np.ndarray,
    events_df: pl.DataFrame,
    pt_sl: list[float],
) -> pl.DataFrame:
    """Apply stop-loss/profit-taking barriers on events (path-dependent loop).

    This function stays as a Python loop because barrier detection is inherently
    path-dependent: we must scan the price path sequentially to find the first
    barrier touch for each event.

    # TODO(numba): evaluate JIT for barrier touch loop

    Args:
        close_np: NumPy array of close prices.
        ts_np: NumPy array of timestamps (int64 microseconds) matching close_np.
        events_df: Polars DataFrame with columns: 'timestamp', 't1', 'trgt', 'side'.
            'timestamp' and 't1' must be Datetime("us") columns.
        pt_sl: [profit_taking_mult, stop_loss_mult].

    Returns:
        Polars DataFrame with 'timestamp', 't1_barrier', 'sl', 'pt' columns
        (timestamps as Datetime, or null if not touched).
    """
    # Pre-convert event timestamps to int64 for fast searchsorted
    ts_int = events_df["timestamp"].cast(pl.Int64).to_numpy()
    t1_int = events_df["t1"].cast(pl.Int64).to_numpy()
    trgt_arr = events_df["trgt"].to_numpy()
    side_arr = events_df["side"].to_numpy()

    t1_barrier_out = []
    sl_out: list[int | None] = []
    pt_out: list[int | None] = []

    for i in range(len(events_df)):
        t0_us = int(ts_int[i])
        t1_us = int(t1_int[i])
        trgt = float(trgt_arr[i])
        side = float(side_arr[i])

        t0_idx = int(np.searchsorted(ts_np, t0_us, side="left"))
        t1_idx = int(np.searchsorted(ts_np, t1_us, side="right"))
        t1_idx = min(t1_idx, len(close_np) - 1)

        path = close_np[t0_idx : t1_idx + 1]
        t1_barrier_out.append(int(ts_np[t1_idx]))

        if len(path) < 2:
            sl_out.append(None)
            pt_out.append(None)
            continue

        ret_path = (path / path[0] - 1) * side
        pt_level = pt_sl[0] * trgt if pt_sl[0] > 0 else None
        sl_level = -pt_sl[1] * trgt if pt_sl[1] > 0 else None
        bar_ts = ts_np[t0_idx : t1_idx + 1]

        sl_time: int | None = None
        pt_time: int | None = None

        if sl_level is not None:
            sl_hits = np.where(ret_path < sl_level)[0]
            if len(sl_hits) > 0:
                sl_time = int(bar_ts[sl_hits[0]])

        if pt_level is not None:
            pt_hits = np.where(ret_path > pt_level)[0]
            if len(pt_hits) > 0:
                pt_time = int(bar_ts[pt_hits[0]])

        sl_out.append(sl_time)
        pt_out.append(pt_time)

    return pl.DataFrame(
        {
            "timestamp": events_df["timestamp"],
            "t1_barrier": pl.Series(t1_barrier_out, dtype=pl.Int64).cast(pl.Datetime("us")),
            "sl": pl.Series(sl_out, dtype=pl.Int64).cast(pl.Datetime("us")),
            "pt": pl.Series(pt_out, dtype=pl.Int64).cast(pl.Datetime("us")),
        }
    )


def get_events(
    close: pl.DataFrame,
    t_events: pl.Series,
    pt_sl: float,
    trgt: pl.DataFrame,
    min_ret: float,
    t1: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Find time of first barrier touch for each event (triple-barrier core).

    Args:
        close: DataFrame with 'timestamp' and 'close' columns.
        t_events: Series of event timestamps.
        pt_sl: Symmetric barrier width multiplier (applied to both pt and sl).
        trgt: DataFrame with 'timestamp' and 'volatility' columns (barrier width).
        min_ret: Minimum volatility threshold to include an event.
        t1: Optional DataFrame with 'timestamp' and 't1' columns for vertical barriers.
            If None, vertical barrier defaults to last bar in close.

    Returns:
        DataFrame with 'timestamp', 't1', 'trgt' columns. t1 is the first barrier
        touch time; trgt is the volatility target used.

    Reference:
        Snippet 3.3, AFML Chapter 3
    """
    _validate_close(close)

    # Join trgt onto t_events
    events = pl.DataFrame({"timestamp": t_events}).join(
        trgt.rename({"volatility": "trgt"}),
        on="timestamp",
        how="left",
    )

    # Filter by min_ret
    events = events.filter(pl.col("trgt") > min_ret).drop_nulls("trgt")

    # Add vertical barrier (t1 column)
    if t1 is not None:
        events = events.join(t1.rename({"t1": "t1_vert"}), on="timestamp", how="left")
        # Fill missing t1 with last bar
        last_ts = close["timestamp"][-1]
        events = events.with_columns(pl.col("t1_vert").fill_null(last_ts).alias("t1_vert"))
    else:
        last_ts = close["timestamp"][-1]
        events = events.with_columns(pl.lit(last_ts).alias("t1_vert"))

    # Add symmetric side
    events = events.with_columns(pl.lit(1.0).alias("side"))

    # Rename t1_vert to t1 for apply_pt_sl_on_t1
    events_for_loop = events.rename({"t1_vert": "t1"})

    # Extract NumPy arrays for the loop
    close_np = close["close"].to_numpy()
    ts_np = close["timestamp"].cast(pl.Int64).to_numpy()

    # Run barrier detection loop
    barrier_results = apply_pt_sl_on_t1(
        close_np=close_np,
        ts_np=ts_np,
        events_df=events_for_loop,
        pt_sl=[pt_sl, pt_sl],
    )

    # Find earliest barrier touch: min of t1_barrier, sl, pt
    barrier_results = barrier_results.with_columns(
        [
            pl.min_horizontal(
                pl.col("t1_barrier"),
                pl.col("sl").fill_null(pl.col("t1_barrier")),
                pl.col("pt").fill_null(pl.col("t1_barrier")),
            ).alias("t1_first")
        ]
    )

    # Join back to events
    result = events.join(
        barrier_results.select(["timestamp", "t1_first"]),
        on="timestamp",
        how="left",
    ).select(
        [
            "timestamp",
            pl.col("t1_first").alias("t1"),
            "trgt",
        ]
    )

    return result


def get_bins(events: pl.DataFrame, close: pl.DataFrame) -> pl.DataFrame:
    """Generate labels (-1, 0, 1) from triple-barrier events.

    Computes the realized return from event start to first barrier touch,
    then applies sign to generate the label.

    Args:
        events: DataFrame with 'timestamp' and 't1' columns (output of get_events).
        close: DataFrame with 'timestamp' and 'close' columns.

    Returns:
        DataFrame with 'timestamp', 'ret', 'label' columns.

    Reference:
        Snippet 3.5, AFML Chapter 3
    """
    _validate_t1(events)

    # Join close prices at t0 and t1
    result = (
        events.join(close.rename({"close": "close_t0"}), on="timestamp", how="left")
        .join(
            close.rename({"timestamp": "t1", "close": "close_t1"}),
            on="t1",
            how="left",
        )
        .with_columns((pl.col("close_t1") / pl.col("close_t0") - 1).alias("ret"))
        .with_columns(pl.col("ret").sign().cast(pl.Int8).alias("label"))
        .select(["timestamp", "t1", "ret", "label"])
    )
    return result


def triple_barrier_labels(
    close: pl.DataFrame,
    t_events: pl.Series,
    pt_sl: float,
    num_bars: int,
    vol_span: int = 100,
    min_ret: float = 0.0,
) -> pl.DataFrame:
    """Complete triple-barrier labeling pipeline (high-level wrapper).

    Combines volatility estimation, vertical barrier construction, barrier
    detection, and label generation into one call.

    Args:
        close: DataFrame with 'timestamp' (Datetime) and 'close' (Float) columns.
        t_events: Series of event timestamps to label.
        pt_sl: Symmetric barrier width as multiple of volatility.
        num_bars: Number of bars for vertical barrier (max holding period).
        vol_span: EWMA span for volatility. Default 100.
        min_ret: Minimum volatility to include an event. Default 0.0.

    Returns:
        DataFrame with 'timestamp', 't1', 'ret', 'label', 'trgt' columns.
        t1 is guaranteed to have no nulls.

    Reference:
        AFML Chapter 3
    """
    _validate_close(close)

    # Step 1: Compute volatility
    trgt = daily_volatility(close, span=vol_span)

    # Step 2: Add vertical barriers
    t1_df = add_vertical_barrier(t_events, close, num_bars=num_bars)

    # Step 3: Find first barrier touches
    events = get_events(
        close=close,
        t_events=t_events,
        pt_sl=pt_sl,
        trgt=trgt,
        min_ret=min_ret,
        t1=t1_df,
    )

    # Fill any remaining null t1 with last bar (safety guardrail)
    last_ts = close["timestamp"][-1]
    events = events.with_columns(pl.col("t1").fill_null(last_ts))

    # Step 4: Generate labels
    labels = get_bins(events, close)

    return labels


def t_value_linear_trend(close: np.ndarray) -> float:
    """Calculate t-value of slope coefficient in a linear time-trend model.

    Fits: price[t] = β₀ + β₁·t + ε[t]. Returns t-stat for β₁.
    Positive t-value → uptrend; negative → downtrend.

    Args:
        close: Array of prices to fit the linear trend over.

    Returns:
        t-value of the slope coefficient.

    Reference:
        Snippet 5.1, MLAM Section 5.4
    """
    x = np.ones((len(close), 2))
    x[:, 1] = np.arange(len(close))
    ols = sm.OLS(close, x).fit()
    return float(ols.tvalues[1])


def trend_scanning_labels(
    close: pl.DataFrame,
    t_events: pl.Series,
    span: range | tuple[int, int],
) -> pl.DataFrame:
    """Generate labels using the trend-scanning method.

    Scans multiple look-forward horizons for each event, fits a linear
    trend, and labels based on the horizon with the maximum absolute t-value.

    # TODO(numba): OLS per-event loop is sequential; evaluate JIT

    Args:
        close: DataFrame with 'timestamp' and 'close' columns.
        t_events: Series of event timestamps to label.
        span: Range or (start, end) tuple of look-forward horizons to test.

    Returns:
        DataFrame with 'timestamp', 't1', 't_val', 'label' columns.

    Reference:
        Snippet 5.2, MLAM Section 5.4
    """
    _validate_close(close)

    if isinstance(span, tuple):
        hrzns = range(*span)
    else:
        hrzns = span

    close_np = close["close"].to_numpy()
    ts_np = close["timestamp"].cast(pl.Int64).to_numpy()
    event_ts_int64 = t_events.cast(pl.Int64).to_numpy()
    event_ts = t_events.to_numpy()

    rows = []
    for dt0_int, dt0 in zip(event_ts_int64, event_ts):
        iloc0 = int(np.searchsorted(ts_np, dt0_int, side="left"))

        if iloc0 + max(hrzns) > len(close_np):
            continue

        best_t_val = 0.0
        best_t1 = None

        for hrzn in hrzns:
            end_idx = iloc0 + hrzn
            if end_idx >= len(close_np):
                break
            segment = close_np[iloc0 : end_idx + 1]
            t_val = t_value_linear_trend(segment)
            if np.isfinite(t_val) and abs(t_val) > abs(best_t_val):
                best_t_val = t_val
                best_t1 = ts_np[end_idx]

        if best_t1 is not None:
            rows.append(
                {
                    "timestamp": int(dt0_int),
                    "t1": int(best_t1),
                    "t_val": best_t_val,
                    "label": int(np.sign(best_t_val)),
                }
            )

    if not rows:
        return pl.DataFrame(
            schema={
                "timestamp": pl.Datetime("us"),
                "t1": pl.Datetime("us"),
                "t_val": pl.Float64,
                "label": pl.Int8,
            }
        )

    ts_series = pl.Series([r["timestamp"] for r in rows], dtype=pl.Int64)
    t1_series = pl.Series([r["t1"] for r in rows], dtype=pl.Int64)
    result = pl.DataFrame(
        {
            "timestamp": ts_series.cast(pl.Datetime("us")),
            "t1": t1_series.cast(pl.Datetime("us")),
            "t_val": [r["t_val"] for r in rows],
            "label": pl.Series([r["label"] for r in rows], dtype=pl.Int8),
        }
    )

    return result
