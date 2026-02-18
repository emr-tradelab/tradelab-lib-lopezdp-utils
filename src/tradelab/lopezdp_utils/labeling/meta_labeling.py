"""Meta-labeling extensions for the triple-barrier method.

Meta-labeling is a secondary ML layer that learns how to use a primary model.
The primary model decides the side (long/short), while the secondary model
determines whether to act (binary 0/1 labels).

This enables:
- Primary models to focus on recall (find all opportunities)
- Meta-models to focus on precision (filter false positives)
- Integration of discretionary/fundamental side views with ML sizing

Reference: AFML Chapter 3, Section 3.6
"""

from __future__ import annotations

import numpy as np
import polars as pl

from .triple_barrier import apply_pt_sl_on_t1, _validate_close, _validate_t1


def get_events_meta(
    close: pl.DataFrame,
    t_events: pl.Series,
    pt_sl: list[float] | float,
    trgt: pl.DataFrame,
    min_ret: float,
    t1: pl.DataFrame | None = None,
    side: pl.Series | None = None,
) -> pl.DataFrame:
    """Extended get_events with meta-labeling support (asymmetric barriers).

    When `side` is None, symmetric barriers are used (same as get_events).
    When `side` is provided, barriers are asymmetric and the 'side' column
    is preserved in the output for use by get_bins_meta.

    Args:
        close: DataFrame with 'timestamp' and 'close' columns.
        t_events: Series of event timestamps.
        pt_sl: Symmetric float when side=None, or [profit_taking, stop_loss]
            list when side is provided.
        trgt: DataFrame with 'timestamp' and 'volatility' columns.
        min_ret: Minimum volatility threshold to include an event.
        t1: Optional vertical barrier DataFrame with 'timestamp' and 't1' columns.
        side: Optional Series with primary model side predictions (1 or -1).

    Returns:
        DataFrame with 'timestamp', 't1', 'trgt', and optionally 'side' columns.
        t1 is guaranteed to have no nulls.

    Reference:
        Snippet 3.6, AFML Chapter 3
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

    # Add vertical barrier
    if t1 is not None:
        events = events.join(t1.rename({"t1": "t1_vert"}), on="timestamp", how="left")
        last_ts = close["timestamp"][-1]
        events = events.with_columns(pl.col("t1_vert").fill_null(last_ts))
    else:
        last_ts = close["timestamp"][-1]
        events = events.with_columns(pl.lit(last_ts).alias("t1_vert"))

    # Configure side and pt_sl
    if side is None:
        pt_sl_list = [float(pt_sl), float(pt_sl)]
        side_col = pl.lit(1.0)
    else:
        pt_sl_list = [float(pt_sl[0]), float(pt_sl[1])]
        # Align side with filtered events
        side_df = pl.DataFrame({"timestamp": t_events, "side": side})
        events = events.join(side_df, on="timestamp", how="left")
        side_col = pl.col("side")

    if side is None:
        events = events.with_columns(pl.lit(1.0).alias("side"))

    # Rename t1_vert to t1 for apply_pt_sl_on_t1
    events_for_loop = events.rename({"t1_vert": "t1"})

    # Extract NumPy arrays
    close_np = close["close"].to_numpy()
    ts_np = close["timestamp"].cast(pl.Int64).to_numpy()

    # Run barrier detection
    barrier_results = apply_pt_sl_on_t1(
        close_np=close_np,
        ts_np=ts_np,
        events_df=events_for_loop,
        pt_sl=pt_sl_list,
    )

    # Find earliest barrier touch
    barrier_results = barrier_results.with_columns(
        pl.min_horizontal(
            pl.col("t1_barrier"),
            pl.col("sl").fill_null(pl.col("t1_barrier")),
            pl.col("pt").fill_null(pl.col("t1_barrier")),
        ).alias("t1_first")
    )

    # Join back
    result = events.join(
        barrier_results.select(["timestamp", "t1_first"]),
        on="timestamp",
        how="left",
    )

    # Ensure t1 has no nulls
    last_ts = close["timestamp"][-1]
    result = result.with_columns(pl.col("t1_first").fill_null(last_ts).alias("t1"))

    # Build output columns
    keep_cols = ["timestamp", "t1", "trgt"]
    if side is not None:
        keep_cols.append("side")

    return result.select(keep_cols)


def get_bins_meta(events: pl.DataFrame, close: pl.DataFrame) -> pl.DataFrame:
    """Generate labels from triple-barrier events with meta-labeling support.

    Standard mode (no 'side' column): labels are {-1, 0, 1} based on return sign.
    Meta-labeling mode ('side' column present): labels are {0, 1}, where 1 means
    the primary model's predicted direction was profitable.

    Args:
        events: DataFrame with 'timestamp', 't1', and optionally 'side' columns.
        close: DataFrame with 'timestamp' and 'close' columns.

    Returns:
        DataFrame with 'timestamp', 'ret', 'label' columns.

    Reference:
        Snippet 3.7, AFML Chapter 3
    """
    _validate_t1(events)

    is_meta = "side" in events.columns

    # Join close prices at t0 and t1
    result = (
        events.join(close.rename({"close": "close_t0"}), on="timestamp", how="left")
        .join(
            close.rename({"timestamp": "t1", "close": "close_t1"}),
            on="t1",
            how="left",
        )
        .with_columns(
            (pl.col("close_t1") / pl.col("close_t0") - 1).alias("ret")
        )
    )

    if is_meta:
        # Multiply return by side to get PnL perspective
        result = result.with_columns(
            (pl.col("ret") * pl.col("side")).alias("ret")
        )
        # Binary label: 1 if trade was profitable, 0 otherwise
        result = result.with_columns(
            pl.when(pl.col("ret") > 0)
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .cast(pl.Int8)
            .alias("label")
        )
    else:
        result = result.with_columns(
            pl.col("ret").sign().cast(pl.Int8).alias("label")
        )

    return result.select(["timestamp", "ret", "label"])
