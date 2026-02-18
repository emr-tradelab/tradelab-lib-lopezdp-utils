"""Futures contract utilities for roll adjustment and continuous price series.

Reference: Advances in Financial Machine Learning, Chapter 2, Section 2.5.2.2
"""

import polars as pl


def roll_gaps(
    df: pl.DataFrame,
    instrument_col: str = "instrument",
    open_col: str = "open",
    close_col: str = "close",
    match_end: bool = True,
) -> pl.Series:
    """Compute cumulative price gaps at futures contract roll dates.

    Identifies roll dates (where the instrument identifier changes) and computes
    the price gap between the previous contract's close and the new contract's
    open. Gaps are cumulated to form a continuous adjustment series.

    Args:
        df: DataFrame with columns for instrument, open, and close prices.
            Must be sorted by timestamp.
        instrument_col: Column name for the contract identifier.
        open_col: Column name for open prices.
        close_col: Column name for close prices.
        match_end: If True, the rolled series end matches the raw series end
            (backward roll). If False, forward roll.

    Returns:
        Polars Series of cumulative gaps aligned with the input DataFrame index.
        Subtract from raw prices to create a continuous price series.

    Reference:
        AFML, Chapter 2, Snippet 2.2
    """
    n = len(df)
    gaps = [0.0] * n

    instruments = df[instrument_col].to_list()
    opens = df[open_col].to_list()
    closes = df[close_col].to_list()

    # Find roll dates (first occurrence of each unique instrument)
    seen: set = set()
    roll_indices: list[int] = []
    for i, inst in enumerate(instruments):
        if inst not in seen:
            seen.add(inst)
            if i > 0:
                roll_indices.append(i)

    # Compute gap at each roll: new_open - previous_close
    for i in roll_indices:
        gaps[i] = opens[i] - closes[i - 1]

    # Cumulative sum
    cum = 0.0
    for i in range(n):
        cum += gaps[i]
        gaps[i] = cum

    if match_end:
        last = gaps[-1]
        gaps = [g - last for g in gaps]

    return pl.Series("gap", gaps, dtype=pl.Float64)


def roll_and_rebase(
    df: pl.DataFrame,
    instrument_col: str = "instrument",
    open_col: str = "open",
    close_col: str = "close",
) -> pl.DataFrame:
    """Create a non-negative rolled futures price series using total-return indexing.

    Computes roll-adjusted prices and builds a cumulative product total-return
    index from $1 initial investment. Ensures strictly positive prices.

    Args:
        df: DataFrame with instrument, open, and close columns sorted by timestamp.
        instrument_col: Column name for the contract identifier.
        open_col: Column name for open prices.
        close_col: Column name for close prices.

    Returns:
        DataFrame with additional columns:
        - ``rolled_open``, ``rolled_close``: Gap-adjusted prices.
        - ``returns``: Daily returns from rolled close prices.
        - ``r_prices``: Rebased total-return index.

    Reference:
        AFML, Chapter 2, Snippet 2.3
    """
    gaps = roll_gaps(df, instrument_col=instrument_col, open_col=open_col, close_col=close_col)

    rolled = df.with_columns([
        (pl.col(open_col) - gaps).alias("rolled_open"),
        (pl.col(close_col) - gaps).alias("rolled_close"),
    ])

    # Returns: diff of rolled_close / raw_close.shift(1)
    rolled = rolled.with_columns([
        (
            pl.col("rolled_close").diff()
            / pl.col(close_col).shift(1)
        ).alias("returns"),
    ])

    # Cumulative product: (1 + returns).cumprod()
    rolled = rolled.with_columns([
        (1 + pl.col("returns")).cum_prod().alias("r_prices"),
    ])

    return rolled


def get_rolled_series(path_in: str, key: str) -> pl.DataFrame:
    """Load futures data from HDF5 and apply roll gap adjustment.

    Keeps pandas internally (no Polars HDF5 reader). Converts to Polars
    at the output boundary.

    Args:
        path_in: Path to HDF5 file containing futures data.
        key: HDF5 key to read (e.g., 'bars/ES_10k').

    Returns:
        Polars DataFrame with roll-adjusted 'close' and 'vwap' columns.

    Reference:
        AFML, Chapter 2, Snippet 2.2
    """
    import pandas as pd  # local import — pandas only needed here

    series = pd.read_hdf(path_in, key=key)
    series["Time"] = pd.to_datetime(series["Time"], format="%Y%m%d%H%M%S%f")
    series = series.set_index("Time")

    # Convert to Polars for gap computation
    pl_series = pl.from_pandas(series.reset_index())
    gaps = roll_gaps(pl_series).to_numpy()

    for fld in ["Close", "VWAP"]:
        if fld in series.columns:
            series[fld] -= gaps

    return pl.from_pandas(series.reset_index())
