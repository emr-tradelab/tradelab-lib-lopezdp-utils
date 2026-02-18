"""Standard and information-driven bar construction from tick data.

Reference: Advances in Financial Machine Learning, Chapter 2
"""

import numpy as np
import polars as pl


def _aggregate_bar(ticks: list[dict]) -> dict:
    """Aggregate a list of ticks into OHLCV bar data.

    Args:
        ticks: List of tick dicts with 'timestamp', 'price', 'volume' keys.

    Returns:
        Dictionary with aggregated OHLCV + vwap data.
    """
    if not ticks:
        return {}

    prices = [t["price"] for t in ticks]
    volumes = [t["volume"] for t in ticks]
    total_volume = sum(volumes)
    vwap = (
        sum(t["price"] * t["volume"] for t in ticks) / total_volume
        if total_volume > 0
        else 0.0
    )

    return {
        "timestamp": ticks[-1]["timestamp"],
        "open": ticks[0]["price"],
        "high": max(prices),
        "low": min(prices),
        "close": ticks[-1]["price"],
        "volume": total_volume,
        "vwap": vwap,
    }


def _compute_tick_rule(prices: np.ndarray) -> np.ndarray:
    """Compute tick rule sign for each price.

    Args:
        prices: Array of prices.

    Returns:
        Array of +1 / -1 signals (forward-filled where diff == 0).
    """
    diffs = np.diff(prices, prepend=prices[0])
    signs = np.sign(diffs)

    # Forward-fill zeros
    last = 1
    for i in range(len(signs)):
        if signs[i] == 0:
            signs[i] = last
        else:
            last = signs[i]

    return signs.astype(np.float64)


def time_bars(df: pl.DataFrame, frequency: str) -> pl.DataFrame:
    """Sample bars at fixed time intervals.

    Args:
        df: DataFrame with 'timestamp' (Datetime), 'price' (Float64), 'volume' (Float64).
        frequency: Polars duration string (e.g., '1m', '1h', '1d').

    Returns:
        Polars DataFrame with OHLCV bars at the specified frequency.

    Reference:
        AFML, Chapter 2, Section 2.3.1
    """
    bars = (
        df.sort("timestamp")
        .with_columns(pl.col("timestamp").set_sorted())
        .group_by_dynamic("timestamp", every=frequency)
        .agg([
            pl.col("price").first().alias("open"),
            pl.col("price").max().alias("high"),
            pl.col("price").min().alias("low"),
            pl.col("price").last().alias("close"),
            pl.col("volume").sum().alias("volume"),
            (
                (pl.col("price") * pl.col("volume")).sum()
                / pl.col("volume").sum()
            ).alias("vwap"),
        ])
        .filter(pl.col("volume") > 0)
        .sort("timestamp")
    )

    return bars


def tick_bars(df: pl.DataFrame, threshold: int) -> pl.DataFrame:
    """Sample bars every N transactions (ticks).

    Args:
        df: DataFrame with 'timestamp', 'price', 'volume' columns.
        threshold: Number of ticks per bar.

    Returns:
        Polars DataFrame with OHLCV bars.

    Reference:
        AFML, Chapter 2, Section 2.3.2

    # TODO(numba): evaluate JIT for tick bar inner loop
    """
    ticks = df.to_dicts()
    bars: list[dict] = []
    current: list[dict] = []
    count = 0

    for tick in ticks:
        current.append(tick)
        count += 1
        if count >= threshold:
            bar = _aggregate_bar(current)
            if bar:
                bars.append(bar)
            current = []
            count = 0

    return pl.DataFrame(bars) if bars else pl.DataFrame(schema={
        "timestamp": pl.Datetime,
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "volume": pl.Float64,
        "vwap": pl.Float64,
    })


def volume_bars(df: pl.DataFrame, threshold: float) -> pl.DataFrame:
    """Sample bars every N units of volume.

    Args:
        df: DataFrame with 'timestamp', 'price', 'volume' columns.
        threshold: Volume threshold for bar formation.

    Returns:
        Polars DataFrame with OHLCV bars.

    Reference:
        AFML, Chapter 2, Section 2.3.3

    # TODO(numba): evaluate JIT for volume bar inner loop
    """
    ticks = df.to_dicts()
    bars: list[dict] = []
    current: list[dict] = []
    cum_vol = 0.0

    for tick in ticks:
        current.append(tick)
        cum_vol += tick["volume"]
        if cum_vol >= threshold:
            bar = _aggregate_bar(current)
            if bar:
                bars.append(bar)
            current = []
            cum_vol = 0.0

    return pl.DataFrame(bars) if bars else pl.DataFrame(schema={
        "timestamp": pl.Datetime,
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "volume": pl.Float64,
        "vwap": pl.Float64,
    })


def dollar_bars(df: pl.DataFrame, threshold: float) -> pl.DataFrame:
    """Sample bars every N dollars of market value traded.

    Args:
        df: DataFrame with 'timestamp', 'price', 'volume' columns.
        threshold: Dollar value threshold for bar formation.

    Returns:
        Polars DataFrame with OHLCV bars.

    Reference:
        AFML, Chapter 2, Section 2.3.4

    # TODO(numba): evaluate JIT for dollar bar inner loop
    """
    ticks = df.to_dicts()
    bars: list[dict] = []
    current: list[dict] = []
    cum_dollar = 0.0

    for tick in ticks:
        current.append(tick)
        cum_dollar += tick["price"] * tick["volume"]
        if cum_dollar >= threshold:
            bar = _aggregate_bar(current)
            if bar:
                bars.append(bar)
            current = []
            cum_dollar = 0.0

    return pl.DataFrame(bars) if bars else pl.DataFrame(schema={
        "timestamp": pl.Datetime,
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "volume": pl.Float64,
        "vwap": pl.Float64,
    })


def tick_imbalance_bars(
    df: pl.DataFrame,
    expected_ticks_init: int = 100,
    ewma_span: int = 20,
) -> pl.DataFrame:
    """Sample bars when tick imbalance exceeds EWMA-adjusted threshold.

    Fixes the EWMA antipattern: uses incremental formula instead of
    per-iteration pd.Series().ewm().

    Args:
        df: DataFrame with 'timestamp', 'price', 'volume' columns.
        expected_ticks_init: Initial expected ticks per bar.
        ewma_span: Span for EWMA of expected ticks.

    Returns:
        Polars DataFrame with OHLCV bars.

    Reference:
        AFML, Chapter 2, Section 2.4.1

    # TODO(numba): evaluate JIT for imbalance bar inner loop
    """
    alpha = 2.0 / (ewma_span + 1)
    prices = df["price"].to_numpy()
    tick_rule = _compute_tick_rule(prices)

    ticks = df.to_dicts()
    bars: list[dict] = []
    current: list[dict] = []
    tick_imbalance = 0.0
    ewma_expected_ticks = float(expected_ticks_init)
    ewma_tick_imbalance = 0.0

    for i, tick in enumerate(ticks):
        current.append(tick)
        b_t = tick_rule[i]
        tick_imbalance += b_t

        threshold = abs(ewma_expected_ticks * ewma_tick_imbalance)
        if abs(tick_imbalance) >= max(threshold, 1.0):
            bar = _aggregate_bar(current)
            if bar:
                bars.append(bar)
            # Update EWMAs incrementally
            ewma_expected_ticks = (
                alpha * len(current) + (1 - alpha) * ewma_expected_ticks
            )
            ewma_tick_imbalance = (
                alpha * tick_imbalance + (1 - alpha) * ewma_tick_imbalance
            )
            current = []
            tick_imbalance = 0.0

    return pl.DataFrame(bars) if bars else pl.DataFrame(schema={
        "timestamp": pl.Datetime,
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "volume": pl.Float64,
        "vwap": pl.Float64,
    })


def volume_imbalance_bars(
    df: pl.DataFrame,
    expected_ticks_init: int = 100,
    ewma_span: int = 20,
) -> pl.DataFrame:
    """Sample bars when volume imbalance exceeds EWMA-adjusted threshold.

    Args:
        df: DataFrame with 'timestamp', 'price', 'volume' columns.
        expected_ticks_init: Initial expected ticks per bar.
        ewma_span: Span for EWMA of expected ticks.

    Returns:
        Polars DataFrame with OHLCV bars.

    Reference:
        AFML, Chapter 2, Section 2.4.2

    # TODO(numba): evaluate JIT for volume imbalance bar inner loop
    """
    alpha = 2.0 / (ewma_span + 1)
    prices = df["price"].to_numpy()
    tick_rule = _compute_tick_rule(prices)

    ticks = df.to_dicts()
    bars: list[dict] = []
    current: list[dict] = []
    vol_imbalance = 0.0
    ewma_expected_ticks = float(expected_ticks_init)
    ewma_vol_imbalance = 0.0

    for i, tick in enumerate(ticks):
        current.append(tick)
        vol_imbalance += tick_rule[i] * tick["volume"]

        threshold = abs(ewma_expected_ticks * ewma_vol_imbalance)
        if abs(vol_imbalance) >= max(threshold, 1.0):
            bar = _aggregate_bar(current)
            if bar:
                bars.append(bar)
            ewma_expected_ticks = (
                alpha * len(current) + (1 - alpha) * ewma_expected_ticks
            )
            ewma_vol_imbalance = (
                alpha * vol_imbalance + (1 - alpha) * ewma_vol_imbalance
            )
            current = []
            vol_imbalance = 0.0

    return pl.DataFrame(bars) if bars else pl.DataFrame(schema={
        "timestamp": pl.Datetime,
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "volume": pl.Float64,
        "vwap": pl.Float64,
    })


def dollar_imbalance_bars(
    df: pl.DataFrame,
    expected_ticks_init: int = 100,
    ewma_span: int = 20,
) -> pl.DataFrame:
    """Sample bars when dollar imbalance exceeds EWMA-adjusted threshold.

    Args:
        df: DataFrame with 'timestamp', 'price', 'volume' columns.
        expected_ticks_init: Initial expected ticks per bar.
        ewma_span: Span for EWMA of expected ticks.

    Returns:
        Polars DataFrame with OHLCV bars.

    Reference:
        AFML, Chapter 2, Section 2.4.3

    # TODO(numba): evaluate JIT for dollar imbalance bar inner loop
    """
    alpha = 2.0 / (ewma_span + 1)
    prices = df["price"].to_numpy()
    tick_rule = _compute_tick_rule(prices)

    ticks = df.to_dicts()
    bars: list[dict] = []
    current: list[dict] = []
    dollar_imbalance = 0.0
    ewma_expected_ticks = float(expected_ticks_init)
    ewma_dollar_imbalance = 0.0

    for i, tick in enumerate(ticks):
        current.append(tick)
        dollar_imbalance += tick_rule[i] * tick["price"] * tick["volume"]

        threshold = abs(ewma_expected_ticks * ewma_dollar_imbalance)
        if abs(dollar_imbalance) >= max(threshold, 1.0):
            bar = _aggregate_bar(current)
            if bar:
                bars.append(bar)
            ewma_expected_ticks = (
                alpha * len(current) + (1 - alpha) * ewma_expected_ticks
            )
            ewma_dollar_imbalance = (
                alpha * dollar_imbalance + (1 - alpha) * ewma_dollar_imbalance
            )
            current = []
            dollar_imbalance = 0.0

    return pl.DataFrame(bars) if bars else pl.DataFrame(schema={
        "timestamp": pl.Datetime,
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "volume": pl.Float64,
        "vwap": pl.Float64,
    })


def tick_runs_bars(
    df: pl.DataFrame,
    expected_ticks_init: int = 100,
    ewma_span: int = 20,
) -> pl.DataFrame:
    """Sample bars when tick runs exceed EWMA-adjusted threshold.

    Args:
        df: DataFrame with 'timestamp', 'price', 'volume' columns.
        expected_ticks_init: Initial expected ticks per bar.
        ewma_span: Span for EWMA updates.

    Returns:
        Polars DataFrame with OHLCV bars.

    Reference:
        AFML, Chapter 2, Section 2.4.4

    # TODO(numba): evaluate JIT for tick runs bar inner loop
    """
    alpha = 2.0 / (ewma_span + 1)
    prices = df["price"].to_numpy()
    tick_rule = _compute_tick_rule(prices)

    ticks = df.to_dicts()
    bars: list[dict] = []
    current: list[dict] = []
    n_buy = 0
    ewma_expected_ticks = float(expected_ticks_init)
    ewma_buy_prob = 0.5  # Initial buy probability

    for i, tick in enumerate(ticks):
        current.append(tick)
        if tick_rule[i] > 0:
            n_buy += 1

        n = len(current)
        buy_prob = n_buy / n if n > 0 else 0.5
        expected_runs = ewma_expected_ticks * max(buy_prob, 1 - buy_prob)

        if n >= expected_runs:
            bar = _aggregate_bar(current)
            if bar:
                bars.append(bar)
            ewma_expected_ticks = (
                alpha * n + (1 - alpha) * ewma_expected_ticks
            )
            ewma_buy_prob = (
                alpha * buy_prob + (1 - alpha) * ewma_buy_prob
            )
            current = []
            n_buy = 0

    return pl.DataFrame(bars) if bars else pl.DataFrame(schema={
        "timestamp": pl.Datetime,
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "volume": pl.Float64,
        "vwap": pl.Float64,
    })


def volume_runs_bars(
    df: pl.DataFrame,
    expected_ticks_init: int = 100,
    ewma_span: int = 20,
) -> pl.DataFrame:
    """Sample bars when volume runs exceed EWMA-adjusted threshold.

    Args:
        df: DataFrame with 'timestamp', 'price', 'volume' columns.
        expected_ticks_init: Initial expected ticks per bar.
        ewma_span: Span for EWMA updates.

    Returns:
        Polars DataFrame with OHLCV bars.

    Reference:
        AFML, Chapter 2, Section 2.4.4

    # TODO(numba): evaluate JIT for volume runs bar inner loop
    """
    alpha = 2.0 / (ewma_span + 1)
    prices = df["price"].to_numpy()
    tick_rule = _compute_tick_rule(prices)
    volumes = df["volume"].to_numpy()

    ticks = df.to_dicts()
    bars: list[dict] = []
    current: list[dict] = []
    buy_volume = 0.0
    total_volume = 0.0
    ewma_expected_ticks = float(expected_ticks_init)
    ewma_buy_vol = 0.0
    ewma_vol_per_tick = 1.0

    for i, tick in enumerate(ticks):
        current.append(tick)
        vol = tick["volume"]
        total_volume += vol
        if tick_rule[i] > 0:
            buy_volume += vol

        n = len(current)
        buy_vol_frac = buy_volume / total_volume if total_volume > 0 else 0.5
        sell_vol_frac = 1 - buy_vol_frac
        expected_runs = ewma_expected_ticks * ewma_vol_per_tick * max(buy_vol_frac, sell_vol_frac)

        if total_volume >= max(expected_runs, 1.0):
            bar = _aggregate_bar(current)
            if bar:
                bars.append(bar)
            ewma_expected_ticks = (
                alpha * n + (1 - alpha) * ewma_expected_ticks
            )
            ewma_vol_per_tick = (
                alpha * (total_volume / n if n > 0 else 0)
                + (1 - alpha) * ewma_vol_per_tick
            )
            ewma_buy_vol = (
                alpha * buy_volume + (1 - alpha) * ewma_buy_vol
            )
            current = []
            buy_volume = 0.0
            total_volume = 0.0

    return pl.DataFrame(bars) if bars else pl.DataFrame(schema={
        "timestamp": pl.Datetime,
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "volume": pl.Float64,
        "vwap": pl.Float64,
    })


def dollar_runs_bars(
    df: pl.DataFrame,
    expected_ticks_init: int = 100,
    ewma_span: int = 20,
) -> pl.DataFrame:
    """Sample bars when dollar runs exceed EWMA-adjusted threshold.

    Args:
        df: DataFrame with 'timestamp', 'price', 'volume' columns.
        expected_ticks_init: Initial expected ticks per bar.
        ewma_span: Span for EWMA updates.

    Returns:
        Polars DataFrame with OHLCV bars.

    Reference:
        AFML, Chapter 2, Section 2.4.4

    # TODO(numba): evaluate JIT for dollar runs bar inner loop
    """
    alpha = 2.0 / (ewma_span + 1)
    prices = df["price"].to_numpy()
    tick_rule = _compute_tick_rule(prices)

    ticks = df.to_dicts()
    bars: list[dict] = []
    current: list[dict] = []
    buy_dollars = 0.0
    total_dollars = 0.0
    ewma_expected_ticks = float(expected_ticks_init)
    ewma_dollar_per_tick = 1.0

    for i, tick in enumerate(ticks):
        current.append(tick)
        dv = tick["price"] * tick["volume"]
        total_dollars += dv
        if tick_rule[i] > 0:
            buy_dollars += dv

        n = len(current)
        buy_frac = buy_dollars / total_dollars if total_dollars > 0 else 0.5
        expected_runs = ewma_expected_ticks * ewma_dollar_per_tick * max(buy_frac, 1 - buy_frac)

        if total_dollars >= max(expected_runs, 1.0):
            bar = _aggregate_bar(current)
            if bar:
                bars.append(bar)
            ewma_expected_ticks = (
                alpha * n + (1 - alpha) * ewma_expected_ticks
            )
            ewma_dollar_per_tick = (
                alpha * (total_dollars / n if n > 0 else 0)
                + (1 - alpha) * ewma_dollar_per_tick
            )
            current = []
            buy_dollars = 0.0
            total_dollars = 0.0

    return pl.DataFrame(bars) if bars else pl.DataFrame(schema={
        "timestamp": pl.Datetime,
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "volume": pl.Float64,
        "vwap": pl.Float64,
    })
