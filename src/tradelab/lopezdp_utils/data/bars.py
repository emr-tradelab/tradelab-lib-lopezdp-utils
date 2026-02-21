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
    vwap = sum(t["price"] * t["volume"] for t in ticks) / total_volume if total_volume > 0 else 0.0

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
        .agg(
            [
                pl.col("price").first().alias("open"),
                pl.col("price").max().alias("high"),
                pl.col("price").min().alias("low"),
                pl.col("price").last().alias("close"),
                pl.col("volume").sum().alias("volume"),
                ((pl.col("price") * pl.col("volume")).sum() / pl.col("volume").sum()).alias("vwap"),
            ]
        )
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

    return (
        pl.DataFrame(bars)
        if bars
        else pl.DataFrame(
            schema={
                "timestamp": pl.Datetime,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
                "vwap": pl.Float64,
            }
        )
    )


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

    return (
        pl.DataFrame(bars)
        if bars
        else pl.DataFrame(
            schema={
                "timestamp": pl.Datetime,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
                "vwap": pl.Float64,
            }
        )
    )


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

    return (
        pl.DataFrame(bars)
        if bars
        else pl.DataFrame(
            schema={
                "timestamp": pl.Datetime,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
                "vwap": pl.Float64,
            }
        )
    )


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
    ewma_expected_bt = 0.0  # EWMA of per-tick sign E[b_t]

    for i, tick in enumerate(ticks):
        current.append(tick)
        b_t = tick_rule[i]
        tick_imbalance += b_t

        # Threshold: E_0[T] * |2P[b_t=1] - 1| = E_0[T] * |E[b_t]|
        threshold = abs(ewma_expected_ticks * ewma_expected_bt)
        if abs(tick_imbalance) >= max(threshold, 1.0):
            bar = _aggregate_bar(current)
            if bar:
                bars.append(bar)
            # Update EWMAs: E[T] and E[b_t] (per-tick average)
            n = len(current)
            ewma_expected_ticks = alpha * n + (1 - alpha) * ewma_expected_ticks
            ewma_expected_bt = alpha * (tick_imbalance / n) + (1 - alpha) * ewma_expected_bt
            current = []
            tick_imbalance = 0.0

    return (
        pl.DataFrame(bars)
        if bars
        else pl.DataFrame(
            schema={
                "timestamp": pl.Datetime,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
                "vwap": pl.Float64,
            }
        )
    )


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
    ewma_expected_bv = 0.0  # EWMA of per-tick signed volume E[b_t * v_t]

    for i, tick in enumerate(ticks):
        current.append(tick)
        vol_imbalance += tick_rule[i] * tick["volume"]

        # Threshold: E_0[T] * |E[b_t * v_t]|
        threshold = abs(ewma_expected_ticks * ewma_expected_bv)
        if abs(vol_imbalance) >= max(threshold, 1.0):
            bar = _aggregate_bar(current)
            if bar:
                bars.append(bar)
            n = len(current)
            ewma_expected_ticks = alpha * n + (1 - alpha) * ewma_expected_ticks
            ewma_expected_bv = alpha * (vol_imbalance / n) + (1 - alpha) * ewma_expected_bv
            current = []
            vol_imbalance = 0.0

    return (
        pl.DataFrame(bars)
        if bars
        else pl.DataFrame(
            schema={
                "timestamp": pl.Datetime,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
                "vwap": pl.Float64,
            }
        )
    )


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
    ewma_expected_bd = 0.0  # EWMA of per-tick signed dollar E[b_t * p_t * v_t]

    for i, tick in enumerate(ticks):
        current.append(tick)
        dollar_imbalance += tick_rule[i] * tick["price"] * tick["volume"]

        # Threshold: E_0[T] * |E[b_t * p_t * v_t]|
        threshold = abs(ewma_expected_ticks * ewma_expected_bd)
        if abs(dollar_imbalance) >= max(threshold, 1.0):
            bar = _aggregate_bar(current)
            if bar:
                bars.append(bar)
            n = len(current)
            ewma_expected_ticks = alpha * n + (1 - alpha) * ewma_expected_ticks
            ewma_expected_bd = alpha * (dollar_imbalance / n) + (1 - alpha) * ewma_expected_bd
            current = []
            dollar_imbalance = 0.0

    return (
        pl.DataFrame(bars)
        if bars
        else pl.DataFrame(
            schema={
                "timestamp": pl.Datetime,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
                "vwap": pl.Float64,
            }
        )
    )


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
    ewma_buy_prob = 0.5  # EWMA of P[b_t=1] from prior bars

    for i, tick in enumerate(ticks):
        current.append(tick)
        if tick_rule[i] > 0:
            n_buy += 1

        n = len(current)
        # Threshold uses EWMA buy prob from prior bars, not current bar
        expected_runs = ewma_expected_ticks * max(ewma_buy_prob, 1 - ewma_buy_prob)

        if max(n_buy, n - n_buy) >= max(expected_runs, 1.0):
            bar = _aggregate_bar(current)
            if bar:
                bars.append(bar)
            buy_prob = n_buy / n if n > 0 else 0.5
            ewma_expected_ticks = alpha * n + (1 - alpha) * ewma_expected_ticks
            ewma_buy_prob = alpha * buy_prob + (1 - alpha) * ewma_buy_prob
            current = []
            n_buy = 0

    return (
        pl.DataFrame(bars)
        if bars
        else pl.DataFrame(
            schema={
                "timestamp": pl.Datetime,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
                "vwap": pl.Float64,
            }
        )
    )


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

    ticks = df.to_dicts()
    bars: list[dict] = []
    current: list[dict] = []
    buy_volume = 0.0
    sell_volume = 0.0
    ewma_expected_ticks = float(expected_ticks_init)
    ewma_buy_prob = 0.5  # EWMA of P[b_t=1] from prior bars
    ewma_buy_vol_per_tick = 1.0  # EWMA of E[v_t | b_t=1]
    ewma_sell_vol_per_tick = 1.0  # EWMA of E[v_t | b_t=-1]

    for i, tick in enumerate(ticks):
        current.append(tick)
        vol = tick["volume"]
        if tick_rule[i] > 0:
            buy_volume += vol
        else:
            sell_volume += vol

        # Threshold: E[T] * max(P[b=1]*E[v|b=1], P[b=-1]*E[v|b=-1])
        expected_runs = ewma_expected_ticks * max(
            ewma_buy_prob * ewma_buy_vol_per_tick,
            (1 - ewma_buy_prob) * ewma_sell_vol_per_tick,
        )

        if max(buy_volume, sell_volume) >= max(expected_runs, 1.0):
            bar = _aggregate_bar(current)
            if bar:
                bars.append(bar)
            n = len(current)
            n_buy = sum(1 for j in range(i - n + 1, i + 1) if tick_rule[j] > 0)
            n_sell = n - n_buy
            buy_prob = n_buy / n if n > 0 else 0.5
            ewma_expected_ticks = alpha * n + (1 - alpha) * ewma_expected_ticks
            ewma_buy_prob = alpha * buy_prob + (1 - alpha) * ewma_buy_prob
            if n_buy > 0:
                ewma_buy_vol_per_tick = (
                    alpha * (buy_volume / n_buy) + (1 - alpha) * ewma_buy_vol_per_tick
                )
            if n_sell > 0:
                ewma_sell_vol_per_tick = (
                    alpha * (sell_volume / n_sell) + (1 - alpha) * ewma_sell_vol_per_tick
                )
            current = []
            buy_volume = 0.0
            sell_volume = 0.0

    return (
        pl.DataFrame(bars)
        if bars
        else pl.DataFrame(
            schema={
                "timestamp": pl.Datetime,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
                "vwap": pl.Float64,
            }
        )
    )


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
    sell_dollars = 0.0
    ewma_expected_ticks = float(expected_ticks_init)
    ewma_buy_prob = 0.5  # EWMA of P[b_t=1] from prior bars
    ewma_buy_dv_per_tick = 1.0  # EWMA of E[dv_t | b_t=1]
    ewma_sell_dv_per_tick = 1.0  # EWMA of E[dv_t | b_t=-1]

    for i, tick in enumerate(ticks):
        current.append(tick)
        dv = tick["price"] * tick["volume"]
        if tick_rule[i] > 0:
            buy_dollars += dv
        else:
            sell_dollars += dv

        # Threshold: E[T] * max(P[b=1]*E[dv|b=1], P[b=-1]*E[dv|b=-1])
        expected_runs = ewma_expected_ticks * max(
            ewma_buy_prob * ewma_buy_dv_per_tick,
            (1 - ewma_buy_prob) * ewma_sell_dv_per_tick,
        )

        if max(buy_dollars, sell_dollars) >= max(expected_runs, 1.0):
            bar = _aggregate_bar(current)
            if bar:
                bars.append(bar)
            n = len(current)
            n_buy = sum(1 for j in range(i - n + 1, i + 1) if tick_rule[j] > 0)
            n_sell = n - n_buy
            buy_prob = n_buy / n if n > 0 else 0.5
            ewma_expected_ticks = alpha * n + (1 - alpha) * ewma_expected_ticks
            ewma_buy_prob = alpha * buy_prob + (1 - alpha) * ewma_buy_prob
            if n_buy > 0:
                ewma_buy_dv_per_tick = (
                    alpha * (buy_dollars / n_buy) + (1 - alpha) * ewma_buy_dv_per_tick
                )
            if n_sell > 0:
                ewma_sell_dv_per_tick = (
                    alpha * (sell_dollars / n_sell) + (1 - alpha) * ewma_sell_dv_per_tick
                )
            current = []
            buy_dollars = 0.0
            sell_dollars = 0.0

    return (
        pl.DataFrame(bars)
        if bars
        else pl.DataFrame(
            schema={
                "timestamp": pl.Datetime,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
                "vwap": pl.Float64,
            }
        )
    )
