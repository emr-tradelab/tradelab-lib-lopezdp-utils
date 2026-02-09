"""Standard bar sampling methods: time, tick, volume, and dollar bars.

Reference: Advances in Financial Machine Learning, Chapter 2, Section 2.3
"""

from collections.abc import Iterable

import pandas as pd


def _aggregate_bar(ticks: list[dict]) -> dict:
    """Aggregate a list of ticks into OHLCV bar data.

    Args:
        ticks: List of tick dictionaries with 'timestamp', 'price', 'volume' keys.

    Returns:
        Dictionary with aggregated OHLCV data and VWAP.
    """
    if not ticks:
        return {}

    prices = [t["price"] for t in ticks]
    volumes = [t["volume"] for t in ticks]
    total_volume = sum(volumes)

    # Compute VWAP
    vwap = sum(t["price"] * t["volume"] for t in ticks) / total_volume if total_volume > 0 else 0

    return {
        "timestamp": ticks[-1]["timestamp"],  # Bar completion time
        "open": ticks[0]["price"],
        "high": max(prices),
        "low": min(prices),
        "close": ticks[-1]["price"],
        "volume": total_volume,
        "vwap": vwap,
    }


def time_bars(
    tick_data: pd.DataFrame,
    frequency: str,
) -> pd.DataFrame:
    """Sample bars at fixed time intervals.

    Time bars oversample during low-activity periods and undersample during high-activity
    periods, leading to poor statistical properties (heteroscedasticity, non-normality
    of returns). AFML advises against using time bars for ML applications.

    Args:
        tick_data: DataFrame with columns ['timestamp', 'price', 'volume'].
            'timestamp' should be datetime-like.
        frequency: Pandas frequency string (e.g., '1min', '1h', '1D').

    Returns:
        DataFrame with OHLCV bars at the specified frequency.

    Reference:
        - AFML, Chapter 2, Section 2.3.1
        - Critique: Exhibit heteroscedasticity and poor statistical properties

    Example:
        >>> ticks = pd.DataFrame({
        ...     'timestamp': pd.date_range('2020-01-01', periods=100, freq='1s'),
        ...     'price': np.random.randn(100).cumsum() + 100,
        ...     'volume': np.random.randint(1, 100, 100)
        ... })
        >>> bars = time_bars(ticks, frequency='1min')
    """
    df = tick_data.copy()
    df = df.set_index("timestamp")

    # Resample using pandas built-in functionality
    bars = pd.DataFrame(
        {
            "open": df["price"].resample(frequency).first(),
            "high": df["price"].resample(frequency).max(),
            "low": df["price"].resample(frequency).min(),
            "close": df["price"].resample(frequency).last(),
            "volume": df["volume"].resample(frequency).sum(),
        }
    )

    # Compute VWAP
    bars["vwap"] = (df["price"] * df["volume"]).resample(frequency).sum() / bars["volume"]

    # Remove bars with no data
    bars = bars.dropna(subset=["open", "close"])

    return bars.reset_index()


def tick_bars(
    tick_data: Iterable[dict] | pd.DataFrame,
    n: int,
) -> pd.DataFrame:
    """Sample bars every N transactions.

    Tick bars synchronize sampling with information arrival (transaction speed). Research
    shows tick bar price changes are closer to IID Gaussian distributions compared to
    time bars.

    Args:
        tick_data: Either an iterable of tick dictionaries or a DataFrame.
            Each tick should have 'timestamp', 'price', 'volume' fields.
        n: Number of transactions per bar.

    Returns:
        DataFrame with OHLCV bars, one bar per N transactions.

    Reference:
        - AFML, Chapter 2, Section 2.3.2
        - Advantage: Price changes closer to IID Gaussian distributions

    Example:
        >>> ticks = [
        ...     {'timestamp': '2020-01-01 00:00:00', 'price': 100, 'volume': 10},
        ...     {'timestamp': '2020-01-01 00:00:01', 'price': 101, 'volume': 20},
        ...     # ... more ticks
        ... ]
        >>> bars = tick_bars(ticks, n=100)
    """
    if isinstance(tick_data, pd.DataFrame):
        tick_data = tick_data.to_dict("records")

    bars = []
    current_bar_ticks = []
    tick_count = 0

    for tick in tick_data:
        current_bar_ticks.append(tick)
        tick_count += 1

        # Form bar when threshold reached
        if tick_count >= n:
            bar = _aggregate_bar(current_bar_ticks)
            if bar:
                bars.append(bar)

            # Reset for next bar
            current_bar_ticks = []
            tick_count = 0

    return pd.DataFrame(bars)


def volume_bars(
    tick_data: Iterable[dict] | pd.DataFrame,
    n: float,
) -> pd.DataFrame:
    """Sample bars every N units exchanged.

    Volume bars are robust to order fragmentation. One large 100-unit order counts the
    same as 100 separate 1-unit ticks—they're treated identically regardless of exchange
    protocol.

    Args:
        tick_data: Either an iterable of tick dictionaries or a DataFrame.
            Each tick should have 'timestamp', 'price', 'volume' fields.
        n: Threshold volume for bar formation (in units).

    Returns:
        DataFrame with OHLCV bars, one bar per N volume units.

    Reference:
        - AFML, Chapter 2, Section 2.3.3
        - Advantage: Robust to order fragmentation

    Example:
        >>> ticks = pd.DataFrame({
        ...     'timestamp': pd.date_range('2020-01-01', periods=100, freq='1s'),
        ...     'price': np.random.randn(100).cumsum() + 100,
        ...     'volume': np.random.randint(1, 100, 100)
        ... })
        >>> bars = volume_bars(ticks, n=500)
    """
    if isinstance(tick_data, pd.DataFrame):
        tick_data = tick_data.to_dict("records")

    bars = []
    current_bar_ticks = []
    cumulative_volume = 0

    for tick in tick_data:
        current_bar_ticks.append(tick)
        cumulative_volume += tick["volume"]

        # Form bar when threshold reached
        if cumulative_volume >= n:
            bar = _aggregate_bar(current_bar_ticks)
            if bar:
                bars.append(bar)

            # Reset for next bar
            current_bar_ticks = []
            cumulative_volume = 0

    return pd.DataFrame(bars)


def dollar_bars(
    tick_data: Iterable[dict] | pd.DataFrame,
    n: float,
) -> pd.DataFrame:
    """Sample bars every N market value exchanged.

    Dollar bars are the most robust standard bar type. They automatically adjust for
    price fluctuations (price doubling means half as many shares needed) and are robust
    to corporate actions like stock splits. They produce a stable number of bars per day
    over time.

    Args:
        tick_data: Either an iterable of tick dictionaries or a DataFrame.
            Each tick should have 'timestamp', 'price', 'volume' fields.
        n: Threshold dollar value for bar formation.

    Returns:
        DataFrame with OHLCV bars, one bar per N dollar value.

    Reference:
        - AFML, Chapter 2, Section 2.3.4
        - Advantages: Adjusts for price changes, robust to splits, stable bar count

    Example:
        >>> ticks = pd.DataFrame({
        ...     'timestamp': pd.date_range('2020-01-01', periods=100, freq='1s'),
        ...     'price': np.random.randn(100).cumsum() + 100,
        ...     'volume': np.random.randint(1, 100, 100)
        ... })
        >>> bars = dollar_bars(ticks, n=10000)
    """
    if isinstance(tick_data, pd.DataFrame):
        tick_data = tick_data.to_dict("records")

    bars = []
    current_bar_ticks = []
    cumulative_dollar = 0

    for tick in tick_data:
        current_bar_ticks.append(tick)
        cumulative_dollar += tick["price"] * tick["volume"]

        # Form bar when threshold reached
        if cumulative_dollar >= n:
            bar = _aggregate_bar(current_bar_ticks)
            if bar:
                bars.append(bar)

            # Reset for next bar
            current_bar_ticks = []
            cumulative_dollar = 0

    return pd.DataFrame(bars)
