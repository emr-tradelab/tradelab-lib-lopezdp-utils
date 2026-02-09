"""Information-driven imbalance bars: tick, volume, and dollar imbalance bars.

Reference: Advances in Financial Machine Learning, Chapter 2, Section 2.4
"""

from collections.abc import Iterable

import numpy as np
import pandas as pd


def _compute_tick_rule(prices: list[float], prev_b: int = 1) -> list[int]:
    """Compute tick rule (b_t) for a sequence of prices.

    The tick rule determines whether a trade is a buy (+1) or sell (-1):
    - If Δp_t ≠ 0: b_t = |Δp_t| / Δp_t (sign of price change)
    - If Δp_t = 0: b_t = b_{t-1} (carry forward previous sign)

    Args:
        prices: List of prices.
        prev_b: Previous bar's terminal tick rule value. Default: 1.

    Returns:
        List of tick rule values (+1 or -1) for each tick.
    """
    b_t = [prev_b]  # Initialize with boundary condition

    for i in range(1, len(prices)):
        delta_p = prices[i] - prices[i - 1]
        if delta_p != 0:
            b_t.append(int(np.sign(delta_p)))
        else:
            b_t.append(b_t[-1])  # Carry forward

    return b_t


def tick_imbalance_bars(
    tick_data: Iterable[dict] | pd.DataFrame,
    expected_ticks_init: int = 100,
    ewma_span: int = 20,
) -> pd.DataFrame:
    """Sample tick imbalance bars (TIBs).

    TIBs sample when the accumulation of signed ticks deviates from expectations.
    They bucket equal amounts of information regardless of price or volume, capturing
    asymmetric information from one-sided trading.

    A new bar forms when: |θ_T| ≥ E[T] × |2P[b_t = 1] - 1|

    Args:
        tick_data: Either an iterable of tick dictionaries or a DataFrame.
            Each tick should have 'timestamp', 'price', 'volume' fields.
        expected_ticks_init: Initial expectation for number of ticks per bar.
        ewma_span: Span for exponentially weighted moving average of expectations.

    Returns:
        DataFrame with OHLCV bars sampled at imbalance events.

    Reference:
        - AFML, Chapter 2, Section 2.4.1
        - Imbalance: θ_T = Σ(b_t)
        - Expected imbalance: E[θ_T] = E[T] × |2P[b_t = 1] - 1|

    Example:
        >>> ticks = pd.DataFrame({
        ...     'timestamp': pd.date_range('2020-01-01', periods=1000, freq='1s'),
        ...     'price': np.random.randn(1000).cumsum() + 100,
        ...     'volume': np.random.randint(1, 100, 1000)
        ... })
        >>> bars = tick_imbalance_bars(ticks)
    """
    if isinstance(tick_data, pd.DataFrame):
        tick_data = tick_data.to_dict("records")

    bars = []
    current_bar_ticks = []
    theta = 0  # Cumulative imbalance
    prev_b = 1  # Previous bar's terminal b_t
    expected_ticks = expected_ticks_init
    expected_imbalance = expected_ticks  # Initial expectation

    # Track for EWMA updates
    bar_tick_counts = []
    bar_buy_proportions = []

    for tick in tick_data:
        current_bar_ticks.append(tick)

        # Compute tick rule for current tick
        if len(current_bar_ticks) == 1:
            b_t = prev_b  # Use previous bar's terminal value
        else:
            delta_p = tick["price"] - current_bar_ticks[-2]["price"]
            b_t = int(np.sign(delta_p)) if delta_p != 0 else prev_b

        prev_b = b_t  # Update for next iteration
        theta += b_t  # Accumulate imbalance

        # Check sampling condition: |θ_T| ≥ E[T] × |2P[b=1] - 1|
        if abs(theta) >= expected_imbalance:
            # Aggregate bar
            prices = [t["price"] for t in current_bar_ticks]
            volumes = [t["volume"] for t in current_bar_ticks]
            total_volume = sum(volumes)
            vwap = (
                sum(t["price"] * t["volume"] for t in current_bar_ticks) / total_volume
                if total_volume > 0
                else 0
            )

            bar = {
                "timestamp": current_bar_ticks[-1]["timestamp"],
                "open": current_bar_ticks[0]["price"],
                "high": max(prices),
                "low": min(prices),
                "close": current_bar_ticks[-1]["price"],
                "volume": total_volume,
                "vwap": vwap,
            }
            bars.append(bar)

            # Update expectations using EWMA
            tick_count = len(current_bar_ticks)
            bar_tick_counts.append(tick_count)

            # Compute buy proportion for this bar
            b_values = _compute_tick_rule(prices, prev_b)
            buy_proportion = sum(1 for b in b_values if b == 1) / len(b_values)
            bar_buy_proportions.append(buy_proportion)

            # Update expected ticks
            if len(bar_tick_counts) > 1:
                expected_ticks = (
                    pd.Series(bar_tick_counts).ewm(span=ewma_span, adjust=False).mean().iloc[-1]
                )

            # Update expected imbalance: E[T] × |2P[b=1] - 1|
            if len(bar_buy_proportions) > 1:
                expected_buy_prop = (
                    pd.Series(bar_buy_proportions).ewm(span=ewma_span, adjust=False).mean().iloc[-1]
                )
                expected_imbalance = expected_ticks * abs(2 * expected_buy_prop - 1)
            else:
                expected_imbalance = expected_ticks

            # Reset for next bar
            current_bar_ticks = []
            theta = 0

    return pd.DataFrame(bars)


def volume_imbalance_bars(
    tick_data: Iterable[dict] | pd.DataFrame,
    expected_ticks_init: int = 100,
    ewma_span: int = 20,
) -> pd.DataFrame:
    """Sample volume imbalance bars (VIBs).

    VIBs extend TIBs by incorporating trade volume. They sample when the accumulation
    of signed volumes deviates from expectations.

    A new bar forms when: |θ_T| ≥ E[T] × |2v_+ - E[v_t]|

    Args:
        tick_data: Either an iterable of tick dictionaries or a DataFrame.
            Each tick should have 'timestamp', 'price', 'volume' fields.
        expected_ticks_init: Initial expectation for number of ticks per bar.
        ewma_span: Span for exponentially weighted moving average of expectations.

    Returns:
        DataFrame with OHLCV bars sampled at volume imbalance events.

    Reference:
        - AFML, Chapter 2, Section 2.4.2
        - Imbalance: θ_T = Σ(b_t × v_t)
        - Expected imbalance: E[θ_T] = E[T] × |2v_+ - E[v_t]|
        - v_+ = P[b=1] × E[v_t | b=1] (expected buy volume)

    Example:
        >>> bars = volume_imbalance_bars(ticks)
    """
    if isinstance(tick_data, pd.DataFrame):
        tick_data = tick_data.to_dict("records")

    bars = []
    current_bar_ticks = []
    theta = 0  # Cumulative signed volume
    prev_b = 1
    expected_ticks = expected_ticks_init
    expected_imbalance = expected_ticks * 10  # Initial guess

    # Track for EWMA updates
    bar_tick_counts = []
    bar_buy_volumes = []
    bar_total_volumes = []

    for tick in tick_data:
        current_bar_ticks.append(tick)

        # Compute tick rule
        if len(current_bar_ticks) == 1:
            b_t = prev_b
        else:
            delta_p = tick["price"] - current_bar_ticks[-2]["price"]
            b_t = int(np.sign(delta_p)) if delta_p != 0 else prev_b

        prev_b = b_t
        theta += b_t * tick["volume"]  # Accumulate signed volume

        # Check sampling condition: |θ_T| ≥ E[T] × |2v_+ - E[v_t]|
        if abs(theta) >= expected_imbalance:
            # Aggregate bar
            prices = [t["price"] for t in current_bar_ticks]
            volumes = [t["volume"] for t in current_bar_ticks]
            total_volume = sum(volumes)
            vwap = (
                sum(t["price"] * t["volume"] for t in current_bar_ticks) / total_volume
                if total_volume > 0
                else 0
            )

            bar = {
                "timestamp": current_bar_ticks[-1]["timestamp"],
                "open": current_bar_ticks[0]["price"],
                "high": max(prices),
                "low": min(prices),
                "close": current_bar_ticks[-1]["price"],
                "volume": total_volume,
                "vwap": vwap,
            }
            bars.append(bar)

            # Update expectations
            tick_count = len(current_bar_ticks)
            bar_tick_counts.append(tick_count)

            # Compute buy volume for this bar
            b_values = _compute_tick_rule(prices, prev_b)
            buy_volume = sum(v for b, v in zip(b_values, volumes, strict=False) if b == 1)
            bar_buy_volumes.append(buy_volume / tick_count)  # Per-tick buy volume
            bar_total_volumes.append(total_volume / tick_count)  # Per-tick total volume

            # Update expectations using EWMA
            if len(bar_tick_counts) > 1:
                expected_ticks = (
                    pd.Series(bar_tick_counts).ewm(span=ewma_span, adjust=False).mean().iloc[-1]
                )

            # Update expected imbalance: E[T] × |2v_+ - E[v_t]|
            if len(bar_buy_volumes) > 1:
                expected_buy_vol = (
                    pd.Series(bar_buy_volumes).ewm(span=ewma_span, adjust=False).mean().iloc[-1]
                )
                expected_total_vol = (
                    pd.Series(bar_total_volumes).ewm(span=ewma_span, adjust=False).mean().iloc[-1]
                )
                expected_imbalance = expected_ticks * abs(2 * expected_buy_vol - expected_total_vol)
            else:
                expected_imbalance = expected_ticks * 10

            # Reset for next bar
            current_bar_ticks = []
            theta = 0

    return pd.DataFrame(bars)


def dollar_imbalance_bars(
    tick_data: Iterable[dict] | pd.DataFrame,
    expected_ticks_init: int = 100,
    ewma_span: int = 20,
) -> pd.DataFrame:
    """Sample dollar imbalance bars (DIBs).

    DIBs are identical to VIBs but use dollar value (price × volume) instead of volume.
    They sample when the accumulation of signed dollar amounts deviates from expectations.

    A new bar forms when: |θ_T| ≥ E[T] × |2d_+ - E[d_t]|

    Args:
        tick_data: Either an iterable of tick dictionaries or a DataFrame.
            Each tick should have 'timestamp', 'price', 'volume' fields.
        expected_ticks_init: Initial expectation for number of ticks per bar.
        ewma_span: Span for exponentially weighted moving average of expectations.

    Returns:
        DataFrame with OHLCV bars sampled at dollar imbalance events.

    Reference:
        - AFML, Chapter 2, Section 2.4.3
        - Imbalance: θ_T = Σ(b_t × p_t × v_t)
        - Expected imbalance: E[θ_T] = E[T] × |2d_+ - E[d_t]|

    Example:
        >>> bars = dollar_imbalance_bars(ticks)
    """
    if isinstance(tick_data, pd.DataFrame):
        tick_data = tick_data.to_dict("records")

    bars = []
    current_bar_ticks = []
    theta = 0  # Cumulative signed dollars
    prev_b = 1
    expected_ticks = expected_ticks_init
    expected_imbalance = expected_ticks * 1000  # Initial guess

    # Track for EWMA updates
    bar_tick_counts = []
    bar_buy_dollars = []
    bar_total_dollars = []

    for tick in tick_data:
        current_bar_ticks.append(tick)

        # Compute tick rule
        if len(current_bar_ticks) == 1:
            b_t = prev_b
        else:
            delta_p = tick["price"] - current_bar_ticks[-2]["price"]
            b_t = int(np.sign(delta_p)) if delta_p != 0 else prev_b

        prev_b = b_t
        theta += b_t * tick["price"] * tick["volume"]  # Accumulate signed dollars

        # Check sampling condition: |θ_T| ≥ E[T] × |2d_+ - E[d_t]|
        if abs(theta) >= expected_imbalance:
            # Aggregate bar
            prices = [t["price"] for t in current_bar_ticks]
            volumes = [t["volume"] for t in current_bar_ticks]
            total_volume = sum(volumes)
            vwap = (
                sum(t["price"] * t["volume"] for t in current_bar_ticks) / total_volume
                if total_volume > 0
                else 0
            )

            bar = {
                "timestamp": current_bar_ticks[-1]["timestamp"],
                "open": current_bar_ticks[0]["price"],
                "high": max(prices),
                "low": min(prices),
                "close": current_bar_ticks[-1]["price"],
                "volume": total_volume,
                "vwap": vwap,
            }
            bars.append(bar)

            # Update expectations
            tick_count = len(current_bar_ticks)
            bar_tick_counts.append(tick_count)

            # Compute buy dollars for this bar
            b_values = _compute_tick_rule(prices, prev_b)
            buy_dollars = sum(
                p * v for b, p, v in zip(b_values, prices, volumes, strict=False) if b == 1
            )
            total_dollars = sum(p * v for p, v in zip(prices, volumes, strict=False))

            bar_buy_dollars.append(buy_dollars / tick_count)  # Per-tick buy dollars
            bar_total_dollars.append(total_dollars / tick_count)  # Per-tick total dollars

            # Update expectations using EWMA
            if len(bar_tick_counts) > 1:
                expected_ticks = (
                    pd.Series(bar_tick_counts).ewm(span=ewma_span, adjust=False).mean().iloc[-1]
                )

            # Update expected imbalance: E[T] × |2d_+ - E[d_t]|
            if len(bar_buy_dollars) > 1:
                expected_buy_dollars = (
                    pd.Series(bar_buy_dollars).ewm(span=ewma_span, adjust=False).mean().iloc[-1]
                )
                expected_total_dollars = (
                    pd.Series(bar_total_dollars).ewm(span=ewma_span, adjust=False).mean().iloc[-1]
                )
                expected_imbalance = expected_ticks * abs(
                    2 * expected_buy_dollars - expected_total_dollars
                )
            else:
                expected_imbalance = expected_ticks * 1000

            # Reset for next bar
            current_bar_ticks = []
            theta = 0

    return pd.DataFrame(bars)
