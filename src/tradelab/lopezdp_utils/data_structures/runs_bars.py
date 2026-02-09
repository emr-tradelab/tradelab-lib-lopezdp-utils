"""Information-driven runs bars: tick, volume, and dollar runs bars.

Runs bars monitor the absolute accumulation of buy or sell activity separately,
focusing on persistence rather than net imbalance.

Reference: Advances in Financial Machine Learning, Chapter 2, Section 2.4
"""

from collections.abc import Iterable

import numpy as np
import pandas as pd


def _compute_tick_rule(prices: list[float], prev_b: int = 1) -> list[int]:
    """Compute tick rule (b_t) for a sequence of prices.

    Args:
        prices: List of prices.
        prev_b: Previous bar's terminal tick rule value. Default: 1.

    Returns:
        List of tick rule values (+1 or -1) for each tick.
    """
    b_t = [prev_b]

    for i in range(1, len(prices)):
        delta_p = prices[i] - prices[i - 1]
        if delta_p != 0:
            b_t.append(int(np.sign(delta_p)))
        else:
            b_t.append(b_t[-1])

    return b_t


def tick_runs_bars(
    tick_data: Iterable[dict] | pd.DataFrame,
    expected_ticks_init: int = 100,
    ewma_span: int = 20,
) -> pd.DataFrame:
    """Sample tick runs bars (TRBs).

    TRBs sample based on the count of buy or sell transactions. They focus on the
    maximum accumulation of either side separately (max{Buys, Sells}), capturing
    persistence of activity by one type of aggressor.

    A new bar forms when: θ_T ≥ E[T] × max{P[b_t=1], 1-P[b_t=1]}

    Args:
        tick_data: Either an iterable of tick dictionaries or a DataFrame.
            Each tick should have 'timestamp', 'price', 'volume' fields.
        expected_ticks_init: Initial expectation for number of ticks per bar.
        ewma_span: Span for exponentially weighted moving average of expectations.

    Returns:
        DataFrame with OHLCV bars sampled at run events.

    Reference:
        - AFML, Chapter 2, Section 2.4 (Runs Bars)
        - Runs: θ_T = max{Σ(b_t=1)|b_t, -Σ(b_t=-1)|b_t}
        - Expected runs: E[θ_T] = E[T] × max{P[b=1], 1-P[b=1]}

    Note:
        Runs bars are more sensitive to overall persistence of activity by one
        aggressor type, even without extreme net imbalance. They capture execution
        tactics like "sweeping the book" or sliced orders (TWAP).

    Example:
        >>> ticks = pd.DataFrame({
        ...     'timestamp': pd.date_range('2020-01-01', periods=1000, freq='1s'),
        ...     'price': np.random.randn(1000).cumsum() + 100,
        ...     'volume': np.random.randint(1, 100, 1000)
        ... })
        >>> bars = tick_runs_bars(ticks)
    """
    if isinstance(tick_data, pd.DataFrame):
        tick_data = tick_data.to_dict("records")

    bars = []
    current_bar_ticks = []
    buy_count = 0
    sell_count = 0
    prev_b = 1
    expected_ticks = expected_ticks_init
    expected_runs = expected_ticks * 0.5  # Initial: 50% buy probability

    # Track for EWMA updates
    bar_tick_counts = []
    bar_buy_proportions = []

    for tick in tick_data:
        current_bar_ticks.append(tick)

        # Compute tick rule
        if len(current_bar_ticks) == 1:
            b_t = prev_b
        else:
            delta_p = tick["price"] - current_bar_ticks[-2]["price"]
            b_t = int(np.sign(delta_p)) if delta_p != 0 else prev_b

        prev_b = b_t

        # Accumulate buy/sell counts separately
        if b_t == 1:
            buy_count += 1
        else:
            sell_count += 1

        # Compute runs: max of buy or sell accumulation
        theta = max(buy_count, sell_count)

        # Check sampling condition: θ_T ≥ E[θ_T]
        if theta >= expected_runs:
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

            # Compute buy proportion
            buy_proportion = buy_count / tick_count
            bar_buy_proportions.append(buy_proportion)

            # Update expectations using EWMA
            if len(bar_tick_counts) > 1:
                expected_ticks = (
                    pd.Series(bar_tick_counts).ewm(span=ewma_span, adjust=False).mean().iloc[-1]
                )

            # Update expected runs: E[T] × max{P[b=1], 1-P[b=1]}
            if len(bar_buy_proportions) > 1:
                expected_buy_prop = (
                    pd.Series(bar_buy_proportions).ewm(span=ewma_span, adjust=False).mean().iloc[-1]
                )
                expected_runs = expected_ticks * max(expected_buy_prop, 1 - expected_buy_prop)
            else:
                expected_runs = expected_ticks * 0.5

            # Reset for next bar
            current_bar_ticks = []
            buy_count = 0
            sell_count = 0

    return pd.DataFrame(bars)


def volume_runs_bars(
    tick_data: Iterable[dict] | pd.DataFrame,
    expected_ticks_init: int = 100,
    ewma_span: int = 20,
) -> pd.DataFrame:
    """Sample volume runs bars (VRBs).

    VRBs extend TRB concept to include trade size. They sample based on the maximum
    total volume accumulated on either the buy or sell side.

    A new bar forms when: θ_T ≥ E[T] × max{P[b=1]×E[v|b=1], (1-P[b=1])×E[v|b=-1]}

    Args:
        tick_data: Either an iterable of tick dictionaries or a DataFrame.
            Each tick should have 'timestamp', 'price', 'volume' fields.
        expected_ticks_init: Initial expectation for number of ticks per bar.
        ewma_span: Span for exponentially weighted moving average of expectations.

    Returns:
        DataFrame with OHLCV bars sampled at volume run events.

    Reference:
        - AFML, Chapter 2, Section 2.4 (Volume Runs Bars)
        - Runs: θ_T = max{Σ(b=1) b_t×v_t, -Σ(b=-1) b_t×v_t}

    Example:
        >>> bars = volume_runs_bars(ticks)
    """
    if isinstance(tick_data, pd.DataFrame):
        tick_data = tick_data.to_dict("records")

    bars = []
    current_bar_ticks = []
    buy_volume = 0
    sell_volume = 0
    prev_b = 1
    expected_ticks = expected_ticks_init
    expected_runs = expected_ticks * 10  # Initial guess

    # Track for EWMA updates
    bar_tick_counts = []
    bar_buy_volumes = []
    bar_sell_volumes = []

    for tick in tick_data:
        current_bar_ticks.append(tick)

        # Compute tick rule
        if len(current_bar_ticks) == 1:
            b_t = prev_b
        else:
            delta_p = tick["price"] - current_bar_ticks[-2]["price"]
            b_t = int(np.sign(delta_p)) if delta_p != 0 else prev_b

        prev_b = b_t

        # Accumulate buy/sell volumes separately
        if b_t == 1:
            buy_volume += tick["volume"]
        else:
            sell_volume += tick["volume"]

        # Compute runs: max of buy or sell volume
        theta = max(buy_volume, sell_volume)

        # Check sampling condition
        if theta >= expected_runs:
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

            # Per-tick averages for buy/sell volumes
            bar_buy_volumes.append(buy_volume / tick_count)
            bar_sell_volumes.append(sell_volume / tick_count)

            # Update expectations using EWMA
            if len(bar_tick_counts) > 1:
                expected_ticks = (
                    pd.Series(bar_tick_counts).ewm(span=ewma_span, adjust=False).mean().iloc[-1]
                )

            # Update expected runs: E[T] × max{P[b=1]×E[v|b=1], (1-P[b=1])×E[v|b=-1]}
            if len(bar_buy_volumes) > 1:
                expected_buy_vol = (
                    pd.Series(bar_buy_volumes).ewm(span=ewma_span, adjust=False).mean().iloc[-1]
                )
                expected_sell_vol = (
                    pd.Series(bar_sell_volumes).ewm(span=ewma_span, adjust=False).mean().iloc[-1]
                )
                expected_runs = expected_ticks * max(expected_buy_vol, expected_sell_vol)
            else:
                expected_runs = expected_ticks * 10

            # Reset for next bar
            current_bar_ticks = []
            buy_volume = 0
            sell_volume = 0

    return pd.DataFrame(bars)


def dollar_runs_bars(
    tick_data: Iterable[dict] | pd.DataFrame,
    expected_ticks_init: int = 100,
    ewma_span: int = 20,
) -> pd.DataFrame:
    """Sample dollar runs bars (DRBs).

    DRBs have the same structure as VRBs but use market value (price × volume) instead
    of volume. They sample based on the maximum total dollar value accumulated on either
    the buy or sell side.

    A new bar forms when: θ_T ≥ E[T] × max{P[b=1]×E[d|b=1], (1-P[b=1])×E[d|b=-1]}

    Args:
        tick_data: Either an iterable of tick dictionaries or a DataFrame.
            Each tick should have 'timestamp', 'price', 'volume' fields.
        expected_ticks_init: Initial expectation for number of ticks per bar.
        ewma_span: Span for exponentially weighted moving average of expectations.

    Returns:
        DataFrame with OHLCV bars sampled at dollar run events.

    Reference:
        - AFML, Chapter 2, Section 2.4 (Dollar Runs Bars)
        - Runs: θ_T = max{Σ(b=1) b_t×d_t, -Σ(b=-1) b_t×d_t} where d_t = p_t×v_t

    Example:
        >>> bars = dollar_runs_bars(ticks)
    """
    if isinstance(tick_data, pd.DataFrame):
        tick_data = tick_data.to_dict("records")

    bars = []
    current_bar_ticks = []
    buy_dollars = 0
    sell_dollars = 0
    prev_b = 1
    expected_ticks = expected_ticks_init
    expected_runs = expected_ticks * 1000  # Initial guess

    # Track for EWMA updates
    bar_tick_counts = []
    bar_buy_dollar_avgs = []
    bar_sell_dollar_avgs = []

    for tick in tick_data:
        current_bar_ticks.append(tick)

        # Compute tick rule
        if len(current_bar_ticks) == 1:
            b_t = prev_b
        else:
            delta_p = tick["price"] - current_bar_ticks[-2]["price"]
            b_t = int(np.sign(delta_p)) if delta_p != 0 else prev_b

        prev_b = b_t

        # Accumulate buy/sell dollars separately
        dollar_value = tick["price"] * tick["volume"]
        if b_t == 1:
            buy_dollars += dollar_value
        else:
            sell_dollars += dollar_value

        # Compute runs: max of buy or sell dollars
        theta = max(buy_dollars, sell_dollars)

        # Check sampling condition
        if theta >= expected_runs:
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

            # Per-tick averages for buy/sell dollars
            bar_buy_dollar_avgs.append(buy_dollars / tick_count)
            bar_sell_dollar_avgs.append(sell_dollars / tick_count)

            # Update expectations using EWMA
            if len(bar_tick_counts) > 1:
                expected_ticks = (
                    pd.Series(bar_tick_counts).ewm(span=ewma_span, adjust=False).mean().iloc[-1]
                )

            # Update expected runs: E[T] × max{P[b=1]×E[d|b=1], (1-P[b=1])×E[d|b=-1]}
            if len(bar_buy_dollar_avgs) > 1:
                expected_buy_dollars = (
                    pd.Series(bar_buy_dollar_avgs).ewm(span=ewma_span, adjust=False).mean().iloc[-1]
                )
                expected_sell_dollars = (
                    pd.Series(bar_sell_dollar_avgs)
                    .ewm(span=ewma_span, adjust=False)
                    .mean()
                    .iloc[-1]
                )
                expected_runs = expected_ticks * max(expected_buy_dollars, expected_sell_dollars)
            else:
                expected_runs = expected_ticks * 1000

            # Reset for next bar
            current_bar_ticks = []
            buy_dollars = 0
            sell_dollars = 0

    return pd.DataFrame(bars)
