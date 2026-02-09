"""ETF Trick for modeling complex baskets as synthetic total-return products.

Reference: Advances in Financial Machine Learning, Chapter 2, Section 2.5.2.2
"""

import pandas as pd


def etf_trick(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    point_values: pd.DataFrame | None = None,
    dividends: pd.DataFrame | None = None,
    rebalance_dates: list[pd.Timestamp] | None = None,
) -> pd.Series:
    """Model a complex basket as a single non-expiring, total-return cash product.

    The ETF Trick solves several challenges when modeling multi-instrument baskets:
    1. Weight-induced convergence (spreads converging without price changes)
    2. Negative prices in spreads (problematic for many models)
    3. Futures rolls creating spurious price jumps
    4. Complex accounting for dividends, coupons, and rebalancing costs

    This function creates a synthetic series K_t representing the value of $1 invested
    in the basket, accounting for rebalancing, reinvestment, and transaction costs.

    Args:
        prices: DataFrame with DatetimeIndex and columns for each instrument.
            Should contain both 'open' and 'close' prices, with multi-level columns
            like ('open', 'instrument1'), ('close', 'instrument1'), etc.
        weights: DataFrame with DatetimeIndex and columns for each instrument.
            Allocation weights (can change over time).
        point_values: DataFrame with USD point values (φ) for each instrument.
            If None, assumes point value of 1.0 for all instruments.
        dividends: DataFrame with dividends/carry (d) for each instrument.
            If None, assumes zero dividends.
        rebalance_dates: List of dates when rebalancing occurs (futures rolls,
            weight adjustments). If None, assumes no rebalancing.

    Returns:
        Series K_t representing the cumulative value of $1 invested in the basket.
        This synthetic ETF price can be used directly in ML models.

    Reference:
        - AFML, Chapter 2, Section 2.5.2.2 (The ETF Trick)
        - Mathematical formulation:
          K_t = K_{t-1} + Σ_i[h_{i,t-1} × φ_{i,t} × (δ_{i,t} + d_{i,t})]
        - Use case: Model futures spreads, bond portfolios, or any basket as a
          single, well-behaved synthetic price series

    Note:
        This is a v1 implementation. The book provides the mathematical framework
        but the exact implementation depends on your data structure. This function
        assumes a specific DataFrame layout and may need adjustment for your use case.

    Example:
        >>> prices = pd.DataFrame({
        ...     ('open', 'ES'): [100, 101, 102],
        ...     ('close', 'ES'): [101, 102, 103],
        ...     ('open', 'NQ'): [200, 202, 204],
        ...     ('close', 'NQ'): [202, 204, 206]
        ... }, index=pd.date_range('2020-01-01', periods=3))
        >>> weights = pd.DataFrame({
        ...     'ES': [1, 1, 1],
        ...     'NQ': [-1, -1, -1]
        ... }, index=prices.index)
        >>> k_t = etf_trick(prices, weights)
    """
    # Initialize
    instruments = weights.columns.tolist()
    timestamps = prices.index
    k_t = pd.Series(index=timestamps, dtype=float)
    k_t.iloc[0] = 1.0  # Start with $1

    # Default values
    if point_values is None:
        point_values = pd.DataFrame(1.0, index=timestamps, columns=instruments)
    if dividends is None:
        dividends = pd.DataFrame(0.0, index=timestamps, columns=instruments)
    if rebalance_dates is None:
        rebalance_dates = []

    # Convert rebalance dates to set for fast lookup
    rebalance_set = set(rebalance_dates)

    # Initialize holdings
    holdings = pd.DataFrame(index=timestamps, columns=instruments, dtype=float)

    for i, t in enumerate(timestamps):
        if i == 0:
            # Initial holdings: allocate $1 according to weights
            sum_abs_weights = weights.loc[t].abs().sum()
            for inst in instruments:
                if ("open", inst) in prices.columns:
                    o_t = prices.loc[t, ("open", inst)]
                else:
                    o_t = prices.loc[t, inst]  # Fallback if no multi-level columns

                phi_t = point_values.loc[t, inst]
                holdings.loc[t, inst] = (weights.loc[t, inst] * k_t.iloc[0]) / (
                    o_t * phi_t * sum_abs_weights
                )
            continue

        # Check if rebalancing occurs
        is_rebalance = t in rebalance_set
        t_prev = timestamps[i - 1]

        # Update holdings if rebalancing
        if is_rebalance:
            sum_abs_weights = weights.loc[t].abs().sum()
            for inst in instruments:
                if ("open", inst) in prices.columns:
                    o_next = prices.loc[t, ("open", inst)]
                else:
                    o_next = prices.loc[t, inst]

                phi_t = point_values.loc[t, inst]
                holdings.loc[t, inst] = (weights.loc[t, inst] * k_t.iloc[i - 1]) / (
                    o_next * phi_t * sum_abs_weights
                )
        else:
            # Carry forward holdings
            holdings.loc[t] = holdings.loc[t_prev]

        # Compute market value change (δ)
        pnl = 0
        for inst in instruments:
            h_prev = holdings.loc[t_prev, inst]
            phi_t = point_values.loc[t, inst]
            d_t = dividends.loc[t, inst]

            # Price change
            if is_rebalance:
                # Use open-to-close on rebalance days
                if ("open", inst) in prices.columns:
                    o_t = prices.loc[t, ("open", inst)]
                    p_t = prices.loc[t, ("close", inst)]
                else:
                    # Fallback: assume previous close and current close
                    p_t = prices.loc[t, inst]
                    o_t = prices.loc[t_prev, inst]
                delta = p_t - o_t
            else:
                # Regular price change
                if ("close", inst) in prices.columns:
                    p_t = prices.loc[t, ("close", inst)]
                    p_prev = prices.loc[t_prev, ("close", inst)]
                else:
                    p_t = prices.loc[t, inst]
                    p_prev = prices.loc[t_prev, inst]
                delta = p_t - p_prev

            pnl += h_prev * phi_t * (delta + d_t)

        # Update K_t
        k_t.iloc[i] = k_t.iloc[i - 1] + pnl

    return k_t
