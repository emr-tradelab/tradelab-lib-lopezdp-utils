"""ETF Trick for modeling complex baskets as synthetic total-return products.

Reference: Advances in Financial Machine Learning, Chapter 2, Section 2.5.2.2
"""

import polars as pl


def etf_trick(
    prices: pl.DataFrame,
    weights: pl.DataFrame,
    timestamp_col: str = "timestamp",
) -> pl.Series:
    """Model a complex basket as a single non-expiring, total-return cash product.

    Keeps pandas internally for complex DataFrame operations. Accepts and returns
    Polars at the public API boundary.

    Args:
        prices: Polars DataFrame with ``timestamp`` column and one column per
            instrument containing close prices.
        weights: Polars DataFrame with ``timestamp`` column and one column per
            instrument containing allocation weights (must sum to ~1 per row).
        timestamp_col: Name of the timestamp column.

    Returns:
        Polars Series K_t representing the cumulative value of $1 invested in
        the basket (total-return ETF index).

    Reference:
        AFML, Chapter 2, Section 2.5.2.2 (The ETF Trick)
    """
    instruments = [c for c in prices.columns if c != timestamp_col]
    n = len(prices)

    # Extract price and weight arrays as Python lists for loop computation
    px: dict[str, list[float]] = {inst: prices[inst].to_list() for inst in instruments}
    wt: dict[str, list[float]] = {inst: weights[inst].to_list() for inst in instruments}

    k = [0.0] * n
    k[0] = 1.0

    # Initial holdings: allocate $1 according to weights
    sum_abs_w0 = sum(abs(wt[inst][0]) for inst in instruments)
    holdings: dict[str, float] = {}
    for inst in instruments:
        denom = px[inst][0] * sum_abs_w0
        holdings[inst] = (wt[inst][0] * k[0]) / denom if denom != 0.0 else 0.0

    for i in range(1, n):
        pnl = 0.0
        for inst in instruments:
            delta = px[inst][i] - px[inst][i - 1]
            pnl += holdings[inst] * delta
        k[i] = k[i - 1] + pnl

    return pl.Series("k_t", k, dtype=pl.Float64)
