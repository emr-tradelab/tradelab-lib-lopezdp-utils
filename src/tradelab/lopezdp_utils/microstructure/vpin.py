"""Volume-Synchronized Probability of Informed Trading (VPIN) from AFML Chapter 19.

VPIN is a third-generation microstructural feature that estimates the
probability of informed trading in real-time using volume-clock sampling.
Rising VPIN indicates increasing order flow toxicity, serving as an
early warning for adverse selection and liquidity crises.

Reference:
    AFML Chapter 19, Section 19.3.3
    Easley, López de Prado & O'Hara (2012)
"""

import pandas as pd

from tradelab.lopezdp_utils.microstructure.trade_classification import tick_rule


def _classify_volume_in_bucket(prices: pd.Series, volumes: pd.Series) -> tuple[float, float]:
    """Classify volume within a bucket as buy or sell using tick rule.

    Args:
        prices: Transaction prices within the bucket.
        volumes: Transaction volumes within the bucket.

    Returns:
        Tuple of (buy_volume, sell_volume).
    """
    signs = tick_rule(prices)
    buy_vol = volumes[signs > 0].sum()
    sell_vol = volumes[signs < 0].sum()
    return float(buy_vol), float(sell_vol)


def volume_bucket(
    prices: pd.Series,
    volumes: pd.Series,
    bucket_size: float,
) -> pd.DataFrame:
    """Partition trade data into equal-volume buckets.

    Each bucket accumulates trades until the total volume reaches
    bucket_size. Partial trades at bucket boundaries are split across
    buckets.

    Args:
        prices: Transaction prices.
        volumes: Transaction volumes.
        bucket_size: Target volume per bucket (V).

    Returns:
        DataFrame with columns: 'buy_volume', 'sell_volume', 'bucket_end_time'.
    """
    buckets: list[dict] = []
    cum_vol = 0.0
    bucket_buy = 0.0
    bucket_sell = 0.0
    signs = tick_rule(prices)

    for i in range(len(prices)):
        vol = volumes.iloc[i]
        sign = signs.iloc[i]

        remaining = vol
        while remaining > 0:
            space = bucket_size - cum_vol
            fill = min(remaining, space)

            if sign > 0:
                bucket_buy += fill
            else:
                bucket_sell += fill

            cum_vol += fill
            remaining -= fill

            if cum_vol >= bucket_size:
                buckets.append(
                    {
                        "buy_volume": bucket_buy,
                        "sell_volume": bucket_sell,
                        "bucket_end_time": prices.index[i],
                    }
                )
                cum_vol = 0.0
                bucket_buy = 0.0
                bucket_sell = 0.0

    return pd.DataFrame(buckets)


def vpin(
    prices: pd.Series,
    volumes: pd.Series,
    bucket_size: float,
    n_buckets: int = 50,
) -> pd.Series:
    """Compute Volume-Synchronized Probability of Informed Trading.

    VPIN estimates the fraction of informed trading by measuring order
    flow imbalance under volume clock. A rolling window of n_buckets
    is used to smooth the estimate.

    Formula:
        VPIN_τ = (1/n) * Σ_{i=1}^{n} |V^B_{τ-i+1} - V^S_{τ-i+1}| / V

    where V is the bucket size and n is the number of buckets in the
    rolling window.

    Args:
        prices: Transaction prices.
        volumes: Transaction volumes.
        bucket_size: Target volume per bucket (V). A good starting point
            is total daily volume / 200 (for ~200 buckets per day).
        n_buckets: Number of buckets in the rolling window for VPIN
            estimation.

    Returns:
        Series of VPIN values indexed by bucket end time.

    Reference:
        AFML Section 19.3.3, Easley, López de Prado & O'Hara (2012)
    """
    buckets = volume_bucket(prices, volumes, bucket_size)
    if buckets.empty:
        return pd.Series(dtype=float)

    imbalance = (buckets["buy_volume"] - buckets["sell_volume"]).abs()
    vpin_values = imbalance.rolling(window=n_buckets).sum() / (n_buckets * bucket_size)
    vpin_series = pd.Series(vpin_values.values, index=buckets["bucket_end_time"])
    return vpin_series.dropna()
