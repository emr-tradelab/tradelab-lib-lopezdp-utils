"""Microstructural features: spread estimators, price impact, and VPIN.

Merges trade_classification, spread_estimators, price_impact, and vpin
from the v1 microstructure package.

Reference: Advances in Financial Machine Learning, Chapter 19
"""

import numpy as np
import polars as pl
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------------------------------
# Trade Classification
# ---------------------------------------------------------------------------


def tick_rule(prices: pl.Series) -> pl.Series:
    """Classify trades as buyer- or seller-initiated using the tick rule.

    A trade is classified as +1 (buy) if the price is higher than the
    previous trade, -1 (sell) if lower. Zero-tick (no change) carries
    forward the prior classification.

    Args:
        prices: Polars Series of transaction prices.

    Returns:
        Polars Series of trade signs (+1/-1, Integer type).

    Reference:
        AFML Section 19.3.1
    """
    arr = prices.to_numpy()
    diffs = np.diff(arr, prepend=arr[0])
    raw = np.sign(diffs).astype(np.float64)

    # Forward-fill zeros (zero-tick rule)
    last = 1.0
    for i in range(len(raw)):
        if raw[i] == 0.0:
            raw[i] = last
        else:
            last = raw[i]

    return pl.Series(prices.name, raw.astype(np.int8), dtype=pl.Int8)


# ---------------------------------------------------------------------------
# Corwin-Schultz Spread Estimator
# ---------------------------------------------------------------------------


def _get_beta(high: pl.Series, low: pl.Series, sl: int = 1) -> pl.Series:
    """Compute beta for Corwin-Schultz: rolling mean of pairwise log hl^2.

    Reference: AFML Snippet 19.1
    """
    log_hl2 = (high / low).log().pow(2)
    beta = log_hl2.rolling_sum(window_size=2).rolling_mean(window_size=sl)
    return beta


def _get_gamma(high: pl.Series, low: pl.Series) -> pl.Series:
    """Compute gamma for Corwin-Schultz: squared log of 2-bar HL range.

    Reference: AFML Snippet 19.1
    """
    h2 = high.rolling_max(window_size=2)
    l2 = low.rolling_min(window_size=2)
    gamma = (h2 / l2).log().pow(2)
    return gamma


def _get_alpha(beta: pl.Series, gamma: pl.Series) -> pl.Series:
    """Compute alpha from beta and gamma for Corwin-Schultz.

    Reference: AFML Snippet 19.1
    """
    den = 3 - 2 * (2**0.5)
    alpha = (2**0.5 - 1) * beta.sqrt() / den - (gamma / den).sqrt()
    # Clip negatives to zero per Corwin-Schultz (2012)
    alpha = alpha.clip(lower_bound=0.0)
    return alpha


def corwin_schultz_spread(df: pl.DataFrame, sl: int = 1) -> pl.DataFrame:
    """Estimate bid-ask spread from high-low prices (Corwin-Schultz).

    Args:
        df: DataFrame with 'high' and 'low' columns.
        sl: Sample length for beta estimation (rolling window).

    Returns:
        Polars DataFrame with 'spread' column.

    Reference:
        AFML Snippet 19.1, Corwin & Schultz (2012)
    """
    beta = _get_beta(df["high"], df["low"], sl)
    gamma = _get_gamma(df["high"], df["low"])
    alpha = _get_alpha(beta, gamma)
    spread = 2 * (alpha.exp() - 1) / (1 + alpha.exp())
    return pl.DataFrame({"spread": spread})


def becker_parkinson_volatility(df: pl.DataFrame, sl: int = 1) -> pl.Series:
    """Estimate volatility via Becker-Parkinson method.

    Args:
        df: DataFrame with 'high' and 'low' columns.
        sl: Sample length for beta estimation.

    Returns:
        Polars Series of estimated volatility values.

    Reference:
        AFML Snippet 19.2
    """
    beta = _get_beta(df["high"], df["low"], sl)
    gamma = _get_gamma(df["high"], df["low"])
    k2 = (8 / np.pi) ** 0.5
    den = 3 - 2 * (2**0.5)
    sigma = (2**-0.5 - 1) * beta.sqrt() / (k2 * den) + (gamma / (k2**2 * den)).sqrt()
    return sigma.clip(lower_bound=0.0)


# ---------------------------------------------------------------------------
# Roll Model (keeps NumPy internally — needs serial covariance)
# ---------------------------------------------------------------------------


def roll_model(prices: pl.Series) -> dict[str, float]:
    """Estimate effective bid-ask spread using Roll's model.

    Keeps NumPy/pandas internally (autocorr computation).

    Args:
        prices: Polars Series of transaction prices.

    Returns:
        Dictionary with 'spread', 'half_spread', 'noise_variance'.

    Reference:
        AFML Section 19.3.1, Roll (1984)
    """
    import pandas as pd  # local import — autocorr not available in Polars

    px = pd.Series(prices.to_numpy())
    dp = px.diff().dropna()
    serial_cov = dp.autocorr(lag=1) * dp.var()
    c = np.sqrt(max(0.0, -serial_cov))
    sigma_u_sq = dp.var() + 2 * serial_cov
    return {
        "spread": 2 * c,
        "half_spread": c,
        "noise_variance": max(0.0, sigma_u_sq),
    }


def high_low_volatility(high: pl.Series, low: pl.Series) -> float:
    """Estimate volatility using Parkinson's high-low range estimator.

    Args:
        high: Series of high prices per bar.
        low: Series of low prices per bar.

    Returns:
        Estimated volatility (scalar).

    Reference:
        AFML Section 19.3.1, Parkinson (1980)
    """
    k1 = 4 * np.log(2)
    h = high.to_numpy()
    lo = low.to_numpy()
    log_hl_sq = np.log(h / lo) ** 2
    return float(np.sqrt(log_hl_sq.mean() / k1))


# ---------------------------------------------------------------------------
# Price Impact Models
# ---------------------------------------------------------------------------


def kyle_lambda(
    prices: pl.Series,
    volume: pl.Series,
    signs: pl.Series,
) -> dict[str, float]:
    """Estimate Kyle's Lambda (price impact coefficient).

    Regression: Δp_t = λ * (b_t * V_t) + ε_t

    Args:
        prices: Polars Series of transaction prices.
        volume: Polars Series of traded volume.
        signs: Polars Series of trade signs (+1/-1).

    Returns:
        Dictionary with 'lambda' and 't_stat'.

    Reference:
        AFML Section 19.3.2, Kyle (1985)
    """
    dp = prices.diff().drop_nulls().to_numpy()
    sv = (signs * volume).slice(1).to_numpy()  # align with diff

    min_len = min(len(dp), len(sv))
    dp = dp[:min_len]
    sv = sv[:min_len]

    x = sv.reshape(-1, 1)
    reg = LinearRegression(fit_intercept=True).fit(x, dp)
    y_hat = reg.predict(x)
    residuals = dp - y_hat
    n = len(dp)
    denom = np.sum((x - x.mean()) ** 2)
    se = np.sqrt(np.sum(residuals**2) / max(n - 2, 1) / max(denom, 1e-10))
    t_stat = reg.coef_[0] / se if se > 0 else np.nan

    return {"lambda": float(reg.coef_[0]), "t_stat": float(t_stat)}


def amihud_lambda(
    prices: pl.Series,
    dollar_volume: pl.Series,
) -> float:
    """Estimate Amihud's illiquidity ratio.

    Formula: λ = mean( |Δlog(p_t)| / DollarVolume_t )

    Args:
        prices: Polars Series of bar closing prices.
        dollar_volume: Polars Series of dollar volume per bar.

    Returns:
        Amihud illiquidity ratio (scalar).

    Reference:
        AFML Section 19.3.2, Amihud (2002)
    """
    log_ret = (prices / prices.shift(1)).log().abs()
    dv = dollar_volume
    # Align: log_ret has null at position 0
    log_ret_arr = log_ret.slice(1).to_numpy()
    dv_arr = dv.slice(1).to_numpy()
    mask = dv_arr > 0
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(log_ret_arr[mask] / dv_arr[mask]))


def hasbrouck_lambda(
    prices: pl.Series,
    dollar_volume: pl.Series,
    signs: pl.Series,
) -> dict[str, float]:
    """Estimate Hasbrouck's Lambda (effective trading cost).

    Regression: Δlog(p_t) = λ * b_t * √(DollarVolume_t) + ε_t

    Args:
        prices: Polars Series of bar closing prices.
        dollar_volume: Polars Series of dollar volume per bar.
        signs: Polars Series of aggregated trade signs (+1/-1).

    Returns:
        Dictionary with 'lambda' and 't_stat'.

    Reference:
        AFML Section 19.3.2, Hasbrouck (2009)
    """
    log_ret = (prices / prices.shift(1)).log().drop_nulls().to_numpy()
    signed_root_dv = (signs * dollar_volume.sqrt()).slice(1).to_numpy()

    min_len = min(len(log_ret), len(signed_root_dv))
    log_ret = log_ret[:min_len]
    signed_root_dv = signed_root_dv[:min_len]

    x = signed_root_dv.reshape(-1, 1)
    reg = LinearRegression(fit_intercept=True).fit(x, log_ret)
    y_hat = reg.predict(x)
    residuals = log_ret - y_hat
    n = len(log_ret)
    denom = np.sum((x - x.mean()) ** 2)
    se = np.sqrt(np.sum(residuals**2) / max(n - 2, 1) / max(denom, 1e-10))
    t_stat = reg.coef_[0] / se if se > 0 else np.nan

    return {"lambda": float(reg.coef_[0]), "t_stat": float(t_stat)}


# ---------------------------------------------------------------------------
# VPIN
# ---------------------------------------------------------------------------


def volume_bucket(
    prices: pl.Series,
    volumes: pl.Series,
    bucket_size: float,
) -> pl.DataFrame:
    """Partition trades into equal-volume buckets using the tick rule.

    Args:
        prices: Polars Series of transaction prices.
        volumes: Polars Series of transaction volumes.
        bucket_size: Target volume per bucket.

    Returns:
        Polars DataFrame with 'buy_volume' and 'sell_volume' per bucket.
    """
    signs = tick_rule(prices).to_numpy().astype(float)
    vols = volumes.to_numpy()
    n = len(prices)

    buckets: list[dict] = []
    cum_vol = 0.0
    bucket_buy = 0.0
    bucket_sell = 0.0

    for i in range(n):
        vol = vols[i]
        sign = signs[i]
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
                buckets.append({"buy_volume": bucket_buy, "sell_volume": bucket_sell})
                cum_vol = 0.0
                bucket_buy = 0.0
                bucket_sell = 0.0

    if not buckets:
        return pl.DataFrame(
            {
                "buy_volume": pl.Series([], dtype=pl.Float64),
                "sell_volume": pl.Series([], dtype=pl.Float64),
            }
        )
    return pl.DataFrame(buckets)


def vpin(
    prices: pl.Series,
    volumes: pl.Series,
    bucket_size: float,
    n_buckets: int = 50,
) -> pl.Series:
    """Compute Volume-Synchronized Probability of Informed Trading.

    Formula: VPIN = (1/n) * Σ |V^B - V^S| / bucket_size

    Args:
        prices: Polars Series of transaction prices.
        volumes: Polars Series of transaction volumes.
        bucket_size: Target volume per bucket.
        n_buckets: Rolling window of buckets for VPIN estimation.

    Returns:
        Polars Series of VPIN values.

    Reference:
        AFML Section 19.3.3, Easley, López de Prado & O'Hara (2012)
    """
    buckets = volume_bucket(prices, volumes, bucket_size)
    if len(buckets) == 0:
        return pl.Series("vpin", [], dtype=pl.Float64)

    imbalance = (buckets["buy_volume"] - buckets["sell_volume"]).abs()
    vpin_vals = imbalance.rolling_sum(window_size=n_buckets) / (n_buckets * bucket_size)
    return vpin_vals.alias("vpin")
