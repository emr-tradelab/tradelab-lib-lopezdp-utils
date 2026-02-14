"""Bid-ask spread estimators from AFML Chapter 19.

Implements first-generation microstructural features for estimating
effective bid-ask spreads and volatility from price data.

References:
    - AFML Chapter 19, Sections 19.3.1
    - Corwin & Schultz (2012) for high-low spread estimator
    - Roll (1984) for serial covariance spread model
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Corwin-Schultz Spread Estimator (Snippet 19.1)
# ---------------------------------------------------------------------------


def get_beta(series: pd.DataFrame, sl: int = 1) -> pd.Series:
    """Compute beta component for Corwin-Schultz estimator.

    Beta is the rolling mean of pairwise sums of squared log high-to-low
    ratios. It captures the volatility component that increases with the
    sampling interval.

    Args:
        series: DataFrame with 'High' and 'Low' columns.
        sl: Sample length for beta estimation (rolling window for mean).

    Returns:
        Beta series with NaN entries dropped.

    Reference:
        AFML Snippet 19.1
    """
    hl = series[["High", "Low"]].values
    hl = np.log(hl[:, 0] / hl[:, 1]) ** 2
    hl = pd.Series(hl, index=series.index)
    beta = hl.rolling(window=2).sum()
    beta = beta.rolling(window=sl).mean()
    return beta.dropna()


def get_gamma(series: pd.DataFrame) -> pd.Series:
    """Compute gamma component for Corwin-Schultz estimator.

    Gamma is the squared log of the 2-bar high-to-low range. Unlike beta,
    the spread component of gamma does not increase with the sampling
    interval.

    Args:
        series: DataFrame with 'High' and 'Low' columns.

    Returns:
        Gamma series with NaN entries dropped.

    Reference:
        AFML Snippet 19.1
    """
    h2 = series["High"].rolling(window=2).max()
    l2 = series["Low"].rolling(window=2).min()
    gamma = np.log(h2.values / l2.values) ** 2
    gamma = pd.Series(gamma, index=h2.index)
    return gamma.dropna()


def get_alpha(beta: pd.Series, gamma: pd.Series) -> pd.Series:
    """Compute alpha from beta and gamma for Corwin-Schultz estimator.

    Alpha separates the spread component from the volatility component
    by exploiting the different scaling properties of beta and gamma.

    Args:
        beta: Beta series from get_beta().
        gamma: Gamma series from get_gamma().

    Returns:
        Alpha series (non-negative, NaN dropped).

    Reference:
        AFML Snippet 19.1
    """
    den = 3 - 2 * 2**0.5
    alpha = (2**0.5 - 1) * (beta**0.5) / den
    alpha -= (gamma / den) ** 0.5
    alpha[alpha < 0] = 0  # Per Corwin-Schultz (2012)
    return alpha.dropna()


def corwin_schultz_spread(series: pd.DataFrame, sl: int = 1) -> pd.DataFrame:
    """Estimate bid-ask spread from high-low prices (Corwin-Schultz).

    Exploits the fact that high prices are typically set by buyer-initiated
    trades (at the ask) and low prices by seller-initiated trades (at the
    bid). The spread component is separated from volatility because the
    volatility component scales with the sampling interval while the spread
    component does not.

    Especially useful for illiquid markets (e.g., corporate bonds) where
    direct spread observation is unavailable.

    Args:
        series: DataFrame with 'High' and 'Low' columns indexed by time.
        sl: Sample length for beta estimation.

    Returns:
        DataFrame with 'Spread' and 'Start_Time' columns.

    Reference:
        AFML Snippet 19.1, Corwin & Schultz (2012)
    """
    beta = get_beta(series, sl)
    gamma = get_gamma(series)
    alpha = get_alpha(beta, gamma)
    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    start_time = pd.Series(series.index[0 : spread.shape[0]], index=spread.index)
    spread = pd.concat([spread, start_time], axis=1)
    spread.columns = ["Spread", "Start_Time"]
    return spread


# ---------------------------------------------------------------------------
# Becker-Parkinson Volatility (Snippet 19.2)
# ---------------------------------------------------------------------------


def becker_parkinson_volatility(series: pd.DataFrame, sl: int = 1) -> pd.Series:
    """Estimate volatility using Becker-Parkinson method.

    A byproduct of the Corwin-Schultz decomposition. Uses the same beta
    and gamma components but extracts the volatility portion instead of
    the spread portion.

    Args:
        series: DataFrame with 'High' and 'Low' columns.
        sl: Sample length for beta estimation.

    Returns:
        Estimated volatility series (non-negative).

    Reference:
        AFML Snippet 19.2
    """
    beta = get_beta(series, sl)
    gamma = get_gamma(series)
    k2 = (8 / np.pi) ** 0.5
    den = 3 - 2 * 2**0.5
    sigma = (2**-0.5 - 1) * beta**0.5 / (k2 * den)
    sigma += (gamma / (k2**2 * den)) ** 0.5
    sigma[sigma < 0] = 0
    return sigma


# ---------------------------------------------------------------------------
# Roll Model (Section 19.3.1)
# ---------------------------------------------------------------------------


def roll_model(prices: pd.Series) -> dict[str, float]:
    """Estimate effective bid-ask spread using Roll's model.

    Assumes observed prices are the sum of an efficient price (random walk)
    and a bid-ask bounce component. The spread is recovered from the negative
    serial covariance of price changes.

    Args:
        prices: Series of observed transaction prices.

    Returns:
        Dictionary with:
            - 'spread': Estimated effective bid-ask spread (2c).
            - 'half_spread': Estimated half-spread (c).
            - 'noise_variance': Estimated variance of the efficient price
              innovation (sigma_u^2).

    Reference:
        AFML Section 19.3.1, Roll (1984)
    """
    dp = prices.diff().dropna()
    serial_cov = dp.autocorr(lag=1) * dp.var()
    c = np.sqrt(max(0.0, -serial_cov))
    sigma_u_sq = dp.var() + 2 * serial_cov
    return {
        "spread": 2 * c,
        "half_spread": c,
        "noise_variance": max(0.0, sigma_u_sq),
    }


# ---------------------------------------------------------------------------
# High-Low Volatility — Parkinson (Section 19.3.1)
# ---------------------------------------------------------------------------


def high_low_volatility(high: pd.Series, low: pd.Series) -> float:
    """Estimate volatility using Parkinson's high-low range estimator.

    More efficient than close-to-close estimators because it uses the full
    price range observed during each bar.

    Formula:
        sigma_HL = sqrt( (1/T) * sum( ln(H_t/L_t)^2 ) / (4*ln(2)) )

    Args:
        high: Series of high prices per bar.
        low: Series of low prices per bar.

    Returns:
        Estimated volatility (scalar).

    Reference:
        AFML Section 19.3.1, Parkinson (1980)
    """
    k1 = 4 * np.log(2)
    log_hl_sq = np.log(high / low) ** 2
    sigma = np.sqrt(log_hl_sq.mean() / k1)
    return float(sigma)
