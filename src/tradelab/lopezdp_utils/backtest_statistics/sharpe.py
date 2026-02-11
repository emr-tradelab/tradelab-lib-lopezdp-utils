"""Sharpe ratio variants from AFML Chapter 14.

Implements the standard Sharpe ratio, Probabilistic Sharpe Ratio (PSR),
and Deflated Sharpe Ratio (DSR). PSR adjusts for non-normality and
short track records. DSR further corrects for selection bias under
multiple testing (SBuMT).

Reference: AFML Sections 14.5-14.7
"""

import numpy as np
from scipy.stats import norm


def sharpe_ratio(returns: np.ndarray, periods_per_year: float = 252.0) -> float:
    """Compute annualized Sharpe ratio from excess returns.

    Args:
        returns: Array of excess returns (already net of risk-free rate).
        periods_per_year: Annualization factor (252 for daily, 12 for monthly).

    Returns:
        Annualized Sharpe ratio.

    Reference:
        AFML Section 14.5
    """
    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1)
    if sigma == 0:
        return 0.0
    return (mu / sigma) * np.sqrt(periods_per_year)


def probabilistic_sharpe_ratio(
    observed_sr: float,
    benchmark_sr: float,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Compute Probabilistic Sharpe Ratio (PSR).

    Adjusts the observed SR for non-normality (skewness/kurtosis) and
    finite sample size. Returns the probability that the true SR exceeds
    the benchmark SR.

    Args:
        observed_sr: Observed (non-annualized) Sharpe ratio.
        benchmark_sr: Benchmark SR to test against (often 0).
        n_obs: Number of return observations.
        skew: Skewness of returns (0 for Gaussian).
        kurtosis: Kurtosis of returns (3 for Gaussian).

    Returns:
        PSR as a probability in [0, 1].

    Reference:
        AFML Section 14.6
    """
    z = (observed_sr - benchmark_sr) * np.sqrt(n_obs - 1)
    z /= np.sqrt(1 - skew * observed_sr + (kurtosis - 1) / 4.0 * observed_sr**2)
    return float(norm.cdf(z))


def deflated_sharpe_ratio(
    observed_sr: float,
    sr_estimates: np.ndarray,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Compute Deflated Sharpe Ratio (DSR).

    Corrects the observed SR for selection bias under multiple testing
    (SBuMT). Uses the expected maximum SR across N trials as the
    benchmark for PSR.

    Args:
        observed_sr: Observed (non-annualized) Sharpe ratio of best strategy.
        sr_estimates: Array of Sharpe ratios from all N backtesting trials.
        n_obs: Number of return observations per trial.
        skew: Skewness of returns.
        kurtosis: Kurtosis of returns.

    Returns:
        DSR as a probability in [0, 1]. Values near 0 suggest the
        observed SR is likely due to overfitting.

    Reference:
        AFML Section 14.7
    """
    n_trials = len(sr_estimates)
    sr_std = np.std(sr_estimates, ddof=1)

    # Expected maximum SR under multiple testing (Euler-Mascheroni approx)
    euler_mascheroni = 0.5772156649015329
    sr_max = sr_std * (
        (1 - euler_mascheroni) * norm.ppf(1 - 1.0 / n_trials)
        + euler_mascheroni * norm.ppf(1 - 1.0 / (n_trials * np.e))
    )

    return probabilistic_sharpe_ratio(
        observed_sr=observed_sr,
        benchmark_sr=sr_max,
        n_obs=n_obs,
        skew=skew,
        kurtosis=kurtosis,
    )
