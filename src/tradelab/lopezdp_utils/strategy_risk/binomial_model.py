"""Binomial strategy risk model — AFML Chapter 15.

Framework for computing implied precision, implied betting frequency,
and strategy failure probability using a binomial model of IID bets.

References:
    López de Prado, M. (2018). Advances in Financial Machine Learning.
    Chapter 15: Understanding Strategy Risk.
"""

import numpy as np
import scipy.stats as ss


def sharpe_ratio_symmetric(p: float, n: float) -> float:
    """Annualized Sharpe ratio for symmetric payouts (+π / -π).

    Under symmetric payouts, the Sharpe ratio simplifies to a function
    of precision (p) and betting frequency (n) only.

    Args:
        p: Precision (probability of a winning bet), 0 < p < 1.
        n: Number of bets per year.

    Returns:
        Annualized Sharpe ratio.

    Reference:
        AFML Section 15.4.1, Equation 15.4.
    """
    return n**0.5 * (2.0 * p - 1.0) / (4.0 * p * (1.0 - p)) ** 0.5


def implied_precision_symmetric(n: float, target_sr: float) -> float:
    """Implied precision for symmetric payouts given a target Sharpe ratio.

    Inverts the symmetric Sharpe ratio formula to find the minimum precision
    required to achieve the target annualized Sharpe ratio. This is a special
    case of ``bin_hr`` with sl=-1 and pt=1.

    Args:
        n: Number of bets per year.
        target_sr: Target annualized Sharpe ratio.

    Returns:
        Minimum precision p required, in (0.5, 1).

    Reference:
        AFML Section 15.4.1, Equation 15.5.
    """
    return bin_hr(sl=-1.0, pt=1.0, freq=n, target_sr=target_sr)


def sharpe_ratio_asymmetric(p: float, n: float, sl: float, pt: float) -> float:
    """Annualized Sharpe ratio for asymmetric payouts.

    Generalizes the Sharpe ratio formula to allow different profit-taking
    (π+) and stop-loss (π-) thresholds.

    Args:
        p: Precision (probability of a winning bet).
        n: Number of bets per year.
        sl: Stop-loss threshold (negative value, e.g. -0.01).
        pt: Profit-taking threshold (positive value, e.g. 0.02).

    Returns:
        Annualized Sharpe ratio.

    Reference:
        AFML Section 15.4.2, Equation 15.6.
    """
    mean = (pt - sl) * p + sl
    var = (pt - sl) ** 2 * p * (1.0 - p)
    return n**0.5 * mean / var**0.5


def bin_hr(sl: float, pt: float, freq: float, target_sr: float) -> float:
    """Compute implied precision to achieve a target Sharpe ratio.

    Given a trading rule characterized by stop-loss, profit-taking, and
    betting frequency, computes the minimum precision (hit rate) required
    to achieve the target annualized Sharpe ratio.

    Args:
        sl: Stop-loss threshold (negative value, e.g. -0.01).
        pt: Profit-taking threshold (positive value, e.g. 0.02).
        freq: Number of bets per year.
        target_sr: Target annualized Sharpe ratio.

    Returns:
        Minimum required precision p.

    Reference:
        AFML Snippet 15.3.
    """
    a = (freq + target_sr**2) * (pt - sl) ** 2
    b = (2.0 * freq * sl - target_sr**2 * (pt - sl)) * (pt - sl)
    c = freq * sl**2
    p = (-b + (b**2 - 4.0 * a * c) ** 0.5) / (2.0 * a)
    return p


def bin_freq(sl: float, pt: float, p: float, target_sr: float) -> float:
    """Compute implied betting frequency to achieve a target Sharpe ratio.

    Given a trading rule characterized by stop-loss, profit-taking, and
    precision, computes the number of bets per year needed to achieve the
    target annualized Sharpe ratio.

    Args:
        sl: Stop-loss threshold (negative value, e.g. -0.01).
        pt: Profit-taking threshold (positive value, e.g. 0.02).
        p: Precision (probability of a winning bet).
        target_sr: Target annualized Sharpe ratio.

    Returns:
        Number of bets per year required.

    Reference:
        AFML Snippet 15.4.
    """
    freq = (target_sr * (pt - sl)) ** 2 * p * (1.0 - p) / ((pt - sl) * p + sl) ** 2
    return freq


def mix_gaussians(
    mu1: float,
    mu2: float,
    sigma1: float,
    sigma2: float,
    prob1: float,
    n_obs: int,
) -> np.ndarray:
    """Generate random draws from a mixture of two Gaussians.

    Used to simulate realistic return distributions with separate regimes
    for positive and negative returns.

    Args:
        mu1: Mean of the first Gaussian component.
        mu2: Mean of the second Gaussian component.
        sigma1: Std deviation of the first component.
        sigma2: Std deviation of the second component.
        prob1: Mixing probability for the first component.
        n_obs: Total number of observations to generate.

    Returns:
        Array of shuffled random draws from the mixture.

    Reference:
        AFML Snippet 15.5.
    """
    ret1 = np.random.normal(mu1, sigma1, size=int(n_obs * prob1))
    ret2 = np.random.normal(mu2, sigma2, size=int(n_obs) - ret1.shape[0])
    ret = np.append(ret1, ret2, axis=0)
    np.random.shuffle(ret)
    return ret


def prob_failure(ret: np.ndarray, freq: float, target_sr: float) -> float:
    """Estimate probability that a strategy fails to meet its Sharpe target.

    Derives empirical precision and payout structure from a return series,
    computes the precision hurdle via ``bin_hr``, and estimates the
    probability that the true precision falls below this hurdle using a
    Gaussian approximation.

    Args:
        ret: Array of strategy returns (positive and negative).
        freq: Number of bets per year.
        target_sr: Target annualized Sharpe ratio.

    Returns:
        Probability of strategy failure (0 to 1).

    Reference:
        AFML Snippet 15.5.
    """
    r_pos = ret[ret > 0].mean()
    r_neg = ret[ret <= 0].mean()
    p = ret[ret > 0].shape[0] / float(ret.shape[0])
    thres_p = bin_hr(r_neg, r_pos, freq, target_sr)
    risk = ss.norm.cdf(thres_p, p, (p * (1.0 - p) / freq) ** 0.5)
    return risk
