"""Binomial strategy risk model — AFML Chapter 15.

References:
    López de Prado, "Advances in Financial Machine Learning", Chapter 15
"""

import numpy as np
import polars as pl
import scipy.stats as ss


def sharpe_ratio_symmetric(p: float, n: float) -> float:
    """Annualized Sharpe ratio for symmetric payouts (+pi / -pi).

    Args:
        p: Precision (probability of a winning bet), 0 < p < 1.
        n: Number of bets per year.

    Returns:
        Annualized Sharpe ratio.
    """
    return n**0.5 * (2.0 * p - 1.0) / (4.0 * p * (1.0 - p)) ** 0.5


def implied_precision_symmetric(n: float, target_sr: float) -> float:
    """Implied precision for symmetric payouts given a target Sharpe ratio.

    Args:
        n: Number of bets per year.
        target_sr: Target annualized Sharpe ratio.

    Returns:
        Minimum precision p required.
    """
    return bin_hr(sl=-1.0, pt=1.0, freq=n, target_sr=target_sr)


def sharpe_ratio_asymmetric(p: float, n: float, sl: float, pt: float) -> float:
    """Annualized Sharpe ratio for asymmetric payouts.

    Args:
        p: Precision.
        n: Number of bets per year.
        sl: Stop-loss threshold (negative value).
        pt: Profit-taking threshold (positive value).

    Returns:
        Annualized Sharpe ratio.
    """
    mean = (pt - sl) * p + sl
    var = (pt - sl) ** 2 * p * (1.0 - p)
    return n**0.5 * mean / var**0.5


def bin_hr(sl: float, pt: float, freq: float, target_sr: float) -> float:
    """Compute implied precision to achieve a target Sharpe ratio.

    Args:
        sl: Stop-loss threshold (negative value).
        pt: Profit-taking threshold (positive value).
        freq: Number of bets per year.
        target_sr: Target annualized Sharpe ratio.

    Returns:
        Minimum required precision p.
    """
    a = (freq + target_sr**2) * (pt - sl) ** 2
    b = (2.0 * freq * sl - target_sr**2 * (pt - sl)) * (pt - sl)
    c = freq * sl**2
    p = (-b + (b**2 - 4.0 * a * c) ** 0.5) / (2.0 * a)
    return p


def bin_freq(sl: float, pt: float, p: float, target_sr: float) -> float:
    """Compute implied betting frequency to achieve a target Sharpe ratio.

    Args:
        sl: Stop-loss threshold (negative value).
        pt: Profit-taking threshold (positive value).
        p: Precision.
        target_sr: Target annualized Sharpe ratio.

    Returns:
        Number of bets per year required.
    """
    return (target_sr * (pt - sl)) ** 2 * p * (1.0 - p) / ((pt - sl) * p + sl) ** 2


def mix_gaussians(
    mu1: float,
    mu2: float,
    sigma1: float,
    sigma2: float,
    prob1: float,
    n_obs: int,
    seed: int | None = None,
) -> np.ndarray:
    """Generate random draws from a mixture of two Gaussians.

    Args:
        mu1: Mean of the first component.
        mu2: Mean of the second component.
        sigma1: Std of the first component.
        sigma2: Std of the second component.
        prob1: Mixing probability for the first component.
        n_obs: Total number of observations.
        seed: Random seed for reproducibility.

    Returns:
        Array of shuffled random draws.
    """
    rng = np.random.default_rng(seed)
    n1 = int(n_obs * prob1)
    ret1 = rng.normal(mu1, sigma1, size=n1)
    ret2 = rng.normal(mu2, sigma2, size=int(n_obs) - n1)
    ret = np.append(ret1, ret2, axis=0)
    rng.shuffle(ret)
    return ret


def prob_failure(ret: pl.Series, freq: float, target_sr: float) -> float:
    """Estimate probability that a strategy fails to meet its Sharpe target.

    Args:
        ret: Polars Series of strategy returns.
        freq: Number of bets per year.
        target_sr: Target annualized Sharpe ratio.

    Returns:
        Probability of strategy failure (0 to 1).
    """
    r = ret.to_numpy()
    pos_mask = r > 0
    neg_mask = r <= 0
    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        return float("nan")
    r_pos = r[pos_mask].mean()
    r_neg = r[neg_mask].mean()
    p = pos_mask.sum() / float(r.shape[0])
    if p == 0 or p == 1:
        return float("nan")
    thres_p = bin_hr(r_neg, r_pos, freq, target_sr)
    risk = ss.norm.cdf(thres_p, p, (p * (1.0 - p) / freq) ** 0.5)
    return float(risk)
