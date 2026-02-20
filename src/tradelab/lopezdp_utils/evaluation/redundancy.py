"""Strategy redundancy via ONC clustering — AFML Chapter 14 / MLAM Section 6.

Reduces the effective number of independent trials for Deflated Sharpe Ratio
by clustering correlated strategies with ONC and aggregating via minimum-variance.

References:
    López de Prado, "Advances in Financial Machine Learning", Chapter 14
    López de Prado, "Machine Learning for Asset Managers", Section 6
"""

import numpy as np
import pandas as pd
import polars as pl

from tradelab.lopezdp_utils.evaluation.statistics import deflated_sharpe_ratio
from tradelab.lopezdp_utils.features.importance import cluster_kmeans_top


def get_effective_trials(trial_returns: pl.DataFrame) -> dict:
    """Cluster strategy trials by correlation and compute effective trial count.

    Uses ONC to discover K independent strategy groups from N correlated trials,
    then forms minimum-variance aggregate returns per cluster.

    Args:
        trial_returns: DataFrame with 'timestamp' column and one column per
            strategy trial containing returns.

    Returns:
        Dictionary with:
        - 'n_effective': Number of independent clusters (K).
        - 'cluster_srs': List of non-annualized Sharpe ratios per cluster.
        - 'sr_variance': Variance of cluster Sharpe ratios.
        - 'clusters': Dict mapping cluster ID to list of trial column names.
    """
    cols = [c for c in trial_returns.columns if c != "timestamp"]
    if len(cols) < 2:
        raise ValueError("Need at least 2 strategy trials")

    # Build pandas DataFrame for ONC (it requires pd.DataFrame with named columns)
    ret_pd = pd.DataFrame({c: trial_returns[c].to_numpy() for c in cols}, columns=cols)

    # Correlation matrix
    corr = ret_pd.corr()

    # ONC clustering
    _, clstrs, _ = cluster_kmeans_top(corr)

    # Minimum-variance aggregate per cluster
    cluster_srs = []
    for _cid, members in clstrs.items():
        cluster_ret = ret_pd[members]
        cov = cluster_ret.cov()

        # Minimum-variance weights: w = Σ^{-1} 1 / (1' Σ^{-1} 1)
        try:
            inv_cov = np.linalg.inv(cov.values)
        except np.linalg.LinAlgError:
            inv_cov = np.linalg.pinv(cov.values)

        ones = np.ones(len(members))
        w = inv_cov @ ones
        w = w / w.sum()

        agg_ret = cluster_ret.values @ w
        mu = agg_ret.mean()
        sigma = agg_ret.std()
        sr = mu / sigma if sigma > 0 else 0.0
        cluster_srs.append(float(sr))

    return {
        "n_effective": len(clstrs),
        "cluster_srs": cluster_srs,
        "sr_variance": float(np.var(cluster_srs, ddof=1)) if len(cluster_srs) > 1 else 0.0,
        "clusters": clstrs,
    }


def deflated_sharpe_ratio_clustered(
    observed_sr: float,
    trial_returns: pl.DataFrame,
    n_obs: int = 252,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Deflated Sharpe Ratio with ONC-based redundancy correction.

    Instead of treating all N trials as independent, discovers K effective
    independent clusters and uses their SR variance for the expected-max
    SR hurdle.

    Args:
        observed_sr: Observed **non-annualized** Sharpe ratio of best strategy.
        trial_returns: DataFrame with 'timestamp' and trial return columns.
        n_obs: Number of return observations per trial.
        skew: Skewness of returns (0 for Gaussian).
        kurtosis: Raw kurtosis of returns (3 for Gaussian, NOT excess kurtosis).

    Returns:
        DSR as a probability in [0, 1].
    """
    eff = get_effective_trials(trial_returns)
    return deflated_sharpe_ratio(
        observed_sr=observed_sr,
        sr_estimates=eff["cluster_srs"],
        n_obs=n_obs,
        skew=skew,
        kurtosis=kurtosis,
    )
