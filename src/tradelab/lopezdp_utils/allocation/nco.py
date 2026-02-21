"""Nested Clustered Optimization (NCO) — MLAM Chapter 7.

Modular portfolio optimization that clusters assets, optimizes within clusters,
then optimizes across clusters using a reduced covariance matrix. Can wrap
any optimizer (minimum variance, max Sharpe, etc.).

Depends on ONC clustering from features/importance.py.

Reference: ML for Asset Managers, Snippets 7.3-7.6.
"""

import numpy as np
import pandas as pd

from tradelab.lopezdp_utils.allocation.denoising import _cov2corr
from tradelab.lopezdp_utils.features.importance import cluster_kmeans_base


def _min_var_port(cov: np.ndarray) -> np.ndarray:
    """Compute minimum variance portfolio via inverse covariance."""
    inv_cov = np.linalg.inv(cov)
    ones = np.ones((cov.shape[0], 1))
    w = np.dot(inv_cov, ones)
    w /= np.dot(ones.T, w)
    return w


def _opt_port(cov: np.ndarray, mu: np.ndarray | None = None) -> np.ndarray:
    """General portfolio optimization.

    If mu is None, computes minimum variance portfolio.
    If mu is provided, computes maximum Sharpe ratio portfolio.
    """
    inv_cov = np.linalg.inv(cov)
    if mu is None:
        ones = np.ones((cov.shape[0], 1))
        w = np.dot(inv_cov, ones)
        w /= np.dot(ones.T, w)
    else:
        w = np.dot(inv_cov, mu)
        ones = np.ones((cov.shape[0], 1))
        w /= np.dot(ones.T, w)  # sum-to-one per Snippet 2.10
    return w


def opt_port_nco(
    cov: pd.DataFrame,
    mu: pd.Series | None = None,
    max_num_clusters: int | None = None,
) -> pd.Series:
    """Nested Clustered Optimization.

    Three-step process:
        1. Cluster assets using ONC on the correlation matrix
        2. Optimize within each cluster independently (intracluster)
        3. Optimize across clusters using reduced covariance (intercluster)

    Final weights = intracluster weights x intercluster weights.

    Args:
        cov: Covariance matrix (DataFrame with asset labels).
        mu: Expected returns (Series). If None, min-variance optimization.
        max_num_clusters: Maximum K for K-means. If None, auto-detect.

    Returns:
        Portfolio weights indexed by asset labels.

    Reference:
        MLAM Snippets 7.3-7.6.
    """
    cov = pd.DataFrame(cov)
    if mu is not None:
        mu = pd.Series(mu)

    # step 1: clustering
    corr1 = pd.DataFrame(_cov2corr(cov.values), index=cov.index, columns=cov.columns)
    if max_num_clusters is None:
        max_num_clusters = max(2, corr1.shape[0] // 2)
    corr1, clstrs, _ = cluster_kmeans_base(corr1, max_num_clusters=max_num_clusters, n_init=10)

    # step 2: intracluster allocations
    w_intra = pd.DataFrame(0.0, index=cov.index, columns=clstrs.keys())
    for i in clstrs:
        cov_ = cov.loc[clstrs[i], clstrs[i]].values
        if mu is None:
            mu_ = None
        else:
            mu_ = mu.loc[clstrs[i]].values.reshape(-1, 1)
        w_intra.loc[clstrs[i], i] = _opt_port(cov_, mu_).flatten()

    # step 3: intercluster allocations
    cov_ = w_intra.T.dot(np.dot(cov, w_intra))  # reduced covariance
    mu_ = None if mu is None else w_intra.T.dot(mu)
    if mu_ is not None:
        mu_ = mu_.values.reshape(-1, 1)
    w_inter = pd.Series(_opt_port(cov_.values, mu_).flatten(), index=cov_.index)

    # final: product of intra and inter
    nco = w_intra.mul(w_inter, axis=1).sum(axis=1)
    return nco.sort_index()
