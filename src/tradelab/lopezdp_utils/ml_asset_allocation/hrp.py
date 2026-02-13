"""Hierarchical Risk Parity (HRP) — AFML Chapter 16.

Graph-theory and ML-based portfolio allocation that bypasses covariance matrix
inversion, addressing instability of traditional mean-variance optimization.

Three stages:
    1. Tree clustering — group assets by correlation distance
    2. Quasi-diagonalization — reorder so similar assets are adjacent
    3. Recursive bisection — top-down weight allocation by inverse cluster variance

Reference: AFML Snippets 16.1–16.4.
"""

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch


def correl_dist(corr: pd.DataFrame) -> pd.DataFrame:
    """Compute correlation-based distance matrix.

    Converts correlation to a proper metric: d(i,j) = sqrt(0.5 * (1 - ρ(i,j))).
    Distance is 0 when ρ=1 (identical) and 1 when ρ=-1 (opposite).

    Args:
        corr: Correlation matrix.

    Returns:
        Distance matrix with values in [0, 1].

    Reference:
        AFML Snippet 16.1.
    """
    dist = ((1 - corr) / 2.0) ** 0.5
    return dist


def tree_clustering(
    corr: pd.DataFrame, method: str = "single"
) -> np.ndarray:
    """Hierarchical tree clustering on correlation distance matrix.

    Computes the linkage matrix from a correlation-based distance metric
    using scipy's hierarchical clustering.

    Args:
        corr: Correlation matrix.
        method: Linkage method (default 'single' as in the book).

    Returns:
        Linkage matrix from scipy.cluster.hierarchy.linkage.

    Reference:
        AFML Snippet 16.1.
    """
    dist = correl_dist(corr)
    link = sch.linkage(dist, method)
    return link


def get_quasi_diag(link: np.ndarray) -> list[int]:
    """Quasi-diagonalize the covariance matrix via sorted tree traversal.

    Reorders items so that similar assets are placed next to each other,
    producing a quasi-diagonal covariance matrix where largest values
    concentrate along the diagonal.

    Args:
        link: Linkage matrix from tree_clustering.

    Returns:
        Sorted list of original item indices.

    Reference:
        AFML Snippet 16.2.
    """
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]  # number of original items
    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)  # make space
        df0 = sort_ix[sort_ix >= num_items]  # find clusters
        i = df0.index
        j = df0.values - num_items
        sort_ix[i] = link[j, 0]  # item 1
        df0 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df0])  # item 2
        sort_ix = sort_ix.sort_index()  # re-sort
        sort_ix.index = range(sort_ix.shape[0])  # re-index
    return sort_ix.tolist()


def get_ivp(cov: np.ndarray | pd.DataFrame) -> np.ndarray:
    """Compute inverse-variance portfolio weights.

    Allocates in inverse proportion to each asset's variance:
    w_n = (1/V_{n,n}) / Σ(1/V_{i,i}).

    Args:
        cov: Covariance matrix (array or DataFrame).

    Returns:
        Weight array summing to 1.

    Reference:
        AFML Snippet 16.4.
    """
    if isinstance(cov, pd.DataFrame):
        cov = cov.values
    ivp = 1.0 / np.diag(cov)
    ivp /= ivp.sum()
    return ivp


def get_cluster_var(cov: pd.DataFrame, c_items: list) -> float:
    """Compute variance of a cluster using IVP weights.

    Calculates w' * Σ * w where w are inverse-variance weights
    for the cluster's sub-covariance matrix.

    Args:
        cov: Full covariance matrix (DataFrame).
        c_items: List of item labels belonging to the cluster.

    Returns:
        Cluster variance (scalar).

    Reference:
        AFML Snippet 16.4.
    """
    cov_ = cov.loc[c_items, c_items]  # matrix slice
    w_ = get_ivp(cov_).reshape(-1, 1)
    c_var = np.dot(np.dot(w_.T, cov_), w_)[0, 0]
    return c_var


def get_rec_bipart(cov: pd.DataFrame, sort_ix: list) -> pd.Series:
    """Allocate weights via top-down recursive bisection.

    Splits the sorted asset list into halves, allocating weight to each
    half in inverse proportion to its cluster variance. Recurses until
    each cluster contains a single asset.

    Args:
        cov: Covariance matrix (DataFrame).
        sort_ix: Sorted list of asset labels from get_quasi_diag.

    Returns:
        Portfolio weights indexed by asset labels.

    Reference:
        AFML Snippet 16.3.
    """
    w = pd.Series(1.0, index=sort_ix)
    c_items = [sort_ix]  # initialize all items in one cluster
    while len(c_items) > 0:
        # bisect each cluster
        c_items = [
            i[j:k]
            for i in c_items
            for j, k in ((0, len(i) // 2), (len(i) // 2, len(i)))
            if len(i) > 1
        ]
        # iterate through pairs
        for i in range(0, len(c_items), 2):
            c_items0 = c_items[i]
            c_items1 = c_items[i + 1]
            c_var0 = get_cluster_var(cov, c_items0)
            c_var1 = get_cluster_var(cov, c_items1)
            alpha = 1 - c_var0 / (c_var0 + c_var1)
            w[c_items0] *= alpha
            w[c_items1] *= 1 - alpha
    return w


def hrp_alloc(cov: pd.DataFrame, corr: pd.DataFrame | None = None) -> pd.Series:
    """Full Hierarchical Risk Parity allocation.

    Combines all three HRP stages: tree clustering, quasi-diagonalization,
    and recursive bisection to produce portfolio weights.

    Args:
        cov: Covariance matrix (DataFrame with asset labels).
        corr: Correlation matrix. If None, derived from cov.

    Returns:
        Portfolio weights indexed by asset labels, sorted by index.

    Reference:
        AFML Snippet 16.4.
    """
    if corr is None:
        # derive correlation from covariance
        std = np.sqrt(np.diag(cov.values))
        corr = pd.DataFrame(
            cov.values / np.outer(std, std),
            index=cov.index,
            columns=cov.columns,
        )
    # stage 1: tree clustering
    link = tree_clustering(corr)
    # stage 2: quasi-diagonalization
    sort_ix = get_quasi_diag(link)
    sort_ix = corr.index[sort_ix].tolist()  # recover original labels
    # stage 3: recursive bisection
    hrp = get_rec_bipart(cov, sort_ix)
    return hrp.sort_index()
