"""Orthogonal features and PCA-based portfolio weights.

Provides:
- PCA-based feature orthogonalization to alleviate linear substitution effects
- Weighted Kendall's tau consistency check between supervised and unsupervised rankings
- PCA-derived portfolio weights for covariance-based allocation

All functions operate on NumPy arrays or pandas DataFrames (no Polars needed — consumers
are sklearn classifiers and portfolio optimizers).

Reference:
    López de Prado, "Advances in Financial Machine Learning", Chapter 2 (pca_weights)
    and Chapter 8 (orthogonal features).
"""

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import weightedtau


def pca_weights(
    cov: NDArray[np.float64],
    risk_dist: NDArray[np.float64] | None = None,
    risk_target: float = 1.0,
) -> NDArray[np.float64]:
    """Derive portfolio allocation weights based on PCA risk distribution.

    Computes allocation weights based on how risk is distributed among the
    covariance matrix's principal components. Default behavior allocates all
    risk to the principal component with the smallest eigenvalue.

    Args:
        cov: Covariance matrix of assets (symmetric). Shape: (N, N).
        risk_dist: Desired distribution of risks across principal components.
            Shape: (N,). Defaults to allocating 100% to the smallest eigenvalue.
        risk_target: Target total portfolio risk (sigma). Default 1.0.

    Returns:
        Portfolio weights as a column vector. Shape: (N, 1).

    Reference:
        AFML Chapter 2, Snippet 2.1.
    """
    eig_val, eig_vec = np.linalg.eigh(cov)

    indices = eig_val.argsort()[::-1]
    eig_val, eig_vec = eig_val[indices], eig_vec[:, indices]

    if risk_dist is None:
        risk_dist = np.zeros(cov.shape[0])
        risk_dist[-1] = 1.0

    loads = risk_target * (risk_dist / eig_val) ** 0.5
    weights = np.dot(eig_vec, np.reshape(loads, (-1, 1)))
    return weights


def get_e_vec(
    dot: pd.DataFrame,
    var_thres: float,
) -> tuple[pd.Series, pd.DataFrame]:
    """Compute eigenvectors from dot product matrix with dimensionality reduction.

    Performs spectral decomposition on Z'Z and retains enough principal
    components to explain the specified cumulative variance threshold.

    Args:
        dot: Dot product matrix (Z'Z) of standardized features.
        var_thres: Minimum cumulative variance ratio to retain (e.g., 0.95).

    Returns:
        Tuple of (eigenvalues, eigenvectors) as (pd.Series, pd.DataFrame),
        both reduced to the components meeting the variance threshold.

    Reference:
        AFML Snippet 8.5 (helper).
    """
    e_val, e_vec = np.linalg.eigh(dot)
    idx = e_val.argsort()[::-1]
    e_val, e_vec = e_val[idx], e_vec[:, idx]

    e_val = pd.Series(e_val, index=["PC_" + str(i + 1) for i in range(e_val.shape[0])])
    e_vec = pd.DataFrame(e_vec, index=dot.index, columns=e_val.index)
    e_vec = e_vec.loc[:, e_val.index]

    cum_var = e_val.cumsum() / e_val.sum()
    dim = cum_var.values.searchsorted(var_thres)
    e_val, e_vec = e_val.iloc[: dim + 1], e_vec.iloc[:, : dim + 1]
    return e_val, e_vec


def get_ortho_feats(
    dfX: pd.DataFrame,
    var_thres: float = 0.95,
) -> np.ndarray:
    """Compute orthogonal features via PCA.

    Standardizes features, computes the dot product matrix, performs spectral
    decomposition, and projects standardized features onto selected eigenvectors.
    The result is orthogonal principal components that alleviate linear
    substitution effects.

    Args:
        dfX: DataFrame of stationary features (observations x features).
        var_thres: Minimum cumulative variance ratio to retain. Default 0.95.

    Returns:
        NumPy array of orthogonal features (P = Z @ W).

    Note:
        Only alleviates *linear* substitution effects. Consider Clustered Feature
        Importance (MLAM) as an alternative that preserves interpretability.

    Reference:
        AFML Snippet 8.5.
    """
    dfZ = dfX.sub(dfX.mean(), axis=1).div(dfX.std(), axis=1)
    dot = pd.DataFrame(np.dot(dfZ.T, dfZ), index=dfX.columns, columns=dfX.columns)
    _e_val, e_vec = get_e_vec(dot, var_thres)
    dfP = np.dot(dfZ, e_vec)
    return dfP


def weighted_kendall_tau(
    feat_imp: pd.Series | np.ndarray,
    pc_rank: pd.Series | np.ndarray,
) -> float:
    """Weighted Kendall's tau consistency check.

    Computes the correlation between supervised feature importance rankings
    (MDI, MDA, or SFI) and unsupervised PCA rankings (eigenvalue order).
    Uses inverse PCA rank as weights to prioritize concordance among the most
    important components.

    A high value (close to 1) indicates ML importance rankings are consistent
    with the principal component structure.

    Args:
        feat_imp: Feature importance scores (e.g., from MDI or MDA).
        pc_rank: PCA ranks for each feature (1 = most important PC).

    Returns:
        Weighted Kendall's tau correlation coefficient.

    Reference:
        AFML Snippet 8.6.
    """
    result = weightedtau(feat_imp, pc_rank**-1.0)
    return float(result.statistic)
