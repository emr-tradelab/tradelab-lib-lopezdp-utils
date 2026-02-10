"""Orthogonal features and consistency validation utilities.

Provides PCA-based feature orthogonalization to alleviate linear substitution
effects (multicollinearity) and a weighted Kendall's tau check to validate
that supervised importance rankings are consistent with unsupervised PCA rankings.

Reference:
    López de Prado, M. (2018). *Advances in Financial Machine Learning*. Chapter 8.
"""

import numpy as np
import pandas as pd
from scipy.stats import weightedtau


def get_e_vec(
    dot: pd.DataFrame,
    var_thres: float,
) -> tuple[pd.Series, pd.DataFrame]:
    """Compute eigenvectors from dot product matrix with dimensionality reduction.

    Performs spectral decomposition on the dot product matrix Z'Z and retains
    only enough principal components to explain the specified cumulative variance
    threshold.

    Args:
        dot: Dot product matrix (Z'Z) of standardized features.
        var_thres: Minimum cumulative variance ratio to retain (e.g., 0.95 for 95%).

    Returns:
        Tuple of (eigenvalues, eigenvectors) as (pd.Series, pd.DataFrame),
        both reduced to the components meeting the variance threshold.

    Reference:
        AFML Snippet 8.5 (helper).
    """
    e_val, e_vec = np.linalg.eigh(dot)
    idx = e_val.argsort()[::-1]  # sort eigenvalues descending
    e_val, e_vec = e_val[idx], e_vec[:, idx]

    e_val = pd.Series(e_val, index=["PC_" + str(i + 1) for i in range(e_val.shape[0])])
    e_vec = pd.DataFrame(e_vec, index=dot.index, columns=e_val.index)
    e_vec = e_vec.loc[:, e_val.index]

    # Reduce dimension: retain components up to var_thres cumulative variance
    cum_var = e_val.cumsum() / e_val.sum()
    dim = cum_var.values.searchsorted(var_thres)
    e_val, e_vec = e_val.iloc[: dim + 1], e_vec.iloc[:, : dim + 1]
    return e_val, e_vec


def get_ortho_feats(
    dfX: pd.DataFrame,
    var_thres: float = 0.95,
) -> np.ndarray:
    """Compute orthogonal features via PCA.

    Standardizes features (center and scale), computes the dot product matrix,
    performs spectral decomposition, and projects the standardized features onto
    the selected eigenvectors. The result is a set of orthogonal principal
    components that alleviate linear substitution effects.

    Args:
        dfX: DataFrame of stationary features (observations x features).
        var_thres: Minimum cumulative variance ratio to retain (default 0.95).

    Returns:
        numpy array of orthogonal features (P = Z @ W), where Z is the
        standardized features matrix and W are the selected eigenvectors.

    Note:
        Only alleviates *linear* substitution effects. Nonlinear redundancy
        persists. Principal components also lack intuitive real-world
        interpretation. Consider Clustered Feature Importance (MLAM) as an
        alternative that preserves interpretability.

    Reference:
        AFML Snippet 8.5.
    """
    # Standardize: center and scale
    dfZ = dfX.sub(dfX.mean(), axis=1).div(dfX.std(), axis=1)

    # Compute dot product matrix
    dot = pd.DataFrame(np.dot(dfZ.T, dfZ), index=dfX.columns, columns=dfX.columns)

    # Get eigenvectors and project
    _e_val, e_vec = get_e_vec(dot, var_thres)
    dfP = np.dot(dfZ, e_vec)
    return dfP


def weighted_kendall_tau(
    feat_imp: np.ndarray,
    pc_rank: np.ndarray,
) -> tuple[float, float]:
    """Weighted Kendall's tau consistency check.

    Computes the correlation between supervised feature importance rankings
    (from MDI, MDA, or SFI) and unsupervised PCA rankings (eigenvalue order).
    Uses inverse PCA rank as weights to prioritize concordance among the most
    important features.

    A high value (close to 1) indicates that ML importance rankings are consistent
    with the principal component structure, providing evidence that the identified
    patterns are structural rather than overfit flukes.

    Args:
        feat_imp: Array of feature importance scores (e.g., from MDI or MDA).
        pc_rank: Array of PCA ranks for each feature (1 = most important PC).

    Returns:
        Tuple of (correlation, p_value) from scipy's weightedtau.

    Reference:
        AFML Snippet 8.6.
    """
    return weightedtau(feat_imp, pc_rank**-1.0)
