"""Covariance matrix denoising and detoning - MLAM Sections 2.4-2.9.

Random Matrix Theory (RMT) tools for cleaning empirical covariance matrices.
The Marcenko-Pastur theorem identifies the eigenvalue distribution of a purely
random matrix, allowing separation of signal from noise eigenvalues.

Denoising: replace noise eigenvalues with their average (constant residual)
    or apply targeted shrinkage to noise eigenvectors.
Detoning: remove market component (first eigenvector) to amplify sector signals.

Reference: ML for Asset Managers, Snippets 2.1, 2.4, 2.5, 2.6, 2.9, Section 2.6.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.neighbors import KernelDensity


def mp_pdf(var: float, q: float, pts: int = 1000) -> pd.Series:
    """Marcenko-Pastur probability density function.

    Describes the eigenvalue distribution of a random matrix with aspect
    ratio q = T/N. Eigenvalues outside [λ-, λ+] are signal.

    Args:
        var: Variance of the random matrix entries.
        q: Aspect ratio T/N (observations / variables).
        pts: Number of points for PDF evaluation.

    Returns:
        PDF values indexed by eigenvalue.

    Reference:
        MLAM Snippet 2.1.
    """
    e_min = var * (1 - (1.0 / q) ** 0.5) ** 2
    e_max = var * (1 + (1.0 / q) ** 0.5) ** 2
    e_val = np.linspace(e_min, e_max, pts)
    pdf = q / (2 * np.pi * var * e_val) * ((e_max - e_val) * (e_val - e_min)) ** 0.5
    pdf = pd.Series(pdf, index=e_val)
    return pdf


def _fit_kde(obs: np.ndarray, bwidth: float, x: np.ndarray | None = None) -> pd.Series:
    """Fit Kernel Density Estimator to observations."""
    if len(obs.shape) == 1:
        obs = obs.reshape(-1, 1)
    kde = KernelDensity(kernel="gaussian", bandwidth=bwidth).fit(obs)
    if x is None:
        x = obs.flatten()
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    log_prob = kde.score_samples(x)
    pdf = pd.Series(np.exp(log_prob), index=x.flatten())
    return pdf


def _get_pca(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract eigenvalues and eigenvectors, sorted descending."""
    e_val, e_vec = np.linalg.eigh(matrix)
    idx = e_val.argsort()[::-1]
    e_val = np.diag(e_val[idx])
    e_vec = e_vec[:, idx]
    return e_val, e_vec


def _cov2corr(cov: np.ndarray) -> np.ndarray:
    """Convert covariance matrix to correlation matrix."""
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    corr[corr < -1] = -1  # numerical fix
    corr[corr > 1] = 1
    np.fill_diagonal(corr, 1.0)
    return corr


def _corr2cov(corr: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Convert correlation matrix to covariance matrix."""
    cov = corr * np.outer(std, std)
    return cov


def _err_pdfs(var: float, e_val: np.ndarray, q: float, bwidth: float, pts: int = 1000) -> float:
    """Compute fitting error between theoretical MP and empirical PDFs."""
    var = float(np.asarray(var).flat[0])
    pdf0 = mp_pdf(var, q, pts)  # theoretical
    pdf1 = _fit_kde(e_val, bwidth, x=pdf0.index.values)  # empirical
    sse = np.sum((pdf1 - pdf0) ** 2)
    return sse


def find_max_eval(e_val: np.ndarray, q: float, bwidth: float) -> tuple[float, float]:
    """Find maximum random eigenvalue by fitting Marcenko-Pastur distribution.

    Optimizes the variance parameter to best fit the empirical eigenvalue
    distribution, then computes λ+ = σ² * (1 + sqrt(1/q))².

    Args:
        e_val: Empirical eigenvalues (1D array).
        q: Aspect ratio T/N.
        bwidth: Bandwidth for KDE fitting.

    Returns:
        Tuple of (eMax, var) — maximum noise eigenvalue and fitted variance.

    Reference:
        MLAM Snippet 2.4.
    """
    out = minimize(
        lambda *x: _err_pdfs(*x),
        0.5,
        args=(e_val, q, bwidth),
        bounds=((1e-5, 1 - 1e-5),),
    )
    if out["success"]:
        var = out["x"][0]
    else:
        var = 1.0
    e_max = var * (1 + (1.0 / q) ** 0.5) ** 2
    return e_max, var


def denoised_corr(e_val: np.ndarray, e_vec: np.ndarray, n_facts: int) -> np.ndarray:
    """Denoise correlation matrix using constant residual eigenvalue method.

    Replaces noise eigenvalues (those below λ+) with their average,
    preserving the trace of the correlation matrix.

    Args:
        e_val: Eigenvalues as diagonal matrix from _get_pca.
        e_vec: Eigenvectors from _get_pca.
        n_facts: Number of signal factors to keep.

    Returns:
        Denoised correlation matrix.

    Reference:
        MLAM Snippet 2.5.
    """
    e_val_ = np.diag(e_val).copy()
    e_val_[n_facts:] = e_val_[n_facts:].sum() / float(e_val_.shape[0] - n_facts)
    e_val_ = np.diag(e_val_)
    corr1 = np.dot(e_vec, e_val_).dot(e_vec.T)
    corr1 = _cov2corr(corr1)
    return corr1


def denoised_corr_shrinkage(
    e_val: np.ndarray, e_vec: np.ndarray, n_facts: int, alpha: float = 0
) -> np.ndarray:
    """Denoise correlation matrix via targeted shrinkage of noise eigenvectors.

    Blends the noise reconstruction: alpha=0 keeps only diagonal (full shrinkage),
    alpha=1 keeps the full noise reconstruction (no shrinkage).

    Args:
        e_val: Eigenvalues as diagonal matrix from _get_pca.
        e_vec: Eigenvectors from _get_pca.
        n_facts: Number of signal factors to keep.
        alpha: Shrinkage parameter in [0, 1].

    Returns:
        Denoised correlation matrix.

    Reference:
        MLAM Snippet 2.6.
    """
    e_val_l, e_vec_l = e_val[:n_facts, :n_facts], e_vec[:, :n_facts]
    e_val_r, e_vec_r = e_val[n_facts:, n_facts:], e_vec[:, n_facts:]

    corr0 = np.dot(e_vec_l, e_val_l).dot(e_vec_l.T)
    corr1 = np.dot(e_vec_r, e_val_r).dot(e_vec_r.T)

    corr2 = corr0 + alpha * corr1 + (1 - alpha) * np.diag(np.diag(corr1))
    return corr2


def denoise_cov(
    cov0: np.ndarray | pd.DataFrame, q: float, bwidth: float
) -> np.ndarray | pd.DataFrame:
    """Denoise an empirical covariance matrix using Marcenko-Pastur.

    High-level wrapper: covariance → correlation → PCA → denoise → covariance.

    Args:
        cov0: Empirical covariance matrix.
        q: Aspect ratio T/N.
        bwidth: Bandwidth for KDE fitting.

    Returns:
        Denoised covariance matrix (same type as input).

    Reference:
        MLAM Snippet 2.9.
    """
    is_df = isinstance(cov0, pd.DataFrame)
    if is_df:
        cols, idx = cov0.columns, cov0.index
        cov0 = cov0.values
    corr0 = _cov2corr(cov0)
    e_val0, e_vec0 = _get_pca(corr0)
    e_max0, _var0 = find_max_eval(np.diag(e_val0), q, bwidth)
    n_facts0 = e_val0.shape[0] - np.diag(e_val0)[::-1].searchsorted(e_max0)
    corr1 = denoised_corr(e_val0, e_vec0, n_facts0)
    cov1 = _corr2cov(corr1, np.diag(cov0) ** 0.5)
    if is_df:
        cov1 = pd.DataFrame(cov1, columns=cols, index=idx)
    return cov1


def detone_corr(corr: np.ndarray, n_facts: int = 1) -> np.ndarray:
    """Remove market component from correlation matrix (detoning).

    Removes the first n_facts eigenvectors (market mode) to amplify
    sector/industry signals for clustering purposes.

    Args:
        corr: Correlation matrix (typically already denoised).
        n_facts: Number of market factors to remove (default 1).

    Returns:
        Detoned correlation matrix.

    Reference:
        MLAM Section 2.6.
    """
    e_val, e_vec = _get_pca(corr)
    e_val_m = e_val[:n_facts, :n_facts]
    e_vec_m = e_vec[:, :n_facts]
    corr2 = corr - np.dot(e_vec_m, e_val_m).dot(e_vec_m.T)
    corr2 = _cov2corr(corr2)
    return corr2
