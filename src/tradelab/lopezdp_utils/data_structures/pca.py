"""PCA-based portfolio allocation utilities.

Reference: Advances in Financial Machine Learning, Chapter 2, Section 2.5.2.1
"""

import numpy as np
from numpy.typing import NDArray


def pca_weights(
    cov: NDArray[np.float64],
    risk_dist: NDArray[np.float64] | None = None,
    risk_target: float = 1.0,
) -> NDArray[np.float64]:
    """Derive portfolio allocation weights based on PCA risk distribution.

    This function computes allocation weights for a portfolio of N instruments based on
    how risk is distributed among the covariance matrix's principal components, rather
    than individual assets. The allocation is performed in the orthogonal basis (eigenvector
    space), then transformed back to the original asset basis.

    The default behavior allocates all risk to the principal component with the smallest
    eigenvalue, effectively minimizing variance from the most volatile components.

    Args:
        cov: Covariance matrix of the assets (must be Hermitian/symmetric).
            Shape: (N, N) where N is the number of assets.
        risk_dist: User-defined vector representing the desired distribution of risks
            across the principal components. Shape: (N,). If None, defaults to allocating
            100% of risk to the component with the smallest eigenvalue.
        risk_target: Target total portfolio risk (σ) to be achieved. Default: 1.0.

    Returns:
        Portfolio weights as a column vector. Shape: (N, 1).

    Reference:
        - AFML, Chapter 2, Snippet 2.1
        - Use case: Create portfolios with specific risk profiles, such as for the ETF
          Trick to model futures spreads or baskets of securities.

    Example:
        >>> cov = np.array([[0.01, 0.002], [0.002, 0.04]])
        >>> weights = pca_weights(cov)
    """
    # Spectral decomposition (eigenvalue decomposition)
    eig_val, eig_vec = np.linalg.eigh(cov)  # must be Hermitian

    # Sort eigenvalues and eigenvectors in descending order
    indices = eig_val.argsort()[::-1]
    eig_val, eig_vec = eig_val[indices], eig_vec[:, indices]

    # Default risk distribution: 100% to smallest eigenvalue component
    if risk_dist is None:
        risk_dist = np.zeros(cov.shape[0])
        risk_dist[-1] = 1.0

    # Compute loadings in orthogonal basis
    loads = risk_target * (risk_dist / eig_val) ** 0.5

    # Transform back to original basis: ω = W * β
    weights = np.dot(eig_vec, np.reshape(loads, (-1, 1)))

    return weights
