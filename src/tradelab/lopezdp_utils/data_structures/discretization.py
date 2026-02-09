"""Optimal binning (discretization) for entropy and information-theoretic metrics.

Reference: Machine Learning for Asset Managers, Section 3.9
"""

import numpy as np
import scipy.stats as ss
from numpy.typing import NDArray
from sklearn.metrics import mutual_info_score


def num_bins(n_obs: int, corr: float | None = None) -> int:
    """Compute optimal number of bins for discretization.

    Shannon's entropy and information-theoretic metrics are defined only for discrete
    random variables. When discretizing continuous financial data, the number of bins
    is critical—too many or too few bins creates bias in entropy estimation.

    This function provides mathematically derived formulas to determine the optimal
    number of bins that minimize bias, based on sample size and (for bivariate cases)
    correlation between variables.

    Args:
        n_obs: Number of observations (sample size).
        corr: Correlation between two variables (for bivariate case).
            If None, computes for univariate case.

    Returns:
        Optimal number of bins (B_X or B_XY).

    Reference:
        - MLAM, Section 3.9
        - Univariate: B_X = round[ζ/6 + 2/(3ζ) + 1/3]
          where ζ = (8 + 324N + 12√(36N + 729N²))^(1/3)
        - Bivariate: B_X = B_Y = round(2^(-1/2) × √(1 + √(1 + 24N/(1-ρ̂²))))

    Example:
        >>> # Univariate case
        >>> bins = num_bins(1000)
        >>> # Bivariate case with correlation 0.5
        >>> bins = num_bins(1000, corr=0.5)
    """
    if corr is None:
        # Univariate case
        z = (8 + 324 * n_obs + 12 * (36 * n_obs + 729 * n_obs**2) ** 0.5) ** (1 / 3.0)
        b = round(z / 6.0 + 2.0 / (3 * z) + 1.0 / 3)
    else:
        # Bivariate case
        b = round(2**-0.5 * (1 + (1 + 24 * n_obs / (1.0 - corr**2)) ** 0.5) ** 0.5)

    return int(b)


def discretize_optimal(
    x: NDArray[np.float64],
    n_bins: int | None = None,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Discretize continuous data using optimal binning.

    Uses equal-width bins (range divided into B equal-sized segments), which adapts
    to the actual distribution and provides lower-bias entropy estimates compared to
    equal-frequency (quantile) binning.

    Args:
        x: Array of continuous values to discretize.
        n_bins: Number of bins. If None, computed automatically using num_bins().

    Returns:
        Tuple of (bin_indices, bin_edges).
        - bin_indices: Integer array of bin assignments for each observation
        - bin_edges: Array of bin edge values

    Reference:
        - MLAM, Section 3.9
        - Use case: Preprocessing for entropy, mutual information, variation of
          information, and other information-theoretic metrics

    Example:
        >>> x = np.random.randn(1000)
        >>> bin_indices, bin_edges = discretize_optimal(x)
        >>> # Use for entropy estimation
        >>> from scipy.stats import entropy
        >>> hist, _ = np.histogram(x, bins=bin_edges)
        >>> h_x = entropy(hist)
    """
    if n_bins is None:
        n_bins = num_bins(len(x))

    _, bin_edges = np.histogram(x, bins=n_bins)
    bin_indices = np.digitize(x, bin_edges[1:-1])  # Assign each value to a bin

    return bin_indices, bin_edges


def variation_of_information(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    normalize: bool = False,
) -> float:
    """Compute variation of information (VI) between two variables using optimal binning.

    Variation of information is a true metric (satisfies triangle inequality) for
    measuring the distance between two random variables. It captures nonlinear
    dependencies that correlation would miss (e.g., symmetric relationships).

    VI(X,Y) = H(X) + H(Y) - 2×I(X,Y)

    where H is entropy and I is mutual information.

    Args:
        x: First variable (continuous).
        y: Second variable (continuous).
        normalize: If True, returns normalized VI (divided by joint entropy H(X,Y)).

    Returns:
        Variation of information between X and Y. If normalize=True, returns value
        in [0, 1] where 0 means identical and 1 means completely independent.

    Reference:
        - MLAM, Section 3.9
        - Use cases: Cluster comparison, feature selection, nonlinear dependencies

    Example:
        >>> x = np.random.randn(1000)
        >>> y = x**2  # Nonlinear relationship (correlation ≈ 0)
        >>> vi = variation_of_information(x, y)
        >>> # VI will be low despite near-zero correlation
    """
    # Compute optimal bins for bivariate case
    b_xy = num_bins(x.shape[0], corr=np.corrcoef(x, y)[0, 1])

    # Create 2D histogram (contingency table)
    c_xy, _, _ = np.histogram2d(x, y, bins=b_xy)

    # Compute mutual information
    i_xy = mutual_info_score(None, None, contingency=c_xy)

    # Compute marginal entropies
    h_x = ss.entropy(np.histogram(x, bins=b_xy)[0])
    h_y = ss.entropy(np.histogram(y, bins=b_xy)[0])

    # Variation of information: H(X) + H(Y) - 2×I(X,Y)
    v_xy = h_x + h_y - 2 * i_xy

    if normalize:
        # Normalize by joint entropy: H(X,Y) = H(X) + H(Y) - I(X,Y)
        h_xy = h_x + h_y - i_xy
        if h_xy > 0:
            v_xy /= h_xy

    return float(v_xy)


def mutual_information_optimal(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
) -> float:
    """Compute mutual information between two variables using optimal binning.

    Mutual information measures the amount of information obtained about one random
    variable through observing another. Unlike correlation, it captures nonlinear
    dependencies.

    I(X,Y) = H(X) + H(Y) - H(X,Y)

    Args:
        x: First variable (continuous).
        y: Second variable (continuous).

    Returns:
        Mutual information between X and Y (in nats if using natural log).

    Reference:
        - MLAM, Section 3.9
        - Use cases: Feature selection, dependency detection, information gain

    Example:
        >>> x = np.random.randn(1000)
        >>> y = 2 * x + np.random.randn(1000) * 0.1
        >>> mi = mutual_information_optimal(x, y)
    """
    # Compute optimal bins for bivariate case
    b_xy = num_bins(x.shape[0], corr=np.corrcoef(x, y)[0, 1])

    # Create 2D histogram (contingency table)
    c_xy, _, _ = np.histogram2d(x, y, bins=b_xy)

    # Compute mutual information
    i_xy = mutual_info_score(None, None, contingency=c_xy)

    return float(i_xy)
