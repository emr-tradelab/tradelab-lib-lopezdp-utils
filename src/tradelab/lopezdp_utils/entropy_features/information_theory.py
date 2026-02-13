"""Information-theoretic distance metrics — MLAM Chapter 3.

Complementary tools from Machine Learning for Asset Managers: KL divergence,
cross-entropy, and normalized mutual information. Core utilities (num_bins,
variation_of_information, mutual_information_optimal) are imported from
data_structures.discretization where they were originally extracted.

Reference: Machine Learning for Asset Managers, Sections 3.5-3.6
"""

import numpy as np
import scipy.stats as ss
from numpy.typing import NDArray

from tradelab.lopezdp_utils.data_structures.discretization import (
    mutual_information_optimal,
    num_bins,
    variation_of_information,
)


def kl_divergence(
    p: NDArray[np.float64],
    q: NDArray[np.float64],
) -> float:
    """Compute Kullback-Leibler divergence D_KL(p || q).

    Measures how much distribution p diverges from reference distribution q.
    Not a true metric (asymmetric, doesn't satisfy triangle inequality), but
    useful for measuring distributional shift in financial data.

    D_KL(p || q) = sum(p(x) * log(p(x) / q(x)))

    Args:
        p: "True" probability distribution (must sum to 1).
        q: Reference probability distribution (must sum to 1).

    Returns:
        KL divergence in nats. Returns inf if q has zeros where p doesn't.

    Reference:
        MLAM Section 3.5
    """
    return float(ss.entropy(p, q))


def cross_entropy(
    p: NDArray[np.float64],
    q: NDArray[np.float64],
) -> float:
    """Compute cross-entropy H_C(p || q).

    Cross-entropy is the expected number of bits needed to encode data from
    distribution p using a code optimized for distribution q. Used as a
    scoring function for financial classification models.

    H_C(p || q) = H(p) + D_KL(p || q) = -sum(p(x) * log(q(x)))

    Args:
        p: True probability distribution (must sum to 1).
        q: Predicted probability distribution (must sum to 1).

    Returns:
        Cross-entropy in nats.

    Reference:
        MLAM Section 3.6
    """
    h_p = float(ss.entropy(p))
    d_kl = kl_divergence(p, q)
    return h_p + d_kl


# Re-export core utilities for convenient access from this submodule
__all__ = [
    "cross_entropy",
    "kl_divergence",
    "mutual_information_optimal",
    "num_bins",
    "variation_of_information",
]
