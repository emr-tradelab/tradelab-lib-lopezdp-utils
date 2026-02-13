"""Financial applications of entropy — AFML Chapter 18.

Entropy-based features for measuring market efficiency, portfolio concentration,
and adverse selection probability.

Reference: Advances in Financial Machine Learning, Section 18.6
"""

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from tradelab.lopezdp_utils.entropy_features.encoding import (
    encode_binary,
    encode_quantile,
)
from tradelab.lopezdp_utils.entropy_features.estimators import konto


def market_efficiency_metric(
    returns: NDArray[np.float64] | pd.Series,
    encoding: str = "quantile",
    bins: int = 10,
) -> dict[str, float]:
    """Quantify market efficiency using the redundancy of encoded returns.

    High entropy = "decompressed" (efficient) market where returns are
    unpredictable. Low entropy = "compressed" (inefficient) market where
    patterns exist and may indicate bubbles or exploitable structure.

    Redundancy R = 1 - H[X] / log₂(|A|), where |A| is alphabet size.

    Args:
        returns: Array of price returns.
        encoding: Encoding scheme — "binary" or "quantile".
        bins: Number of quantile bins (only used if encoding="quantile").

    Returns:
        Dictionary with 'entropy_rate' (H) and 'redundancy' (R ∈ [0, 1]).

    Reference:
        AFML Section 18.6 — Market efficiency application
    """
    if encoding == "binary":
        msg = encode_binary(returns)
    else:
        msg = encode_quantile(returns, num_letters=bins)

    res = konto(msg)
    return {"entropy_rate": res["h"], "redundancy": res["r"]}


def portfolio_concentration(
    w: NDArray[np.float64],
    cov: NDArray[np.float64],
) -> float:
    """Compute Meucci's entropy-based portfolio concentration.

    Measures how concentrated risk is across principal components. High
    concentration means risk is driven by few factors; low concentration
    means well-diversified risk.

    Formula: C = 1 - (1/N) * exp(sum(theta_i * log(theta_i)))
    where theta_i = risk contribution from i-th principal component.

    Args:
        w: Portfolio weight vector (N assets).
        cov: Covariance matrix (N x N).

    Returns:
        Concentration value in [0, 1]. 0 = maximally diversified, 1 = fully
        concentrated in one component.

    Reference:
        AFML Section 18.6 — Portfolio concentration application
    """
    # Eigen-decomposition
    e_val, e_vec = np.linalg.eigh(cov)
    # Sort descending
    idx = e_val.argsort()[::-1]
    e_val, e_vec = e_val[idx], e_vec[:, idx]

    # Component risk contributions
    f_w = np.dot(e_vec.T, w)  # factor loadings
    total_risk = np.dot(f_w**2, e_val)
    theta = (f_w**2 * e_val) / total_risk

    # Concentration via entropy of risk contributions
    theta = theta[theta > 0]
    h_theta = -np.sum(theta * np.log(theta))
    concentration = 1 - (1.0 / len(w)) * np.exp(h_theta)

    return float(concentration)


def adverse_selection_feature(
    buy_vol: NDArray[np.float64] | pd.Series,
    total_vol: NDArray[np.float64] | pd.Series,
    q_bins: int = 10,
    window: int | None = 100,
) -> float:
    """Derive adverse selection probability from order flow entropy.

    High entropy in order flow implies "surprised" market makers — the
    information content of trades is high, indicating likely adverse selection.
    Low entropy means predictable (uninformed) flow.

    Args:
        buy_vol: Volume classified as buyer-initiated.
        total_vol: Total volume.
        q_bins: Number of quantile bins for discretization.
        window: Window size for Kontoyiannis estimator. None for expanding.

    Returns:
        Estimated entropy rate of order flow imbalance.

    Reference:
        AFML Section 18.6 — Adverse selection application
    """
    v_b = np.asarray(buy_vol) / np.asarray(total_vol)
    encoded = pd.qcut(v_b, q=q_bins, labels=False, duplicates="drop")
    msg = "".join(map(str, encoded.astype(int)))
    result = konto(msg, window=window)
    return float(result["h"])
