"""Entropy estimation, encoding, and information theory — AFML Ch.18 + MLAM Ch.3.

Merges:
- entropy_features/estimators.py — plug-in, LZ, Kontoyiannis estimators
- entropy_features/encoding.py — binary, quantile, sigma encoding
- entropy_features/applications.py — market efficiency, concentration, adverse selection
- entropy_features/information_theory.py — KL divergence, cross-entropy
- data_structures/discretization.py — optimal binning, VI, MI

References:
    López de Prado, "Advances in Financial Machine Learning", Chapter 18
    López de Prado, "Machine Learning for Asset Managers", Chapter 3
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import scipy.stats as ss
from numpy.typing import NDArray
from sklearn.metrics import mutual_info_score

# ===========================================================================
# Entropy estimators (pure string operations)
# ===========================================================================


def pmf1(msg: str, w: int) -> dict[str, float]:
    """Compute empirical PMF for all substrings of length w.

    Args:
        msg: Input message as a string of symbols.
        w: Word length (window size).

    Returns:
        Dictionary mapping each unique word to its empirical probability.

    Reference:
        AFML Snippet 18.1
    """
    if not isinstance(msg, str):
        msg = "".join(map(str, msg))

    lib: dict[str, list[int]] = {}
    for i in range(w, len(msg)):
        word = msg[i - w : i]
        if word not in lib:
            lib[word] = [i - w]
        else:
            lib[word].append(i - w)

    total = float(len(msg) - w)
    return {word: len(positions) / total for word, positions in lib.items()}


def plug_in(msg: str, w: int) -> tuple[float, dict[str, float]]:
    """Plug-in (maximum likelihood) entropy rate estimator.

    H = -1/w * sum(p(w) * log2(p(w)))

    Args:
        msg: Input message as a string of symbols.
        w: Word length for PMF estimation.

    Returns:
        Tuple of (entropy_rate, pmf_dict).

    Reference:
        AFML Snippet 18.1
    """
    pmf = pmf1(msg, w)
    out = -sum(p * np.log2(p) for p in pmf.values()) / w
    return float(out), pmf


def lempel_ziv_lib(msg: str) -> list[str]:
    """Build library of non-redundant substrings using Lempel-Ziv parsing.

    Larger library = higher entropy (less compressible).

    Args:
        msg: Input message as a string of symbols.

    Returns:
        List of unique substrings forming the LZ decomposition.

    Reference:
        AFML Snippet 18.2
    """
    # TODO(numba): evaluate JIT for LZ decomposition
    i = 1
    lib = [msg[0:1]]
    while i < len(msg):
        for j in range(i, len(msg)):
            sub = msg[i : j + 1]
            if sub not in lib:
                lib.append(sub)
                break
        i = j + 1
    return lib


def match_length(msg: str, i: int, n: int) -> tuple[int, str]:
    """Find longest matching substring at position i within prior n positions.

    Args:
        msg: Input message.
        i: Current position (i >= n).
        n: Number of preceding positions to search.

    Returns:
        Tuple of (matched_length + 1, matched_substring).

    Reference:
        AFML Snippet 18.3
    """
    sub_s = ""
    for length in range(n):
        msg1 = msg[i : i + length + 1]
        for j in range(i - n, i):
            msg0 = msg[j : j + length + 1]
            if msg1 == msg0:
                sub_s = msg1
                break
    return len(sub_s) + 1, sub_s


def konto(msg: str, window: int | None = None) -> dict:
    """Kontoyiannis' LZ entropy estimator.

    Estimates entropy rate as the inverse of average non-redundant match length.
    Also computes redundancy R = 1 - H/log2(N).

    Args:
        msg: Input message.
        window: Fixed window size. None for expanding window.

    Returns:
        Dict with 'h' (entropy rate), 'r' (redundancy), 'num', 'sum', 'subS'.

    Reference:
        AFML Snippet 18.4
    """
    out: dict = {"num": 0, "sum": 0.0, "subS": []}
    if not isinstance(msg, str):
        msg = "".join(map(str, msg))

    if window is None:
        points = range(1, len(msg) // 2 + 1)
    else:
        window = min(window, len(msg) // 2)
        points = range(window, len(msg) - window + 1)

    for i in points:
        if window is None:
            length, matched = match_length(msg, i, i)
            out["sum"] += np.log2(i + 1) / length
        else:
            length, matched = match_length(msg, i, window)
            out["sum"] += np.log2(window + 1) / length
        out["subS"].append(matched)
        out["num"] += 1

    out["h"] = out["sum"] / out["num"]
    out["r"] = 1 - out["h"] / np.log2(len(msg))
    return out


# ===========================================================================
# Encoding schemes (Polars Series → string)
# ===========================================================================


def encode_binary(returns: pl.Series | NDArray[np.float64]) -> str:
    """Encode returns as binary string: 1 if positive, 0 if negative.

    Zero returns are removed before encoding.

    Args:
        returns: Return series (Polars Series or numpy array).

    Returns:
        Binary string (e.g., "110010011...").

    Reference:
        AFML Section 18.5.1
    """
    r = returns.to_numpy() if isinstance(returns, pl.Series) else np.asarray(returns)
    r = r[r != 0]
    encoded = np.where(r > 0, 1, 0)
    return "".join(map(str, encoded))


def encode_quantile(
    returns: pl.Series | NDArray[np.float64],
    num_letters: int = 10,
) -> str:
    """Discretize returns into equal-frequency (quantile) bins.

    Args:
        returns: Return series.
        num_letters: Number of quantile bins (alphabet size).

    Returns:
        String of digit symbols representing quantile bin assignments.

    Reference:
        AFML Section 18.5.2
    """
    r = returns.to_numpy() if isinstance(returns, pl.Series) else np.asarray(returns)
    encoded = pd.qcut(r, q=num_letters, labels=False, duplicates="drop")
    return "".join(map(str, encoded.astype(int)))


def encode_sigma(
    returns: pl.Series | NDArray[np.float64],
    sigma_step: float,
) -> str:
    """Discretize returns into equal-width bins of size sigma_step.

    Args:
        returns: Return series.
        sigma_step: Width of each bin.

    Returns:
        String of integer symbols representing bin assignments.

    Reference:
        AFML Section 18.5.3
    """
    r = returns.to_numpy() if isinstance(returns, pl.Series) else np.asarray(returns)
    min_r = r.min()
    encoded = ((r - min_r) / sigma_step).astype(int)
    return "".join(map(str, encoded))


# ===========================================================================
# Financial applications
# ===========================================================================


def market_efficiency_metric(
    returns: pl.Series | NDArray[np.float64],
    encoding: str = "quantile",
    bins: int = 10,
) -> dict[str, float]:
    """Quantify market efficiency via redundancy of encoded returns.

    High redundancy = inefficient (predictable). Low redundancy = efficient.

    Args:
        returns: Return series.
        encoding: "binary" or "quantile".
        bins: Number of quantile bins (if encoding="quantile").

    Returns:
        Dict with 'entropy_rate' and 'redundancy' in [0, 1].

    Reference:
        AFML Section 18.6
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
    """Meucci's entropy-based portfolio concentration.

    Args:
        w: Portfolio weight vector (N assets).
        cov: Covariance matrix (N x N).

    Returns:
        Concentration in [0, 1]. 0 = diversified, 1 = concentrated.

    Reference:
        AFML Section 18.6
    """
    e_val, e_vec = np.linalg.eigh(cov)
    idx = e_val.argsort()[::-1]
    e_val, e_vec = e_val[idx], e_vec[:, idx]

    f_w = np.dot(e_vec.T, w)
    total_risk = np.dot(f_w**2, e_val)
    theta = (f_w**2 * e_val) / total_risk

    theta = theta[theta > 0]
    h_theta = -np.sum(theta * np.log(theta))
    concentration = 1 - (1.0 / len(w)) * np.exp(h_theta)

    return float(concentration)


def adverse_selection_feature(
    buy_vol: NDArray[np.float64] | pl.Series,
    total_vol: NDArray[np.float64] | pl.Series,
    q_bins: int = 10,
    window: int | None = 100,
) -> float:
    """Estimate adverse selection probability from order flow entropy.

    Args:
        buy_vol: Volume classified as buyer-initiated.
        total_vol: Total volume.
        q_bins: Number of quantile bins.
        window: Window size for Kontoyiannis estimator.

    Returns:
        Estimated entropy rate of order flow imbalance.

    Reference:
        AFML Section 18.6
    """
    bv = buy_vol.to_numpy() if isinstance(buy_vol, pl.Series) else np.asarray(buy_vol)
    tv = total_vol.to_numpy() if isinstance(total_vol, pl.Series) else np.asarray(total_vol)
    v_b = bv / tv
    encoded = pd.qcut(v_b, q=q_bins, labels=False, duplicates="drop")
    msg = "".join(map(str, encoded.astype(int)))
    result = konto(msg, window=window)
    return float(result["h"])


# ===========================================================================
# Information theory (scipy wrappers)
# ===========================================================================


def kl_divergence(
    p: NDArray[np.float64],
    q: NDArray[np.float64],
) -> float:
    """Kullback-Leibler divergence D_KL(p || q).

    Args:
        p: "True" probability distribution.
        q: Reference probability distribution.

    Returns:
        KL divergence in nats.

    Reference:
        MLAM Section 3.5
    """
    return float(ss.entropy(p, q))


def cross_entropy(
    p: NDArray[np.float64],
    q: NDArray[np.float64],
) -> float:
    """Cross-entropy H_C(p || q) = H(p) + D_KL(p || q).

    Args:
        p: True probability distribution.
        q: Predicted probability distribution.

    Returns:
        Cross-entropy in nats.

    Reference:
        MLAM Section 3.6
    """
    h_p = float(ss.entropy(p))
    d_kl = kl_divergence(p, q)
    return h_p + d_kl


# ===========================================================================
# Optimal binning and discretization (from data_structures/discretization.py)
# ===========================================================================


def num_bins(n_obs: int, corr: float | None = None) -> int:
    """Compute optimal number of bins for discretization.

    Args:
        n_obs: Number of observations.
        corr: Correlation between two variables (bivariate case). None for univariate.

    Returns:
        Optimal number of bins.

    Reference:
        MLAM Section 3.9
    """
    if corr is None:
        z = (8 + 324 * n_obs + 12 * (36 * n_obs + 729 * n_obs**2) ** 0.5) ** (1 / 3.0)
        b = round(z / 6.0 + 2.0 / (3 * z) + 1.0 / 3)
    else:
        b = round(2**-0.5 * (1 + (1 + 24 * n_obs / (1.0 - corr**2)) ** 0.5) ** 0.5)
    return int(b)


def discretize_optimal(
    x: NDArray[np.float64],
    n_bins: int | None = None,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Discretize continuous data using optimal equal-width binning.

    Args:
        x: Continuous values to discretize.
        n_bins: Number of bins. If None, computed via num_bins().

    Returns:
        Tuple of (bin_indices, bin_edges).

    Reference:
        MLAM Section 3.9
    """
    if n_bins is None:
        n_bins = num_bins(len(x))
    _, bin_edges = np.histogram(x, bins=n_bins)
    bin_indices = np.digitize(x, bin_edges[1:-1])
    return bin_indices, bin_edges


def variation_of_information(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    normalize: bool = False,
) -> float:
    """Variation of information — true metric distance between two variables.

    VI(X,Y) = H(X) + H(Y) - 2*I(X,Y)

    Args:
        x: First variable (continuous).
        y: Second variable (continuous).
        normalize: If True, divide by joint entropy H(X,Y).

    Returns:
        VI value. If normalized, in [0, 1].

    Reference:
        MLAM Section 3.9
    """
    b_xy = num_bins(x.shape[0], corr=np.corrcoef(x, y)[0, 1])
    c_xy, _, _ = np.histogram2d(x, y, bins=b_xy)
    i_xy = mutual_info_score(None, None, contingency=c_xy)
    h_x = ss.entropy(np.histogram(x, bins=b_xy)[0])
    h_y = ss.entropy(np.histogram(y, bins=b_xy)[0])

    v_xy = h_x + h_y - 2 * i_xy

    if normalize:
        h_xy = h_x + h_y - i_xy
        if h_xy > 0:
            v_xy /= h_xy

    return float(v_xy)


def mutual_information_optimal(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
) -> float:
    """Mutual information using optimal binning.

    I(X,Y) = H(X) + H(Y) - H(X,Y)

    Args:
        x: First variable (continuous).
        y: Second variable (continuous).

    Returns:
        Mutual information in nats.

    Reference:
        MLAM Section 3.9
    """
    b_xy = num_bins(x.shape[0], corr=np.corrcoef(x, y)[0, 1])
    c_xy, _, _ = np.histogram2d(x, y, bins=b_xy)
    return float(mutual_info_score(None, None, contingency=c_xy))
