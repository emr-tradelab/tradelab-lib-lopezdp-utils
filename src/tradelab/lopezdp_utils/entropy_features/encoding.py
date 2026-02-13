"""Encoding schemes for discretizing financial time series — AFML Chapter 18.

Before applying entropy estimators, continuous return series must be discretized
into symbol sequences. These encoding schemes control the alphabet size and
mapping from returns to symbols.

Reference: Advances in Financial Machine Learning, Sections 18.5.1-18.5.3
"""

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def encode_binary(returns: NDArray[np.float64] | pd.Series) -> str:
    """Encode returns as a binary string: 1 if positive, 0 if negative.

    Simplest encoding with alphabet size |A| = 2. Zero returns are removed
    before encoding to avoid ambiguity.

    Args:
        returns: Array of price returns.

    Returns:
        Binary string (e.g., "110010011...").

    Reference:
        AFML Section 18.5.1
    """
    r = np.asarray(returns)
    r = r[r != 0]
    encoded = np.where(r > 0, 1, 0)
    return "".join(map(str, encoded))


def encode_quantile(returns: NDArray[np.float64] | pd.Series, num_letters: int = 10) -> str:
    """Discretize returns into equal-frequency (quantile) bins.

    Each bin contains approximately the same number of observations. Alphabet
    size equals num_letters.

    Args:
        returns: Array of price returns.
        num_letters: Number of quantile bins (alphabet size).

    Returns:
        String of digit symbols representing quantile bin assignments.

    Reference:
        AFML Section 18.5.2
    """
    encoded = pd.qcut(returns, q=num_letters, labels=False, duplicates="drop")
    return "".join(map(str, encoded.astype(int)))


def encode_sigma(returns: NDArray[np.float64] | pd.Series, sigma_step: float) -> str:
    """Discretize returns into equal-width bins of size sigma_step.

    Total number of codes = ceil((max(r) - min(r)) / sigma_step). This encoding
    preserves the magnitude structure of returns.

    Args:
        returns: Array of price returns.
        sigma_step: Width of each bin (e.g., 0.01 for 1% steps).

    Returns:
        String of integer symbols representing bin assignments.

    Reference:
        AFML Section 18.5.3
    """
    r = np.asarray(returns)
    min_r = r.min()
    encoded = ((r - min_r) / sigma_step).astype(int)
    return "".join(map(str, encoded))
