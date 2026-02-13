"""Entropy estimation algorithms — AFML Chapter 18.

Provides plug-in (maximum likelihood), Lempel-Ziv, and Kontoyiannis estimators
for entropy rate of discrete sequences. These estimators quantify the
information content and compressibility of financial time series.

Reference: Advances in Financial Machine Learning, Chapter 18
"""

import numpy as np


def pmf1(msg: str, w: int) -> dict[str, float]:
    """Compute probability mass function for a one-dimensional discrete random variable.

    Counts empirical frequency of all "words" (substrings) of length w in the message.
    Used as input for the plug-in entropy estimator.

    Args:
        msg: Input message as a string of symbols.
        w: Word length (window size for substring extraction).

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
    pmf = {word: len(positions) / total for word, positions in lib.items()}
    return pmf


def plug_in(msg: str, w: int) -> tuple[float, dict[str, float]]:
    """Compute plug-in (maximum likelihood) entropy rate.

    The plug-in estimator directly uses the empirical PMF to compute Shannon's
    entropy rate. Simple but biased for short sequences — tends to underestimate
    entropy because unobserved words get zero probability.

    Formula: H = -1/w * sum(p(w) * log2(p(w)))

    Args:
        msg: Input message as a string of symbols.
        w: Word length for PMF estimation.

    Returns:
        Tuple of (entropy_rate, pmf_dict).
        - entropy_rate: Estimated entropy rate in bits per symbol.
        - pmf_dict: The empirical PMF used for computation.

    Reference:
        AFML Snippet 18.1
    """
    pmf = pmf1(msg, w)
    out = -sum(p * np.log2(p) for p in pmf.values()) / w
    return float(out), pmf


def lempel_ziv_lib(msg: str) -> list[str]:
    """Build library of non-redundant substrings using the Lempel-Ziv algorithm.

    Decomposes the message into the smallest set of unique segments. The size of
    this library relates to the compressibility of the sequence — more unique
    segments means higher entropy (less compressible).

    Args:
        msg: Input message as a string of symbols.

    Returns:
        List of unique substrings forming the LZ decomposition.

    Reference:
        AFML Snippet 18.2
    """
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

    Scans backward through the message to find the longest substring starting at
    position i that also appears in the preceding n characters. Used by the
    Kontoyiannis entropy estimator.

    Args:
        msg: Input message as a string of symbols.
        i: Current position in the message (i >= n).
        n: Number of preceding positions to search.

    Returns:
        Tuple of (matched_length_plus_1, matched_substring).
        Returns length + 1 to avoid the Doeblin condition (log(0)).

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

    Estimates entropy rate as the inverse of the average length of the shortest
    non-redundant substring. If non-redundant substrings are short, the text is
    highly entropic (random). If they are long, the text has structure.

    Also computes redundancy R = 1 - H/log₂(N), where R ∈ [0,1]. High redundancy
    means the sequence is highly compressible (structured/predictable).

    Args:
        msg: Input message as a string of symbols.
        window: Fixed window size for searching matches. If None, uses expanding
            window (message length must be even in this case). If the end of the
            message is more relevant, try konto(msg[::-1]).

    Returns:
        Dictionary with keys:
        - 'h': Estimated entropy rate (bits per symbol).
        - 'r': Redundancy (0 = maximum entropy, 1 = fully redundant).
        - 'num': Number of points evaluated.
        - 'sum': Running sum of log₂(i+1)/l terms.
        - 'subS': List of matched substrings.

    Reference:
        AFML Snippet 18.4 — Kontoyiannis (2013) centered window version
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
