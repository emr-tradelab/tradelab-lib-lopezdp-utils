"""Utilities for handling class imbalance in labeled datasets.

This module provides functions to address class imbalance issues that commonly
arise in financial machine learning, where certain market outcomes may be rare
but still present enough to distract classifiers from learning meaningful patterns.

Reference: AFML Chapter 3, Section 3.9
"""

import pandas as pd


def drop_labels(events: pd.DataFrame, min_pct: float = 0.05) -> pd.DataFrame:
    """Recursively drop under-populated labels to handle class imbalance.

    Many ML classifiers perform poorly when classes are extremely imbalanced.
    A model trained on a dataset where one class represents 99% of observations
    may simply learn to always predict that class, achieving high "accuracy"
    but zero predictive power for rare, catalytic events.

    This function iteratively removes the rarest label class until all remaining
    classes meet a minimum frequency threshold. This forces the ML algorithm to
    focus on common, statistically relevant outcomes rather than being distracted
    by noise or outliers that appear too infrequently to form reliable patterns.

    The function preserves at least binary classification (stops at 2 classes).

    Args:
        events: DataFrame containing generated labels, typically in a 'bin' column.
            Usually the output from get_bins() or get_bins_meta().
        min_pct: Minimum threshold fraction for a class to be retained.
            Default is 0.05 (5%). Classes below this frequency are dropped.

    Returns:
        Filtered DataFrame with only labels meeting the minimum frequency
        threshold. All observations associated with dropped labels are removed.

    Reference:
        Snippet 3.8 in AFML Chapter 3, Section 3.9

    Example:
        >>> labels = get_bins(events, close)
        >>> print(labels['bin'].value_counts(normalize=True))
        # 1     0.85
        # -1    0.13
        # 0     0.02
        >>> balanced = drop_labels(labels, min_pct=0.05)
        >>> print(balanced['bin'].value_counts(normalize=True))
        # 1     0.87
        # -1    0.13
        # (0 was dropped as it was below 5%)
    """
    while True:
        # Calculate normalized frequency of each label
        df0 = events["bin"].value_counts(normalize=True)

        # Stop if minimum frequency meets threshold or fewer than 3 classes remain
        if df0.min() > min_pct or df0.shape[0] < 3:
            break

        # Identify and drop the rarest label
        print(f"Dropped label {df0.idxmin()}, frequency: {df0.min():.4f}")
        events = events[events["bin"] != df0.idxmin()]

    return events
