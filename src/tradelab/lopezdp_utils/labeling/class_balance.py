"""Utilities for handling class imbalance in labeled financial datasets.

Reference: AFML Chapter 3, Section 3.9 and Chapter 4, Section 4.7
"""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.utils.class_weight import compute_class_weight


def drop_labels(events: pl.DataFrame, min_pct: float = 0.05) -> pl.DataFrame:
    """Recursively drop under-populated label classes to handle class imbalance.

    Iteratively removes the rarest class until all remaining classes meet the
    minimum frequency threshold, preserving at least binary classification.

    Args:
        events: DataFrame with a 'label' column containing class labels.
        min_pct: Minimum fraction for a class to be retained. Default 0.05 (5%).

    Returns:
        Filtered DataFrame with only labels meeting the threshold.

    Reference:
        Snippet 3.8, AFML Chapter 3
    """
    while True:
        # Compute value counts (normalized)
        vc = (
            events["label"]
            .value_counts(sort=True)
            .with_columns((pl.col("count") / pl.col("count").sum()).alias("pct"))
        )

        min_pct_val = float(vc["pct"].min())
        n_classes = len(vc)

        # Stop if threshold met or only 1 class remains
        if min_pct_val > min_pct or n_classes <= 1:
            break

        # Drop the rarest class
        rarest = vc.sort("pct")["label"][0]
        events = events.filter(pl.col("label") != rarest)

    return events


def get_class_weights(
    y: pl.Series | np.ndarray,
    method: str = "balanced",
) -> dict:
    """Compute class weights to correct for imbalanced datasets.

    Rare classes receive higher weights so the model doesn't ignore them.

    Args:
        y: Series or array of target labels.
        method: Weighting method — "balanced" or "balanced_subsample".

    Returns:
        Dictionary mapping class label to weight.

    Reference:
        AFML Chapter 4, Section 4.7
    """
    if isinstance(y, pl.Series):
        y_array = y.to_numpy()
    else:
        y_array = y

    classes = np.unique(y_array)
    weights = compute_class_weight(class_weight=method, classes=classes, y=y_array)
    return dict(zip(classes, weights))
