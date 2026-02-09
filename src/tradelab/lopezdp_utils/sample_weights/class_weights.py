"""
Class Weight Utilities

This module provides utilities for handling class imbalance in financial ML.
Class imbalance occurs when certain labels (e.g., buy signals) are much more
frequent than others (e.g., sell signals), causing models to maximize accuracy
on common labels while ignoring rare but critical events.

References:
    - AFML Chapter 4, Section 4.7: Class Weights
"""

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight


def get_class_weights(y: pd.Series | np.ndarray, method: str = "balanced") -> dict:
    """
    Compute class weights to correct for imbalanced datasets.

    Class imbalance in financial data prevents models from learning rare but important
    events (e.g., flash crashes, liquidity crises). This function computes weights that
    penalize errors in rare classes more heavily, effectively reweighting observations
    so all classes appear equally frequent during training.

    Args:
        y: Target labels (e.g., -1, 0, 1 for sell/neutral/buy signals)
        method: Weighting method. Options:
            - "balanced": Inversely proportional to class frequencies
            - "balanced_subsample": Similar to balanced but computed for each bootstrap

    Returns:
        Dictionary mapping class labels to their weights

    Mathematical Logic:
        For "balanced" method:
            weight_i = n_samples / (n_classes * n_samples_i)

        Where:
            n_samples = total number of observations
            n_classes = number of unique classes
            n_samples_i = number of observations in class i

    Example:
        >>> y = pd.Series([1, 1, 1, 1, 1, -1, -1, 0])  # Imbalanced
        >>> weights = get_class_weights(y)
        >>> print(weights)
        {-1: 2.0, 0: 4.0, 1: 0.8}
        # Rare class (0) gets highest weight (4.0)
        # Common class (1) gets lowest weight (0.8)

    Usage with scikit-learn:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> clf = RandomForestClassifier(class_weight=weights)
        >>> clf.fit(X, y)

    Reference:
        AFML Chapter 4, Section 4.7
    """
    # Convert to numpy array if pandas Series
    if isinstance(y, pd.Series):
        y_array = y.values
    else:
        y_array = y

    # Get unique classes
    classes = np.unique(y_array)

    # Compute class weights
    if method == "balanced":
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_array)
    elif method == "balanced_subsample":
        weights = compute_class_weight(
            class_weight="balanced_subsample", classes=classes, y=y_array
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    # Return as dictionary
    return dict(zip(classes, weights))
