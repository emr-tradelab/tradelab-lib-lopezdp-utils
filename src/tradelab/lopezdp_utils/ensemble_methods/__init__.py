"""Ensemble Methods — AFML Chapter 6.

Utilities for constructing ensemble classifiers adapted for financial data,
where observations are non-IID due to overlapping labels and low signal-to-noise ratios.
"""

from tradelab.lopezdp_utils.ensemble_methods.ensemble import (
    bagging_accuracy,
    bagging_classifier_factory,
    build_random_forest,
)

__all__ = [
    "bagging_accuracy",
    "bagging_classifier_factory",
    "build_random_forest",
]
