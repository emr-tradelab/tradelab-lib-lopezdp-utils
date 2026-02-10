"""Cross-Validation in Finance — AFML Chapter 7.

Time-aware cross-validation utilities that prevent information leakage from
overlapping financial labels. Standard k-fold CV assumes IID observations,
which is violated in finance where labels span time intervals and exhibit
serial correlation. This module provides purging, embargoing, and a
PurgedKFold splitter to address these issues.
"""

from tradelab.lopezdp_utils.cross_validation.purging import (
    PurgedKFold,
    cv_score,
    get_embargo_times,
    get_train_times,
)
from tradelab.lopezdp_utils.cross_validation.scoring import (
    probability_weighted_accuracy,
)

__all__ = [
    "PurgedKFold",
    "cv_score",
    "get_embargo_times",
    "get_train_times",
    "probability_weighted_accuracy",
]
