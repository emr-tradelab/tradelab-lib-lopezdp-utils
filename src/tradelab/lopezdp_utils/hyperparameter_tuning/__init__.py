"""Hyper-Parameter Tuning with Cross-Validation — AFML Chapter 9.

Financial-aware hyperparameter optimization that uses purged k-fold
cross-validation to prevent information leakage from overlapping labels.
Includes grid search, randomized search with log-uniform distributions,
and optional bagging of the tuned estimator.
"""

from tradelab.lopezdp_utils.hyperparameter_tuning.distributions import (
    log_uniform,
)
from tradelab.lopezdp_utils.hyperparameter_tuning.tuning import (
    MyPipeline,
    clf_hyper_fit,
)

__all__ = [
    "MyPipeline",
    "clf_hyper_fit",
    "log_uniform",
]
