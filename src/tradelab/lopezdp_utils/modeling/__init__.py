"""Model training, cross-validation, and tuning — AFML Chapters 6-7, 9.

This package covers the fifth stage of López de Prado's pipeline:
features → model training with proper cross-validation that respects
temporal dependencies via purging and embargoing.

PurgedKFold is the cornerstone: it prevents information leakage by
removing training observations whose label windows overlap the test set,
and optionally adds an embargo buffer for serial correlation.

Reference:
    López de Prado, "Advances in Financial Machine Learning", Chapters 6-7, 9
    López de Prado, "Machine Learning for Asset Managers", Section 6.4
"""

from tradelab.lopezdp_utils.modeling.cross_validation import (
    PurgedKFold,
    cv_score,
    get_embargo_times,
    get_train_times,
    probability_weighted_accuracy,
)
from tradelab.lopezdp_utils.modeling.ensemble import (
    bagging_accuracy,
    bagging_classifier_factory,
    build_random_forest,
)
from tradelab.lopezdp_utils.modeling.hyperparameter_tuning import (
    MyPipeline,
    clf_hyper_fit,
    log_uniform,
)

__all__ = [
    "MyPipeline",
    "PurgedKFold",
    "bagging_accuracy",
    "bagging_classifier_factory",
    "build_random_forest",
    "clf_hyper_fit",
    "cv_score",
    "get_embargo_times",
    "get_train_times",
    "log_uniform",
    "probability_weighted_accuracy",
]
