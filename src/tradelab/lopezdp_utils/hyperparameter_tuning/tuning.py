"""Hyperparameter tuning with purged cross-validation.

Standard grid/randomized search assumes IID observations, which is violated
in finance where labels span overlapping time intervals and exhibit serial
correlation. This module wraps scikit-learn's search utilities with
PurgedKFold CV to prevent information leakage during hyperparameter selection.

Reference:
    López de Prado, M. (2018). *Advances in Financial Machine Learning*.
    Chapter 9, Snippets 9.1-9.3.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline


class MyPipeline(Pipeline):
    """Pipeline subclass that correctly routes ``sample_weight`` to the final estimator.

    Scikit-learn's default ``Pipeline.fit`` does not always pass
    ``sample_weight`` through to the last step. This subclass fixes that
    by injecting it into ``fit_params`` with the proper step-name prefix.

    Reference:
        AFML Chapter 9, Snippet 9.2.
    """

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series | None = None,
        sample_weight: np.ndarray | pd.Series | None = None,
        **fit_params: Any,
    ) -> "MyPipeline":
        if sample_weight is not None:
            fit_params[self.steps[-1][0] + "__sample_weight"] = sample_weight
        return super().fit(X, y, **fit_params)


def clf_hyper_fit(
    feat: pd.DataFrame,
    lbl: pd.Series,
    t1: pd.Series,
    pipe_clf: Pipeline,
    param_grid: dict[str, Any],
    cv: int = 3,
    bagging: list[float] = [0, None, 1.0],
    rnd_search_iter: int = 0,
    n_jobs: int = -1,
    pct_embargo: float = 0,
    **fit_params: Any,
) -> Pipeline:
    """Find optimal hyperparameters using purged k-fold cross-validation.

    Performs grid search (when ``rnd_search_iter=0``) or randomized search
    over a parameter space, using ``PurgedKFold`` as the internal CV splitter
    to prevent information leakage from overlapping financial labels.

    Scoring is automatically selected:
    - **F1** when labels are ``{0, 1}`` (meta-labeling).
    - **Negative log-loss** otherwise (general financial models).

    Optionally wraps the best estimator in a ``BaggingClassifier`` for
    additional variance reduction.

    Args:
        feat: Feature matrix (observations x features).
        lbl: Label series aligned with ``feat``.
        t1: Series of label end-dates for purging (index = label start).
        pipe_clf: Scikit-learn Pipeline to tune.
        param_grid: Parameter grid (dict for grid search, distributions
            for randomized search).
        cv: Number of cross-validation folds.
        bagging: ``[n_estimators, max_samples, max_features]``. Set
            ``bagging[0] > 0`` to enable bagging around the best estimator.
        rnd_search_iter: Number of randomized search iterations. Use 0
            for exhaustive grid search.
        n_jobs: Number of parallel jobs.
        pct_embargo: Fraction of observations to embargo after each test set.
        **fit_params: Additional keyword arguments passed to ``fit``
            (e.g., ``sample_weight``).

    Returns:
        Fitted Pipeline (or Pipeline wrapping a BaggingClassifier).

    Reference:
        AFML Chapter 9, Snippets 9.1 & 9.3.
    """
    from tradelab.lopezdp_utils.cross_validation import PurgedKFold

    # Auto-select scoring metric
    if set(lbl.values) == {0, 1}:
        scoring = "f1"  # meta-labeling
    else:
        scoring = "neg_log_loss"  # general case

    # Purged k-fold CV
    inner_cv = PurgedKFold(n_splits=cv, t1=t1, pct_embargo=pct_embargo)

    # Grid or randomized search
    if rnd_search_iter == 0:
        gs = GridSearchCV(
            estimator=pipe_clf,
            param_grid=param_grid,
            scoring=scoring,
            cv=inner_cv,
            n_jobs=n_jobs,
        )
    else:
        gs = RandomizedSearchCV(
            estimator=pipe_clf,
            param_distributions=param_grid,
            scoring=scoring,
            cv=inner_cv,
            n_jobs=n_jobs,
            n_iter=rnd_search_iter,
        )

    gs = gs.fit(feat, lbl, **fit_params).best_estimator_

    # Optional bagging
    if bagging[0] > 0:
        gs = BaggingClassifier(
            estimator=MyPipeline(gs.steps),
            n_estimators=int(bagging[0]),
            max_samples=float(bagging[1]),
            max_features=float(bagging[2]),
            n_jobs=n_jobs,
        )
        gs = gs.fit(
            feat,
            lbl,
            sample_weight=fit_params.get(gs.estimator.steps[-1][0] + "__sample_weight"),
        )
        gs = Pipeline([("bag", gs)])

    return gs
