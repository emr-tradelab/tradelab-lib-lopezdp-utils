"""Purging and embargoing utilities for cross-validation in finance.

Implements time-aware sample removal to prevent information leakage when
cross-validating models on financial data with overlapping labels.

Reference:
    López de Prado, M. (2018). *Advances in Financial Machine Learning*. Chapter 7.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import KFold


def get_train_times(
    t1: pd.Series,
    test_times: pd.Series,
) -> pd.Series:
    """Remove training observations whose labels overlap with the test set.

    Given a set of test observation times, this function identifies and removes
    all training observations that share informational overlap with the test set.
    Three overlap conditions are checked:

    1. Training observation *starts* within a test label interval.
    2. Training observation *ends* within a test label interval.
    3. Training observation completely *envelops* a test label interval.

    This is the core "purging" operation that prevents leakage from concurrent
    labels — the most common source of backtest overfitting in financial ML.

    Args:
        t1: pandas Series where the index is the observation start time and the
            value is the observation end time (label "through date").
        test_times: pandas Series of test set observation times (index = start,
            value = end).

    Returns:
        Subset of ``t1`` with all overlapping training observations removed.

    Reference:
        AFML Snippet 7.1.
    """
    trn = t1.copy(deep=True)
    for i, j in test_times.items():
        # Train starts within test interval
        df0 = trn[(i <= trn.index) & (trn.index <= j)].index
        # Train ends within test interval
        df1 = trn[(i <= trn) & (trn <= j)].index
        # Train envelops test interval
        df2 = trn[(trn.index <= i) & (j <= trn)].index
        trn = trn.drop(df0.union(df1).union(df2))
    return trn


def get_embargo_times(
    times: pd.Index,
    pct_embargo: float,
) -> pd.Series:
    """Compute embargo timestamps for each observation.

    Creates a mapping from each observation time to the earliest time at which
    a subsequent training observation may begin. This "embargo" period guards
    against serial-correlation leakage (e.g., ARMA effects) that purging alone
    cannot address.

    The embargo is applied only to training observations that *follow* the test
    set. Observations preceding the test set are not affected.

    Args:
        times: pandas DatetimeIndex of all observation timestamps.
        pct_embargo: Fraction of total observations to embargo (e.g., 0.01
            for a 1% embargo). Typical values are 0.01-0.02.

    Returns:
        pandas Series mapping each observation time to its embargo boundary time.

    Reference:
        AFML Snippet 7.2.
    """
    step = int(times.shape[0] * pct_embargo)
    if step == 0:
        mbrg = pd.Series(times, index=times)
    else:
        mbrg = pd.Series(times[step:], index=times[:-step])
        mbrg = pd.concat([mbrg, pd.Series(times[-1], index=times[-step:])])
    return mbrg


class PurgedKFold(KFold):
    """K-Fold cross-validator with purging and embargoing for financial labels.

    Extends scikit-learn's ``KFold`` to handle labels that span time intervals
    (i.e., where each observation has a start and end date). Training folds are
    purged of any observations whose label intervals overlap with the test fold,
    and an optional embargo period is enforced after each test set.

    Key constraints:

    - ``shuffle`` is forced to ``False`` — shuffling defeats the purpose by
      scattering serially dependent observations across folds.
    - Test sets are always contiguous blocks of consecutive observations.
    - The index of ``X`` must exactly match the index of ``t1``.

    Args:
        n_splits: Number of folds (default 3).
        t1: pandas Series of label "through dates" (index = start, value = end).
            Required.
        pct_embargo: Fraction of observations to embargo after each test set
            (default 0.0).

    Reference:
        AFML Snippet 7.3.
    """

    def __init__(
        self,
        n_splits: int = 3,
        t1: pd.Series | None = None,
        pct_embargo: float = 0.0,
    ):
        if not isinstance(t1, pd.Series):
            raise ValueError("Label Through Dates must be a pd.Series")
        super().__init__(n_splits=n_splits, shuffle=False, random_state=None)
        self.t1 = t1
        self.pct_embargo = pct_embargo

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
        groups: np.ndarray | None = None,
    ):
        """Generate purged and embargoed train/test index splits.

        Yields:
            Tuple of (train_indices, test_indices) as numpy arrays.
        """
        if (X.index == self.t1.index).sum() != len(self.t1):
            raise ValueError("X and ThruDateValues must have the same index")

        indices = np.arange(X.shape[0])
        mbrg = int(X.shape[0] * self.pct_embargo)
        test_starts = [
            (i[0], i[-1] + 1) for i in np.array_split(np.arange(X.shape[0]), self.n_splits)
        ]

        for i, j in test_starts:
            t0 = self.t1.index[i]  # start of test set
            test_indices = indices[i:j]
            # End time of the test set labels
            max_t1_idx = self.t1.index.searchsorted(self.t1[test_indices].max())
            # Training set predating the test set (Purging)
            train_indices = self.t1.index.searchsorted(self.t1[self.t1 <= t0].index)
            # Training set following the test set (Purging + Embargo)
            if max_t1_idx < X.shape[0]:
                train_indices = np.concatenate((train_indices, indices[max_t1_idx + mbrg :]))

            yield train_indices, test_indices


def cv_score(
    clf: object,
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: pd.Series,
    scoring: str = "neg_log_loss",
    t1: pd.Series | None = None,
    cv: int | None = None,
    cv_gen: PurgedKFold | None = None,
    pct_embargo: float | None = None,
) -> np.ndarray:
    """Cross-validate a classifier with purging and embargoing.

    Drop-in replacement for scikit-learn's ``cross_val_score`` that fixes two
    bugs in the library:

    1. **classes_ attribute**: When using non-shuffled pandas data, scikit-learn
       may not observe all classes in each fold. This function explicitly passes
       ``labels=clf.classes_`` to ``log_loss``.

    2. **Weight inconsistency**: scikit-learn passes ``sample_weight`` to
       ``.fit()`` but not to the scoring metric. This function passes weights
       to both.

    Args:
        clf: Classifier with ``.fit()``, ``.predict()``, and
            ``.predict_proba()`` methods.
        X: Feature matrix (pandas DataFrame).
        y: Labels (pandas Series).
        sample_weight: Observation weights (pandas Series).
        scoring: Scoring method — ``'neg_log_loss'`` or ``'accuracy'``.
        t1: Label through dates (required if ``cv_gen`` is not provided).
        cv: Number of folds (required if ``cv_gen`` is not provided).
        cv_gen: Pre-configured ``PurgedKFold`` instance. If provided,
            ``t1``, ``cv``, and ``pct_embargo`` are ignored.
        pct_embargo: Embargo fraction (required if ``cv_gen`` is not provided).

    Returns:
        Array of scores, one per fold.

    Raises:
        ValueError: If ``scoring`` is not ``'neg_log_loss'`` or ``'accuracy'``.

    Reference:
        AFML Snippet 7.4.
    """
    if scoring not in ("neg_log_loss", "accuracy"):
        raise ValueError(f"scoring must be 'neg_log_loss' or 'accuracy', got '{scoring}'")

    if cv_gen is None:
        cv_gen = PurgedKFold(n_splits=cv, t1=t1, pct_embargo=pct_embargo)

    score = []
    for train, test in cv_gen.split(X=X):
        fit = clf.fit(
            X=X.iloc[train, :],
            y=y.iloc[train],
            sample_weight=sample_weight.iloc[train].values,
        )
        if scoring == "neg_log_loss":
            prob = fit.predict_proba(X.iloc[test, :])
            score_ = -log_loss(
                y.iloc[test],
                prob,
                sample_weight=sample_weight.iloc[test].values,
                labels=clf.classes_,
            )
        else:
            pred = fit.predict(X.iloc[test, :])
            score_ = accuracy_score(
                y.iloc[test],
                pred,
                sample_weight=sample_weight.iloc[test].values,
            )
        score.append(score_)

    return np.array(score)
