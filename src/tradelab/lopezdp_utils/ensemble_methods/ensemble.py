"""Ensemble methods for financial machine learning.

Implements bagging accuracy calculation and Random Forest configurations
adapted for non-IID financial data, following AFML Chapter 6.

Reference:
    López de Prado, M. (2018). *Advances in Financial Machine Learning*. Chapter 6.
"""

from scipy.special import comb
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


def bagging_accuracy(n: int, p: float, k: int = 2) -> float:
    """Calculate probability that a bagging ensemble classifies correctly via majority voting.

    Computes the theoretical accuracy of an ensemble of N independent classifiers,
    each with individual accuracy p, classifying into k classes. The ensemble uses
    majority voting: it is correct when more than N/k classifiers agree on the
    correct label.

    This demonstrates why ensembles outperform individual learners: even with
    base accuracy barely above random (p > 1/k), a sufficiently large ensemble
    converges toward perfect accuracy.

    Args:
        n: Number of independent classifiers in the ensemble.
        p: Accuracy of each individual classifier (must be > 1/k for improvement).
        k: Number of classes (default 2 for binary classification).

    Returns:
        Probability of correct classification by the ensemble.

    Reference:
        AFML Snippet 6.1.
    """
    p_fail = 0.0
    for i in range(0, int(n / k) + 1):
        p_fail += comb(n, i, exact=True) * p**i * (1 - p) ** (n - i)
    return 1 - p_fail


def build_random_forest(
    avg_uniqueness: float | None = None,
    n_estimators: int = 1000,
    method: int = 0,
) -> BaggingClassifier | RandomForestClassifier:
    """Build a Random Forest classifier adapted for financial data.

    Provides three configurations from AFML Snippet 6.2, each addressing the
    non-IID nature of financial observations differently:

    - **Method 0**: Standard ``RandomForestClassifier`` with balanced subsamples.
      Simplest approach but trees may become near-identical when data is redundant.

    - **Method 1**: ``BaggingClassifier`` wrapping ``DecisionTreeClassifier`` with
      ``max_samples`` set to average uniqueness. Controls how much redundant data
      each tree sees.

    - **Method 2**: ``BaggingClassifier`` wrapping a single-tree
      ``RandomForestClassifier`` with ``max_samples`` set to average uniqueness.
      Combines BaggingClassifier's sample control with RandomForest's internal
      ``class_weight='balanced_subsample'``.

    Methods 1 and 2 are recommended for financial data because they explicitly
    account for overlapping/redundant observations through ``avg_uniqueness``.

    Args:
        avg_uniqueness: Average uniqueness of the dataset, computed via
            ``sample_weights.mp_sample_tw``. Required for methods 1 and 2.
            Should be a float between 0 and 1.
        n_estimators: Number of estimators in the ensemble.
        method: Configuration to use (0, 1, or 2). See description above.

    Returns:
        A fitted-ready scikit-learn classifier (call ``.fit(X, y)`` on it).

    Raises:
        ValueError: If method is 1 or 2 and avg_uniqueness is not provided,
            or if method is not in {0, 1, 2}.

    Reference:
        AFML Snippet 6.2.
    """
    if method == 0:
        return RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight="balanced_subsample",
            criterion="entropy",
        )

    if avg_uniqueness is None:
        raise ValueError(
            f"avg_uniqueness is required for method {method}. "
            "Compute it via sample_weights.mp_sample_tw."
        )

    if method == 1:
        base = DecisionTreeClassifier(
            criterion="entropy",
            max_features="sqrt",
            class_weight="balanced",
        )
        return BaggingClassifier(
            estimator=base,
            n_estimators=n_estimators,
            max_samples=avg_uniqueness,
        )

    if method == 2:
        base = RandomForestClassifier(
            n_estimators=1,
            criterion="entropy",
            bootstrap=False,
            class_weight="balanced_subsample",
        )
        return BaggingClassifier(
            estimator=base,
            n_estimators=n_estimators,
            max_samples=avg_uniqueness,
            max_features=1.0,
        )

    raise ValueError(f"method must be 0, 1, or 2, got {method}")


def bagging_classifier_factory(
    base_estimator: object,
    n_estimators: int = 1000,
    max_samples: float = 1.0,
    max_features: float = 1.0,
    n_jobs: int = -1,
) -> BaggingClassifier:
    """Create a BaggingClassifier with financial-data-aware defaults.

    Factory for wrapping any base estimator (e.g., SVM, logistic regression)
    in a BaggingClassifier for scalability and variance reduction. Useful for
    algorithms that don't scale well with sample size — bagging parallelizes
    training across subsamples.

    For financial data, set ``max_samples`` to the dataset's average uniqueness
    (computed via ``sample_weights.mp_sample_tw``) to avoid oversampling
    redundant observations.

    Args:
        base_estimator: The base estimator to bag (e.g., SVM, DecisionTree).
        n_estimators: Number of base estimators in the ensemble.
        max_samples: Fraction of samples to draw for each base estimator.
            Set to average uniqueness for financial data.
        max_features: Fraction of features to draw for each base estimator.
        n_jobs: Number of parallel jobs (-1 uses all cores).

    Returns:
        Configured BaggingClassifier ready for ``.fit(X, y)``.

    Reference:
        AFML Chapter 6, Section 6.6 (Bagging for Scalability).
    """
    return BaggingClassifier(
        estimator=base_estimator,
        n_estimators=n_estimators,
        max_samples=max_samples,
        max_features=max_features,
        n_jobs=n_jobs,
    )
