"""Alternative scoring metrics for cross-validation in finance.

Reference:
    López de Prado, M. (2020). *Machine Learning for Asset Managers*. Section 6.4.
"""

import numpy as np


def probability_weighted_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prob: np.ndarray,
    n_classes: int = 2,
) -> float:
    """Compute Probability-Weighted Accuracy (PWA).

    An alternative to standard accuracy that penalizes high-confidence wrong
    predictions more severely. While standard accuracy treats all correct/incorrect
    predictions equally, PWA weights each prediction by the classifier's excess
    confidence (how much the predicted probability exceeds random chance).

    This means:
    - A confident correct prediction contributes more than a hesitant one.
    - A confident wrong prediction drags the score down significantly.

    PWA is bounded in [-1, 1], where 1 is perfect confident classification and
    negative values indicate the classifier is confidently wrong.

    Formula::

        PWA = Σ(pₙ - 1/K) · yₙ / Σ(pₙ - 1/K)

    where pₙ is the max predicted probability for observation n, yₙ is 1 if
    correct and 0 otherwise, and K is the number of classes.

    Args:
        y_true: True labels, shape (n_samples,).
        y_pred: Predicted labels, shape (n_samples,).
        prob: Maximum predicted probability for each observation (i.e.,
            ``clf.predict_proba(X).max(axis=1)``), shape (n_samples,).
        n_classes: Number of classes (default 2).

    Returns:
        Probability-weighted accuracy score.

    Reference:
        ML for Asset Managers, Section 6.4.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    prob = np.asarray(prob)

    correct = (y_pred == y_true).astype(float)
    excess_confidence = prob - 1.0 / n_classes

    denominator = excess_confidence.sum()
    if denominator == 0:
        return 0.0

    return float((excess_confidence * correct).sum() / denominator)
