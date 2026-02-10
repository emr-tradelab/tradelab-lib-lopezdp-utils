"""Log-uniform distribution for hyperparameter search.

Many ML hyperparameters (e.g., SVM's C and gamma, regularization strength)
respond non-linearly to changes: going from 0.01 to 1 has a similar impact
as going from 1 to 100. A log-uniform distribution ensures efficient
exploration by sampling proportionally across orders of magnitude.

Reference:
    López de Prado, M. (2018). *Advances in Financial Machine Learning*.
    Chapter 9, Snippet 9.4.
"""

import numpy as np
from scipy.stats import rv_continuous


class _LogUniformGen(rv_continuous):
    """Log-uniform continuous random variable.

    Random numbers are distributed log-uniformly between bounds *a* and *b*.
    This means that ``log(X)`` is uniformly distributed on ``[log(a), log(b)]``.

    The CDF is::

        F(x) = log(x / a) / log(b / a)

    and the PDF is::

        f(x) = 1 / (x * log(b / a))

    Reference:
        AFML Snippet 9.4.
    """

    def _cdf(self, x: np.ndarray) -> np.ndarray:
        return np.log(x / self.a) / np.log(self.b / self.a)

    def _pdf(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (x * np.log(self.b / self.a))


def log_uniform(a: float = 1, b: float = np.e) -> _LogUniformGen:
    """Create a log-uniform distribution on [a, b].

    Use this as a ``param_distributions`` value in ``RandomizedSearchCV``
    or ``clf_hyper_fit`` to sample hyperparameters that span several orders
    of magnitude (e.g., regularization strength, kernel bandwidth).

    Args:
        a: Lower bound (must be > 0).
        b: Upper bound (must be > a).

    Returns:
        A frozen scipy continuous distribution that can be sampled via
        ``.rvs(size=n)`` or used directly in randomized search.

    Reference:
        AFML Chapter 9, Snippet 9.4.
    """
    return _LogUniformGen(a=a, b=b, name="log_uniform")
