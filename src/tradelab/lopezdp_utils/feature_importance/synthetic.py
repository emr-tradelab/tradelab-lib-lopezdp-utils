"""Synthetic data experimental suite for testing feature importance methods.

Provides utilities to generate controlled datasets with informative, redundant,
and noise features, run importance analysis (MDI/MDA/SFI), and visualize results.
This suite validates that importance algorithms correctly identify signal vs noise.

Reference:
    López de Prado, M. (2018). *Advances in Financial Machine Learning*. Chapter 8.
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

from tradelab.lopezdp_utils.cross_validation import PurgedKFold
from tradelab.lopezdp_utils.feature_importance.importance import (
    feat_imp_mda,
    feat_imp_mdi,
    feat_imp_sfi,
)


def get_test_data(
    n_features: int = 40,
    n_informative: int = 10,
    n_redundant: int = 10,
    n_samples: int = 10000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic dataset for testing feature importance algorithms.

    Creates a classification dataset with three categories of features:
    - **Informative** (I_*): Features with genuine predictive power.
    - **Redundant** (R_*): Linear combinations of informative features.
    - **Noise** (N_*): Random features with no predictive value.

    Args:
        n_features: Total number of features.
        n_informative: Number of informative features.
        n_redundant: Number of redundant features (linear combos of informative).
        n_samples: Number of observations.

    Returns:
        Tuple of (trns_x, cont):
        - trns_x: DataFrame of features with columns named I_0..., R_0..., N_0...
        - cont: DataFrame with ``'bin'`` (labels), ``'w'`` (equal weights), and
          ``'t1'`` (through dates, set to self for no overlap in synthetic data).

    Reference:
        AFML Snippet 8.7.
    """
    trns_x, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        random_state=0,
        shuffle=False,
    )
    df0 = pd.DatetimeIndex(periods=n_samples, freq="h", start=pd.Timestamp("2015-01-01"))
    trns_x, y = pd.DataFrame(trns_x, index=df0), pd.Series(y, index=df0)

    # Feature names based on category
    cols = ["I_" + str(i) for i in range(n_informative)]
    cols += ["R_" + str(i) for i in range(n_redundant)]
    cols += ["N_" + str(i) for i in range(n_features - len(cols))]
    trns_x.columns = cols

    # Target and weights container
    cont = pd.DataFrame({"bin": y})
    cont["w"] = 1.0 / n_samples  # equal weights for synthetic test
    cont["t1"] = pd.Series(trns_x.index, index=trns_x.index)  # no overlap
    return trns_x, cont


def feat_importance(
    trns_x: pd.DataFrame,
    cont: pd.DataFrame,
    n_estimators: int = 1000,
    cv: int = 10,
    max_samples: float = 1.0,
    num_threads: int = 24,
    pct_embargo: float = 0.0,
    scoring: str = "neg_log_loss",
    method: str = "MDI",
    min_weight_fraction_leaf: float = 0.0,
) -> tuple[pd.DataFrame, float]:
    """Unified wrapper to compute feature importance using MDI, MDA, or SFI.

    Fits a BaggingClassifier with DecisionTreeClassifier base estimators
    (max_features=1 to prevent masking) and delegates to the appropriate
    importance method.

    Args:
        trns_x: Feature matrix (pandas DataFrame).
        cont: DataFrame with ``'bin'`` (labels), ``'w'`` (weights), ``'t1'``
            (through dates).
        n_estimators: Number of base estimators in the bagging ensemble.
        cv: Number of cross-validation folds (for MDA/SFI).
        max_samples: Fraction of samples per base estimator.
        num_threads: Number of parallel jobs (-1 if > 1, else 1).
        pct_embargo: Embargo fraction for purged CV.
        scoring: ``'neg_log_loss'`` or ``'accuracy'``.
        method: ``'MDI'``, ``'MDA'``, or ``'SFI'``.
        min_weight_fraction_leaf: Minimum weighted fraction for leaf nodes.

    Returns:
        Tuple of (imp, oob_score):
        - imp: DataFrame with ``mean`` and ``std`` columns.
        - oob_score: Out-of-bag score from the fitted ensemble.

    Reference:
        AFML Snippet 8.8.
    """
    n_jobs = -1 if num_threads > 1 else 1

    # Prepare classifier
    ds = DecisionTreeClassifier(
        criterion="entropy",
        max_features=1,
        class_weight="balanced",
        min_weight_fraction_leaf=min_weight_fraction_leaf,
    )
    fit = BaggingClassifier(
        estimator=ds,
        n_estimators=n_estimators,
        max_features=1.0,
        max_samples=max_samples,
        bootstrap=True,
        n_jobs=n_jobs,
        oob_score=True,
    )
    fit.fit(trns_x, cont["bin"], sample_weight=cont["w"].values)
    oob = fit.oob_score_

    # Compute importance
    if method == "MDI":
        imp = feat_imp_mdi(fit, feat_names=trns_x.columns)
    elif method == "MDA":
        imp, _oos = feat_imp_mda(
            fit,
            trns_x,
            cont["bin"],
            cv=cv,
            sample_weight=cont["w"],
            t1=cont["t1"],
            pct_embargo=pct_embargo,
            scoring=scoring,
        )
    elif method == "SFI":
        cv_gen = PurgedKFold(n_splits=cv, t1=cont["t1"], pct_embargo=pct_embargo)
        imp = feat_imp_sfi(trns_x.columns, fit, trns_x, cont, scoring, cv_gen)
    else:
        raise ValueError(f"method must be 'MDI', 'MDA', or 'SFI', got '{method}'")

    return imp, oob


def test_func(
    n_features: int = 40,
    n_informative: int = 10,
    n_redundant: int = 10,
    n_samples: int = 10000,
    n_estimators: int = 1000,
    cv: int = 10,
    method: str = "MDI",
    **kwargs,
) -> None:
    """Master execution component: generate data, compute importance, and plot.

    Automates the entire feature importance testing pipeline from synthetic data
    generation through analysis and visualization.

    Args:
        n_features: Total number of features.
        n_informative: Number of informative features.
        n_redundant: Number of redundant features.
        n_samples: Number of observations.
        n_estimators: Number of base estimators.
        cv: Number of CV folds.
        method: ``'MDI'``, ``'MDA'``, or ``'SFI'``.
        **kwargs: Additional keyword arguments passed to ``feat_importance``.

    Reference:
        AFML Snippet 8.9.
    """
    trns_x, cont = get_test_data(n_features, n_informative, n_redundant, n_samples)

    dict0 = {"min_weight_fraction_leaf": 0.0}
    dict0.update(kwargs)
    imp, oob = feat_importance(
        trns_x, cont, n_estimators=n_estimators, cv=cv, method=method, **dict0
    )

    plot_feat_importance(imp, oob, method, n_informative, n_redundant)


def plot_feat_importance(
    imp: pd.DataFrame,
    oob: float,
    method: str,
    n_informative: int,
    n_redundant: int,
    save_fig: str | None = None,
) -> None:
    """Visualize feature importance rankings as horizontal bar chart.

    Displays mean importance with standard deviation error bars. Features are
    sorted by importance and colored to distinguish informative, redundant,
    and noise categories.

    Args:
        imp: DataFrame with ``mean`` and ``std`` columns (from any importance method).
        oob: Out-of-bag score to display in the title.
        method: Importance method name for axis label.
        n_informative: Number of informative features (for reference in title).
        n_redundant: Number of redundant features (for reference in title).
        save_fig: If provided, save figure to this path instead of showing.

    Reference:
        AFML Snippet 8.10.
    """
    plt.figure(figsize=(10, imp.shape[0] * 0.25))
    imp = imp.sort_values("mean", ascending=True)
    imp["mean"].plot(
        kind="barh",
        color="b",
        alpha=0.25,
        xerr=imp["std"],
        error_kw={"ecolor": "r"},
    )

    plt.xlabel(f"Importance ({method})")
    plt.title(f"Method: {method} | OOB Score: {round(oob, 4)}")

    if save_fig is not None:
        plt.savefig(save_fig)
        plt.close()
    else:
        plt.show()
