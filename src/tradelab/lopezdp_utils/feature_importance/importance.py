"""Feature importance methods: MDI, MDA, and SFI.

Three complementary approaches to measuring feature importance in financial ML:
- MDI (Mean Decrease Impurity): in-sample, explanatory importance for tree ensembles
- MDA (Mean Decrease Accuracy): out-of-sample, permutation-based predictive importance
- SFI (Single Feature Importance): out-of-sample, isolates each feature individually

All methods report mean importance and standard error (CLT-scaled) across estimators
or folds.

Reference:
    López de Prado, M. (2018). *Advances in Financial Machine Learning*. Chapter 8.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from tradelab.lopezdp_utils.cross_validation import PurgedKFold, cv_score


def feat_imp_mdi(
    fit: object,
    feat_names: list[str] | pd.Index,
) -> pd.DataFrame:
    """Mean Decrease Impurity feature importance.

    In-sample (explanatory) importance method for tree-based ensemble classifiers.
    Computes the weighted average of impurity decrease across all nodes where a
    feature is used to split data, aggregated over all trees in the ensemble.

    The method replaces zero importances with NaN to handle the recommended
    ``max_features=1`` setting, which prevents masking effects where informative
    features are hidden by redundant ones during tree building.

    Args:
        fit: Fitted ensemble of tree-based classifiers (e.g., RandomForestClassifier,
            BaggingClassifier with DecisionTreeClassifier base). Must have an
            ``estimators_`` attribute.
        feat_names: Feature names corresponding to the columns of the training data.

    Returns:
        DataFrame with columns ``mean`` (normalized average importance) and ``std``
        (standard error via CLT). Importances are normalized to sum to 1.

    Note:
        MDI is biased: every feature receives some importance even with zero
        predictive power. It is also susceptible to substitution effects where
        correlated features dilute each other's importance. Use alongside MDA/SFI
        for a complete picture.

    Reference:
        AFML Snippet 8.2.
    """
    df0 = {i: tree.feature_importances_ for i, tree in enumerate(fit.estimators_)}
    df0 = pd.DataFrame.from_dict(df0, orient="index")
    df0.columns = feat_names
    df0 = df0.replace(0, np.nan)  # because max_features=1
    imp = pd.concat({"mean": df0.mean(), "std": df0.std() * df0.shape[0] ** -0.5}, axis=1)  # CLT
    imp /= imp["mean"].sum()
    return imp


def feat_imp_mda(
    clf: object,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int,
    sample_weight: pd.Series,
    t1: pd.Series,
    pct_embargo: float,
    scoring: str = "neg_log_loss",
) -> tuple[pd.DataFrame, float]:
    """Mean Decrease Accuracy (permutation) feature importance.

    Out-of-sample predictive importance method. For each cross-validation fold,
    fits the classifier and records baseline OOS performance. Then, for each
    feature, shuffles that column and measures the resulting performance decay.
    Features whose permutation causes larger decay are more important.

    Uses purged k-fold cross-validation to prevent information leakage from
    overlapping labels.

    Args:
        clf: Classifier with ``.fit()``, ``.predict()``, and ``.predict_proba()``
            methods.
        X: Feature matrix (pandas DataFrame).
        y: Labels (pandas Series).
        cv: Number of cross-validation folds.
        sample_weight: Observation weights (pandas Series).
        t1: Label "through dates" for purging (index = start, value = end).
        pct_embargo: Fraction of observations to embargo after test set.
        scoring: ``'neg_log_loss'`` or ``'accuracy'``.

    Returns:
        Tuple of (imp, mean_score):
        - imp: DataFrame with ``mean`` and ``std`` columns (importance per feature).
        - mean_score: Average baseline OOS performance across folds.

    Note:
        MDA is susceptible to substitution effects: highly correlated features may
        both appear unimportant because shuffling one is compensated by its twin.
        Use Clustered MDA (MLAM) to address this.

    Reference:
        AFML Snippet 8.3.
    """
    if scoring not in ("neg_log_loss", "accuracy"):
        raise ValueError(f"scoring must be 'neg_log_loss' or 'accuracy', got '{scoring}'")

    cv_gen = PurgedKFold(n_splits=cv, t1=t1, pct_embargo=pct_embargo)

    scr0 = pd.Series(dtype=float)
    scr1 = pd.DataFrame(columns=X.columns)
    for i, (train, test) in enumerate(cv_gen.split(X=X)):
        X0, y0, w0 = X.iloc[train, :], y.iloc[train], sample_weight.iloc[train]
        X1, y1, w1 = X.iloc[test, :], y.iloc[test], sample_weight.iloc[test]
        fit = clf.fit(X=X0, y=y0, sample_weight=w0.values)

        if scoring == "neg_log_loss":
            prob = fit.predict_proba(X1)
            scr0.loc[i] = -log_loss(y1, prob, sample_weight=w1.values, labels=clf.classes_)
        else:
            pred = fit.predict(X1)
            scr0.loc[i] = accuracy_score(y1, pred, sample_weight=w1.values)

        for j in X.columns:
            X1_ = X1.copy(deep=True)
            np.random.shuffle(X1_[j].values)  # permutation of a single column
            if scoring == "neg_log_loss":
                prob = fit.predict_proba(X1_)
                scr1.loc[i, j] = -log_loss(y1, prob, sample_weight=w1.values, labels=clf.classes_)
            else:
                pred = fit.predict(X1_)
                scr1.loc[i, j] = accuracy_score(y1, pred, sample_weight=w1.values)

    imp = (-scr1).add(scr0, axis=0)
    if scoring == "neg_log_loss":
        imp = imp / -scr1
    else:
        imp = imp / (1.0 - scr1)

    imp = pd.concat({"mean": imp.mean(), "std": imp.std() * imp.shape[0] ** -0.5}, axis=1)
    return imp, scr0.mean()


def feat_imp_sfi(
    feat_names: list[str] | pd.Index,
    clf: object,
    X: pd.DataFrame,
    cont: pd.DataFrame,
    scoring: str,
    cv_gen: PurgedKFold,
) -> pd.DataFrame:
    """Single Feature Importance.

    Out-of-sample predictive importance method that evaluates each feature in
    isolation. For each feature, fits a classifier using only that single column
    and computes its cross-validated OOS performance. This approach is immune to
    the substitution effects that affect MDI and MDA, since no other features
    compete for importance.

    Args:
        feat_names: Feature names to evaluate.
        clf: Classifier with ``.fit()``, ``.predict()``, and ``.predict_proba()``
            methods.
        X: Full feature matrix (pandas DataFrame).
        cont: DataFrame with ``'bin'`` (labels) and ``'w'`` (weights) columns.
        scoring: Scoring method (``'neg_log_loss'`` or ``'accuracy'``).
        cv_gen: Pre-configured ``PurgedKFold`` instance.

    Returns:
        DataFrame with ``mean`` (average OOS score) and ``std`` (standard error)
        for each feature.

    Note:
        SFI misses joint effects and hierarchical interactions — a feature may be
        useless alone but powerful in combination. Best used alongside MDI/MDA for
        a complete picture.

    Reference:
        AFML Snippet 8.4.
    """
    imp = pd.DataFrame(columns=["mean", "std"])
    for feat_name in feat_names:
        df0 = cv_score(
            clf,
            X=X[[feat_name]],
            y=cont["bin"],
            sample_weight=cont["w"],
            scoring=scoring,
            cv_gen=cv_gen,
        )
        imp.loc[feat_name, "mean"] = df0.mean()
        imp.loc[feat_name, "std"] = df0.std() * df0.shape[0] ** -0.5  # CLT
    return imp
