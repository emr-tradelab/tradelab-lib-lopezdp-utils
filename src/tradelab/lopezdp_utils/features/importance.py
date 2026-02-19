"""Feature importance methods — MDI, MDA, SFI, clustered variants, and synthetic tests.

Three complementary approaches to measuring feature importance in financial ML:
- MDI (Mean Decrease Impurity): in-sample, explanatory importance for tree ensembles
- MDA (Mean Decrease Accuracy): out-of-sample, permutation-based predictive importance
- SFI (Single Feature Importance): out-of-sample, isolates each feature individually
- Clustered MDI/MDA (MLAM): group correlated features via ONC before importance scoring

All methods use pandas/numpy/sklearn internally — feature importance consumers
are sklearn classifiers that operate on pandas DataFrames.

Reference:
    López de Prado, "Advances in Financial Machine Learning", Chapter 8
    López de Prado, "Machine Learning for Asset Managers", Sections 4, 6.5
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import make_classification
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, log_loss, silhouette_samples
from sklearn.tree import DecisionTreeClassifier

from tradelab.lopezdp_utils.cross_validation import PurgedKFold, cv_score


# ---------------------------------------------------------------------------
# MDI — Mean Decrease Impurity
# ---------------------------------------------------------------------------


def feat_imp_mdi(
    fit: object,
    feat_names: list[str] | pd.Index,
) -> pd.DataFrame:
    """Mean Decrease Impurity feature importance.

    In-sample (explanatory) importance for tree-based ensemble classifiers.
    Computes the weighted average of impurity decrease across all nodes where a
    feature is used to split data, aggregated over all trees.

    Args:
        fit: Fitted ensemble of tree-based classifiers (e.g., RandomForestClassifier
            or BaggingClassifier). Must have an ``estimators_`` attribute.
        feat_names: Feature names corresponding to the columns of the training data.

    Returns:
        DataFrame with columns ``mean`` (normalized average importance) and
        ``std`` (standard error via CLT). Importances are normalized to sum to 1.

    Reference:
        AFML Snippet 8.2.
    """
    df0 = {i: tree.feature_importances_ for i, tree in enumerate(fit.estimators_)}
    df0 = pd.DataFrame.from_dict(df0, orient="index")
    df0.columns = feat_names
    df0 = df0.replace(0, np.nan)  # because max_features=1
    imp = pd.concat({"mean": df0.mean(), "std": df0.std() * df0.shape[0] ** -0.5}, axis=1)
    imp /= imp["mean"].sum()
    return imp


# ---------------------------------------------------------------------------
# MDA — Mean Decrease Accuracy
# ---------------------------------------------------------------------------


def feat_imp_mda(
    clf: object,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int | object = 10,
    sample_weight: pd.Series | None = None,
    t1: pd.Series | None = None,
    pct_embargo: float = 0.0,
    scoring: str = "neg_log_loss",
) -> tuple[pd.DataFrame, float]:
    """Mean Decrease Accuracy (permutation) feature importance.

    Out-of-sample predictive importance method. For each CV fold, fits the
    classifier and records baseline OOS performance. Then, for each feature,
    shuffles that column and measures the resulting performance decay.

    Args:
        clf: Classifier with ``.fit()``, ``.predict()``, and ``.predict_proba()``.
        X: Feature matrix (pandas DataFrame).
        y: Labels (pandas Series).
        cv: Number of folds (int, uses PurgedKFold if t1 provided, else KFold)
            or a pre-constructed sklearn CV splitter object. Default 10.
        sample_weight: Observation weights. Uniform if None.
        t1: Label end-dates for purging (required for PurgedKFold with int cv).
        pct_embargo: Fraction of observations to embargo. Default 0.0.
        scoring: ``'neg_log_loss'`` or ``'accuracy'``.

    Returns:
        Tuple of (imp, mean_score):
        - imp: DataFrame with ``mean`` and ``std`` columns (importance per feature).
        - mean_score: Average baseline OOS performance across folds.

    Reference:
        AFML Snippet 8.3.
    """
    if scoring not in ("neg_log_loss", "accuracy"):
        raise ValueError(f"scoring must be 'neg_log_loss' or 'accuracy', got '{scoring}'")

    if sample_weight is None:
        sample_weight = pd.Series(np.ones(len(X)), index=X.index)

    # Build CV splitter
    if isinstance(cv, int):
        if t1 is not None:
            cv_gen = PurgedKFold(n_splits=cv, t1=t1, pct_embargo=pct_embargo)
        else:
            from sklearn.model_selection import KFold

            cv_gen = KFold(n_splits=cv)
    else:
        cv_gen = cv  # already a splitter

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
            X1_[j] = np.random.permutation(X1_[j].values)  # permutation of a single column
            if scoring == "neg_log_loss":
                prob = fit.predict_proba(X1_)
                scr1.loc[i, j] = -log_loss(
                    y1, prob, sample_weight=w1.values, labels=clf.classes_
                )
            else:
                pred = fit.predict(X1_)
                scr1.loc[i, j] = accuracy_score(y1, pred, sample_weight=w1.values)

    imp = (-scr1).add(scr0, axis=0)
    if scoring == "neg_log_loss":
        imp = imp / -scr1
    else:
        imp = imp / (1.0 - scr1)

    imp = pd.concat({"mean": imp.mean(), "std": imp.std() * imp.shape[0] ** -0.5}, axis=1)
    return imp, float(scr0.mean())


# ---------------------------------------------------------------------------
# SFI — Single Feature Importance
# ---------------------------------------------------------------------------


def feat_imp_sfi(
    feat_names: list[str] | pd.Index,
    clf: object,
    X: pd.DataFrame,
    cont: pd.DataFrame,
    scoring: str,
    cv_gen: PurgedKFold,
) -> pd.DataFrame:
    """Single Feature Importance.

    Evaluates each feature in isolation — immune to substitution effects since
    no other features compete. For each feature, fits a classifier using only
    that single column and computes CV OOS performance.

    Args:
        feat_names: Feature names to evaluate.
        clf: Classifier with ``.fit()``, ``.predict()``, and ``.predict_proba()``.
        X: Full feature matrix (pandas DataFrame).
        cont: DataFrame with ``'bin'`` (labels) and ``'w'`` (weights) columns.
        scoring: Scoring method (``'neg_log_loss'`` or ``'accuracy'``).
        cv_gen: Pre-configured ``PurgedKFold`` instance.

    Returns:
        DataFrame with ``mean`` (average OOS score) and ``std`` (standard error)
        for each feature.

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
        imp.loc[feat_name, "std"] = df0.std() * df0.shape[0] ** -0.5
    return imp


# ---------------------------------------------------------------------------
# ONC Clustering
# ---------------------------------------------------------------------------


def cluster_kmeans_base(
    corr0: pd.DataFrame,
    max_num_clusters: int = 10,
    n_init: int = 10,
) -> tuple[pd.DataFrame, dict[int, list[str]], pd.Series]:
    """Base ONC clustering via KMeans on correlation distance matrix.

    Finds the best partition by maximizing the silhouette score t-statistic
    across multiple cluster counts and random initializations.

    Args:
        corr0: Correlation matrix (pandas DataFrame, symmetric).
        max_num_clusters: Maximum number of clusters to try.
        n_init: Number of random initializations per cluster count.

    Returns:
        Tuple of (corr1, clstrs, silh):
        - corr1: Reordered correlation matrix (rows/cols grouped by cluster).
        - clstrs: Dict mapping cluster ID to list of feature names.
        - silh: Silhouette scores per feature (pandas Series).

    Reference:
        MLAM Snippet 4.1.
    """
    x = ((1 - corr0.fillna(0)) / 2.0) ** 0.5  # correlation distance
    silh = pd.Series(dtype=float)
    kmeans = None

    for _init in range(n_init):
        for i in range(2, min(max_num_clusters, len(x) - 1) + 1):
            kmeans_ = KMeans(n_clusters=i, n_init=1)
            kmeans_ = kmeans_.fit(x)
            try:
                silh_ = silhouette_samples(x, kmeans_.labels_)
            except ValueError:
                continue
            stat = silh_.mean() / silh_.std()
            stat_prev = silh.mean() / silh.std() if len(silh) > 0 else np.nan
            if np.isnan(stat_prev) or stat > stat_prev:
                silh, kmeans = pd.Series(silh_, index=x.index), kmeans_

    new_idx = np.argsort(kmeans.labels_)
    corr1 = corr0.iloc[new_idx].iloc[:, new_idx]
    clstrs = {
        i: corr0.columns[np.where(kmeans.labels_ == i)[0]].tolist()
        for i in np.unique(kmeans.labels_)
    }
    silh = pd.Series(silh.values, index=x.index)
    return corr1, clstrs, silh


def _make_new_outputs(
    corr0: pd.DataFrame,
    clstrs: dict[int, list[str]],
    clstrs2: dict[int, list[str]],
) -> tuple[pd.DataFrame, dict[int, list[str]], pd.Series]:
    """Merge high-quality clusters with redone clusters (helper for cluster_kmeans_top)."""
    clstrs_new: dict[int, list[str]] = {}
    for i in clstrs:
        clstrs_new[i] = list(clstrs[i])
    for i in clstrs2:
        clstrs_new[len(clstrs_new)] = list(clstrs2[i])

    new_idx = [j for i in clstrs_new for j in clstrs_new[i]]
    corr_new = corr0.loc[new_idx, new_idx]

    x = ((1 - corr0.fillna(0)) / 2.0) ** 0.5
    kmeans_labels = np.zeros(len(x.columns))
    for i in clstrs_new:
        idxs = [x.index.get_loc(k) for k in clstrs_new[i]]
        kmeans_labels[idxs] = i
    silh_new = pd.Series(silhouette_samples(x, kmeans_labels), index=x.index)
    return corr_new, clstrs_new, silh_new


def cluster_kmeans_top(
    corr0: pd.DataFrame,
    max_num_clusters: int | None = None,
    n_init: int = 10,
) -> tuple[pd.DataFrame, dict[int, list[str]], pd.Series]:
    """Recursive top-level ONC clustering with quality-based refinement.

    Performs base clustering, then recursively refines clusters whose silhouette
    t-statistic falls below the mean.

    Args:
        corr0: Correlation matrix (pandas DataFrame, symmetric).
        max_num_clusters: Maximum clusters to try (default: n_features - 1).
        n_init: Number of random initializations per cluster count.

    Returns:
        Tuple of (corr1, clstrs, silh):
        - corr1: Reordered correlation matrix.
        - clstrs: Dict mapping cluster ID to list of feature names.
        - silh: Silhouette scores per feature.

    Reference:
        MLAM Snippet 4.2.
    """
    if max_num_clusters is None:
        max_num_clusters = corr0.shape[1] - 1

    corr1, clstrs, silh = cluster_kmeans_base(
        corr0,
        max_num_clusters=min(max_num_clusters, corr0.shape[1] - 1),
        n_init=n_init,
    )

    cluster_tstats = {i: np.mean(silh[clstrs[i]]) / np.std(silh[clstrs[i]]) for i in clstrs}
    tstat_mean = sum(cluster_tstats.values()) / len(cluster_tstats)
    redo_clusters = [i for i in cluster_tstats if cluster_tstats[i] < tstat_mean]

    if len(redo_clusters) <= 1:
        return corr1, clstrs, silh

    keys_redo = [j for i in redo_clusters for j in clstrs[i]]
    corr_tmp = corr0.loc[keys_redo, keys_redo]
    _corr2, clstrs2, _silh2 = cluster_kmeans_top(
        corr_tmp,
        max_num_clusters=min(max_num_clusters, corr_tmp.shape[1] - 1),
        n_init=n_init,
    )

    good_clstrs = {i: clstrs[i] for i in clstrs if i not in redo_clusters}
    corr_new, clstrs_new, silh_new = _make_new_outputs(corr0, good_clstrs, clstrs2)

    new_tstat_mean = np.mean(
        [
            np.mean(silh_new[clstrs_new[i]]) / np.std(silh_new[clstrs_new[i]])
            for i in clstrs_new
        ]
    )

    if new_tstat_mean <= tstat_mean:
        return corr1, clstrs, silh
    return corr_new, clstrs_new, silh_new


# ---------------------------------------------------------------------------
# Clustered MDI / MDA
# ---------------------------------------------------------------------------


def _group_mean_std(
    df0: pd.DataFrame,
    clstrs: dict[int, list[str]],
) -> pd.DataFrame:
    """Aggregate feature-level importance into cluster-level importance."""
    out = pd.DataFrame(columns=["mean", "std"])
    for i, j in clstrs.items():
        df1 = df0[j].sum(axis=1)
        out.loc["C_" + str(i), "mean"] = df1.mean()
        out.loc["C_" + str(i), "std"] = df1.std() * df1.shape[0] ** -0.5
    return out


def feat_imp_mdi_clustered(
    fit: object,
    feat_names: list[str] | pd.Index,
    clstrs: dict[int, list[str]],
) -> pd.DataFrame:
    """Clustered Mean Decrease Impurity.

    Sums MDI importance of all features within each cluster to measure the
    total explanatory power of each signal group.

    Args:
        fit: Fitted ensemble of tree-based classifiers.
        feat_names: Feature names corresponding to training data columns.
        clstrs: Dict mapping cluster ID to list of feature names (from ONC).

    Returns:
        DataFrame with ``mean`` and ``std`` columns indexed by cluster name.
        Normalized so means sum to 1.

    Reference:
        MLAM Snippet 6.4.
    """
    df0 = {i: tree.feature_importances_ for i, tree in enumerate(fit.estimators_)}
    df0 = pd.DataFrame.from_dict(df0, orient="index")
    df0.columns = feat_names
    df0 = df0.replace(0, np.nan)
    imp = _group_mean_std(df0, clstrs)
    imp /= imp["mean"].sum()
    return imp


def feat_imp_mda_clustered(
    clf: object,
    X: pd.DataFrame,
    y: pd.Series,
    clstrs: dict[int, list[str]],
    n_splits: int = 10,
) -> pd.DataFrame:
    """Clustered Mean Decrease Accuracy.

    Shuffles all features within each cluster simultaneously and measures
    OOS performance decay.

    Args:
        clf: Classifier with ``.fit()``, ``.predict_proba()``.
        X: Feature matrix (pandas DataFrame).
        y: Labels (pandas Series).
        clstrs: Dict mapping cluster ID to list of feature names (from ONC).
        n_splits: Number of cross-validation folds.

    Returns:
        DataFrame with ``mean`` and ``std`` columns indexed by cluster name.

    Reference:
        MLAM Snippet 6.5.
    """
    from sklearn.model_selection import KFold

    cv_gen = KFold(n_splits=n_splits)
    scr0 = pd.Series(dtype=float)
    scr1 = pd.DataFrame(columns=list(clstrs.keys()))

    for i, (train, test) in enumerate(cv_gen.split(X=X)):
        X0, y0 = X.iloc[train, :], y.iloc[train]
        X1, y1 = X.iloc[test, :], y.iloc[test]
        fit = clf.fit(X=X0, y=y0)
        prob = fit.predict_proba(X1)
        scr0.loc[i] = -log_loss(y1, prob, labels=clf.classes_)

        for j in scr1.columns:
            X1_ = X1.copy(deep=True)
            for k in clstrs[j]:
                np.random.shuffle(X1_[k].values)
            prob = fit.predict_proba(X1_)
            scr1.loc[i, j] = -log_loss(y1, prob, labels=clf.classes_)

    imp = (-1 * scr1).add(scr0, axis=0)
    imp = imp / (-1 * scr1)
    imp = pd.concat({"mean": imp.mean(), "std": imp.std() * imp.shape[0] ** -0.5}, axis=1)
    imp.index = ["C_" + str(i) for i in imp.index]
    return imp


# ---------------------------------------------------------------------------
# Synthetic data + pipeline
# ---------------------------------------------------------------------------


def get_test_data(
    n_features: int = 40,
    n_informative: int = 10,
    n_redundant: int = 10,
    n_samples: int = 10000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic dataset for testing feature importance algorithms.

    Creates a classification dataset with informative, redundant, and noise features.

    Args:
        n_features: Total number of features.
        n_informative: Number of informative features.
        n_redundant: Number of redundant features (linear combos of informative).
        n_samples: Number of observations.

    Returns:
        Tuple of (trns_x, cont):
        - trns_x: DataFrame of features with columns named I_0..., R_0..., N_0...
        - cont: DataFrame with ``'bin'`` (labels), ``'w'`` (equal weights), and
          ``'t1'`` (through dates).

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
    df0 = pd.date_range(start=pd.Timestamp("2015-01-01"), periods=n_samples, freq="h")
    trns_x, y = pd.DataFrame(trns_x, index=df0), pd.Series(y, index=df0)

    cols = ["I_" + str(i) for i in range(n_informative)]
    cols += ["R_" + str(i) for i in range(n_redundant)]
    cols += ["N_" + str(i) for i in range(n_features - len(cols))]
    trns_x.columns = cols

    cont = pd.DataFrame({"bin": y})
    cont["w"] = 1.0 / n_samples
    cont["t1"] = pd.Series(trns_x.index, index=trns_x.index)
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

    Args:
        trns_x: Feature matrix (pandas DataFrame).
        cont: DataFrame with ``'bin'`` (labels), ``'w'`` (weights), ``'t1'`` (dates).
        n_estimators: Number of base estimators in the bagging ensemble.
        cv: Number of cross-validation folds.
        max_samples: Fraction of samples per base estimator.
        num_threads: Number of parallel jobs (-1 if > 1, else 1).
        pct_embargo: Embargo fraction for purged CV.
        scoring: ``'neg_log_loss'`` or ``'accuracy'``.
        method: ``'MDI'``, ``'MDA'``, or ``'SFI'``.
        min_weight_fraction_leaf: Minimum weighted fraction for leaf nodes.

    Returns:
        Tuple of (imp, oob_score).

    Reference:
        AFML Snippet 8.8.
    """
    n_jobs = -1 if num_threads > 1 else 1

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


def plot_feat_importance(
    imp: pd.DataFrame,
    oob: float,
    method: str,
    n_informative: int,
    n_redundant: int,
    save_fig: str | None = None,
) -> None:
    """Visualize feature importance rankings as a horizontal bar chart.

    Args:
        imp: DataFrame with ``mean`` and ``std`` columns.
        oob: Out-of-bag score to display in the title.
        method: Importance method name for axis label.
        n_informative: Number of informative features (for reference in title).
        n_redundant: Number of redundant features (for reference in title).
        save_fig: If provided, save figure to this path instead of showing.

    Reference:
        AFML Snippet 8.10.
    """
    import matplotlib.pyplot as plt

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
