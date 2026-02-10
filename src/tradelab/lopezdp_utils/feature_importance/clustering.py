"""Clustered feature importance using ONC (Optimal Number of Clusters).

Addresses the substitution effect (multicollinearity) in standard MDI/MDA by
grouping correlated features into disjoint clusters and computing importance
at the cluster level. This prevents redundant features from diluting each
other's importance scores.

The ONC algorithm finds the optimal number of clusters by maximizing the
silhouette score t-statistic, with recursive refinement of below-average
clusters.

Reference:
    López de Prado, M. (2020). *Machine Learning for Asset Managers*.
    Section 4 (ONC) and Section 6.5.2 (Clustered Feature Importance).
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import log_loss, silhouette_samples


def cluster_kmeans_base(
    corr0: pd.DataFrame,
    max_num_clusters: int = 10,
    n_init: int = 10,
) -> tuple[pd.DataFrame, dict[int, list[str]], pd.Series]:
    """Base ONC clustering via KMeans on correlation distance matrix.

    Finds the best partition by maximizing the silhouette score t-statistic
    across multiple cluster counts and random initializations. Transforms
    the correlation matrix into a distance matrix via d = sqrt((1 - corr) / 2).

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
        for i in range(2, max_num_clusters + 1):
            kmeans_ = KMeans(n_clusters=i, n_init=1)
            kmeans_ = kmeans_.fit(x)
            silh_ = silhouette_samples(x, kmeans_.labels_)
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
    """Merge high-quality clusters with redone clusters.

    Utility function for ``cluster_kmeans_top`` that combines clusters that
    passed quality checks with newly refined clusters from recursive calls.

    Args:
        corr0: Original correlation matrix.
        clstrs: High-quality clusters to keep.
        clstrs2: Refined clusters from recursive redo.

    Returns:
        Tuple of (corr_new, clstrs_new, silh_new).

    Reference:
        MLAM Snippet 4.2 (helper).
    """
    clstrs_new = {}
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
    t-statistic falls below the mean. Accepts the refinement only if it improves
    overall quality.

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

    # Identify below-average clusters
    cluster_tstats = {i: np.mean(silh[clstrs[i]]) / np.std(silh[clstrs[i]]) for i in clstrs}
    tstat_mean = sum(cluster_tstats.values()) / len(cluster_tstats)
    redo_clusters = [i for i in cluster_tstats if cluster_tstats[i] < tstat_mean]

    if len(redo_clusters) <= 1:
        return corr1, clstrs, silh

    # Recursive refinement
    keys_redo = [j for i in redo_clusters for j in clstrs[i]]
    corr_tmp = corr0.loc[keys_redo, keys_redo]
    _corr2, clstrs2, _silh2 = cluster_kmeans_top(
        corr_tmp,
        max_num_clusters=min(max_num_clusters, corr_tmp.shape[1] - 1),
        n_init=n_init,
    )

    # Merge good clusters with refined clusters
    good_clstrs = {i: clstrs[i] for i in clstrs if i not in redo_clusters}
    corr_new, clstrs_new, silh_new = _make_new_outputs(corr0, good_clstrs, clstrs2)

    new_tstat_mean = np.mean(
        [np.mean(silh_new[clstrs_new[i]]) / np.std(silh_new[clstrs_new[i]]) for i in clstrs_new]
    )

    if new_tstat_mean <= tstat_mean:
        return corr1, clstrs, silh
    return corr_new, clstrs_new, silh_new


def _group_mean_std(
    df0: pd.DataFrame,
    clstrs: dict[int, list[str]],
) -> pd.DataFrame:
    """Aggregate feature-level importance into cluster-level importance.

    Args:
        df0: Feature-level importance matrix (estimators x features).
        clstrs: Dict mapping cluster ID to list of feature names.

    Returns:
        DataFrame with ``mean`` and ``std`` columns indexed by cluster name (C_0, ...).

    Reference:
        MLAM Snippet 6.4 (helper).
    """
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

    Sums the MDI importance of all features within each cluster to measure
    the total explanatory power of each signal group. This prevents correlated
    features from diluting each other's individual importance scores.

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

    Shuffles all features within each cluster simultaneously and measures the
    OOS performance decay. This correctly attributes importance to signal groups
    rather than individual features, solving the substitution effect problem
    where redundant features make each other appear irrelevant.

    Args:
        clf: Classifier with ``.fit()``, ``.predict_proba()``.
        X: Feature matrix (pandas DataFrame).
        y: Labels (pandas Series).
        clstrs: Dict mapping cluster ID to list of feature names (from ONC).
        n_splits: Number of cross-validation folds.

    Returns:
        DataFrame with ``mean`` and ``std`` columns indexed by cluster name.

    Note:
        This simplified version uses standard KFold. For financial data with
        overlapping labels, wrap with PurgedKFold instead.

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
                np.random.shuffle(X1_[k].values)  # shuffle entire cluster
            prob = fit.predict_proba(X1_)
            scr1.loc[i, j] = -log_loss(y1, prob, labels=clf.classes_)

    imp = (-1 * scr1).add(scr0, axis=0)
    imp = imp / (-1 * scr1)
    imp = pd.concat({"mean": imp.mean(), "std": imp.std() * imp.shape[0] ** -0.5}, axis=1)
    imp.index = ["C_" + str(i) for i in imp.index]
    return imp
