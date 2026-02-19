"""Tests for features.importance — MDI, MDA, SFI, clustering."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


class TestFeatImpMDI:
    def test_returns_dataframe(self, synthetic_features):
        from tradelab.lopezdp_utils.features.importance import feat_imp_mdi

        X, y, feat_names = synthetic_features
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X, y)
        result = feat_imp_mdi(clf, feat_names)
        assert isinstance(result, pd.DataFrame)
        assert "mean" in result.columns
        assert len(result) == len(feat_names)

    def test_importances_sum_to_one(self, synthetic_features):
        from tradelab.lopezdp_utils.features.importance import feat_imp_mdi

        X, y, feat_names = synthetic_features
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X, y)
        result = feat_imp_mdi(clf, feat_names)
        assert abs(result["mean"].sum() - 1.0) < 0.05


class TestFeatImpMDA:
    def test_returns_dataframe_and_score(self, synthetic_features):
        from sklearn.model_selection import KFold

        from tradelab.lopezdp_utils.features.importance import feat_imp_mda

        X, y, _feat_names = synthetic_features
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        cv = KFold(n_splits=3)
        result, score = feat_imp_mda(
            clf, X, y, cv=cv, scoring="accuracy",
        )
        assert isinstance(result, pd.DataFrame)
        assert "mean" in result.columns
        assert isinstance(score, float)


class TestClusterKMeansBase:
    def test_returns_clusters(self):
        from tradelab.lopezdp_utils.features.importance import cluster_kmeans_base

        np.random.seed(42)
        # Build a correlation matrix with 2 clear clusters
        n = 100
        x1 = np.random.randn(n)
        x2 = x1 + np.random.randn(n) * 0.1  # correlated with x1
        x3 = np.random.randn(n)  # independent
        x4 = x3 + np.random.randn(n) * 0.1  # correlated with x3
        data = pd.DataFrame({"a": x1, "b": x2, "c": x3, "d": x4})
        corr = data.corr()
        _, clstrs, _ = cluster_kmeans_base(corr, max_num_clusters=4)
        assert isinstance(clstrs, dict)
        assert len(clstrs) >= 2


class TestGetTestData:
    def test_returns_expected_shape(self):
        from tradelab.lopezdp_utils.features.importance import get_test_data

        trns_x, cont = get_test_data(
            n_features=10, n_informative=3, n_redundant=2, n_samples=200,
        )
        assert isinstance(trns_x, pd.DataFrame)
        assert trns_x.shape == (200, 10)
        assert "bin" in cont.columns
        assert "t1" in cont.columns
