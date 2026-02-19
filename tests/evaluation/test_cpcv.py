"""Tests for evaluation.cpcv — Combinatorial Purged Cross-Validation."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


class TestGetNumSplits:
    def test_known_values(self):
        from tradelab.lopezdp_utils.evaluation.cpcv import get_num_splits

        assert get_num_splits(n_groups=6, k_test_groups=2) == 15
        assert get_num_splits(n_groups=10, k_test_groups=2) == 45


class TestGetNumBacktestPaths:
    def test_known_values(self):
        from tradelab.lopezdp_utils.evaluation.cpcv import get_num_backtest_paths

        assert get_num_backtest_paths(n_groups=6, k_test_groups=2) == 5
        assert get_num_backtest_paths(n_groups=10, k_test_groups=2) == 9


class TestCombinatorialPurgedKFold:
    def test_correct_number_of_splits(self):
        from tradelab.lopezdp_utils.evaluation.cpcv import CombinatorialPurgedKFold

        np.random.seed(42)
        n = 200
        X = pd.DataFrame(np.random.randn(n, 5), columns=[f"f{i}" for i in range(5)])
        idx = pd.date_range("2024-01-01", periods=n, freq="min")
        X.index = idx
        t1 = pd.Series(
            pd.date_range("2024-01-01 00:05", periods=n, freq="min").map(lambda x: min(x, idx[-1])),
            index=idx,
        )

        cpkf = CombinatorialPurgedKFold(
            n_splits=6,
            k_test_groups=2,
            t1=t1,
            pct_embargo=0.01,
        )
        splits = list(cpkf.split(X))
        assert len(splits) == 15

    def test_purging_reduces_train_set(self):
        from tradelab.lopezdp_utils.evaluation.cpcv import CombinatorialPurgedKFold

        np.random.seed(42)
        n = 120
        X = pd.DataFrame(np.random.randn(n, 3), columns=["a", "b", "c"])
        idx = pd.date_range("2024-01-01", periods=n, freq="min")
        X.index = idx
        t1 = pd.Series(
            pd.date_range("2024-01-01 00:03", periods=n, freq="min").map(lambda x: min(x, idx[-1])),
            index=idx,
        )

        cpkf = CombinatorialPurgedKFold(
            n_splits=6,
            k_test_groups=2,
            t1=t1,
            pct_embargo=0.01,
        )
        for train_idx, test_idx in cpkf.split(X):
            # Train + test should not cover all indices (purging removes some)
            assert len(train_idx) + len(test_idx) < n
            # No train index should be in test
            assert len(set(train_idx) & set(test_idx)) == 0

    def test_all_indices_appear_in_test(self):
        from tradelab.lopezdp_utils.evaluation.cpcv import CombinatorialPurgedKFold

        np.random.seed(42)
        n = 120
        X = pd.DataFrame(np.random.randn(n, 3), columns=["a", "b", "c"])
        idx = pd.date_range("2024-01-01", periods=n, freq="min")
        X.index = idx
        t1 = pd.Series(
            pd.date_range("2024-01-01 00:03", periods=n, freq="min").map(lambda x: min(x, idx[-1])),
            index=idx,
        )

        cpkf = CombinatorialPurgedKFold(
            n_splits=6,
            k_test_groups=2,
            t1=t1,
            pct_embargo=0.0,
        )
        all_test = set()
        for _, test_idx in cpkf.split(X):
            all_test.update(test_idx.tolist())
        assert all_test == set(range(n))


class TestAssembleBacktestPaths:
    def test_returns_correct_count(self):
        from tradelab.lopezdp_utils.evaluation.cpcv import (
            CombinatorialPurgedKFold,
            assemble_backtest_paths,
            get_num_backtest_paths,
        )

        np.random.seed(42)
        n = 120
        n_groups = 6
        k_test = 2
        idx = pd.date_range("2024-01-01", periods=n, freq="min")
        t1 = pd.Series(
            pd.date_range("2024-01-01 00:03", periods=n, freq="min").map(lambda x: min(x, idx[-1])),
            index=idx,
        )

        predictions = {}
        cpkf = CombinatorialPurgedKFold(
            n_splits=n_groups,
            k_test_groups=k_test,
            t1=t1,
        )
        X = pd.DataFrame(np.random.randn(n, 3), columns=["a", "b", "c"], index=idx)
        y = pd.Series(np.random.randint(0, 2, n), index=idx)

        for i, (train_idx, test_idx) in enumerate(cpkf.split(X)):
            clf = RandomForestClassifier(n_estimators=10, random_state=42)
            clf.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = clf.predict_proba(X.iloc[test_idx])
            predictions[i] = pd.DataFrame(preds, index=X.index[test_idx])

        paths = assemble_backtest_paths(predictions, n_groups, k_test, n)
        expected_n_paths = get_num_backtest_paths(n_groups, k_test)
        assert len(paths) == expected_n_paths
