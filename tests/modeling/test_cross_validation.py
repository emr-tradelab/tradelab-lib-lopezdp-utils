"""Tests for modeling.cross_validation — PurgedKFold, embargo, cv_score."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier


class TestGetTrainTimes:
    def test_removes_overlapping_observations(self):
        from tradelab.lopezdp_utils.modeling.cross_validation import get_train_times

        # 20 observations, t1 spans 3 periods each
        idx = pd.date_range("2024-01-01", periods=20, freq="min")
        raw = pd.Series(
            pd.date_range("2024-01-01 00:03", periods=20, freq="min"),
            index=idx,
        )
        t1 = raw.clip(upper=idx[-1])
        # Test set: observations 8-12
        test_times = t1.iloc[8:13]
        result = get_train_times(t1, test_times)
        # No training observation should overlap with test
        for train_start, train_end in result.items():
            for test_start, test_end in test_times.items():
                assert not (train_start <= test_end and train_end >= test_start)


class TestGetEmbargoTimes:
    def test_returns_series(self):
        from tradelab.lopezdp_utils.modeling.cross_validation import get_embargo_times

        times = pd.date_range("2024-01-01", periods=100, freq="min")
        result = get_embargo_times(times, pct_embargo=0.01)
        assert isinstance(result, pd.Series)
        assert len(result) == len(times)

    def test_embargo_offset(self):
        from tradelab.lopezdp_utils.modeling.cross_validation import get_embargo_times

        times = pd.date_range("2024-01-01", periods=100, freq="min")
        result = get_embargo_times(times, pct_embargo=0.05)
        # First observation's embargo should be ~ 5 steps ahead
        assert result.iloc[0] >= times[4]


class TestPurgedKFold:
    def test_no_leakage(self, classification_data):
        from tradelab.lopezdp_utils.modeling.cross_validation import PurgedKFold

        X, y, t1 = classification_data
        pkf = PurgedKFold(n_splits=3, t1=t1, pct_embargo=0.01)
        for train_idx, test_idx in pkf.split(X):
            train_times = t1.iloc[train_idx]
            test_start = X.index[test_idx].min()
            test_end = X.index[test_idx].max()
            # No train observation's label should strictly overlap with test period
            # (t1 == test_start is allowed — label ends exactly as test begins)
            assert not any(
                (train_times > test_start) & (train_times.index < test_start)
            )

    def test_correct_number_of_splits(self, classification_data):
        from tradelab.lopezdp_utils.modeling.cross_validation import PurgedKFold

        X, y, t1 = classification_data
        pkf = PurgedKFold(n_splits=5, t1=t1, pct_embargo=0.0)
        splits = list(pkf.split(X))
        assert len(splits) == 5

    def test_raises_without_t1(self, classification_data):
        from tradelab.lopezdp_utils.modeling.cross_validation import PurgedKFold

        X, y, _ = classification_data
        with pytest.raises((ValueError, TypeError)):
            pkf = PurgedKFold(n_splits=3, t1=None)
            list(pkf.split(X))

    def test_all_indices_covered(self, classification_data):
        """Every observation should appear in exactly one test fold."""
        from tradelab.lopezdp_utils.modeling.cross_validation import PurgedKFold

        X, y, t1 = classification_data
        pkf = PurgedKFold(n_splits=3, t1=t1)
        all_test = []
        for _, test_idx in pkf.split(X):
            all_test.extend(test_idx.tolist())
        assert sorted(all_test) == list(range(len(X)))


class TestCVScore:
    def test_returns_array_of_scores(self, classification_data, sample_weights):
        from tradelab.lopezdp_utils.modeling.cross_validation import cv_score

        X, y, t1 = classification_data
        clf = RandomForestClassifier(n_estimators=20, random_state=42)
        scores = cv_score(
            clf, X, y,
            sample_weight=sample_weights,
            scoring="accuracy",
            t1=t1,
            cv=3,
        )
        assert isinstance(scores, np.ndarray)
        assert len(scores) == 3
        assert all(0 <= s <= 1 for s in scores)

    def test_neg_log_loss_scoring(self, classification_data, sample_weights):
        from tradelab.lopezdp_utils.modeling.cross_validation import cv_score

        X, y, t1 = classification_data
        clf = RandomForestClassifier(n_estimators=20, random_state=42)
        scores = cv_score(
            clf, X, y,
            sample_weight=sample_weights,
            scoring="neg_log_loss",
            t1=t1,
            cv=3,
        )
        assert isinstance(scores, np.ndarray)
        assert all(s <= 0 for s in scores)  # neg_log_loss is negative


class TestProbabilityWeightedAccuracy:
    def test_perfect_confident_prediction(self):
        from tradelab.lopezdp_utils.modeling.cross_validation import (
            probability_weighted_accuracy,
        )

        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0])
        prob = np.array([0.9, 0.8, 0.85, 0.95])
        result = probability_weighted_accuracy(y_true, y_pred, prob)
        assert result > 0.5

    def test_random_predictions_near_zero(self):
        from tradelab.lopezdp_utils.modeling.cross_validation import (
            probability_weighted_accuracy,
        )

        np.random.seed(42)
        n = 200
        y_true = np.random.randint(0, 2, n)
        y_pred = np.random.randint(0, 2, n)
        prob = np.ones(n) * 0.5  # no excess confidence
        result = probability_weighted_accuracy(y_true, y_pred, prob)
        assert abs(result) < 0.3  # should be near zero
