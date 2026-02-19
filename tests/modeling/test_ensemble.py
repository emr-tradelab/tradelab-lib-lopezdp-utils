"""Tests for modeling.ensemble — bagging accuracy, RF builders."""

import numpy as np
import pytest
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier


class TestBaggingAccuracy:
    def test_single_perfect_classifier(self):
        from tradelab.lopezdp_utils.modeling.ensemble import bagging_accuracy

        result = bagging_accuracy(n=1, p=1.0)
        assert abs(result - 1.0) < 1e-10

    def test_many_good_classifiers_high_accuracy(self):
        from tradelab.lopezdp_utils.modeling.ensemble import bagging_accuracy

        result = bagging_accuracy(n=101, p=0.6)
        assert result > 0.8

    def test_random_classifiers_near_half(self):
        from tradelab.lopezdp_utils.modeling.ensemble import bagging_accuracy

        result = bagging_accuracy(n=101, p=0.5)
        assert abs(result - 0.5) < 0.1


class TestBuildRandomForest:
    def test_method_0_returns_rf(self):
        from tradelab.lopezdp_utils.modeling.ensemble import build_random_forest

        clf = build_random_forest(method=0)
        assert isinstance(clf, RandomForestClassifier)

    def test_method_1_returns_bagging(self):
        from tradelab.lopezdp_utils.modeling.ensemble import build_random_forest

        clf = build_random_forest(avg_uniqueness=0.5, method=1)
        assert isinstance(clf, BaggingClassifier)

    def test_method_1_requires_uniqueness(self):
        from tradelab.lopezdp_utils.modeling.ensemble import build_random_forest

        with pytest.raises((ValueError, TypeError)):
            build_random_forest(method=1, avg_uniqueness=None)


class TestBaggingClassifierFactory:
    def test_returns_bagging_classifier(self):
        from tradelab.lopezdp_utils.modeling.ensemble import bagging_classifier_factory
        from sklearn.tree import DecisionTreeClassifier

        clf = bagging_classifier_factory(
            base_estimator=DecisionTreeClassifier(),
            n_estimators=10,
        )
        assert isinstance(clf, BaggingClassifier)
