"""Tests for modeling.hyperparameter_tuning."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class TestLogUniform:
    def test_samples_in_range(self):
        from tradelab.lopezdp_utils.modeling.hyperparameter_tuning import log_uniform

        dist = log_uniform(a=1e-3, b=1e3)
        samples = dist.rvs(size=1000)
        assert np.all(samples >= 1e-3)
        assert np.all(samples <= 1e3)

    def test_log_is_uniform(self):
        """Log of samples should be approximately uniformly distributed."""
        from tradelab.lopezdp_utils.modeling.hyperparameter_tuning import log_uniform

        np.random.seed(42)
        dist = log_uniform(a=1e-2, b=1e2)
        samples = np.log(dist.rvs(size=5000))
        # Check that the log-samples span the range reasonably uniformly
        hist, _ = np.histogram(samples, bins=10)
        # No bin should have < 10% of samples (200 of 2000)
        assert all(h > 200 for h in hist)


class TestMyPipeline:
    def test_propagates_sample_weight(self, classification_data):
        from tradelab.lopezdp_utils.modeling.hyperparameter_tuning import MyPipeline

        X, y, _t1 = classification_data
        pipe = MyPipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=10, random_state=42)),
        ])
        # Should not raise when sample_weight is passed
        sw = np.ones(len(y))
        pipe.fit(X, y, sample_weight=sw)
        score = pipe.score(X, y)
        assert score > 0.5


class TestClfHyperFit:
    def test_returns_fitted_pipeline(self, classification_data):
        from tradelab.lopezdp_utils.modeling.hyperparameter_tuning import clf_hyper_fit

        X, y, t1 = classification_data
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(random_state=42)),
        ])
        param_grid = {"clf__n_estimators": [10, 20]}
        result = clf_hyper_fit(
            feat=X, lbl=y, t1=t1,
            pipe_clf=pipe,
            param_grid=param_grid,
            cv=3,
            bagging=[0, None, 1.0],
        )
        # Should return a fitted estimator
        preds = result.predict(X)
        assert len(preds) == len(y)

    def test_with_randomized_search(self, classification_data):
        from tradelab.lopezdp_utils.modeling.hyperparameter_tuning import (
            clf_hyper_fit,
        )

        X, y, t1 = classification_data
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(random_state=42)),
        ])
        param_grid = {"clf__n_estimators": [10, 20, 50]}
        result = clf_hyper_fit(
            feat=X, lbl=y, t1=t1,
            pipe_clf=pipe,
            param_grid=param_grid,
            cv=3,
            rnd_search_iter=5,
            bagging=[0, None, 1.0],
        )
        preds = result.predict(X)
        assert len(preds) == len(y)
