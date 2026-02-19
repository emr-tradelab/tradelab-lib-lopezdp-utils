"""Integration tests: modeling pipeline."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class TestPurgedCVPipeline:
    """End-to-end: PurgedKFold → cv_score → hyperparameter tuning."""

    def test_cv_score_with_purged_kfold(self, classification_data, sample_weights):
        from tradelab.lopezdp_utils.modeling import cv_score

        X, y, t1 = classification_data
        clf = RandomForestClassifier(n_estimators=20, random_state=42)
        scores = cv_score(
            clf, X, y,
            sample_weight=sample_weights,
            scoring="accuracy",
            t1=t1,
            cv=3,
            pct_embargo=0.01,
        )
        assert len(scores) == 3
        assert np.mean(scores) > 0.4  # better than random

    def test_hyper_fit_then_predict(self, classification_data):
        from tradelab.lopezdp_utils.modeling import clf_hyper_fit

        X, y, t1 = classification_data
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(random_state=42)),
        ])
        param_grid = {"clf__n_estimators": [10, 30]}
        best = clf_hyper_fit(
            feat=X, lbl=y, t1=t1,
            pipe_clf=pipe,
            param_grid=param_grid,
            cv=3,
        )
        preds = best.predict(X)
        acc = np.mean(preds == y)
        assert acc > 0.5

    def test_ensemble_with_uniqueness(self, classification_data, sample_weights):
        """Build RF with avg_uniqueness and cross-validate with PurgedKFold."""
        from tradelab.lopezdp_utils.modeling import (
            build_random_forest,
            cv_score,
        )

        X, y, t1 = classification_data
        clf = build_random_forest(avg_uniqueness=0.7, method=1, n_estimators=20)
        scores = cv_score(
            clf, X, y,
            sample_weight=sample_weights,
            scoring="accuracy",
            t1=t1,
            cv=3,
        )
        assert len(scores) == 3
