# Phase 2 Session 5: `modeling/` — Cross-Validation, Ensembles, Hyperparameter Tuning

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Migrate `cross_validation/` (2 files, 302 lines) + `ensemble_methods/` (1 file, 167 lines) + `hyperparameter_tuning/` (2 files, 204 lines) into `modeling/` package (3 files). This covers AFML Chapters 6-7, 9 + MLAM Section 6.4.

**Architecture:** `PurgedKFold` is the cornerstone — it enforces purging + embargoing to prevent information leakage. All other modules consume it. This session also updates `features/importance.py` to import from the new `modeling/` path instead of the v1 `cross_validation/`.

**Tech Stack:** numpy, pandas, sklearn (KFold, metrics, GridSearchCV, RandomizedSearchCV, Pipeline, BaggingClassifier, RandomForestClassifier, DecisionTreeClassifier), scipy.special (comb), pydantic, pytest

**Depends on:** Session 4 (`features/` merged to main — `features/importance.py` uses `PurgedKFold`)

> **Note:** This module is **sklearn-native** throughout. No Polars migration — all functions interface with sklearn classifiers that expect pandas/numpy. The value here is tests, Pydantic validation, and consolidation.

---

## Pre-Session Checklist

- [ ] Session 4 merged to main
- [ ] Branch from main: `git checkout -b phase2/modeling`
- [ ] `uv sync --all-extras --dev`

---

## File Mapping: Old → New

| Old file | New location | Action |
|----------|-------------|--------|
| `cross_validation/purging.py` | `modeling/cross_validation.py` | Migrate |
| `cross_validation/scoring.py` | `modeling/cross_validation.py` | Merge into |
| `ensemble_methods/ensemble.py` | `modeling/ensemble.py` | Migrate |
| `hyperparameter_tuning/distributions.py` | `modeling/hyperparameter_tuning.py` | Merge into |
| `hyperparameter_tuning/tuning.py` | `modeling/hyperparameter_tuning.py` | Merge into |

---

## Polars Migration Decisions

**None.** This package is 100% sklearn-native. All functions accept/return pandas DataFrames, numpy arrays, or sklearn objects. Converting to Polars would add complexity with zero benefit since every downstream consumer is sklearn.

---

## Pydantic Validation

```python
from pydantic import BaseModel, field_validator


class PurgedCVConfig(BaseModel):
    """Configuration for purged cross-validation."""
    n_splits: int = 3
    pct_embargo: float = 0.0

    @field_validator("n_splits")
    @classmethod
    def at_least_two_splits(cls, v):
        if v < 2:
            raise ValueError("n_splits must be >= 2")
        return v

    @field_validator("pct_embargo")
    @classmethod
    def embargo_in_range(cls, v):
        if not 0 <= v < 1:
            raise ValueError("pct_embargo must be in [0, 1)")
        return v
```

**Critical validation at function boundaries:**
- `PurgedKFold.split()` must validate that `t1` is provided and has no nulls
- `cv_score` must validate `scoring` is one of the supported values
- `clf_hyper_fit` must validate `bagging` parameter shape

---

## Tasks

### Task 1: Create branch, package structure, and test fixtures

**Files:**
- Create: `src/tradelab/lopezdp_utils/modeling/__init__.py`
- Create: `tests/modeling/__init__.py`
- Create: `tests/modeling/conftest.py`

**Step 1: Create branch**

Run: `git checkout -b phase2/modeling`

**Step 2: Create directories**

Run: `mkdir -p tests/modeling`

**Step 3: Create shared fixtures**

```python
"""Shared fixtures for modeling package tests."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification


@pytest.fixture
def classification_data() -> tuple:
    """Synthetic classification dataset with t1 for purged CV."""
    np.random.seed(42)
    n_samples = 200
    X, y = make_classification(
        n_samples=n_samples,
        n_features=10,
        n_informative=5,
        random_state=42,
    )
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
    y = pd.Series(y, name="label")

    # Create t1 timestamps: each label spans 5 periods
    timestamps = pd.date_range("2024-01-01", periods=n_samples, freq="min")
    X.index = timestamps
    y.index = timestamps
    t1 = pd.Series(
        pd.date_range("2024-01-01 00:05", periods=n_samples, freq="min"),
        index=timestamps,
        name="t1",
    )
    # Clip t1 to not exceed the last timestamp
    t1 = t1.clip(upper=timestamps[-1])

    return X, y, t1


@pytest.fixture
def sample_weights(classification_data) -> pd.Series:
    """Uniform sample weights for testing."""
    X, y, t1 = classification_data
    return pd.Series(np.ones(len(y)), index=y.index, name="weight")
```

**Step 4: Commit**

```bash
git add tests/modeling/ src/tradelab/lopezdp_utils/modeling/
git commit -m "test(modeling): add test skeleton and fixtures"
```

---

### Task 2: Migrate `cross_validation.py` — PurgedKFold, cv_score, embargo, PWA

**Files:**
- Create: `src/tradelab/lopezdp_utils/modeling/cross_validation.py`
- Create: `tests/modeling/test_cross_validation.py`

**Step 1: Write failing tests**

```python
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
        t1 = pd.Series(
            pd.date_range("2024-01-01 00:03", periods=20, freq="min").clip(
                upper=idx[-1]
            ),
            index=idx,
        )
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
            # No train observation's t1 should fall within test period
            assert not any(
                (train_times >= test_start) & (train_times.index <= test_end)
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
```

**Step 2: Run to verify fail**

Run: `uv run pytest tests/modeling/test_cross_validation.py -v`

**Step 3: Implement `modeling/cross_validation.py`**

Implementation notes:
- `get_train_times`, `get_embargo_times`: Copy from v1, clean up style.
- `PurgedKFold(KFold)`: Copy from v1. Add validation that `t1` is not None and has no nulls. Force `shuffle=False`.
- `cv_score`: Copy from v1. Fix the known sklearn `log_loss` / `labels` bug. Accept `sample_weight`.
- `probability_weighted_accuracy`: Copy from `scoring.py`. Pure numpy.
- All functions stay pandas/numpy — no Polars migration.

**Step 4: Run tests**

Run: `uv run pytest tests/modeling/test_cross_validation.py -v`

**Step 5: Commit**

```bash
git add src/tradelab/lopezdp_utils/modeling/cross_validation.py tests/modeling/test_cross_validation.py
git commit -m "feat(modeling): migrate cross_validation with PurgedKFold and cv_score"
```

---

### Task 3: Migrate `ensemble.py` — RF configs, bagging factory

**Files:**
- Create: `src/tradelab/lopezdp_utils/modeling/ensemble.py`
- Create: `tests/modeling/test_ensemble.py`

**Step 1: Write failing tests**

```python
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
```

**Step 2: Run to verify fail**

Run: `uv run pytest tests/modeling/test_ensemble.py -v`

**Step 3: Implement `modeling/ensemble.py`**

Implementation notes:
- Copy from v1 with cleanup.
- `bagging_accuracy`: Pure math with `scipy.special.comb`.
- `build_random_forest`: 3 methods, validate that `avg_uniqueness` is provided for methods 1-2.
- `bagging_classifier_factory`: Simple sklearn wrapper.

**Step 4: Run tests**

Run: `uv run pytest tests/modeling/test_ensemble.py -v`

**Step 5: Commit**

```bash
git add src/tradelab/lopezdp_utils/modeling/ensemble.py tests/modeling/test_ensemble.py
git commit -m "feat(modeling): migrate ensemble methods"
```

---

### Task 4: Migrate `hyperparameter_tuning.py` — log_uniform, MyPipeline, clf_hyper_fit

**Files:**
- Create: `src/tradelab/lopezdp_utils/modeling/hyperparameter_tuning.py`
- Create: `tests/modeling/test_hyperparameter_tuning.py`

**Step 1: Write failing tests**

```python
"""Tests for modeling.hyperparameter_tuning."""

import numpy as np
import pandas as pd
import pytest
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
        # No bin should have < 10% or > 20% of samples
        assert all(h > 200 for h in hist)


class TestMyPipeline:
    def test_propagates_sample_weight(self, classification_data):
        from tradelab.lopezdp_utils.modeling.hyperparameter_tuning import MyPipeline

        X, y, t1 = classification_data
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
            log_uniform,
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
```

**Step 2: Run to verify fail**

Run: `uv run pytest tests/modeling/test_hyperparameter_tuning.py -v`

**Step 3: Implement `modeling/hyperparameter_tuning.py`**

Implementation notes:
- `_LogUniformGen(rv_continuous)` + `log_uniform(a, b)`: Copy from v1 — pure scipy.
- `MyPipeline(Pipeline)`: Copy from v1 — overrides `fit()` to propagate `sample_weight`.
- `clf_hyper_fit`: Copy from v1. Update the internal import of `PurgedKFold` to point to `modeling.cross_validation` instead of `cross_validation`. Auto-selects scoring based on label values. Supports GridSearchCV and RandomizedSearchCV. Optional bagging wrapper.

**Step 4: Run tests**

Run: `uv run pytest tests/modeling/test_hyperparameter_tuning.py -v`

**Step 5: Commit**

```bash
git add src/tradelab/lopezdp_utils/modeling/hyperparameter_tuning.py tests/modeling/test_hyperparameter_tuning.py
git commit -m "feat(modeling): migrate hyperparameter_tuning with PurgedKFold integration"
```

---

### Task 5: Create `modeling/__init__.py` with public exports

```python
"""Model training, cross-validation, and tuning — AFML Chapters 6-7, 9.

This package covers the fifth stage of López de Prado's pipeline:
features → model training with proper cross-validation that respects
temporal dependencies via purging and embargoing.

PurgedKFold is the cornerstone: it prevents information leakage by
removing training observations whose label windows overlap the test set,
and optionally adds an embargo buffer for serial correlation.

Reference:
    López de Prado, "Advances in Financial Machine Learning", Chapters 6-7, 9
    López de Prado, "Machine Learning for Asset Managers", Section 6.4
"""

from tradelab.lopezdp_utils.modeling.cross_validation import (
    PurgedKFold,
    cv_score,
    get_embargo_times,
    get_train_times,
    probability_weighted_accuracy,
)
from tradelab.lopezdp_utils.modeling.ensemble import (
    bagging_accuracy,
    bagging_classifier_factory,
    build_random_forest,
)
from tradelab.lopezdp_utils.modeling.hyperparameter_tuning import (
    MyPipeline,
    clf_hyper_fit,
    log_uniform,
)

__all__ = [
    # Cross-validation
    "PurgedKFold",
    "cv_score",
    "get_train_times",
    "get_embargo_times",
    "probability_weighted_accuracy",
    # Ensembles
    "bagging_accuracy",
    "build_random_forest",
    "bagging_classifier_factory",
    # Hyperparameter tuning
    "log_uniform",
    "MyPipeline",
    "clf_hyper_fit",
]
```

---

### Task 6: Update `features/importance.py` imports

Update the import of `PurgedKFold` and `cv_score` in `features/importance.py` to point to the new `modeling.cross_validation` module:

```python
# OLD:
from tradelab.lopezdp_utils.cross_validation import PurgedKFold, cv_score

# NEW:
from tradelab.lopezdp_utils.modeling.cross_validation import PurgedKFold, cv_score
```

Run: `uv run pytest tests/features/test_importance.py -v` to verify.

```bash
git add src/tradelab/lopezdp_utils/features/importance.py
git commit -m "refactor(features): update importance.py to import from modeling/"
```

---

### Task 7: Integration test — PurgedKFold → cv_score → hyperparameter tuning

**Files:**
- Create: `tests/modeling/test_integration.py`

```python
"""Integration tests: modeling pipeline."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class TestPurgedCVPipeline:
    """End-to-end: PurgedKFold → cv_score → hyperparameter tuning."""

    def test_cv_score_with_purged_kfold(self, classification_data, sample_weights):
        from tradelab.lopezdp_utils.modeling import PurgedKFold, cv_score

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
```

**Steps: Write → Run → Verify pass → Commit**

```bash
git commit -m "test(modeling): add integration tests for purged CV pipeline"
```

---

### Task 8: Delete old directories and verify

**Step 1: Delete old modules**

```bash
rm -rf src/tradelab/lopezdp_utils/cross_validation/
rm -rf src/tradelab/lopezdp_utils/ensemble_methods/
rm -rf src/tradelab/lopezdp_utils/hyperparameter_tuning/
```

**Step 2: Verify imports**

Run: `uv run python -c "from tradelab.lopezdp_utils.modeling import PurgedKFold, cv_score, build_random_forest, clf_hyper_fit; print('OK')"`

**Step 3: Run full test suite**

Run: `uv run pytest tests/ -v`

**Step 4: Lint**

Run: `uvx ruff check --fix . && uvx ruff format .`

**Step 5: Commit**

```bash
git add -A
git commit -m "refactor(modeling): remove old cross_validation/, ensemble_methods/, hyperparameter_tuning/"
```

---

### Task 9: Merge to main

Run: `git checkout main && git merge phase2/modeling`
Verify: `uv run pytest tests/ -v`
