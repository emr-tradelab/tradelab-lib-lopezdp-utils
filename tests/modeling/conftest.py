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
    _X, y, _t1 = classification_data
    return pd.Series(np.ones(len(y)), index=y.index, name="weight")
