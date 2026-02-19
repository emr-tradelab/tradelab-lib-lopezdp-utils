"""Shared fixtures for features package tests."""

import numpy as np
import pandas as pd
import polars as pl
import pytest


@pytest.fixture
def price_series() -> pl.DataFrame:
    """500-bar close price series with realistic random walk for fracdiff/structural tests."""
    np.random.seed(42)
    n = 500
    timestamps = pl.datetime_range(
        pl.datetime(2024, 1, 1),
        pl.datetime(2024, 1, 1, 8, 19),
        interval="1m",
        eager=True,
    )
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.3)
    return pl.DataFrame({"timestamp": timestamps, "close": close})


@pytest.fixture
def log_price_series(price_series) -> pl.Series:
    """Log prices for structural break tests."""
    return price_series.select(pl.col("close").log().alias("log_close"))["log_close"]


@pytest.fixture
def return_series() -> pl.Series:
    """1000-observation return series for entropy tests."""
    np.random.seed(42)
    returns = np.random.randn(1000) * 0.01
    return pl.Series("returns", returns)


@pytest.fixture
def explosive_series() -> pl.DataFrame:
    """Price series with an explosive (bubble) regime for SADF tests."""
    np.random.seed(42)
    timestamps = pl.datetime_range(
        pl.datetime(2024, 1, 1),
        pl.datetime(2024, 1, 1, 4, 59),
        interval="1m",
        eager=True,
    )
    # Random walk for 200 bars, then exponential growth for 100
    rw = 100.0 + np.cumsum(np.random.randn(200) * 0.3)
    bubble = rw[-1] * np.exp(np.linspace(0, 0.5, 100))
    close = np.concatenate([rw, bubble])
    return pl.DataFrame({"timestamp": timestamps, "close": close})


@pytest.fixture
def synthetic_features() -> tuple:
    """Synthetic classification data for feature importance tests."""
    from sklearn.datasets import make_classification

    np.random.seed(42)
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=3,
        n_redundant=2,
        n_clusters_per_class=1,
        random_state=42,
    )

    feat_names = [f"f_{i}" for i in range(10)]
    X_df = pd.DataFrame(X, columns=feat_names)
    return X_df, pd.Series(y, name="label"), feat_names
