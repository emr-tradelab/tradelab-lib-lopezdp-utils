"""Shared fixtures for labeling package tests."""

import numpy as np
import polars as pl
import pytest


@pytest.fixture
def close_prices() -> pl.DataFrame:
    """500-bar close price series with realistic random walk."""
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
def close_with_trend() -> pl.DataFrame:
    """Price series with a clear uptrend then downtrend for labeling tests."""
    np.random.seed(42)
    timestamps = pl.datetime_range(
        pl.datetime(2024, 1, 1),
        pl.datetime(2024, 1, 1, 3, 19),
        interval="1m",
        eager=True,
    )
    # Up for 100 bars, down for 100 bars
    up = np.linspace(100, 110, 100) + np.random.randn(100) * 0.1
    down = np.linspace(110, 95, 100) + np.random.randn(100) * 0.1
    close = np.concatenate([up, down])
    return pl.DataFrame({"timestamp": timestamps, "close": close})


@pytest.fixture
def events_with_t1(close_prices) -> pl.DataFrame:
    """Pre-built events DataFrame with t1 column for weight tests."""
    np.random.seed(42)
    # Sample 50 events from the close prices
    event_indices = np.sort(np.random.choice(400, size=50, replace=False))
    timestamps = close_prices["timestamp"]
    t0 = timestamps.gather(event_indices)
    # t1 = t0 + random 5-20 bars
    offsets = np.random.randint(5, 20, size=50)
    t1_indices = np.minimum(event_indices + offsets, 499)
    t1 = timestamps.gather(t1_indices)
    labels = np.random.choice([-1, 0, 1], size=50)

    return pl.DataFrame(
        {
            "timestamp": t0,
            "t1": t1,
            "label": labels,
        }
    )
