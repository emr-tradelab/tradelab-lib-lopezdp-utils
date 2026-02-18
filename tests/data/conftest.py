"""Shared fixtures for data package tests."""

import numpy as np
import polars as pl
import pytest


@pytest.fixture
def ohlcv_1min() -> pl.DataFrame:
    """Synthetic 1-minute OHLCV data (100 bars)."""
    np.random.seed(42)
    n = 100
    timestamps = pl.datetime_range(
        pl.datetime(2024, 1, 1),
        pl.datetime(2024, 1, 1, 1, 39),  # 100 minutes
        interval="1m",
        eager=True,
    )
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.2)
    low = close - np.abs(np.random.randn(n) * 0.2)
    open_ = close + np.random.randn(n) * 0.1
    volume = np.random.randint(100, 10000, size=n).astype(float)

    return pl.DataFrame({
        "timestamp": timestamps,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


@pytest.fixture
def tick_data() -> pl.DataFrame:
    """Synthetic tick data (1000 ticks) for bar construction."""
    np.random.seed(42)
    n = 1000
    timestamps = pl.datetime_range(
        pl.datetime(2024, 1, 1),
        pl.datetime(2024, 1, 1, 0, 16, 39),  # ~1000 seconds
        interval="1s",
        eager=True,
    )
    prices = 100.0 + np.cumsum(np.random.randn(n) * 0.1)
    volumes = np.random.randint(1, 100, size=n).astype(float)

    return pl.DataFrame({
        "timestamp": timestamps,
        "price": prices,
        "volume": volumes,
    })


@pytest.fixture
def close_series() -> pl.DataFrame:
    """Close price series as a Polars DataFrame with timestamp index."""
    np.random.seed(42)
    n = 500
    timestamps = pl.datetime_range(
        pl.datetime(2024, 1, 1),
        pl.datetime(2024, 1, 1, 8, 19),  # 500 minutes
        interval="1m",
        eager=True,
    )
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.3)

    return pl.DataFrame({
        "timestamp": timestamps,
        "close": close,
    })
