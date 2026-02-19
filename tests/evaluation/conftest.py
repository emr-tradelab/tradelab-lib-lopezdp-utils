"""Shared fixtures for evaluation package tests."""

import numpy as np
import polars as pl
import pytest


@pytest.fixture
def daily_returns() -> pl.Series:
    """252-day return series for Sharpe ratio / drawdown tests."""
    np.random.seed(42)
    returns = np.random.randn(252) * 0.01 + 0.0003  # slight positive drift
    return pl.Series("returns", returns)


@pytest.fixture
def pnl_series() -> pl.DataFrame:
    """Cumulative PnL series with a drawdown episode."""
    np.random.seed(42)
    n = 252
    timestamps = pl.datetime_range(
        pl.datetime(2024, 1, 1),
        pl.datetime(2024, 9, 8),
        interval="1d",
        eager=True,
    )[:n]
    # Up for 150 days, down for 50, recover for 52
    up = np.linspace(100, 130, 150) + np.random.randn(150) * 0.5
    down = np.linspace(130, 115, 50) + np.random.randn(50) * 0.5
    recover = np.linspace(115, 135, 52) + np.random.randn(52) * 0.5
    pnl = np.concatenate([up, down, recover])
    return pl.DataFrame({"timestamp": timestamps, "pnl": pnl})


@pytest.fixture
def position_series() -> pl.DataFrame:
    """Target position series with flips and flattenings for bet timing tests."""
    np.random.seed(42)
    n = 100
    timestamps = pl.datetime_range(
        pl.datetime(2024, 1, 1),
        pl.datetime(2024, 4, 9),
        interval="1d",
        eager=True,
    )[:n]
    # Long, flat, short, flat, long
    pos = np.concatenate(
        [
            np.ones(20),  # long
            np.zeros(10),  # flat
            -np.ones(25),  # short
            np.zeros(10),  # flat
            np.ones(35),  # long
        ]
    )
    return pl.DataFrame({"timestamp": timestamps, "position": pos})


@pytest.fixture
def trial_returns() -> pl.DataFrame:
    """Multiple strategy trial return series for CSCV/PBO tests."""
    np.random.seed(42)
    n_obs = 200
    n_trials = 10
    data = {
        "timestamp": pl.datetime_range(
            pl.datetime(2024, 1, 1),
            pl.datetime(2024, 7, 18),
            interval="1d",
            eager=True,
        )[:n_obs]
    }
    for i in range(n_trials):
        data[f"trial_{i}"] = np.random.randn(n_obs) * 0.01
    return pl.DataFrame(data)


@pytest.fixture
def price_series_ou() -> pl.Series:
    """Mean-reverting price series for O-U tests."""
    np.random.seed(42)
    n = 500
    phi = 0.95
    mu = 100.0
    sigma = 0.5
    prices = np.zeros(n)
    prices[0] = mu
    for t in range(1, n):
        prices[t] = mu + phi * (prices[t - 1] - mu) + np.random.randn() * sigma
    return pl.Series("price", prices)
