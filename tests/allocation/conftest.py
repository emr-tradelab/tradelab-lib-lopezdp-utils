"""Shared fixtures for allocation package tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_cov() -> pd.DataFrame:
    """5x5 positive-definite covariance matrix."""
    np.random.seed(42)
    n_assets = 5
    n_obs = 500
    data = np.random.randn(n_obs, n_assets)
    # Add some correlation structure
    data[:, 1] = data[:, 0] * 0.8 + data[:, 1] * 0.6
    data[:, 3] = data[:, 2] * 0.7 + data[:, 3] * 0.7
    cov = pd.DataFrame(
        np.cov(data, rowvar=False),
        columns=[f"asset_{i}" for i in range(n_assets)],
        index=[f"asset_{i}" for i in range(n_assets)],
    )
    return cov


@pytest.fixture
def sample_corr(sample_cov) -> pd.DataFrame:
    """Correlation matrix derived from sample_cov."""
    std = np.sqrt(np.diag(sample_cov.values))
    corr = np.array(sample_cov.values / np.outer(std, std))
    np.clip(corr, -1, 1, out=corr)
    np.fill_diagonal(corr, 1.0)
    return pd.DataFrame(corr, columns=sample_cov.columns, index=sample_cov.index)


@pytest.fixture
def large_cov() -> pd.DataFrame:
    """50x50 covariance matrix for denoising tests (high q = T/N ratio)."""
    np.random.seed(42)
    n_assets = 50
    n_obs = 500
    data = np.random.randn(n_obs, n_assets)
    # Add block correlation structure
    for i in range(0, n_assets, 10):
        block = data[:, i : i + 10]
        common = np.random.randn(n_obs, 1) * 0.5
        data[:, i : i + 10] = block + common
    cols = [f"asset_{i}" for i in range(n_assets)]
    return pd.DataFrame(np.cov(data, rowvar=False), columns=cols, index=cols)
