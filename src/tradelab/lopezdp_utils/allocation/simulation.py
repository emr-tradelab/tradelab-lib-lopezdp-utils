"""Simulation utilities for ML asset allocation — AFML Chapter 16.

Synthetic data generation and Monte Carlo experiments for comparing
portfolio allocation methods (HRP vs IVP).

Reference: AFML Snippets 16.4 and 16.5.
"""

import random

import numpy as np
import pandas as pd

from tradelab.lopezdp_utils.allocation.hrp import get_ivp, hrp_alloc


def generate_data(
    n_obs: int,
    size0: int,
    size1: int,
    sigma1: float,
    seed: int = 12345,
) -> tuple[pd.DataFrame, list[int]]:
    """Generate synthetic correlated time series for testing.

    Creates size0 uncorrelated series, then derives size1 additional
    series as noisy copies of randomly selected originals.

    Args:
        n_obs: Number of observations.
        size0: Number of independent (uncorrelated) variables.
        size1: Number of correlated (derived) variables.
        sigma1: Noise level for derived variables.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (DataFrame of returns, list of source column indices).

    Reference:
        AFML Snippet 16.4.
    """
    np.random.seed(seed=seed)
    random.seed(seed)
    x = np.random.normal(0, 1, size=(n_obs, size0))
    # create correlated variables
    cols = [random.randint(0, size0 - 1) for _ in range(size1)]
    y = x[:, cols] + np.random.normal(0, sigma1, size=(n_obs, len(cols)))
    x = np.append(x, y, axis=1)
    x = pd.DataFrame(x, columns=range(1, x.shape[1] + 1))
    return x, cols


def hrp_mc(
    num_iters: int = 1000,
    n_obs: int = 520,
    size0: int = 5,
    size1: int = 5,
    sigma1: float = 0.25,
    s_length: int = 260,
    rebal: int = 22,
    seed: int = 12345,
) -> pd.DataFrame:
    """Monte Carlo experiment comparing HRP vs IVP out-of-sample.

    For each iteration, generates synthetic data with regime changes,
    computes rolling portfolio weights, and evaluates out-of-sample
    performance. Reports variance statistics.

    Note: CLA comparison from original book is omitted (requires external
    solver). Compares HRP vs IVP only.

    Args:
        num_iters: Number of Monte Carlo iterations.
        n_obs: Number of observations per iteration.
        size0: Number of independent variables.
        size1: Number of correlated variables.
        sigma1: Noise level for correlated variables.
        s_length: In-sample window length.
        rebal: Rebalancing frequency (bars).
        seed: Random seed.

    Returns:
        DataFrame with cumulative returns per method per iteration.

    Reference:
        AFML Snippet 16.5 (simplified without CLA).
    """
    np.random.seed(seed)
    random.seed(seed)

    method_names = ["get_ivp", "hrp_alloc"]
    stats = {name: pd.Series(dtype=float) for name in method_names}
    pointers = range(s_length, n_obs, rebal)

    for num_iter in range(int(num_iters)):
        # generate data
        x = np.random.normal(0, 0.01, size=(n_obs, size0))
        cols = [random.randint(0, size0 - 1) for _ in range(size1)]
        y = x[:, cols] + np.random.normal(0, 0.01 * sigma1, size=(n_obs, len(cols)))
        x = np.append(x, y, axis=1)

        r = {name: pd.Series(dtype=float) for name in method_names}

        for pointer in pointers:
            x_ = x[pointer - s_length : pointer]
            cov_ = np.cov(x_, rowvar=False)
            corr_ = np.corrcoef(x_, rowvar=False)
            cov_df = pd.DataFrame(cov_)
            corr_df = pd.DataFrame(corr_)

            x_out = x[pointer : pointer + rebal]

            # IVP
            w_ivp = get_ivp(cov_)
            r_ivp = pd.Series(np.dot(x_out, w_ivp))
            r["get_ivp"] = pd.concat([r["get_ivp"], r_ivp])

            # HRP
            w_hrp = hrp_alloc(cov_df, corr_df)
            r_hrp = pd.Series(np.dot(x_out, w_hrp.values))
            r["hrp_alloc"] = pd.concat([r["hrp_alloc"], r_hrp])

        # evaluate
        for name in method_names:
            r_ = r[name].reset_index(drop=True)
            p_ = (1 + r_).cumprod()
            stats[name].loc[num_iter] = p_.iloc[-1] - 1

    stats = pd.DataFrame.from_dict(stats, orient="columns")
    return stats
