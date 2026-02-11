"""Optimal Trading Rules (OTR) via Monte Carlo simulation.

Generates synthetic price paths under the Ornstein-Uhlenbeck process and
evaluates trading rules (profit-taking and stop-loss thresholds) by computing
Sharpe ratios across a parameter mesh.

Reference: López de Prado, "Advances in Financial Machine Learning", Ch. 13, Snippets 13.1 & 13.2.
"""

from itertools import product
from random import gauss

import numpy as np
import pandas as pd


def otr_batch(
    coeffs: dict[str, float],
    n_iter: int = 100_000,
    max_hp: int = 100,
    r_pt: np.ndarray | None = None,
    r_slm: np.ndarray | None = None,
    seed: float = 0,
) -> pd.DataFrame:
    """Monte Carlo simulation of trading rules under the O-U process.

    For each (profit-taking, stop-loss) pair in the threshold mesh, generates
    n_iter synthetic price paths under the discrete-time O-U process:
        P_t = (1 - phi) * forecast + phi * P_{t-1} + sigma * eps_t
    Each path exits when hitting the profit-taking barrier, stop-loss barrier,
    or the maximum holding period (vertical barrier).

    Args:
        coeffs: O-U process parameters with keys:
            - 'forecast': Target price level E_0[P_T].
            - 'hl': Half-life of convergence (τ).
            - 'sigma': Volatility of the process.
        n_iter: Number of Monte Carlo paths per trading rule.
        max_hp: Maximum holding period (vertical barrier).
        r_pt: Array of profit-taking thresholds. Defaults to linspace(0.5, 10, 20).
        r_slm: Array of stop-loss thresholds. Defaults to linspace(0.5, 10, 20).
        seed: Initial price level P_0.

    Returns:
        DataFrame with columns: 'r_pt', 'r_slm', 'mean_pnl', 'std_pnl', 'sharpe'.

    Reference:
        AFML Ch. 13, Snippet 13.2.
    """
    if r_pt is None:
        r_pt = np.linspace(0.5, 10, 20)
    if r_slm is None:
        r_slm = np.linspace(0.5, 10, 20)

    phi = 2 ** (-1.0 / coeffs["hl"])
    results = []

    for pt, sl in product(r_pt, r_slm):
        pnls = []
        for _ in range(n_iter):
            p = seed
            hp = 0
            while True:
                p = (1 - phi) * coeffs["forecast"] + phi * p + coeffs["sigma"] * gauss(0, 1)
                cp = p - seed
                hp += 1
                if cp > pt or cp < -sl or hp > max_hp:
                    pnls.append(cp)
                    break

        mean_pnl = np.mean(pnls)
        std_pnl = np.std(pnls)
        sharpe = mean_pnl / std_pnl if std_pnl > 0 else 0.0
        results.append(
            {
                "r_pt": pt,
                "r_slm": sl,
                "mean_pnl": mean_pnl,
                "std_pnl": std_pnl,
                "sharpe": sharpe,
            }
        )

    return pd.DataFrame(results)


def otr_main(
    forecasts: list[float] | None = None,
    half_lives: list[float] | None = None,
    sigma: float = 1,
    n_iter: int = 100_000,
    max_hp: int = 100,
) -> dict[tuple[float, float], pd.DataFrame]:
    """Run OTR experiment across market regimes.

    Generates a Cartesian product of forecast levels and half-lives, then
    runs otr_batch for each regime to find optimal trading rules.

    Args:
        forecasts: List of forecast price levels. Defaults to [10, 5, 0, -5, -10].
        half_lives: List of half-lives. Defaults to [1, 5].
        sigma: Process volatility (constant across regimes).
        n_iter: Number of Monte Carlo paths per rule per regime.
        max_hp: Maximum holding period.

    Returns:
        Dictionary mapping (forecast, half_life) tuples to DataFrames of results
        from otr_batch.

    Reference:
        AFML Ch. 13, Snippet 13.1.
    """
    if forecasts is None:
        forecasts = [10, 5, 0, -5, -10]
    if half_lives is None:
        half_lives = [1, 5]

    r_pt = np.linspace(0, 10, 21)
    r_slm = np.linspace(0, 10, 21)

    outputs = {}
    for forecast, hl in product(forecasts, half_lives):
        coeffs = {"forecast": forecast, "hl": hl, "sigma": sigma}
        outputs[(forecast, hl)] = otr_batch(
            coeffs, n_iter=n_iter, max_hp=max_hp, r_pt=r_pt, r_slm=r_slm
        )

    return outputs
