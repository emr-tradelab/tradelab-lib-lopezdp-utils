"""Synthetic O-U process and Optimal Trading Rules — AFML Chapter 13.

References:
    López de Prado, "Advances in Financial Machine Learning", Chapter 13
"""

from itertools import product
from random import gauss

import numpy as np
import polars as pl


def ou_half_life(phi: float) -> float:
    """Compute half-life of convergence from O-U speed parameter.

    Args:
        phi: Speed of convergence parameter, must be in (0, 1).

    Returns:
        Half-life in number of periods.
    """
    if phi <= 0 or phi >= 1:
        raise ValueError(f"phi must be in (0, 1), got {phi}")
    return -np.log(2) / np.log(phi)


def ou_fit(prices: pl.Series, forecast: float) -> dict[str, float]:
    """Estimate O-U process parameters from a price series via OLS.

    Args:
        prices: Polars Series of prices.
        forecast: Target price level E_0[P_T].

    Returns:
        Dictionary with 'phi', 'sigma', 'half_life' keys.
    """
    vals = prices.to_numpy()
    y = vals[1:]
    x = vals[:-1] - forecast

    phi = float(np.cov(x, y, bias=True)[0, 1] / np.var(x))
    residuals = y - forecast - phi * x
    sigma = float(np.std(residuals))

    result = {"phi": phi, "sigma": sigma}
    if 0 < phi < 1:
        result["half_life"] = ou_half_life(phi)
    else:
        result["half_life"] = float("nan")

    return result


def otr_batch(
    coeffs: dict[str, float],
    n_iter: int = 100_000,
    max_hp: int = 100,
    r_pt: np.ndarray | None = None,
    r_slm: np.ndarray | None = None,
    seed: float = 0,
) -> pl.DataFrame:
    """Monte Carlo simulation of trading rules under the O-U process.

    Args:
        coeffs: O-U parameters with keys 'forecast', 'hl', 'sigma'.
        n_iter: Number of Monte Carlo paths per trading rule.
        max_hp: Maximum holding period (vertical barrier).
        r_pt: Array of profit-taking thresholds.
        r_slm: Array of stop-loss thresholds.
        seed: Initial price level P_0.

    Returns:
        Polars DataFrame with columns: 'r_pt', 'r_slm', 'mean_pnl', 'std_pnl', 'sharpe'.
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

    return pl.DataFrame(results)


def otr_main(
    forecasts: list[float] | None = None,
    half_lives: list[float] | None = None,
    sigma: float = 1,
    n_iter: int = 100_000,
    max_hp: int = 100,
) -> dict[tuple[float, float], pl.DataFrame]:
    """Run OTR experiment across market regimes.

    Args:
        forecasts: List of forecast price levels.
        half_lives: List of half-lives.
        sigma: Process volatility.
        n_iter: Number of Monte Carlo paths per rule per regime.
        max_hp: Maximum holding period.

    Returns:
        Dictionary mapping (forecast, half_life) to DataFrames of results.
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
