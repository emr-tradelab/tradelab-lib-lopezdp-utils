"""Ornstein-Uhlenbeck process parameter estimation.

Provides utilities to estimate O-U process parameters from price series and
compute derived quantities like the half-life of mean reversion.

Reference: López de Prado, "Advances in Financial Machine Learning", Ch. 13, Section 13.5.1.
"""

import numpy as np
import pandas as pd


def ou_half_life(phi: float) -> float:
    """Compute half-life of convergence from O-U speed parameter.

    The half-life measures how many periods it takes for a price divergence
    from the long-run mean to decay by half under the O-U process.

    Args:
        phi: Speed of convergence parameter, must be in (0, 1).
            Values closer to 1 indicate slower mean reversion.

    Returns:
        Half-life in number of periods.

    Reference:
        AFML Ch. 13, Section 13.5.1: τ = -log(2) / log(φ)
    """
    if phi <= 0 or phi >= 1:
        raise ValueError(f"phi must be in (0, 1), got {phi}")
    return -np.log(2) / np.log(phi)


def ou_fit(
    prices: pd.Series,
    forecast: float,
) -> dict[str, float]:
    """Estimate O-U process parameters from a price series via OLS.

    Fits the linearized O-U specification:
        P_t = E_0[P_T] + φ(P_{t-1} - E_0[P_T]) + ξ_t

    where E_0[P_T] is the forecast (target price level), and estimates
    the speed of convergence (phi) and volatility (sigma) via OLS regression.

    Args:
        prices: Price series (pandas Series with any index).
        forecast: Target/forecast price level E_0[P_T] that the process
            reverts toward.

    Returns:
        Dictionary with keys:
            - 'phi': Estimated speed of convergence.
            - 'sigma': Estimated volatility of residuals.
            - 'half_life': Estimated half-life of convergence (if φ ∈ (0,1)).

    Reference:
        AFML Ch. 13, Section 13.5.1:
            phi_hat = cov(X, Y) / cov(X, X)
            sigma_hat = sqrt(cov(xi_t, xi_t))
        where X = P_{t-1} - forecast, Y = P_t.
    """
    y = prices.values[1:]  # P_t
    x = prices.values[:-1] - forecast  # P_{t-1} - E_0[P_T]

    phi = np.cov(x, y, bias=True)[0, 1] / np.var(x)
    residuals = y - forecast - phi * x
    sigma = np.std(residuals)

    result = {"phi": phi, "sigma": sigma}
    if 0 < phi < 1:
        result["half_life"] = ou_half_life(phi)
    else:
        result["half_life"] = np.nan

    return result
