"""CUSUM tests for structural break detection.

Implements two CUSUM-based tests:
1. Brown-Durbin-Evans CUSUM on recursive residuals (Section 17.3.1)
2. Chu-Stinchcombe-White CUSUM on levels (Section 17.3.2)

Reference: AFML Chapter 17, Sections 17.3.1-17.3.2.
"""

import numpy as np
import pandas as pd


def brown_durbin_evans_cusum(
    series: pd.DataFrame,
    lags: int = 1,
) -> pd.DataFrame:
    """Brown-Durbin-Evans CUSUM test on recursive residuals.

    Detects structural breaks by monitoring the cumulative sum of standardized
    one-step-ahead recursive residuals from an expanding-window OLS regression.
    Under the null hypothesis of constant coefficients, the CUSUM statistic
    follows a standard Brownian bridge.

    The test fits y_t = β'x_t + ε_t on expanding subsamples [1, k+1], ..., [1, T],
    computes standardized forecast errors (recursive residuals), and accumulates
    them. Departures from zero indicate parameter instability.

    Args:
        series: Price series as single-column DataFrame.
        lags: Number of lagged differences to include in the regression. Default 1.

    Returns:
        DataFrame with columns:
        - ``'S_t'``: Cumulative sum of standardized recursive residuals.
        - ``'upper'``: Upper critical boundary (5% significance).
        - ``'lower'``: Lower critical boundary (5% significance).

    Reference:
        AFML Section 17.3.1.
    """
    # Prepare differences and lagged values
    diff = series.diff().dropna()
    k = lags + 1  # number of regressors (lagged level + lags + constant)

    # Build full regressor matrix: lagged level, lagged diffs, constant
    y_full = diff.values[lags:].flatten()
    x_list = []
    # Lagged level
    x_list.append(series.values[lags:-1].flatten())
    # Lagged differences
    for lag in range(1, lags + 1):
        x_list.append(diff.values[lags - lag : -lag].flatten())
    # Constant
    x_list.append(np.ones(len(y_full)))
    x_full = np.column_stack(x_list)

    n = len(y_full)
    min_obs = k + 1  # minimum observations to start recursion

    # Compute recursive residuals
    recursive_residuals = []
    for t in range(min_obs, n):
        x_t = x_full[:t]
        y_t = y_full[:t]
        # OLS on [0, t)
        try:
            beta = np.linalg.lstsq(x_t, y_t, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue
        # One-step-ahead forecast error
        x_next = x_full[t]
        forecast = np.dot(x_next, beta)
        residual = y_full[t] - forecast
        # Standardization factor: sqrt(1 + x'(X'X)^{-1}x) * sigma
        try:
            xx_inv = np.linalg.inv(np.dot(x_t.T, x_t))
        except np.linalg.LinAlgError:
            continue
        f_sq = 1.0 + np.dot(x_next, np.dot(xx_inv, x_next))
        resid_var = np.sum((y_t - np.dot(x_t, beta)) ** 2) / max(t - k, 1)
        std_residual = residual / np.sqrt(resid_var * f_sq)
        recursive_residuals.append(std_residual)

    recursive_residuals = np.array(recursive_residuals)
    if len(recursive_residuals) == 0:
        return pd.DataFrame(columns=["S_t", "upper", "lower"])

    # Standardize by overall std of recursive residuals
    sigma_w = np.std(recursive_residuals, ddof=1) if len(recursive_residuals) > 1 else 1.0
    standardized = recursive_residuals / sigma_w if sigma_w > 0 else recursive_residuals

    # Cumulative sum
    s_t = np.cumsum(standardized)

    # Critical boundaries (5% significance): +/-a*sqrt(T-k) + 2a(t-k)/(T-k)
    # where a ~ 0.948 for 5% level
    a = 0.948
    t_minus_k = np.arange(1, len(s_t) + 1)
    total = len(s_t)
    upper = a * np.sqrt(total) + 2 * a * t_minus_k / total
    lower = -upper

    idx = series.index[-(len(s_t)) :]
    return pd.DataFrame(
        {"S_t": s_t, "upper": upper, "lower": lower},
        index=idx,
    )


def chu_stinchcombe_white_cusum(
    series: pd.DataFrame,
    critical_value: float = 4.6,
) -> pd.DataFrame:
    """Chu-Stinchcombe-White CUSUM test on levels.

    Simplified CUSUM test that assumes a martingale null hypothesis (no expected
    change). Works directly with price levels rather than recursive residuals,
    reducing computational burden. Detects departures from random walk behavior.

    The test statistic at time t measures the standardized departure of the
    current price from each past reference point n, taking the supremum over
    all reference points.

    Args:
        series: Price series as single-column DataFrame.
        critical_value: One-sided critical value boundary parameter b_alpha.
            Default 4.6 (corresponds to 5% significance, derived via Monte Carlo).

    Returns:
        DataFrame with columns:
        - ``'S_t'``: Supremum of standardized departures over all start points.
        - ``'critical'``: Critical value boundary c_alpha[n,t] = b_alpha + log(t-n).

    Reference:
        AFML Section 17.3.2.
    """
    values = series.values.flatten()
    n = len(values)

    # Estimate variance using first differences: σ²_t = (t-1)^{-1} Σ (Δy_i)²
    diffs = np.diff(values)

    s_t_list = []
    crit_list = []

    for t in range(2, n):
        # Running variance estimate
        sigma_sq = np.sum(diffs[:t] ** 2) / t

        if sigma_sq <= 0:
            s_t_list.append(0.0)
            crit_list.append(critical_value)
            continue

        # Compute S_{n,t} for all start points n < t
        sup_s = -np.inf
        for start in range(t):
            span = t - start
            if span < 1:
                continue
            s_nt = (values[t] - values[start]) / np.sqrt(sigma_sq * span)
            if s_nt > sup_s:
                sup_s = s_nt

        s_t_list.append(sup_s)
        # Critical boundary: b_alpha + log(t) (simplified)
        crit_list.append(critical_value + np.log(t))

    idx = series.index[2:]
    return pd.DataFrame(
        {"S_t": s_t_list, "critical": crit_list},
        index=idx,
    )
