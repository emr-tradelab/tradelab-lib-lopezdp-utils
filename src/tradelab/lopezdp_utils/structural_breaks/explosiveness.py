"""Explosiveness tests beyond standard SADF.

Implements Chow-type Dickey-Fuller, Quantile ADF (QADF), and Conditional ADF
(CADF) variants that improve robustness over the standard SADF supremum.

Reference: AFML Chapter 17, Sections 17.4.1, 17.4.3, 17.4.4.
"""

import numpy as np
import pandas as pd

from tradelab.lopezdp_utils.structural_breaks.sadf import get_betas, get_y_x


def chow_type_dickey_fuller(
    log_p: pd.DataFrame,
    min_sl: int,
    constant: str = "c",
    lags: int = 1,
) -> pd.DataFrame:
    """Chow-type Dickey-Fuller test for structural break from random walk to explosive.

    Tests for a regime switch at each candidate break point τ* by including a
    dummy variable D_t[τ*] in the ADF regression. Under H0: δ=0 the series is
    a random walk throughout; under H1: δ>0 the series becomes explosive after τ*.

    The supremum DFC (SDFC) is taken over all candidate break dates.

    Unlike SADF which uses recursive backshifting over expanding windows, the
    Chow-type test uses a dummy variable approach on the full sample, making it
    a single-break-point test.

    Args:
        log_p: Log-price series as single-column DataFrame.
        min_sl: Minimum number of observations before/after break point.
        constant: Deterministic term (``'nc'``, ``'c'``, ``'ct'``, ``'ctt'``).
        lags: Number of lagged differences. Default 1.

    Returns:
        DataFrame with columns ``'tau'`` (break fraction) and ``'DFC'``
        (Dickey-Fuller Chow statistic) for each candidate break point,
        plus the SDFC as the last row attribute.

    Reference:
        AFML Section 17.4.1.
    """
    y, x = get_y_x(log_p, constant=constant, lags=lags)
    n = y.shape[0]
    results: list[dict] = []

    for tau_idx in range(min_sl, n - min_sl):
        tau_frac = tau_idx / n
        # Build dummy: 0 before break, 1 after
        dummy = np.zeros(n)
        dummy[tau_idx:] = 1.0
        # Interaction: δ * y_{t-1} * D_t
        # x[:, 0] is the lagged level
        x_chow = np.column_stack([x, x[:, 0] * dummy])
        try:
            b_mean, b_var = get_betas(y, x_chow)
        except np.linalg.LinAlgError:
            continue
        # The Chow DF stat is the t-stat of the last coefficient (δ)
        delta_idx = b_mean.shape[0] - 1
        b_std = np.sqrt(np.abs(b_var[delta_idx, delta_idx]))
        dfc = b_mean[delta_idx] / b_std if b_std > 0 else 0.0
        results.append({"tau": tau_frac, "DFC": dfc})

    if not results:
        return pd.DataFrame(columns=["tau", "DFC"])

    df = pd.DataFrame(results)
    df.attrs["SDFC"] = df["DFC"].max()
    return df


def qadf_test(
    log_p: pd.DataFrame,
    min_sl: int,
    q: float = 0.95,
    constant: str = "c",
    lags: int = 1,
) -> pd.DataFrame:
    """Quantile ADF (QADF) test.

    Instead of taking the supremum of ADF statistics (as in SADF), takes the
    q-quantile. This provides robustness against outliers — a single extreme
    ADF value cannot dominate the test result.

    Standard SADF is the special case where q=1.0 (maximum).

    Args:
        log_p: Log-price series as single-column DataFrame.
        min_sl: Minimum sample length for each ADF regression.
        q: Quantile to use instead of supremum. Default 0.95.
            Lower values increase robustness at cost of power.
        constant: Deterministic term (``'nc'``, ``'c'``, ``'ct'``, ``'ctt'``).
        lags: Number of lagged differences. Default 1.

    Returns:
        DataFrame with ``'Time'`` as index and ``'qadf'`` column containing
        the q-quantile ADF statistic at each endpoint.

    Reference:
        AFML Section 17.4.3.
    """
    results: list[dict] = []

    for end in range(min_sl, log_p.shape[0]):
        window = log_p.iloc[: end + 1]
        y, x = get_y_x(window, constant=constant, lags=lags)
        start_points = range(0, y.shape[0] + lags - min_sl + 1)
        all_adf: list[float] = []

        for start in start_points:
            y_, x_ = y[start:], x[start:]
            if x_.shape[0] < x_.shape[1] + 1:
                continue
            try:
                b_mean, b_var = get_betas(y_, x_)
            except np.linalg.LinAlgError:
                continue
            b_std = np.sqrt(np.abs(b_var[0, 0]))
            adf_stat = b_mean[0] / b_std if b_std > 0 else 0.0
            all_adf.append(adf_stat)

        if all_adf:
            qadf_val = float(np.quantile(all_adf, q))
        else:
            qadf_val = np.nan

        results.append({"Time": log_p.index[end], "qadf": qadf_val})

    return pd.DataFrame(results).set_index("Time")


def cadf_test(
    log_p: pd.DataFrame,
    min_sl: int,
    q: float = 0.95,
    constant: str = "c",
    lags: int = 1,
) -> pd.DataFrame:
    """Conditional ADF (CADF) test.

    Computes the conditional expected value of ADF statistics above the
    q-quantile threshold, analogous to Expected Shortfall (CVaR) logic but
    applied to the right tail. This provides even more robustness than QADF
    against extreme outliers.

    By definition, C_{t,q} ≤ SADF_t, and the CADF uses information from
    the entire right tail rather than a single point.

    Args:
        log_p: Log-price series as single-column DataFrame.
        min_sl: Minimum sample length for each ADF regression.
        q: Quantile threshold for conditional expectation. Default 0.95.
        constant: Deterministic term (``'nc'``, ``'c'``, ``'ct'``, ``'ctt'``).
        lags: Number of lagged differences. Default 1.

    Returns:
        DataFrame with ``'Time'`` as index and columns:
        - ``'cadf'``: Conditional mean of ADF values above q-quantile.
        - ``'cadf_var'``: Conditional variance (dispersion) of right tail.

    Reference:
        AFML Section 17.4.4.
    """
    results: list[dict] = []

    for end in range(min_sl, log_p.shape[0]):
        window = log_p.iloc[: end + 1]
        y, x = get_y_x(window, constant=constant, lags=lags)
        start_points = range(0, y.shape[0] + lags - min_sl + 1)
        all_adf: list[float] = []

        for start in start_points:
            y_, x_ = y[start:], x[start:]
            if x_.shape[0] < x_.shape[1] + 1:
                continue
            try:
                b_mean, b_var = get_betas(y_, x_)
            except np.linalg.LinAlgError:
                continue
            b_std = np.sqrt(np.abs(b_var[0, 0]))
            adf_stat = b_mean[0] / b_std if b_std > 0 else 0.0
            all_adf.append(adf_stat)

        if all_adf:
            threshold = float(np.quantile(all_adf, q))
            tail = [v for v in all_adf if v >= threshold]
            cadf_val = float(np.mean(tail)) if tail else np.nan
            cadf_var = float(np.var(tail)) if len(tail) > 1 else 0.0
        else:
            cadf_val = np.nan
            cadf_var = np.nan

        results.append(
            {
                "Time": log_p.index[end],
                "cadf": cadf_val,
                "cadf_var": cadf_var,
            }
        )

    return pd.DataFrame(results).set_index("Time")
