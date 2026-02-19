"""Structural break tests — SADF, CUSUM, Chow-DF, QADF, CADF.

Implements tests for detecting explosive behavior and structural breaks in
financial time series: SADF (bubble detection), CUSUM (parameter instability),
Chow-type DF, Quantile ADF, and Conditional ADF.

Public functions accept Polars Series and return Polars DataFrames.
Internal regression helpers use NumPy/pandas.

Reference:
    López de Prado, "Advances in Financial Machine Learning", Chapter 17.
"""

import numpy as np
import pandas as pd
import polars as pl


# ---------------------------------------------------------------------------
# Internal helpers (private)
# ---------------------------------------------------------------------------


def _lag_df(df0: pd.DataFrame, lags: int | list[int]) -> pd.DataFrame:
    """Apply lags to a DataFrame for time-series regression."""
    df1 = pd.DataFrame()
    if isinstance(lags, int):
        lags = range(lags + 1)
    else:
        lags = [int(lag) for lag in lags]
    for lag in lags:
        df_ = df0.shift(lag).copy(deep=True)
        df_.columns = [str(i) + "_" + str(lag) for i in df_.columns]
        df1 = df1.join(df_, how="outer")
    return df1


def _get_y_x(
    series: pd.DataFrame,
    constant: str,
    lags: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare numpy arrays for recursive ADF regressions.

    Args:
        series: Price series as single-column DataFrame.
        constant: Deterministic term: 'nc', 'c', 'ct', or 'ctt'.
        lags: Number of lagged differences to include.

    Returns:
        Tuple of (y, x) numpy arrays ready for OLS regression.

    Reference:
        AFML Snippet 17.2.
    """
    series_ = series.diff().dropna()
    x = _lag_df(series_, lags).dropna()
    x.iloc[:, 0] = series.values[-x.shape[0] - 1 : -1, 0]  # lagged level
    y = series_.iloc[-x.shape[0] :].values.flatten()  # 1D array
    if constant != "nc":
        x = np.append(x, np.ones((x.shape[0], 1)), axis=1)
    if constant[:2] == "ct":
        trend = np.arange(x.shape[0]).reshape(-1, 1)
        x = np.append(x, trend, axis=1)
    if constant == "ctt":
        x = np.append(x, trend**2, axis=1)
    return y, x


def _get_bsadf(
    log_p: pd.DataFrame,
    min_sl: int,
    constant: str,
    lags: int,
) -> float | None:
    """Backward-shifting SADF inner loop — returns supremum ADF statistic."""
    y, x = _get_y_x(log_p, constant=constant, lags=lags)
    start_points = range(0, y.shape[0] + lags - min_sl + 1)
    bsadf = None
    for start in start_points:
        y_, x_ = y[start:], x[start:]
        if x_.shape[0] < x_.shape[1] + 1:
            continue
        b_mean_, b_var_ = get_betas(y_, x_)
        b_std_ = b_var_**0.5
        adf_stat = b_mean_[0] / b_std_[0, 0]
        if bsadf is None or adf_stat > bsadf:
            bsadf = adf_stat
    return bsadf


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_betas(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit OLS regression via normal equations.

    Useful for custom ADF regression loops.

    Args:
        y: Dependent variable array (first differences).
        x: Regressor matrix (lagged level, lagged diffs, deterministic terms).

    Returns:
        Tuple of (b_mean, b_var) where b_mean is the coefficient vector and
        b_var is the variance-covariance matrix of coefficients.

    Reference:
        AFML Snippet 17.4.
    """
    xy = np.dot(x.T, y)
    xx = np.dot(x.T, x)
    xx_inv = np.linalg.inv(xx)
    b_mean = np.dot(xx_inv, xy)
    err = y - np.dot(x, b_mean)
    b_var = np.dot(err.T, err) / (x.shape[0] - x.shape[1]) * xx_inv
    return b_mean, b_var


def sadf_test(
    log_p: pl.Series,
    min_sl: int,
    constant: str = "c",
    lags: int = 1,
) -> pl.DataFrame:
    """Supremum Augmented Dickey-Fuller (SADF) test for bubble detection.

    Runs the backward-shifting ADF test at each endpoint in the series
    (from min_sl onward), collecting the supremum ADF statistic at each point.
    Used to detect bubble-like explosive behavior that standard unit-root tests
    miss.

    Args:
        log_p: Log-price series as Polars Series.
        min_sl: Minimum sample length for each ADF regression window.
        constant: Deterministic term ('nc', 'c', 'ct', 'ctt'). Default 'c'.
        lags: Number of lagged differences. Default 1.

    Returns:
        Polars DataFrame with column 'sadf' (supremum ADF statistic at each
        endpoint).

    Reference:
        AFML Section 17.4.2.
    """
    series_pd = pd.DataFrame({"log_p": log_p.to_numpy()})
    sadf_values: list[float | None] = []
    for end in range(min_sl, series_pd.shape[0]):
        window = series_pd.iloc[: end + 1]
        val = _get_bsadf(window, min_sl=min_sl, constant=constant, lags=lags)
        sadf_values.append(val)
    return pl.DataFrame({"sadf": sadf_values})


def brown_durbin_evans_cusum(
    series: pl.Series,
    lags: int = 1,
) -> pl.DataFrame:
    """Brown-Durbin-Evans CUSUM test on recursive residuals.

    Detects structural breaks by monitoring the cumulative sum of standardized
    one-step-ahead recursive residuals from an expanding-window OLS regression.

    Args:
        series: Price series as Polars Series.
        lags: Number of lagged differences to include. Default 1.

    Returns:
        Polars DataFrame with columns 's_t' (cumulative sum), 'upper', and
        'lower' (critical boundaries at 5% significance).

    Reference:
        AFML Section 17.3.1.
    """
    values = series.to_numpy().flatten()
    diff = np.diff(values)
    k = lags + 1  # number of regressors

    y_full = diff[lags:]
    x_list = []
    x_list.append(values[lags:-1].flatten())
    for lag in range(1, lags + 1):
        x_list.append(diff[lags - lag : -lag if lag > 0 else None].flatten())
    x_list.append(np.ones(len(y_full)))
    x_full = np.column_stack(x_list)

    n = len(y_full)
    min_obs = k + 1

    recursive_residuals = []
    for t in range(min_obs, n):
        x_t = x_full[:t]
        y_t = y_full[:t]
        try:
            beta = np.linalg.lstsq(x_t, y_t, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue
        x_next = x_full[t]
        forecast = np.dot(x_next, beta)
        residual = y_full[t] - forecast
        try:
            xx_inv = np.linalg.inv(np.dot(x_t.T, x_t))
        except np.linalg.LinAlgError:
            continue
        f_sq = 1.0 + np.dot(x_next, np.dot(xx_inv, x_next))
        resid_var = np.sum((y_t - np.dot(x_t, beta)) ** 2) / max(t - k, 1)
        std_residual = residual / np.sqrt(resid_var * f_sq)
        recursive_residuals.append(std_residual)

    recursive_residuals_arr = np.array(recursive_residuals)
    if len(recursive_residuals_arr) == 0:
        return pl.DataFrame({"s_t": [], "upper": [], "lower": []})

    sigma_w = (
        np.std(recursive_residuals_arr, ddof=1) if len(recursive_residuals_arr) > 1 else 1.0
    )
    standardized = recursive_residuals_arr / sigma_w if sigma_w > 0 else recursive_residuals_arr

    s_t = np.cumsum(standardized)
    a = 0.948
    t_minus_k = np.arange(1, len(s_t) + 1)
    total = len(s_t)
    upper = a * np.sqrt(total) + 2 * a * t_minus_k / total
    lower = -upper

    return pl.DataFrame({"s_t": s_t, "upper": upper, "lower": lower})


def chu_stinchcombe_white_cusum(
    series: pl.Series,
    critical_value: float = 4.6,
) -> pl.DataFrame:
    """Chu-Stinchcombe-White CUSUM test on levels.

    Simplified CUSUM test assuming a martingale null hypothesis.
    Works directly with price levels rather than recursive residuals.

    Args:
        series: Price series as Polars Series.
        critical_value: One-sided critical value boundary parameter b_alpha.
            Default 4.6 (5% significance).

    Returns:
        Polars DataFrame with columns 's_t' (supremum of standardized
        departures) and 'critical' (critical value boundary).

    Reference:
        AFML Section 17.3.2.
    """
    values = series.to_numpy().flatten()
    n = len(values)
    diffs = np.diff(values)

    s_t_list = []
    crit_list = []

    for t in range(2, n):
        sigma_sq = np.sum(diffs[:t] ** 2) / t
        if sigma_sq <= 0:
            s_t_list.append(0.0)
            crit_list.append(critical_value)
            continue
        sup_s = -np.inf
        for start in range(t):
            span = t - start
            if span < 1:
                continue
            s_nt = (values[t] - values[start]) / np.sqrt(sigma_sq * span)
            if s_nt > sup_s:
                sup_s = s_nt
        s_t_list.append(sup_s)
        crit_list.append(critical_value + np.log(t))

    return pl.DataFrame({"s_t": s_t_list, "critical": crit_list})


def chow_type_dickey_fuller(
    log_p: pl.Series,
    min_sl: int,
    constant: str = "c",
    lags: int = 1,
) -> pl.DataFrame:
    """Chow-type Dickey-Fuller test for structural break from random walk to explosive.

    Tests for a regime switch at each candidate break point τ* using a dummy
    variable in the ADF regression.

    Args:
        log_p: Log-price series as Polars Series.
        min_sl: Minimum number of observations before/after break point.
        constant: Deterministic term ('nc', 'c', 'ct', 'ctt').
        lags: Number of lagged differences. Default 1.

    Returns:
        Polars DataFrame with columns 'tau' (break fraction) and 'dfc'
        (Dickey-Fuller Chow statistic) for each candidate break point.

    Reference:
        AFML Section 17.4.1.
    """
    series_pd = pd.DataFrame({"log_p": log_p.to_numpy()})
    y, x = _get_y_x(series_pd, constant=constant, lags=lags)
    n = y.shape[0]
    results: list[dict] = []

    for tau_idx in range(min_sl, n - min_sl):
        tau_frac = tau_idx / n
        dummy = np.zeros(n)
        dummy[tau_idx:] = 1.0
        x_chow = np.column_stack([x, x[:, 0] * dummy])
        try:
            b_mean, b_var = get_betas(y, x_chow)
        except np.linalg.LinAlgError:
            continue
        delta_idx = b_mean.shape[0] - 1
        b_std = np.sqrt(np.abs(b_var[delta_idx, delta_idx]))
        dfc = b_mean[delta_idx] / b_std if b_std > 0 else 0.0
        results.append({"tau": tau_frac, "dfc": dfc})

    if not results:
        return pl.DataFrame({"tau": [], "dfc": []})
    return pl.DataFrame(results)


def qadf_test(
    log_p: pl.Series,
    min_sl: int,
    q: float = 0.95,
    constant: str = "c",
    lags: int = 1,
) -> pl.DataFrame:
    """Quantile ADF (QADF) test.

    Instead of taking the supremum of ADF statistics (as in SADF), takes the
    q-quantile, providing robustness against outliers.

    Args:
        log_p: Log-price series as Polars Series.
        min_sl: Minimum sample length for each ADF regression.
        q: Quantile to use instead of supremum. Default 0.95.
        constant: Deterministic term ('nc', 'c', 'ct', 'ctt').
        lags: Number of lagged differences. Default 1.

    Returns:
        Polars DataFrame with column 'qadf' (q-quantile ADF statistic at each
        endpoint).

    Reference:
        AFML Section 17.4.3.
    """
    series_pd = pd.DataFrame({"log_p": log_p.to_numpy()})
    qadf_values: list[float] = []

    for end in range(min_sl, series_pd.shape[0]):
        window = series_pd.iloc[: end + 1]
        y, x = _get_y_x(window, constant=constant, lags=lags)
        all_adf: list[float] = []
        for start in range(0, y.shape[0] + lags - min_sl + 1):
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
        qadf_val = float(np.quantile(all_adf, q)) if all_adf else float("nan")
        qadf_values.append(qadf_val)

    return pl.DataFrame({"qadf": qadf_values})


def cadf_test(
    log_p: pl.Series,
    min_sl: int,
    q: float = 0.95,
    constant: str = "c",
    lags: int = 1,
) -> pl.DataFrame:
    """Conditional ADF (CADF) test.

    Computes the conditional expected value of ADF statistics above the
    q-quantile threshold, analogous to Expected Shortfall logic.

    Args:
        log_p: Log-price series as Polars Series.
        min_sl: Minimum sample length for each ADF regression.
        q: Quantile threshold for conditional expectation. Default 0.95.
        constant: Deterministic term ('nc', 'c', 'ct', 'ctt').
        lags: Number of lagged differences. Default 1.

    Returns:
        Polars DataFrame with columns 'cadf' (conditional mean above
        q-quantile) and 'cadf_var' (conditional variance of right tail).

    Reference:
        AFML Section 17.4.4.
    """
    series_pd = pd.DataFrame({"log_p": log_p.to_numpy()})
    rows: list[dict] = []

    for end in range(min_sl, series_pd.shape[0]):
        window = series_pd.iloc[: end + 1]
        y, x = _get_y_x(window, constant=constant, lags=lags)
        all_adf: list[float] = []
        for start in range(0, y.shape[0] + lags - min_sl + 1):
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
            cadf_val = float(np.mean(tail)) if tail else float("nan")
            cadf_var = float(np.var(tail)) if len(tail) > 1 else 0.0
        else:
            cadf_val = float("nan")
            cadf_var = float("nan")
        rows.append({"cadf": cadf_val, "cadf_var": cadf_var})

    return pl.DataFrame(rows)
