"""SADF (Supremum Augmented Dickey-Fuller) test and supporting utilities.

Implements the backward-shifting SADF test for detecting explosive behavior
(e.g., bubbles) in financial time series, along with the helper functions
for preparing data and fitting ADF regressions.

Reference: AFML Chapter 17, Snippets 17.1-17.4.
"""

import numpy as np
import pandas as pd


def lag_df(df0: pd.DataFrame, lags: int | list[int]) -> pd.DataFrame:
    """Apply lags to a DataFrame for time-series regression.

    Creates a new DataFrame with the original columns and their lagged versions.
    Each lagged column is named ``{original_col}_{lag}``.

    Args:
        df0: Input DataFrame with time-series data.
        lags: If int, creates lags 0 through ``lags`` (inclusive).
            If list, creates the specified lag values.

    Returns:
        DataFrame with original and lagged columns joined via outer join.

    Reference:
        AFML Snippet 17.3.
    """
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


def get_y_x(
    series: pd.DataFrame,
    constant: str,
    lags: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare numpy arrays for recursive ADF regressions.

    Transforms a price series into the dependent variable (first differences)
    and regressors (lagged level, lagged differences, optional constant/trend)
    needed for the ADF regression specification.

    Args:
        series: Price series as single-column DataFrame.
        constant: Deterministic term specification:
            - ``'nc'``: No constant.
            - ``'c'``: Constant only.
            - ``'ct'``: Constant + linear trend.
            - ``'ctt'``: Constant + linear + quadratic trend.
        lags: Number of lagged differences to include.

    Returns:
        Tuple of (y, x) numpy arrays ready for OLS regression.

    Reference:
        AFML Snippet 17.2.
    """
    series_ = series.diff().dropna()
    x = lag_df(series_, lags).dropna()
    x.iloc[:, 0] = series.values[-x.shape[0] - 1 : -1, 0]  # lagged level
    y = series_.iloc[-x.shape[0] :].values
    if constant != "nc":
        x = np.append(x, np.ones((x.shape[0], 1)), axis=1)
    if constant[:2] == "ct":
        trend = np.arange(x.shape[0]).reshape(-1, 1)
        x = np.append(x, trend, axis=1)
    if constant == "ctt":
        x = np.append(x, trend**2, axis=1)
    return y, x


def get_betas(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit ADF regression specification via OLS.

    Solves the normal equations to obtain coefficient estimates and their
    variance-covariance matrix. Returns the coefficient means and variances
    needed to compute the ADF t-statistic.

    Args:
        y: Dependent variable array (first differences).
        x: Regressor matrix (lagged level, lagged diffs, deterministic terms).

    Returns:
        Tuple of (bMean, bVar) where bMean is the coefficient vector and
        bVar is the variance-covariance matrix of coefficients.

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


def get_bsadf(
    log_p: pd.DataFrame,
    min_sl: int,
    constant: str,
    lags: int,
) -> dict:
    """Backward-shifting SADF inner loop.

    For a given endpoint, fits ADF regressions over all possible start points
    (expanding windows backward) and returns the supremum ADF t-statistic.
    This is the core building block of the SADF test.

    Args:
        log_p: Log-price series as single-column DataFrame.
        min_sl: Minimum sample length for each ADF regression.
        constant: Deterministic term (``'nc'``, ``'c'``, ``'ct'``, ``'ctt'``).
        lags: Number of lagged differences in ADF specification.

    Returns:
        Dict with ``'Time'`` (index of endpoint) and ``'gsadf'``
        (supremum ADF statistic across all start points).

    Reference:
        AFML Snippet 17.1.
    """
    y, x = get_y_x(log_p, constant=constant, lags=lags)
    start_points = range(0, y.shape[0] + lags - min_sl + 1)
    bsadf = None
    all_adf: list[float] = []
    for start in start_points:
        y_, x_ = y[start:], x[start:]
        if x_.shape[0] < x_.shape[1] + 1:
            continue
        b_mean_, b_var_ = get_betas(y_, x_)
        b_std_ = b_var_**0.5
        adf_stat = b_mean_[0] / b_std_[0, 0]
        all_adf.append(adf_stat)
        if bsadf is None or adf_stat > bsadf:
            bsadf = adf_stat
    out = {"Time": log_p.index[-1], "gsadf": bsadf}
    return out


def sadf_test(
    log_p: pd.DataFrame,
    min_sl: int,
    constant: str = "c",
    lags: int = 1,
) -> pd.DataFrame:
    """Full Supremum Augmented Dickey-Fuller (SADF) test.

    Runs the backward-shifting ADF test at each endpoint in the series
    (from ``min_sl`` onward), collecting the supremum ADF statistic at each
    point. Used to detect bubble-like explosive behavior that standard
    unit-root tests miss.

    The SADF test addresses the limitation that standard ADF tests have low
    power against explosive alternatives. By computing the supremum of ADF
    statistics over expanding windows, it can detect transitions from
    random walk to explosive regimes.

    Args:
        log_p: Log-price series as single-column DataFrame.
        min_sl: Minimum sample length for each ADF regression window.
        constant: Deterministic term (``'nc'``, ``'c'``, ``'ct'``, ``'ctt'``).
            Default ``'c'``.
        lags: Number of lagged differences. Default 1.

    Returns:
        DataFrame with columns ``'Time'`` and ``'gsadf'`` containing
        the supremum ADF statistic at each endpoint.

    Reference:
        AFML Section 17.4.2.
    """
    results: list[dict] = []
    for end in range(min_sl, log_p.shape[0]):
        window = log_p.iloc[: end + 1]
        result = get_bsadf(window, min_sl=min_sl, constant=constant, lags=lags)
        results.append(result)
    return pd.DataFrame(results).set_index("Time")
