"""Price impact models from AFML Chapter 19.

Implements second-generation microstructural features that measure
how trading activity affects prices (Kyle's Lambda, Amihud's Lambda,
Hasbrouck's Lambda).

Reference:
    AFML Chapter 19, Section 19.3.2
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def kyle_lambda(
    prices: pd.Series,
    volume: pd.Series,
    signs: pd.Series,
) -> dict[str, float]:
    """Estimate Kyle's Lambda (price impact coefficient).

    Measures how much market makers adjust prices per unit of order flow.
    Estimated via OLS regression of price changes on signed volume.

    Regression: Δp_t = λ * (b_t * V_t) + ε_t

    Args:
        prices: Transaction prices.
        volume: Traded volume per observation.
        signs: Trade signs (+1/-1) from tick_rule or similar classifier.

    Returns:
        Dictionary with:
            - 'lambda': Price impact coefficient.
            - 't_stat': t-statistic of lambda estimate.

    Reference:
        AFML Section 19.3.2, Kyle (1985)
    """
    dp = prices.diff().dropna()
    signed_vol = (signs * volume).reindex(dp.index).dropna()
    common = dp.index.intersection(signed_vol.index)
    dp = dp.loc[common]
    signed_vol = signed_vol.loc[common]

    x = signed_vol.values.reshape(-1, 1)
    y = dp.values
    reg = LinearRegression(fit_intercept=True).fit(x, y)

    y_hat = reg.predict(x)
    residuals = y - y_hat
    n = len(y)
    se = np.sqrt(np.sum(residuals**2) / (n - 2) / np.sum((x - x.mean()) ** 2))
    t_stat = reg.coef_[0] / se if se > 0 else np.nan

    return {"lambda": float(reg.coef_[0]), "t_stat": float(t_stat)}


def amihud_lambda(
    prices: pd.Series,
    dollar_volume: pd.Series,
) -> float:
    """Estimate Amihud's Lambda (illiquidity proxy).

    Measures average price response per dollar of trading volume. Simpler
    and more widely used than Kyle's Lambda, requiring only bar-level
    returns and dollar volume.

    Formula: λ = (1/T) * Σ |Δlog(p_t)| / DollarVolume_t

    Args:
        prices: Bar closing prices.
        dollar_volume: Dollar volume per bar (price * volume).

    Returns:
        Amihud illiquidity ratio (scalar).

    Reference:
        AFML Section 19.3.2, Amihud (2002)
    """
    log_returns = np.log(prices / prices.shift(1)).dropna().abs()
    dv = dollar_volume.reindex(log_returns.index)
    # Exclude bars with zero dollar volume
    mask = dv > 0
    ratio = (log_returns[mask] / dv[mask]).mean()
    return float(ratio)


def hasbrouck_lambda(
    prices: pd.Series,
    dollar_volume: pd.Series,
    signs: pd.Series,
) -> dict[str, float]:
    """Estimate Hasbrouck's Lambda (effective trading cost).

    Similar to Kyle's Lambda but uses signed root-dollar volume to account
    for the concavity of the price impact function. Estimated via OLS
    regression of log returns on signed root-dollar volume.

    Regression: Δlog(p_t) = λ * b_t * √(p_t * V_t) + ε_t

    Args:
        prices: Bar closing prices.
        dollar_volume: Dollar volume per bar (price * volume).
        signs: Aggregated trade signs per bar (+1/-1).

    Returns:
        Dictionary with:
            - 'lambda': Price impact coefficient.
            - 't_stat': t-statistic of lambda estimate.

    Reference:
        AFML Section 19.3.2, Hasbrouck (2009)
    """
    log_ret = np.log(prices / prices.shift(1)).dropna()
    signed_root_dv = (signs * np.sqrt(dollar_volume)).reindex(log_ret.index).dropna()
    common = log_ret.index.intersection(signed_root_dv.index)
    log_ret = log_ret.loc[common]
    signed_root_dv = signed_root_dv.loc[common]

    x = signed_root_dv.values.reshape(-1, 1)
    y = log_ret.values
    reg = LinearRegression(fit_intercept=True).fit(x, y)

    y_hat = reg.predict(x)
    residuals = y - y_hat
    n = len(y)
    se = np.sqrt(np.sum(residuals**2) / (n - 2) / np.sum((x - x.mean()) ** 2))
    t_stat = reg.coef_[0] / se if se > 0 else np.nan

    return {"lambda": float(reg.coef_[0]), "t_stat": float(t_stat)}
