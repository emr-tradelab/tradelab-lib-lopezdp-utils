"""Backtest statistics — Sharpe, PSR, DSR, drawdown, HHI, bet timing, strategy metrics.

Covers AFML Chapter 14 (SR variants, drawdown, concentration, bet timing)
and MLAM Section 8 (strategy precision/recall).

References:
    López de Prado, "Advances in Financial Machine Learning", Chapter 14
    López de Prado, "Machine Learning for Asset Managers", Section 8
"""

import numpy as np
import polars as pl
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Sharpe ratio variants (AFML Sections 14.5-14.7)
# ---------------------------------------------------------------------------


def sharpe_ratio(returns: pl.Series, periods_per_year: float = 252.0) -> float:
    """Compute annualized Sharpe ratio from excess returns.

    Args:
        returns: Polars Series of excess returns.
        periods_per_year: Annualization factor (252 for daily, 12 for monthly).

    Returns:
        Annualized Sharpe ratio.
    """
    mu = returns.mean()
    sigma = returns.std()
    if sigma is None or sigma == 0 or mu is None:
        return 0.0
    return float((mu / sigma) * np.sqrt(periods_per_year))


def sharpe_ratio_non_annualized(returns: pl.Series) -> float:
    """Compute non-annualized Sharpe ratio (mean / std).

    Use this when feeding SR into probabilistic_sharpe_ratio or
    deflated_sharpe_ratio, which expect non-annualized values.

    Args:
        returns: Polars Series of excess returns.

    Returns:
        Non-annualized Sharpe ratio.
    """
    mu = returns.mean()
    sigma = returns.std()
    if sigma is None or sigma == 0 or mu is None:
        return 0.0
    return float(mu / sigma)


def probabilistic_sharpe_ratio(
    observed_sr: float,
    benchmark_sr: float = 0.0,
    n_obs: int = 252,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Compute Probabilistic Sharpe Ratio (PSR).

    Args:
        observed_sr: Observed **non-annualized** Sharpe ratio (mean/std).
            Use sharpe_ratio_non_annualized() to compute this.
        benchmark_sr: Benchmark SR to test against (often 0).
        n_obs: Number of return observations.
        skew: Skewness of returns (0 for Gaussian).
        kurtosis: Raw kurtosis of returns (3 for Gaussian, NOT excess kurtosis).

    Returns:
        PSR as a probability in [0, 1].

    Reference:
        AFML Section 14.6
    """
    z = (observed_sr - benchmark_sr) * np.sqrt(n_obs - 1)
    z /= np.sqrt(1 - skew * observed_sr + (kurtosis - 1) / 4.0 * observed_sr**2)
    return float(norm.cdf(z))


def deflated_sharpe_ratio(
    observed_sr: float,
    sr_estimates: list[float],
    n_obs: int = 252,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Compute Deflated Sharpe Ratio (DSR).

    Corrects the observed SR for selection bias under multiple testing.
    Uses the expected maximum SR across N trials as the benchmark for PSR.

    Args:
        observed_sr: Observed **non-annualized** Sharpe ratio of best strategy.
            Use sharpe_ratio_non_annualized() to compute this.
        sr_estimates: List of **non-annualized** Sharpe ratios from all trials.
            No default — caller must provide explicit trial accounting.
        n_obs: Number of return observations per trial.
        skew: Skewness of returns (0 for Gaussian).
        kurtosis: Raw kurtosis of returns (3 for Gaussian, NOT excess kurtosis).

    Returns:
        DSR as a probability in [0, 1].

    Reference:
        AFML Section 14.7
    """
    n_trials = len(sr_estimates)
    sr_std = float(np.std(sr_estimates, ddof=1))

    euler_mascheroni = 0.5772156649015329
    sr_max = sr_std * (
        (1 - euler_mascheroni) * norm.ppf(1 - 1.0 / n_trials)
        + euler_mascheroni * norm.ppf(1 - 1.0 / (n_trials * np.e))
    )

    return probabilistic_sharpe_ratio(
        observed_sr=observed_sr,
        benchmark_sr=sr_max,
        n_obs=n_obs,
        skew=skew,
        kurtosis=kurtosis,
    )


# ---------------------------------------------------------------------------
# Drawdown and time-under-water (AFML Snippet 14.4)
# ---------------------------------------------------------------------------


def compute_dd_tuw(
    series: pl.DataFrame, dollars: bool = False
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Compute drawdown series and time-under-water per drawdown episode.

    Drawdown: running loss from the high-water mark at each timestamp.
    Time-under-water: duration from each peak until recovery (next new HWM).

    Args:
        series: DataFrame with 'timestamp' and 'pnl' columns.
        dollars: If True, DD in dollar terms. If False, in percentage terms.

    Returns:
        Tuple of (dd, tuw) DataFrames.
        - dd: 'timestamp', 'drawdown' — running drawdown at each point (non-negative).
        - tuw: 'peak_timestamp', 'recovery_timestamp', 'tuw_days' — one row per
          drawdown episode (peak to next new HWM).
    """
    timestamps = series["timestamp"].to_list()
    pnl = series["pnl"].to_numpy().astype(float)

    hwm = np.maximum.accumulate(pnl)

    # Running drawdown at each timestamp
    if dollars:
        dd_vals = hwm - pnl
    else:
        dd_vals = np.where(hwm > 0, 1.0 - pnl / hwm, 0.0)

    dd_pl = pl.DataFrame({"timestamp": timestamps, "drawdown": dd_vals.tolist()})

    # Time-under-water: identify episodes (peak → recovery)
    tuw_peaks = []
    tuw_recoveries = []
    tuw_days = []

    peak_idx = 0
    i = 1
    while i < len(hwm):
        if pnl[i] < hwm[i - 1]:
            # Drawdown started at peak_idx (last HWM point)
            peak_idx = i - 1
            # Find recovery: next point where pnl >= hwm at peak
            j = i
            while j < len(pnl) and pnl[j] < hwm[peak_idx]:
                j += 1
            recovery_idx = j if j < len(pnl) else len(pnl) - 1
            ts_peak = timestamps[peak_idx]
            ts_recovery = timestamps[recovery_idx]
            dt = (ts_recovery - ts_peak) / np.timedelta64(1, "D")
            tuw_peaks.append(ts_peak)
            tuw_recoveries.append(ts_recovery)
            tuw_days.append(float(dt))
            i = recovery_idx + 1
        else:
            i += 1

    if tuw_peaks:
        tuw_pl = pl.DataFrame(
            {
                "peak_timestamp": tuw_peaks,
                "recovery_timestamp": tuw_recoveries,
                "tuw_days": tuw_days,
            }
        )
    else:
        tuw_pl = pl.DataFrame(
            schema={
                "peak_timestamp": pl.Datetime,
                "recovery_timestamp": pl.Datetime,
                "tuw_days": pl.Float64,
            }
        )

    return dd_pl, tuw_pl


# ---------------------------------------------------------------------------
# Concentration — HHI (AFML Snippet 14.3)
# ---------------------------------------------------------------------------


def get_hhi(bet_ret: pl.Series) -> float:
    """Compute normalized Herfindahl-Hirschman Index for return concentration.

    Args:
        bet_ret: Series of returns from bets.

    Returns:
        Normalized HHI between 0 (diversified) and 1 (concentrated).
    """
    n = len(bet_ret)
    if n <= 2:
        return float("nan")

    total = bet_ret.abs().sum()
    if total == 0:
        return float("nan")

    wght = bet_ret.abs() / total
    hhi = (wght**2).sum()
    hhi = (hhi - 1.0 / n) / (1.0 - 1.0 / n)
    return float(hhi)


def get_hhi_decomposed(bet_ret: pl.Series) -> dict[str, float]:
    """Compute HHI separately for positive returns, negative returns, and overall.

    López de Prado recommends decomposing concentration to check if PnL
    depends on a few lucky wins or a few large losses.

    Args:
        bet_ret: Series of returns from bets.

    Returns:
        Dictionary with 'hhi_positive', 'hhi_negative', 'hhi_total' keys.
    """
    pos = bet_ret.filter(bet_ret > 0)
    neg = bet_ret.filter(bet_ret < 0)
    return {
        "hhi_positive": get_hhi(pos),
        "hhi_negative": get_hhi(neg),
        "hhi_total": get_hhi(bet_ret),
    }


# ---------------------------------------------------------------------------
# Bet timing and holding period (AFML Snippets 14.1, 14.2)
# ---------------------------------------------------------------------------


def get_bet_timing(t_pos: pl.DataFrame) -> pl.DataFrame:
    """Derive timestamps of independent bets from a position series.

    Args:
        t_pos: DataFrame with 'timestamp' and 'position' columns.

    Returns:
        DataFrame with 'timestamp' column of bet boundary timestamps.
    """
    timestamps = t_pos["timestamp"].to_list()
    pos = t_pos["position"].to_numpy()

    bet_times = []

    for i in range(1, len(pos)):
        if pos[i] == 0 and pos[i - 1] != 0:
            bet_times.append(timestamps[i])
        elif pos[i] * pos[i - 1] < 0:
            bet_times.append(timestamps[i])

    # Always include last timestamp
    if not bet_times or bet_times[-1] != timestamps[-1]:
        bet_times.append(timestamps[-1])

    return pl.DataFrame({"timestamp": sorted(set(bet_times))})


def get_holding_period(t_pos: pl.DataFrame) -> float:
    """Estimate average holding period from a position series.

    Args:
        t_pos: DataFrame with 'timestamp' and 'position' columns.

    Returns:
        Average holding period in days.
    """
    ts_series = t_pos["timestamp"]
    pos = t_pos["position"].to_numpy().astype(float)

    # Compute day offsets from first timestamp
    ms_per_day = 1000 * 60 * 60 * 24
    t_diff_series = (ts_series - ts_series[0]).cast(
        pl.Duration("ms")
    ).dt.total_milliseconds() / ms_per_day
    t_diff = t_diff_series.to_numpy()
    t_entry = 0.0

    hp_dt = []
    hp_w = []

    for i in range(1, len(pos)):
        p_d = pos[i] - pos[i - 1]
        if p_d * pos[i - 1] >= 0:  # Increased or unchanged
            if pos[i] != 0:
                t_entry = (t_entry * pos[i - 1] + t_diff[i] * p_d) / pos[i]
        else:  # Decreased
            if pos[i] * pos[i - 1] < 0:  # Flip
                hp_dt.append(t_diff[i] - t_entry)
                hp_w.append(abs(pos[i - 1]))
                t_entry = t_diff[i]
            else:  # Scaled out
                hp_dt.append(t_diff[i] - t_entry)
                hp_w.append(abs(p_d))

    total_w = sum(hp_w)
    if total_w > 0:
        return float(sum(d * w for d, w in zip(hp_dt, hp_w)) / total_w)
    return float("nan")


# ---------------------------------------------------------------------------
# Strategy precision / recall (MLAM Section 8)
# ---------------------------------------------------------------------------


def strategy_precision(alpha: float, beta: float, theta: float) -> float:
    """Compute precision of strategy discovery process.

    Args:
        alpha: Significance level (Type I error probability).
        beta: Type II error probability.
        theta: Universe odds ratio s_T / s_F.

    Returns:
        Precision in [0, 1].
    """
    return ((1 - beta) * theta) / ((1 - beta) * theta + alpha)


def strategy_recall(beta: float) -> float:
    """Compute recall of strategy discovery process.

    Args:
        beta: Type II error probability.

    Returns:
        Recall (1 - beta) in [0, 1].
    """
    return 1 - beta


def multi_test_precision_recall(
    alpha: float, beta: float, theta: float, k: int
) -> tuple[float, float]:
    """Extend precision/recall to K independent trials with Sidak correction.

    Args:
        alpha: Per-trial significance level.
        beta: Per-trial Type II error probability.
        theta: Universe odds ratio s_T / s_F.
        k: Number of independent trials.

    Returns:
        Tuple of (precision_K, recall_K).
    """
    alpha_k = 1 - (1 - alpha) ** k
    beta_k = beta**k
    precision_k = ((1 - beta_k) * theta) / ((1 - beta_k) * theta + alpha_k)
    recall_k = 1 - beta_k
    return precision_k, recall_k
