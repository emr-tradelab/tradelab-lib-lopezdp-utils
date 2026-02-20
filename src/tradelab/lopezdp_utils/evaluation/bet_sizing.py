"""Signal generation and dynamic position sizing — AFML Chapter 10.

References:
    López de Prado, "Advances in Financial Machine Learning", Chapter 10
"""

import numpy as np
import polars as pl
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Signal pipeline (AFML Snippets 10.1-10.3)
# ---------------------------------------------------------------------------


def get_signal(
    events: pl.DataFrame,
    step_size: float,
    prob: pl.Series,
    num_classes: int,
    pred: pl.Series | None = None,
) -> pl.DataFrame:
    """Translate predicted probabilities into discretized bet sizes.

    Args:
        events: DataFrame with 'timestamp', 't1' columns. Optionally 'side'.
        step_size: Discretization increment. 0.0 for continuous.
        prob: Predicted probabilities.
        num_classes: Number of classes in the classification problem.
        pred: Predicted labels/sides (+1 or -1). If None, derived from prob.

    Returns:
        DataFrame with 'timestamp' and 'signal' columns.
    """
    if "t1" not in events.columns:
        raise ValueError("events must have a 't1' column")

    if len(prob) == 0:
        return pl.DataFrame({"timestamp": [], "signal": []}).cast(
            {"timestamp": pl.Datetime, "signal": pl.Float64}
        )

    prob_np = prob.to_numpy()
    # z-stat from probability relative to uniform prior
    signal0 = (prob_np - 1.0 / num_classes) / np.sqrt(prob_np * (1.0 - prob_np))
    signal0 = 2 * norm.cdf(signal0) - 1

    # Apply predicted side
    if pred is not None:
        signal0 = pred.to_numpy() * signal0

    # Meta-labeling adjustment
    if "side" in events.columns:
        signal0 = events["side"].to_numpy() * signal0

    timestamps = events["timestamp"].to_numpy()
    t1_vals = events["t1"].to_numpy()

    # Build signals DataFrame for averaging
    signals_df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "t1": t1_vals,
            "signal": signal0,
        }
    )

    # Average active signals
    averaged = avg_active_signals(signals_df)

    # Discretize
    if step_size > 0:
        averaged = averaged.with_columns(
            discrete_signal(averaged["signal"], step_size).alias("signal")
        )

    return averaged


def avg_active_signals(signals: pl.DataFrame) -> pl.DataFrame:
    # TODO: vectorize with Polars cross-join for large inputs
    """Compute average signal among concurrently active bets.

    Args:
        signals: DataFrame with 'timestamp', 't1', 'signal' columns.

    Returns:
        DataFrame with 'timestamp' and 'signal' columns.
    """
    n = len(signals)
    ts_list = signals["timestamp"].to_list()
    t1_list = signals["t1"].to_list()
    sig_list = signals["signal"].to_list()

    # Collect all unique time points
    t_pnts = set(ts_list)
    for v in t1_list:
        if v is not None:
            t_pnts.add(v)
    t_pnts = sorted(t_pnts)

    out_ts = []
    out_sig = []
    for loc in t_pnts:
        active_signals = []
        for j in range(n):
            if ts_list[j] <= loc and (t1_list[j] is None or loc < t1_list[j]):
                active_signals.append(sig_list[j])
        if active_signals:
            out_ts.append(loc)
            out_sig.append(sum(active_signals) / len(active_signals))

    return pl.DataFrame({"timestamp": out_ts, "signal": out_sig})


def discrete_signal(signal0: pl.Series, step_size: float) -> pl.Series:
    """Discretize signal by rounding to nearest increment of step_size.

    Args:
        signal0: Raw or averaged bet sizes.
        step_size: Granularity of discretization.

    Returns:
        Discretized signals clipped to [-1, 1].
    """
    result = (signal0 / step_size).round(0) * step_size
    result = result.clip(-1.0, 1.0)
    return result


# ---------------------------------------------------------------------------
# Dynamic sizing (AFML Snippet 10.4)
# ---------------------------------------------------------------------------


def bet_size(w: float, x: float) -> float:
    """Width-regulated sigmoid for position sizing.

    Args:
        w: Width coefficient (omega), must be positive.
        x: Price divergence (f - mP).

    Returns:
        Bet size in (-1, 1).
    """
    return x * (w + x**2) ** -0.5


def get_target_pos(w: float, f: float, m_p: float, max_pos: int) -> int:
    """Calculate target position based on price divergence.

    Args:
        w: Width coefficient.
        f: Forecasted price.
        m_p: Current market price.
        max_pos: Maximum position size.

    Returns:
        Target position size as integer.
    """
    return int(bet_size(w, f - m_p) * max_pos)


def inv_price(f: float, w: float, m: float) -> float:
    """Inverse of sizing function to find price for given bet size.

    Args:
        f: Forecasted price.
        w: Width coefficient.
        m: Desired bet size, |m| < 1.

    Returns:
        Market price corresponding to bet size m.
    """
    return f - m * (w / (1 - m**2)) ** 0.5


def limit_price(t_pos: int, pos: int, f: float, w: float, max_pos: int) -> float:
    """Calculate average limit price for multi-unit order.

    Args:
        t_pos: Target position size.
        pos: Current position size.
        f: Forecasted price.
        w: Width coefficient.
        max_pos: Maximum position size.

    Returns:
        Average limit price for the order.
    """
    if t_pos == pos:
        return f
    sgn = 1 if t_pos >= pos else -1
    l_p = 0.0
    for j in range(abs(pos + sgn), abs(t_pos) + 1):
        l_p += inv_price(f, w, j / float(max_pos))
    l_p /= t_pos - pos
    return l_p


def get_w(x: float, m: float) -> float:
    """Calibrate width coefficient for desired divergence-to-size mapping.

    Args:
        x: Reference price divergence for calibration.
        m: Desired target bet size at divergence x, 0 < |m| < 1.

    Returns:
        Width coefficient omega.
    """
    return x**2 * (m ** (-2) - 1)


# ---------------------------------------------------------------------------
# Strategy-independent sizing (AFML Snippet 10.1 variant)
# ---------------------------------------------------------------------------


def bet_size_mixture(
    concurrent_counts: pl.Series,
    mu1: float,
    mu2: float,
    sigma1: float,
    sigma2: float,
    prob1: float,
) -> pl.Series:
    """Size bets via CDF of a fitted mixture of two Gaussians.

    Strategy-independent approach: fit a mixture model to the distribution
    of concurrent bet counts, then use the CDF to derive sizing.

    Args:
        concurrent_counts: Series of net concurrent bet counts (c_t = c_long - c_short).
        mu1: Mean of the first Gaussian component.
        mu2: Mean of the second Gaussian component.
        sigma1: Std of the first Gaussian component.
        sigma2: Std of the second Gaussian component.
        prob1: Mixing weight for the first component (0 < prob1 < 1).

    Returns:
        Polars Series of bet sizes in (-1, 1).
    """
    c = concurrent_counts.to_numpy().astype(float)
    # Mixture CDF: F(x) = prob1 * Phi((x-mu1)/sigma1) + (1-prob1) * Phi((x-mu2)/sigma2)
    cdf_vals = prob1 * norm.cdf(c, mu1, sigma1) + (1 - prob1) * norm.cdf(c, mu2, sigma2)
    # Map [0, 1] → (-1, 1): m = 2*F(c) - 1 for c >= 0, -(1 - 2*F(c)) for c < 0
    sizes = np.where(c >= 0, 2 * cdf_vals - 1, -(1 - 2 * cdf_vals))
    return pl.Series(concurrent_counts.name, np.clip(sizes, -1, 1))


# ---------------------------------------------------------------------------
# Ensemble sizing (AFML Section 10.3)
# ---------------------------------------------------------------------------


def bet_size_ensemble(avg_prob: float, n_classifiers: int) -> float:
    """Size bet from ensemble of n meta-labeling classifiers via Student-t.

    Args:
        avg_prob: Average predicted probability across n classifiers.
        n_classifiers: Number of classifiers in the ensemble.

    Returns:
        Bet size in (-1, 1).
    """
    from scipy.stats import t as t_dist

    if n_classifiers < 2:
        raise ValueError("Need at least 2 classifiers for ensemble sizing")
    se = (avg_prob * (1 - avg_prob) / n_classifiers) ** 0.5
    if se == 0:
        return 0.0
    t_stat = (avg_prob - 0.5) / se
    df = n_classifiers - 1
    return float(2 * t_dist.cdf(t_stat, df) - 1)
