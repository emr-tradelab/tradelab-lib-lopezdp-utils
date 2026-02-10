"""Signal generation and averaging for bet sizing.

This module implements the core pipeline for translating ML predictions into
actionable bet sizes:
1. Convert predicted probabilities to raw signal sizes (get_signal)
2. Average signals among concurrently active bets (avg_active_signals)
3. Discretize signals to prevent jitter/overtrading (discrete_signal)

The key insight is that bet sizing should be strategy-independent and derived
from the statistical confidence of the model's predictions. The z-statistic
of the predicted probability (relative to a uniform prior) maps directly to
the bet size via the Normal CDF.

Reference: AFML Chapter 10, Snippets 10.1-10.3
"""

import pandas as pd
from scipy.stats import norm


def get_signal(
    events: pd.DataFrame,
    step_size: float,
    prob: pd.Series,
    pred: pd.Series,
    num_classes: int,
) -> pd.Series:
    """Translate predicted probabilities into discretized bet sizes.

    Converts multinomial classifier outputs into bet sizes in [-1, 1] using
    a three-step pipeline:
    1. Compute z-statistic from probability relative to uniform prior (1/K)
    2. Average signals among concurrently active bets
    3. Discretize to prevent signal jitter

    The z-statistic formula:
        z = (p - 1/K) / sqrt(p * (1 - p))
        signal = side * (2 * Phi(z) - 1)

    where K is the number of classes and Phi is the Normal CDF.

    Args:
        events: DataFrame with DatetimeIndex and 't1' column (barrier end times).
            Optionally includes 'side' column for meta-labeling adjustment.
        step_size: Discretization increment in (0, 1]. E.g., 0.1 means signals
            are rounded to nearest 0.1.
        prob: Predicted probabilities from classifier, indexed by event times.
        pred: Predicted labels/sides (+1 or -1), indexed by event times.
        num_classes: Number of classes in the original classification problem.

    Returns:
        Discretized bet sizes in [-1, 1], indexed by all unique time points
        where signals are active.

    Reference:
        AFML Snippet 10.1

    Example:
        >>> events = pd.DataFrame({'t1': barrier_end_times}, index=event_times)
        >>> signal = get_signal(events, step_size=0.1, prob=probs,
        ...                     pred=predictions, num_classes=2)
    """
    if prob.shape[0] == 0:
        return pd.Series(dtype=float)

    # 1) Generate signals from multinomial classification (one-vs-rest)
    signal0 = (prob - 1.0 / num_classes) / (prob * (1.0 - prob)) ** 0.5
    signal0 = pred * (2 * norm.cdf(signal0) - 1)

    # Meta-labeling adjustment: multiply by primary model's side
    if "side" in events:
        signal0 *= events.loc[signal0.index, "side"]

    # 2) Average signals among concurrently active bets
    df0 = signal0.to_frame("signal").join(events[["t1"]], how="left")
    df0 = avg_active_signals(df0)

    # 3) Discretize to prevent jitter
    signal1 = discrete_signal(signal0=df0, step_size=step_size)
    return signal1


def avg_active_signals(signals: pd.DataFrame) -> pd.Series:
    """Compute average signal among concurrently active bets.

    At each unique time point (where a signal starts or ends), computes the
    mean signal across all bets that are still active. A signal is active if:
    - It was issued at or before the time point, AND
    - The time point is before the signal's end time (t1), or end time is NaT.

    This averaging prevents signal concentration when multiple overlapping
    bets point in the same direction, and allows offsetting signals to cancel.

    Args:
        signals: DataFrame with DatetimeIndex, columns 'signal' and 't1'.
            Index = signal start times, 't1' = signal end times.

    Returns:
        Mean signal at each unique time point, indexed by timestamp.

    Reference:
        AFML Snippet 10.2

    Note:
        This is a single-threaded v1 implementation. The book uses
        mpPandasObj (Ch. 20) for parallelization, which will be available
        after Chapter 20 extraction.
    """
    # Collect all unique time points where signals change
    t_pnts = set(signals["t1"].dropna().values)
    t_pnts = t_pnts.union(signals.index.values)
    t_pnts = sorted(t_pnts)

    # At each time point, average all active signals
    out = pd.Series(dtype=float)
    for loc in t_pnts:
        is_active = (signals.index.values <= loc) & (
            (loc < signals["t1"]) | pd.isnull(signals["t1"])
        )
        act = signals[is_active].index
        if len(act) > 0:
            out[loc] = signals.loc[act, "signal"].mean()
        else:
            out[loc] = 0
    return out


def discrete_signal(signal0: pd.Series, step_size: float) -> pd.Series:
    """Discretize signal by rounding to nearest increment of step_size.

    Prevents overtrading caused by minor signal fluctuations (jitter).
    Without discretization, tiny changes in probability estimates would
    trigger position adjustments, increasing transaction costs.

    Args:
        signal0: Raw or averaged bet sizes as a Series.
        step_size: Granularity of discretization, e.g., 0.1 means signals
            are rounded to {-1.0, -0.9, ..., 0.0, ..., 0.9, 1.0}.

    Returns:
        Discretized signals capped at [-1, 1].

    Reference:
        AFML Snippet 10.3

    Example:
        >>> discrete_signal(pd.Series([0.73, -0.42, 1.5]), step_size=0.2)
        0    0.8
        1   -0.4
        2    1.0
    """
    signal1 = (signal0 / step_size).round() * step_size
    signal1[signal1 > 1] = 1
    signal1[signal1 < -1] = -1
    return signal1
