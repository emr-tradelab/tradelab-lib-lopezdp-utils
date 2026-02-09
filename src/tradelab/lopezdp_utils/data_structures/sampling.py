"""Event-based and downsampling utilities for financial time series.

Reference: Advances in Financial Machine Learning, Chapter 2, Section 2.5.3
"""

import numpy as np
import pandas as pd


def get_t_events(g_raw: pd.Series, h: float) -> pd.DatetimeIndex:
    """Apply CUSUM filter for event-based sampling of time series.

    The CUSUM (Cumulative Sum) filter is a quality-control method used to detect shifts
    in the mean of a measured quantity away from a target value. Unlike popular signals
    like Bollinger Bands, the CUSUM filter is not triggered by a price simply hovering
    around a threshold level. It requires a full run of magnitude h for a new event to
    be triggered, which filters out significant amounts of noise.

    This filter identifies sequences of divergences (run-ups or run-downs) from a reset
    level. The threshold is activated when the cumulative divergence from the prior state
    exceeds h in either direction.

    Args:
        g_raw: Raw time series (typically prices or returns) with DatetimeIndex.
        h: Threshold or filter size. An event is triggered when the cumulative
            divergence from the mean exceeds this value.

    Returns:
        DatetimeIndex of timestamps where CUSUM events were triggered.

    Reference:
        - AFML, Chapter 2, Snippet 2.4
        - Use case: Downsample a continuous price series into a subset of catalytic events
        - ML Application: Train classifiers only on these event-driven timestamps to learn
          features associated with structural breaks, entropy spikes, or microstructural
          imbalances

    Theory:
        The filter tracks cumulative positive (sPos) and negative (sNeg) runs. When the
        magnitude exceeds threshold h, an event is recorded and the respective accumulator
        is reset to zero. This ensures the ML algorithm focuses on points where the price
        has drifted significantly from its prior state.

        Mathematically: S_t ≥ h ⟺ ∃τ∈[1,t]: Σ(i=τ to t)[y_i - E_(i-1)[y_t]] ≥ h

    Example:
        >>> prices = pd.Series([100, 101, 102, 103, 102, 101, 100, 99, 98],
        ...                    index=pd.date_range('2020-01-01', periods=9))
        >>> events = get_t_events(prices, h=2.0)
    """
    t_events, s_pos, s_neg = [], 0, 0
    diff = g_raw.diff()

    for i in diff.index[1:]:
        # Accumulate positive and negative runs
        s_pos = max(0, s_pos + diff.loc[i])
        s_neg = min(0, s_neg + diff.loc[i])

        # Trigger event and reset when threshold exceeded
        if s_neg < -h:
            s_neg = 0
            t_events.append(i)
        elif s_pos > h:
            s_pos = 0
            t_events.append(i)

    return pd.DatetimeIndex(t_events)


def sampling_linspace(
    data: pd.DataFrame,
    num_samples: int | None = None,
    step: int | None = None,
) -> pd.DataFrame:
    """Downsample data using constant step size (linspace sampling).

    This is a simple downsampling technique that takes every N-th observation.
    It's useful for reducing dataset size when algorithms don't scale well
    (e.g., SVMs), but has significant limitations.

    Args:
        data: DataFrame with DatetimeIndex to be downsampled.
        num_samples: Target number of samples. If provided, step is computed
            automatically. Either num_samples or step must be provided.
        step: Step size (take every N-th row). If provided, num_samples is ignored.

    Returns:
        Downsampled DataFrame.

    Reference:
        - AFML, Chapter 2, Section 2.5.3.1

    Limitations:
        - Step size is chosen arbitrarily
        - Results vary significantly depending on the "seed bar" (starting point)
        - Does NOT guarantee the sample contains the most informative observations

    Note:
        For better information-preserving sampling, consider using CUSUM filter
        (get_t_events) which samples based on catalytic events.

    Example:
        >>> df = pd.DataFrame({'price': range(100)},
        ...                   index=pd.date_range('2020-01-01', periods=100))
        >>> downsampled = sampling_linspace(df, num_samples=10)
        >>> len(downsampled)
        10
    """
    if step is None and num_samples is None:
        raise ValueError("Either num_samples or step must be provided")

    if step is not None:
        # Use provided step
        return data.iloc[::step]
    else:
        # Compute step from target num_samples
        if num_samples >= len(data):
            return data
        step = len(data) // num_samples
        return data.iloc[::step]


def sampling_uniform(
    data: pd.DataFrame,
    num_samples: int,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Downsample data using random uniform sampling.

    This method draws samples randomly using a uniform distribution across
    the entire dataset. It addresses the seed-dependency problem of linspace
    sampling but still doesn't guarantee capturing the most informative observations.

    Args:
        data: DataFrame with DatetimeIndex to be downsampled.
        num_samples: Number of samples to draw.
        random_state: Random seed for reproducibility. If None, uses random seed.

    Returns:
        Randomly sampled DataFrame, sorted by index (chronological order).

    Reference:
        - AFML, Chapter 2, Section 2.5.3.1

    Advantages over linspace:
        - No seed-dependency (starting point doesn't matter)
        - More representative of the overall dataset

    Limitations:
        - Still arbitrary (doesn't target informative events)
        - May miss structural breaks, catalytic events, etc.

    Note:
        For information-driven sampling, use CUSUM filter (get_t_events) or
        information-driven bars (imbalance/runs bars).

    Example:
        >>> df = pd.DataFrame({'price': range(100)},
        ...                   index=pd.date_range('2020-01-01', periods=100))
        >>> downsampled = sampling_uniform(df, num_samples=10, random_state=42)
        >>> len(downsampled)
        10
    """
    if num_samples >= len(data):
        return data

    rng = np.random.default_rng(random_state)
    indices = rng.choice(len(data), size=num_samples, replace=False)
    indices = np.sort(indices)  # Sort to maintain chronological order

    return data.iloc[indices]
