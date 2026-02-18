"""Event-based and downsampling utilities for financial time series.

Reference: Advances in Financial Machine Learning, Chapter 2, Section 2.5.3
"""

import numpy as np
import polars as pl


def get_t_events(df: pl.DataFrame, threshold: float) -> pl.Series:
    """Apply CUSUM filter for event-based sampling of time series.

    The CUSUM filter detects shifts in the mean of a measured quantity. Unlike
    Bollinger Bands, the filter requires a full run of magnitude `threshold` for
    a new event to be triggered, filtering out significant noise.

    Args:
        df: DataFrame with ``timestamp`` (Datetime) and ``close`` (Float64) columns.
        threshold: CUSUM threshold h. An event triggers when cumulative divergence
            exceeds this value in either direction.

    Returns:
        Polars Series of Datetime values for detected event timestamps.

    Reference:
        AFML, Chapter 2, Snippet 2.4

    # TODO(numba): evaluate JIT for CUSUM loop
    """
    close = df["close"].to_numpy()
    timestamps = df["timestamp"].to_list()

    t_events: list = []
    s_pos, s_neg = 0.0, 0.0

    for i in range(1, len(close)):
        diff = close[i] - close[i - 1]
        s_pos = max(0.0, s_pos + diff)
        s_neg = min(0.0, s_neg + diff)

        if s_neg < -threshold:
            s_neg = 0.0
            t_events.append(timestamps[i])
        elif s_pos > threshold:
            s_pos = 0.0
            t_events.append(timestamps[i])

    return pl.Series("timestamp", t_events, dtype=pl.Datetime)


def sampling_linspace(
    df: pl.DataFrame,
    num_samples: int | None = None,
    step: int | None = None,
) -> pl.DataFrame:
    """Downsample a DataFrame using constant step size (linspace sampling).

    Args:
        df: Input Polars DataFrame.
        num_samples: Target number of samples. Computes step automatically.
        step: Step size — take every N-th row. Takes priority over num_samples.

    Returns:
        Downsampled Polars DataFrame.

    Reference:
        AFML, Chapter 2, Section 2.5.3.1
    """
    if step is None and num_samples is None:
        raise ValueError("Either num_samples or step must be provided")

    if step is not None:
        return df.gather_every(step)

    if num_samples >= len(df):
        return df

    computed_step = len(df) // num_samples
    return df.gather_every(computed_step)


def sampling_uniform(
    df: pl.DataFrame,
    num_samples: int,
    random_state: int | None = None,
) -> pl.DataFrame:
    """Downsample a DataFrame using random uniform sampling.

    Args:
        df: Input Polars DataFrame.
        num_samples: Number of samples to draw.
        random_state: Random seed for reproducibility.

    Returns:
        Randomly sampled Polars DataFrame in chronological order.

    Reference:
        AFML, Chapter 2, Section 2.5.3.1
    """
    if num_samples >= len(df):
        return df

    rng = np.random.default_rng(random_state)
    indices = np.sort(rng.choice(len(df), size=num_samples, replace=False))
    return df[indices]
