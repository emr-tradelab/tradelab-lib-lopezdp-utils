"""Meta-labeling extensions for the triple-barrier method.

Meta-labeling is a secondary ML layer that learns how to use a primary model.
The primary model decides the side (long/short), while the secondary model
determines the size of the bet (binary "act" or "pass").

This allows:
- Primary models to focus on high recall (find opportunities)
- Meta-models to focus on high precision (filter false positives)
- Integration of discretionary/fundamental insights (side) with ML (sizing)

Reference: AFML Chapter 3, Section 3.6
"""

import numpy as np
import pandas as pd

from .triple_barrier import apply_pt_sl_on_t1


def get_events_meta(
    close: pd.Series,
    t_events: pd.DatetimeIndex,
    pt_sl: list[float] | float,
    trgt: pd.Series,
    min_ret: float,
    num_threads: int = 1,
    t1: pd.Series | bool = False,
    side: pd.Series | None = None,
) -> pd.DataFrame:
    """Extended get_events with meta-labeling support (asymmetric barriers).

    This version extends the standard triple-barrier method by allowing the
    researcher to specify the position side via a primary model. When side
    is provided, the profit-taking and stop-loss barriers can be asymmetric
    (different widths).

    When side is None, this function behaves identically to get_events() with
    symmetric barriers. When side is provided, it enables meta-labeling where:
    - Primary model predicts the side (long=1, short=-1)
    - Secondary meta-model learns to predict success probability (act/pass)

    Args:
        close: Series of prices used to track the security's path.
        t_events: DatetimeIndex of timestamps that seed each triple-barrier event.
        pt_sl: If side is None, a single float for symmetric barriers.
            If side is provided, a list of two floats [profit_taking, stop_loss]
            for asymmetric barriers.
        trgt: Series of targets (absolute returns), typically from volatility
            estimator like daily_volatility().
        min_ret: Minimum target return required to initiate a triple-barrier search.
        num_threads: Number of threads for parallel processing. Default 1 (serial).
            Note: Full multiprocessing support via mpPandasObj (Chapter 20) will
            be integrated in production phase.
        t1: Series of vertical barrier timestamps (expiration limits), or False
            to disable vertical barriers.
        side: Optional Series with position sides (1 for long, -1 for short).
            If None, symmetric barriers are used for learning both side and size.
            If provided, enables meta-labeling with asymmetric barriers.

    Returns:
        DataFrame with columns:
            - 't1': Timestamp of first barrier touched
            - 'trgt': Target return (volatility) for the event
            - 'side': Position side (only if side parameter was provided)

    Reference:
        Snippet 3.6 in AFML Chapter 3, Section 3.6

    Example:
        >>> # Standard triple-barrier (symmetric)
        >>> events = get_events_meta(close, t_events, pt_sl=1.0, trgt=vol,
        ...                          min_ret=0.01, t1=t1, side=None)
        >>>
        >>> # Meta-labeling (asymmetric barriers)
        >>> primary_side = pd.Series(1, index=t_events)  # Primary model predictions
        >>> events = get_events_meta(close, t_events, pt_sl=[2.0, 1.0], trgt=vol,
        ...                          min_ret=0.01, t1=t1, side=primary_side)
    """
    # Align target with event timestamps
    trgt = trgt.loc[t_events]

    # Filter for minimum return threshold
    trgt = trgt[trgt > min_ret]

    # Set up vertical barriers
    if t1 is False:
        t1 = pd.Series(pd.NaT, index=t_events)

    # Configure side and barriers
    if side is None:
        # Standard mode: symmetric barriers, unknown side
        side_ = pd.Series(1.0, index=trgt.index)
        pt_sl_ = [pt_sl, pt_sl]
    else:
        # Meta-labeling mode: asymmetric barriers, known side from primary model
        side_ = side.loc[trgt.index]
        pt_sl_ = pt_sl[:2]  # Use first two elements [profit_taking, stop_loss]

    # Create events object
    events = pd.concat({"t1": t1, "trgt": trgt, "side": side_}, axis=1).dropna(subset=["trgt"])

    # Apply barriers to find first touches
    # Note: v1 uses serial processing. Full multiprocessing via mpPandasObj
    # (from Chapter 20) will be integrated in production phase.
    df0 = apply_pt_sl_on_t1(close=close, events=events, pt_sl=pt_sl_, molecule=events.index)

    # Get earliest barrier touch for each event
    events["t1"] = df0.dropna(how="all").min(axis=1)

    # Clean up: drop side column if not in meta-labeling mode
    if side is None:
        events = events.drop("side", axis=1)

    return events


def get_bins_meta(events: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    """Generate labels from triple-barrier events with meta-labeling support.

    This extended version handles two cases:

    Case 1 - Standard labeling ('side' not in events):
        Labels are {-1, 1} based on price action (sign of return).
        Used when learning both side and size from price movements.

    Case 2 - Meta-labeling ('side' in events):
        Labels are {0, 1} based on PnL success/failure.
        - Return is multiplied by side to get PnL
        - 1 = primary model's predicted side was profitable
        - 0 = primary model's prediction resulted in loss or wash
        Used when primary model provides side, meta-model learns sizing.

    Args:
        events: DataFrame output from get_events() or get_events_meta(), with
            index as event start times and 't1' column containing timestamps
            of first barrier touches. May optionally contain 'side' column.
        close: Series of prices used to calculate returns.

    Returns:
        DataFrame with index matching events, containing:
            - 'ret': Realized return (or PnL if side is present)
            - 'bin': Label (-1, 1) for standard mode or (0, 1) for meta-labeling

    Reference:
        Snippet 3.7 in AFML Chapter 3, Section 3.6

    Example:
        >>> # Standard labeling
        >>> events = get_events_meta(close, t_events, pt_sl=1.0, trgt=vol,
        ...                          min_ret=0.01, t1=t1)
        >>> labels = get_bins_meta(events, close)
        >>> # labels['bin'] contains -1 or 1
        >>>
        >>> # Meta-labeling
        >>> events = get_events_meta(close, t_events, pt_sl=[2.0, 1.0], trgt=vol,
        ...                          min_ret=0.01, t1=t1, side=primary_side)
        >>> labels = get_bins_meta(events, close)
        >>> # labels['bin'] contains 0 or 1
    """
    # Drop events with no barrier touch
    events_ = events.dropna(subset=["t1"])

    # Get all unique timestamps (event starts and barrier touches)
    px = events_.index.union(events_["t1"].values).drop_duplicates()

    # Align prices to these timestamps
    px = close.reindex(px, method="bfill")

    # Calculate returns
    out = pd.DataFrame(index=events_.index)
    out["ret"] = px.loc[events_["t1"].values].values / px.loc[events_.index] - 1

    # If side is present, this is meta-labeling: multiply by side to get PnL
    if "side" in events_.columns:
        out["ret"] *= events_["side"]

    # Generate labels
    out["bin"] = np.sign(out["ret"])

    # For meta-labeling: convert to binary {0, 1} where 0 = loss/wash, 1 = profit
    if "side" in events_.columns:
        out.loc[out["ret"] <= 0, "bin"] = 0

    return out
