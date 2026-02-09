"""Futures contract utilities for handling rolls and creating continuous price series.

Reference: Advances in Financial Machine Learning, Chapter 2, Section 2.5.2.2
"""

import numpy as np
import pandas as pd


def roll_gaps(
    series: pd.DataFrame,
    dictio: dict[str, str] | None = None,
    match_end: bool = True,
) -> pd.Series:
    """Compute cumulative gaps at futures contract rolls.

    This function identifies roll dates (when the contract identifier changes) and computes
    the price gap between the previous contract's close and the next contract's open. These
    gaps are cumulated over time to create a continuous adjustment series.

    Because different futures contract months trade at different prices, a price jump occurs
    at expiration that doesn't reflect actual market profit or loss. This function quantifies
    those jumps so they can be removed from the price series.

    Args:
        series: DataFrame containing historical futures data with an index of timestamps
            and columns for instrument identifier, open prices, and close prices.
        dictio: Dictionary mapping required fields to DataFrame column names:
            - 'Instrument': Column name for ticker/contract identifier
            - 'Open': Column name for open price
            - 'Close': Column name for close price
            Default: {'Instrument': 'FUT_CUR_GEN_TICKER', 'Open': 'PX_OPEN', 'Close': 'PX_LAST'}
        match_end: If True, performs backward roll where the end of the rolled series
            matches the end of the raw series. If False, performs forward roll.

    Returns:
        Series of cumulative gaps aligned with the input series index. Subtract this
        from raw prices to create a continuous, "gapless" price curve.

    Reference:
        - AFML, Chapter 2, Snippet 2.2
        - Use case: Create continuous futures price series for PnL simulation

    Note:
        While rolled prices are useful for PnL simulation, raw prices should still be
        used to determine capital consumption and position sizing.

    Example:
        >>> gaps = roll_gaps(futures_df)
        >>> continuous_close = futures_df['Close'] - gaps
    """
    if dictio is None:
        dictio = {
            "Instrument": "FUT_CUR_GEN_TICKER",
            "Open": "PX_OPEN",
            "Close": "PX_LAST",
        }

    # Identify roll dates (where instrument identifier changes)
    roll_dates = series[dictio["Instrument"]].drop_duplicates(keep="first").index

    # Initialize gaps series
    gaps = series[dictio["Close"]] * 0

    # Get integer locations of days prior to roll
    iloc_list = list(series.index)
    iloc_indices = np.array([iloc_list.index(i) - 1 for i in roll_dates])

    # Compute gaps: new_open - previous_close
    gaps.loc[roll_dates[1:]] = (
        series[dictio["Open"]].loc[roll_dates[1:]]
        - series[dictio["Close"]].iloc[iloc_indices[1:]].values
    )

    # Cumulative sum of gaps
    gaps = gaps.cumsum()

    # Roll backward if match_end=True (ending price matches raw data)
    if match_end:
        gaps -= gaps.iloc[-1]

    return gaps


def get_rolled_series(path_in: str, key: str) -> pd.DataFrame:
    """Load futures data from HDF5 and apply roll gap adjustment.

    This is a convenience wrapper around roll_gaps() that loads data from HDF5,
    computes gaps, and adjusts price fields to create a continuous series.

    Args:
        path_in: Path to HDF5 file containing futures data.
        key: HDF5 key to read (e.g., 'bars/ES_10k').

    Returns:
        DataFrame with roll-adjusted prices. 'Close' and 'VWAP' fields are adjusted
        for gaps to create continuous series.

    Reference:
        - AFML, Chapter 2, Snippet 2.2

    Example:
        >>> rolled_df = get_rolled_series('futures.h5', 'bars/ES_10k')
    """
    series = pd.read_hdf(path_in, key=key)
    series["Time"] = pd.to_datetime(series["Time"], format="%Y%m%d%H%M%S%f")
    series = series.set_index("Time")

    gaps = roll_gaps(series)

    # Adjust price fields
    for fld in ["Close", "VWAP"]:
        series[fld] -= gaps

    return series


def roll_and_rebase(
    raw: pd.DataFrame,
    dictio: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Create a non-negative rolled futures price series using total-return indexing.

    This function improves upon simple gap adjustment by computing returns from
    rolled prices and building a cumulative product to represent a $1 investment
    series (total-return index). This ensures the price series remains strictly
    positive, which is important in contango markets and for ML compatibility.

    Args:
        raw: DataFrame containing historical futures data with columns for
            instrument identifier, open, and close prices. Index should be timestamps.
        dictio: Dictionary mapping required fields to DataFrame column names:
            - 'Instrument': Column name for ticker/contract identifier
            - 'Open': Column name for open price
            - 'Close': Column name for close price
            Default: {'Instrument': 'Symbol', 'Open': 'Open', 'Close': 'Close'}

    Returns:
        DataFrame with additional columns:
        - Rolled 'Open' and 'Close' (gap-adjusted)
        - 'Returns': Daily returns computed from rolled prices
        - 'rPrices': Rebased total-return index ($1 initial investment)

    Reference:
        - AFML, Chapter 2, Snippet 2.3
        - Use case: Create strictly positive price series compatible with ML models

    Example:
        >>> rolled_df = roll_and_rebase(futures_df)
        >>> total_return_index = rolled_df['rPrices']
    """
    if dictio is None:
        dictio = {"Instrument": "Symbol", "Open": "Open", "Close": "Close"}

    # Compute gaps
    gaps = roll_gaps(raw, dictio=dictio)

    # Create rolled copy
    rolled = raw.copy(deep=True)

    # Adjust prices for gaps
    for fld in ["Open", "Close"]:
        rolled[fld] -= gaps

    # Compute returns from rolled prices
    rolled["Returns"] = rolled["Close"].diff() / raw["Close"].shift(1)

    # Create rebased total-return index
    rolled["rPrices"] = (1 + rolled["Returns"]).cumprod()

    return rolled
