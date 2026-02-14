"""Trade classification algorithms from AFML Chapter 19.

Implements the tick rule for classifying trade initiation direction,
which is a building block for many microstructural features.

Reference:
    AFML Chapter 19, Section 19.3.1
"""

import numpy as np
import pandas as pd


def tick_rule(prices: pd.Series) -> pd.Series:
    """Classify trades as buyer- or seller-initiated using the tick rule.

    A trade is classified as buyer-initiated (+1) if the price is higher
    than the previous trade, seller-initiated (-1) if lower. When the
    price is unchanged (zero tick), the prior classification is carried
    forward.

    This is the simplest and most widely used trade classification
    algorithm, serving as input to Kyle's Lambda, VPIN, and other
    microstructural features.

    Args:
        prices: Series of transaction prices.

    Returns:
        Series of trade signs (+1 for buy, -1 for sell) with same index.

    Reference:
        AFML Section 19.3.1
    """
    dp = prices.diff()
    signs = pd.Series(np.zeros(len(prices)), index=prices.index)
    signs[dp > 0] = 1.0
    signs[dp < 0] = -1.0
    # Zero-tick rule: carry forward prior classification
    signs.iloc[0] = 1.0  # Initial condition
    signs = signs.replace(0, np.nan).ffill()
    return signs.astype(int)
