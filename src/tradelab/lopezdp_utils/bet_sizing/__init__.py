"""Bet sizing utilities from AFML Chapter 10.

Translates ML predictions into actionable position sizes using two approaches:

1. **Signal-based sizing** (Snippets 10.1-10.3): Converts classifier probabilities
   to bet sizes via z-statistic → Normal CDF mapping, with signal averaging and
   discretization to prevent overtrading.

2. **Dynamic position sizing** (Snippet 10.4): Sigmoid-based framework that adjusts
   position as market price diverges from forecast, with breakeven limit prices.

Note: For meta-labeling bet sizing from ML for Asset Managers (single classifier
and ensemble methods), see ``tradelab.lopezdp_utils.labeling.bet_sizing``.
"""

from tradelab.lopezdp_utils.bet_sizing.dynamic_sizing import (
    bet_size,
    get_target_pos,
    get_w,
    inv_price,
    limit_price,
)
from tradelab.lopezdp_utils.bet_sizing.signals import (
    avg_active_signals,
    discrete_signal,
    get_signal,
)

__all__ = [
    "avg_active_signals",
    "bet_size",
    "discrete_signal",
    "get_signal",
    "get_target_pos",
    "get_w",
    "inv_price",
    "limit_price",
]
