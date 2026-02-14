"""Microstructural Features — AFML Chapter 19.

Market microstructure features organized by generation:
- First Generation: Price sequence models (spreads, volatility)
- Second Generation: Strategic trade models (price impact)
- Third Generation: Sequential trade models (informed trading probability)

Reference:
    López de Prado, "Advances in Financial Machine Learning", Chapter 19
"""

from tradelab.lopezdp_utils.microstructure.price_impact import (
    amihud_lambda,
    hasbrouck_lambda,
    kyle_lambda,
)
from tradelab.lopezdp_utils.microstructure.spread_estimators import (
    becker_parkinson_volatility,
    corwin_schultz_spread,
    get_alpha,
    get_beta,
    get_gamma,
    high_low_volatility,
    roll_model,
)
from tradelab.lopezdp_utils.microstructure.trade_classification import tick_rule
from tradelab.lopezdp_utils.microstructure.vpin import (
    volume_bucket,
    vpin,
)

__all__ = [
    "amihud_lambda",
    "becker_parkinson_volatility",
    "corwin_schultz_spread",
    "get_alpha",
    "get_beta",
    "get_gamma",
    "hasbrouck_lambda",
    "high_low_volatility",
    "kyle_lambda",
    "roll_model",
    "tick_rule",
    "volume_bucket",
    "vpin",
]
