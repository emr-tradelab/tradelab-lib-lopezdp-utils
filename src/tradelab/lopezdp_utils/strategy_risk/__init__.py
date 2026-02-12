"""Strategy risk — AFML Chapter 15.

Binomial framework for understanding strategy risk: implied precision,
implied betting frequency, and probability of strategy failure.
"""

from tradelab.lopezdp_utils.strategy_risk.binomial_model import (
    bin_freq,
    bin_hr,
    implied_precision_symmetric,
    mix_gaussians,
    prob_failure,
    sharpe_ratio_asymmetric,
    sharpe_ratio_symmetric,
)

__all__ = [
    "bin_freq",
    "bin_hr",
    "implied_precision_symmetric",
    "mix_gaussians",
    "prob_failure",
    "sharpe_ratio_asymmetric",
    "sharpe_ratio_symmetric",
]
