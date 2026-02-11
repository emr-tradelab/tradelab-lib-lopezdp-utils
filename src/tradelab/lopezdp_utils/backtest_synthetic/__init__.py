"""Backtesting on Synthetic Data — AFML Chapter 13.

Utilities for generating synthetic price paths using the Ornstein-Uhlenbeck (O-U)
process and evaluating optimal trading rules (OTR) across profit-taking and
stop-loss threshold meshes via Monte Carlo simulation.

Reference: López de Prado, "Advances in Financial Machine Learning", Ch. 13.
"""

from tradelab.lopezdp_utils.backtest_synthetic.otr import (
    otr_batch,
    otr_main,
)
from tradelab.lopezdp_utils.backtest_synthetic.ou_process import (
    ou_fit,
    ou_half_life,
)

__all__ = [
    "otr_batch",
    "otr_main",
    "ou_fit",
    "ou_half_life",
]
