"""Backtest statistics — AFML Chapter 14.

Utilities for evaluating backtesting results: bet timing, holding periods,
return concentration (HHI), drawdowns, and risk-adjusted performance
metrics (SR, PSR, DSR).
"""

from tradelab.lopezdp_utils.backtest_statistics.bet_timing import (
    get_bet_timing,
    get_holding_period,
)
from tradelab.lopezdp_utils.backtest_statistics.concentration import get_hhi
from tradelab.lopezdp_utils.backtest_statistics.drawdown import compute_dd_tuw
from tradelab.lopezdp_utils.backtest_statistics.sharpe import (
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
)
from tradelab.lopezdp_utils.backtest_statistics.strategy_metrics import (
    multi_test_precision_recall,
    strategy_precision,
    strategy_recall,
)

__all__ = [
    "compute_dd_tuw",
    "deflated_sharpe_ratio",
    "get_bet_timing",
    "get_hhi",
    "get_holding_period",
    "multi_test_precision_recall",
    "probabilistic_sharpe_ratio",
    "sharpe_ratio",
    "strategy_precision",
    "strategy_recall",
]
