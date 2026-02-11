"""Backtesting dangers utilities from AFML Chapter 11.

Provides tools for detecting backtest overfitting using Combinatorially Symmetric
Cross-Validation (CSCV). Chapter 11 is primarily conceptual, establishing the
"Seven Sins of Quantitative Investing" and two heuristic laws. The main extractable
algorithm is the CSCV procedure for estimating the Probability of Backtest
Overfitting (PBO).

For multiple testing corrections (Deflated Sharpe Ratio, FWER, Type II errors),
see ``tradelab.lopezdp_utils.sample_weights.strategy_redundancy``.
"""

from tradelab.lopezdp_utils.backtesting_dangers.cscv import (
    probability_of_backtest_overfitting,
)

__all__ = [
    "probability_of_backtest_overfitting",
]
