"""Backtesting through Cross-Validation — AFML Chapter 12.

Combinatorial Purged Cross-Validation (CPCV) generates multiple out-of-sample
backtest paths from a single dataset. Unlike standard Walk-Forward or k-fold CV
which produce a single backtest path, CPCV creates φ[N,k] complete paths by
evaluating all combinatorial train/test splits with purging and embargoing.

This allows researchers to derive an empirical distribution of strategy
performance (e.g., Sharpe ratio) rather than relying on a single, potentially
volatile realization — making backtest overfitting significantly harder.
"""

from tradelab.lopezdp_utils.backtest_cv.cpcv import (
    CombinatorialPurgedKFold,
    assemble_backtest_paths,
    get_num_backtest_paths,
    get_num_splits,
)

__all__ = [
    "CombinatorialPurgedKFold",
    "assemble_backtest_paths",
    "get_num_backtest_paths",
    "get_num_splits",
]
