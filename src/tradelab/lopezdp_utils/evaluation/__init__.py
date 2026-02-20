"""Backtest evaluation, risk assessment, and bet sizing — AFML Chapters 10-15.

This package covers the final stage of López de Prado's pipeline:
model predictions -> bet sizing -> backtest evaluation -> overfitting detection.

Key safety guardrails:
- DSR requires explicit sr_estimates (no default that hides multiple testing)
- CPCV generates complete OOS backtest paths for unbiased evaluation
- PBO quantifies the probability that in-sample performance is spurious

Reference:
    López de Prado, "Advances in Financial Machine Learning", Chapters 10-15
    López de Prado, "Machine Learning for Asset Managers", Section 8
"""

from tradelab.lopezdp_utils.evaluation.bet_sizing import (
    avg_active_signals,
    bet_size,
    bet_size_ensemble,
    bet_size_mixture,
    discrete_signal,
    get_signal,
    get_target_pos,
    get_w,
    inv_price,
    limit_price,
)
from tradelab.lopezdp_utils.evaluation.cpcv import (
    CombinatorialPurgedKFold,
    assemble_backtest_paths,
    get_num_backtest_paths,
    get_num_splits,
)
from tradelab.lopezdp_utils.evaluation.overfitting import (
    probability_of_backtest_overfitting,
)
from tradelab.lopezdp_utils.evaluation.redundancy import (
    deflated_sharpe_ratio_clustered,
    get_effective_trials,
)
from tradelab.lopezdp_utils.evaluation.statistics import (
    compute_dd_tuw,
    deflated_sharpe_ratio,
    get_bet_timing,
    get_hhi,
    get_hhi_decomposed,
    get_holding_period,
    multi_test_precision_recall,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
    sharpe_ratio_non_annualized,
    strategy_precision,
    strategy_recall,
)
from tradelab.lopezdp_utils.evaluation.strategy_risk import (
    bin_freq,
    bin_hr,
    implied_precision_symmetric,
    mix_gaussians,
    prob_failure,
    sharpe_ratio_asymmetric,
    sharpe_ratio_symmetric,
)
from tradelab.lopezdp_utils.evaluation.synthetic import (
    otr_batch,
    otr_main,
    ou_fit,
    ou_half_life,
)

__all__ = [
    "CombinatorialPurgedKFold",
    "assemble_backtest_paths",
    "avg_active_signals",
    "bet_size",
    "bet_size_ensemble",
    "bet_size_mixture",
    "bin_freq",
    "bin_hr",
    "compute_dd_tuw",
    "deflated_sharpe_ratio",
    "deflated_sharpe_ratio_clustered",
    "discrete_signal",
    "get_bet_timing",
    "get_effective_trials",
    "get_hhi",
    "get_hhi_decomposed",
    "get_holding_period",
    "get_num_backtest_paths",
    "get_num_splits",
    "get_signal",
    "get_target_pos",
    "get_w",
    "implied_precision_symmetric",
    "inv_price",
    "limit_price",
    "mix_gaussians",
    "multi_test_precision_recall",
    "otr_batch",
    "otr_main",
    "ou_fit",
    "ou_half_life",
    "prob_failure",
    "probabilistic_sharpe_ratio",
    "probability_of_backtest_overfitting",
    "sharpe_ratio",
    "sharpe_ratio_asymmetric",
    "sharpe_ratio_non_annualized",
    "sharpe_ratio_symmetric",
    "strategy_precision",
    "strategy_recall",
]
