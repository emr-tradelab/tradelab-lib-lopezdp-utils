"""Labeling methods for financial machine learning.

This module provides implementations of labeling methods from AFML Chapter 3
and complementary content from ML for Asset Managers, including:

Core Labeling Methods:
- Triple-barrier method (path-dependent, dynamic thresholds)
- Fixed-time horizon method (standard approach with known flaws)
- Trend-scanning method (statistical significance-based)

Triple-Barrier Components:
- Dynamic volatility thresholds
- Vertical (time) barriers
- Event detection and label generation
- Meta-labeling extensions

Utilities:
- Class imbalance handling
- Bet sizing from meta-labels

Reference:
    AFML Chapter 3: Labeling
    MLAM Section 5: Financial Labels
"""

# Barriers
from .barriers import add_vertical_barrier

# Bet sizing (from MLAM)
from .bet_sizing import (
    bet_size_dynamic,
    bet_size_from_ensemble,
    bet_size_from_probability,
)

# Class balance
from .class_balance import drop_labels

# Fixed-time horizon method
from .fixed_horizon import fixed_time_horizon

# Meta-labeling extensions
from .meta_labeling import get_bins_meta, get_events_meta

# Thresholds
from .thresholds import daily_volatility

# Trend-scanning method (from MLAM)
from .trend_scanning import t_value_linear_trend, trend_scanning_labels

# Triple-barrier method
from .triple_barrier import (
    apply_pt_sl_on_t1,
    get_bins,
    get_events,
    triple_barrier_labels,
)

__all__ = [
    "add_vertical_barrier",
    "apply_pt_sl_on_t1",
    "bet_size_dynamic",
    "bet_size_from_ensemble",
    "bet_size_from_probability",
    "daily_volatility",
    "drop_labels",
    "fixed_time_horizon",
    "get_bins",
    "get_bins_meta",
    "get_events",
    "get_events_meta",
    "t_value_linear_trend",
    "trend_scanning_labels",
    "triple_barrier_labels",
]
