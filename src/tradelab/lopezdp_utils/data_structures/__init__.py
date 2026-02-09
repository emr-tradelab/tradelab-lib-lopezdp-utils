"""Financial data structures from Advances in Financial Machine Learning, Chapter 2.

This module provides implementations of all data structures and utilities from
AFML Chapter 2, including:
- PCA-based portfolio allocation
- Futures contract roll utilities
- Event-based sampling (CUSUM filter)
- Standard bars (time, tick, volume, dollar)
- Information-driven bars (imbalance and runs bars)
- ETF trick for basket modeling
- Sampling utilities (linspace, uniform)
- Optimal binning for entropy estimation
"""

# PCA utilities
# Optimal binning (from MLAM)
from .discretization import (
    discretize_optimal,
    mutual_information_optimal,
    num_bins,
    variation_of_information,
)

# ETF trick
from .etf import etf_trick

# Futures utilities
from .futures import get_rolled_series, roll_and_rebase, roll_gaps

# Information-driven bars - Imbalance
from .imbalance_bars import (
    dollar_imbalance_bars,
    tick_imbalance_bars,
    volume_imbalance_bars,
)
from .pca import pca_weights

# Information-driven bars - Runs
from .runs_bars import dollar_runs_bars, tick_runs_bars, volume_runs_bars

# Sampling utilities
from .sampling import get_t_events, sampling_linspace, sampling_uniform

# Standard bars
from .standard_bars import dollar_bars, tick_bars, time_bars, volume_bars

__all__ = [
    # PCA
    "pca_weights",
    # Futures
    "roll_gaps",
    "get_rolled_series",
    "roll_and_rebase",
    # Sampling
    "get_t_events",
    "sampling_linspace",
    "sampling_uniform",
    # Standard bars
    "time_bars",
    "tick_bars",
    "volume_bars",
    "dollar_bars",
    # Imbalance bars
    "tick_imbalance_bars",
    "volume_imbalance_bars",
    "dollar_imbalance_bars",
    # Runs bars
    "tick_runs_bars",
    "volume_runs_bars",
    "dollar_runs_bars",
    # ETF trick
    "etf_trick",
    # Optimal binning
    "num_bins",
    "discretize_optimal",
    "variation_of_information",
    "mutual_information_optimal",
]
