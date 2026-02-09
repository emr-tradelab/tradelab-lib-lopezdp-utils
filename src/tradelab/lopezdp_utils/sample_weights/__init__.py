"""
Sample Weights Module

This module implements sample weighting techniques from AFML Chapter 4 and complementary
methods from ML for Asset Managers. The core problem addressed is that financial labels
often overlap in time, violating the IID assumption required by standard ML algorithms.

Main categories:
- Concurrency and uniqueness: Measuring and correcting for overlapping labels
- Sequential bootstrap: Non-IID-aware resampling
- Sample weighting: By return attribution and time decay
- Strategy-level redundancy: Multiple testing corrections (MLAM)

References:
    - AFML Chapter 4: Sample Weights
    - MLAM Chapter 8: Strategy Risk
"""

from .class_weights import (
    get_class_weights,
)
from .concurrency import (
    mp_num_co_events,
    mp_sample_tw,
)
from .return_attribution import (
    get_time_decay,
    mp_sample_w,
)
from .sequential_bootstrap import (
    get_avg_uniqueness,
    get_ind_matrix,
    seq_bootstrap,
)
from .strategy_redundancy import (
    estimate_independent_trials,
    false_strategy_theorem,
    familywise_error_rate,
    min_variance_cluster_weights,
    type_ii_error_prob,
)

__all__ = [
    # Concurrency and uniqueness
    "mp_num_co_events",
    "mp_sample_tw",
    # Sequential bootstrap
    "get_ind_matrix",
    "get_avg_uniqueness",
    "seq_bootstrap",
    # Return attribution and decay
    "mp_sample_w",
    "get_time_decay",
    # Class weights
    "get_class_weights",
    # Strategy-level redundancy (MLAM)
    "estimate_independent_trials",
    "min_variance_cluster_weights",
    "false_strategy_theorem",
    "familywise_error_rate",
    "type_ii_error_prob",
]
