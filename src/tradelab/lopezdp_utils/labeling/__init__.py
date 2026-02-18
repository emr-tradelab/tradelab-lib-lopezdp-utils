"""Labeling and sample weighting — AFML Chapters 3-4.

This package handles the second and third stages of López de Prado's pipeline:
price series → labeled observations (with t1 metadata) → sample weights.

The t1 timestamp (first barrier touch time) is the critical metadata that
connects labeling to weighting, purging, and embargoing downstream.

Reference:
    López de Prado, "Advances in Financial Machine Learning", Chapters 3-4
    López de Prado, "Machine Learning for Asset Managers", Chapter 5
"""

from tradelab.lopezdp_utils.labeling.class_balance import (
    drop_labels,
    get_class_weights,
)
from tradelab.lopezdp_utils.labeling.meta_labeling import (
    get_bins_meta,
    get_events_meta,
)
from tradelab.lopezdp_utils.labeling.sample_weights import (
    get_avg_uniqueness,
    get_ind_matrix,
    get_time_decay,
    mp_num_co_events,
    mp_sample_tw,
    mp_sample_w,
    seq_bootstrap,
)
from tradelab.lopezdp_utils.labeling.triple_barrier import (
    add_vertical_barrier,
    daily_volatility,
    fixed_time_horizon,
    get_bins,
    get_events,
    trend_scanning_labels,
    triple_barrier_labels,
)

__all__ = [
    "add_vertical_barrier",
    "daily_volatility",
    "drop_labels",
    "fixed_time_horizon",
    "get_avg_uniqueness",
    "get_bins",
    "get_bins_meta",
    "get_class_weights",
    "get_events",
    "get_events_meta",
    "get_ind_matrix",
    "get_time_decay",
    "mp_num_co_events",
    "mp_sample_tw",
    "mp_sample_w",
    "seq_bootstrap",
    "trend_scanning_labels",
    "triple_barrier_labels",
]
