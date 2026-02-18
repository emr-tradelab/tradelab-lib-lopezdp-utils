"""Data layer — bar construction, sampling, and microstructure features.

This package handles the first stage of López de Prado's pipeline:
raw market data → structured bars → event-driven sampling.

Reference:
    AFML Chapters 2, 19
"""

from tradelab.lopezdp_utils.data.bars import (
    dollar_bars,
    dollar_imbalance_bars,
    dollar_runs_bars,
    tick_bars,
    tick_imbalance_bars,
    tick_runs_bars,
    time_bars,
    volume_bars,
    volume_imbalance_bars,
    volume_runs_bars,
)
from tradelab.lopezdp_utils.data.sampling import (
    get_t_events,
    sampling_linspace,
    sampling_uniform,
)

__all__ = [
    # Standard bars
    "time_bars",
    "tick_bars",
    "volume_bars",
    "dollar_bars",
    # Information-driven bars
    "tick_imbalance_bars",
    "volume_imbalance_bars",
    "dollar_imbalance_bars",
    "tick_runs_bars",
    "volume_runs_bars",
    "dollar_runs_bars",
    # Sampling
    "get_t_events",
    "sampling_linspace",
    "sampling_uniform",
]
