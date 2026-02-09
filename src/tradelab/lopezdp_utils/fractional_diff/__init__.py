"""
Fractional Differentiation Module

This module implements fractional differentiation methods from AFML Chapter 5.
The core problem addressed is the stationarity vs. memory trade-off: standard
integer differentiation (d=1, i.e., log-returns) achieves stationarity but discards
most of the predictive signal (memory) in the series.

Fractional differentiation with d < 1 (typically d ≈ 0.35) achieves stationarity
while preserving significantly more memory, as measured by correlation with the
original price series.

Main categories:
- Weight generation: Binomial series expansion weights for the (1-B)^d operator
- Expanding window fracdiff: Standard method with weight-loss threshold (has drift)
- Fixed-Width Window fracdiff (FFD): Constant window, driftless (recommended)
- Minimum FFD finder: ADF-based optimization to find optimal d*

References:
    - AFML Chapter 5: Fractionally Differentiated Features
"""

from .fracdiff import (
    frac_diff,
    frac_diff_ffd,
    plot_min_ffd,
)
from .weights import (
    get_weights,
    get_weights_ffd,
    plot_weights,
)

__all__ = [
    "frac_diff",
    "frac_diff_ffd",
    "get_weights",
    "get_weights_ffd",
    "plot_min_ffd",
    "plot_weights",
]
