"""
Fractional Differentiation Weights

This module implements the weight generation for the fractional difference operator
(1 - B)^d using binomial series expansion. The weights determine how much "memory"
from past observations is preserved when differentiating a time series.

Key insight: integer differentiation (d=1) removes far more memory than necessary
to achieve stationarity. Fractional d (e.g., d≈0.35) preserves memory while still
rendering the series stationary.

References:
    - AFML Chapter 5, Snippet 5.1
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_weights(d: float, size: int) -> np.ndarray:
    """
    Generate weights for the fractional difference operator (1 - B)^d.

    Computes the binomial series expansion weights iteratively. These weights
    determine how past observations contribute to the fractionally differentiated
    series. For d < 1, the weights decay slowly (long memory); for d = 1, they
    reduce to [1, -1] (standard first difference).

    Args:
        d: Differentiation order. Typically 0 < d < 1 for fractional
            differentiation. d=1 gives standard first difference.
        size: Number of weights to generate (number of lags).

    Returns:
        Array of shape (size, 1) with weights in reverse chronological order
        (most recent observation last).

    Mathematical Logic:
        ω₀ = 1
        ωₖ = -ωₖ₋₁ / k x (d - k + 1)  for k = 1, 2, ..., size-1

        This is the iterative form of the binomial coefficient:
        ωₖ = (-1)^k x C(d, k) where C(d, k) is the generalized binomial coefficient.

    Example:
        >>> weights = get_weights(d=0.5, size=5)
        >>> print(weights.flatten())
        # Shows decaying weights for d=0.5

    Reference:
        AFML Snippet 5.1
    """
    w = [1.0]
    for k in range(1, size):
        w_ = -w[-1] / k * (d - k + 1)
        w.append(w_)
    w = np.array(w[::-1]).reshape(-1, 1)
    return w


def get_weights_ffd(d: float, thres: float = 1e-5) -> np.ndarray:
    """
    Generate weights for Fixed-Width Window fractional differentiation (FFD).

    Unlike get_weights which generates a fixed number of weights, this function
    generates weights until their absolute value falls below the threshold τ.
    This determines the constant window width used by frac_diff_ffd.

    Args:
        d: Differentiation order. Typically 0 < d < 1.
        thres: Cut-off threshold for weight magnitude. Weights with
            |ωₖ| < thres are dropped. Default 1e-5.

    Returns:
        Array of shape (n_weights, 1) with weights in reverse chronological
        order. The number of weights is determined by the threshold.

    Mathematical Logic:
        Same iterative formula as get_weights, but the loop terminates when
        |ωₖ| < τ instead of at a fixed size. This ensures the window width
        is determined by the significance of the weights, not an arbitrary size.

    Example:
        >>> weights = get_weights_ffd(d=0.5, thres=1e-5)
        >>> print(f"Window width: {len(weights)}")

    Reference:
        AFML Snippet 5.3 (helper function)
    """
    w = [1.0]
    k = 1
    while True:
        w_ = -w[-1] / k * (d - k + 1)
        if abs(w_) < thres:
            break
        w.append(w_)
        k += 1
    w = np.array(w[::-1]).reshape(-1, 1)
    return w


def plot_weights(
    d_range: list[float],
    n_plots: int,
    size: int,
) -> plt.Axes:
    """
    Visualize how fractional differentiation weights decay for different d values.

    Plots weight decay curves across a range of differentiation orders, illustrating
    how lower d preserves more memory (slower decay) while higher d removes more
    memory (faster decay). This helps build intuition for choosing d.

    Args:
        d_range: Two-element list [d_min, d_max] defining the range of
            differentiation orders to plot.
        n_plots: Number of curves to generate between d_range boundaries.
        size: Number of lags (weight indices) to display.

    Returns:
        Matplotlib Axes object with the weight decay plot.

    Example:
        >>> ax = plot_weights(d_range=[0, 1], n_plots=6, size=6)
        >>> # Shows 6 weight decay curves from d=0 to d=1

    Reference:
        AFML Snippet 5.1 (plotWeights)
    """
    w = pd.DataFrame()
    for d in np.linspace(d_range[0], d_range[1], n_plots):
        w_ = get_weights(d, size=size)
        w_ = pd.DataFrame(w_, index=range(w_.shape[0])[::-1], columns=[d])
        w = w.join(w_, how="outer")
    ax = w.plot()
    ax.legend(loc="upper left")
    return ax
