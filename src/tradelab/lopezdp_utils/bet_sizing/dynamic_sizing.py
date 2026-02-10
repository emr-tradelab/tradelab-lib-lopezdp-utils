"""Dynamic position sizing based on price divergence.

This module implements a sigmoid-based framework for dynamically adjusting
position size as the market price diverges from the forecast. As the market
moves toward the forecast, the position scales down asymptotically to zero,
ensuring the algorithm doesn't realize losses while scaling out.

The core function is a width-regulated sigmoid:
    m(x) = x / sqrt(w + x^2)

where x = f - mP (forecast minus market price) and w controls the slope.

Key properties:
- m → 0 as x → 0 (no position when market equals forecast)
- m → ±1 as x → ±∞ (full position at extreme divergence)
- w calibrates how quickly position size responds to divergence

The limit_price function ensures that multi-unit orders achieve a
breakeven or better average entry price.

Reference: AFML Chapter 10, Snippet 10.4
"""



def bet_size(w: float, x: float) -> float:
    """Width-regulated sigmoid function for position sizing.

    Maps price divergence to bet size using a sigmoid that approaches ±1
    asymptotically. The width parameter w controls the sensitivity:
    larger w → slower growth (more conservative sizing).

    Args:
        w: Width coefficient (omega), must be positive. Controls sigmoid slope.
        x: Price divergence (f - mP), forecast minus market price.

    Returns:
        Bet size in (-1, 1), approaching ±1 as |x| increases.

    Reference:
        AFML Snippet 10.4 — betSize function
    """
    return x * (w + x**2) ** -0.5


def get_target_pos(w: float, f: float, m_p: float, max_pos: int) -> int:
    """Calculate target position based on price divergence.

    Scales the sigmoid bet size by maximum position to get a discrete
    target number of units. Position scales down as market price
    approaches the forecast.

    Args:
        w: Width coefficient (calibrated via get_w).
        f: Forecasted price.
        m_p: Current market price.
        max_pos: Maximum position size Q (number of units).

    Returns:
        Target position size as integer.

    Reference:
        AFML Snippet 10.4 — getTPos function
    """
    return int(bet_size(w, f - m_p) * max_pos)


def inv_price(f: float, w: float, m: float) -> float:
    """Inverse of sizing function to find price for given bet size.

    Given a desired bet size m, calculates what market price would
    produce that size. Used internally by limit_price to compute
    breakeven entry points during scaling.

    Args:
        f: Forecasted price.
        w: Width coefficient.
        m: Desired bet size, must satisfy |m| < 1.

    Returns:
        Market price corresponding to bet size m.

    Reference:
        AFML Snippet 10.4 — invPrice function
    """
    return f - m * (w / (1 - m**2)) ** 0.5


def limit_price(
    t_pos: int, pos: int, f: float, w: float, max_pos: int
) -> float:
    """Calculate average limit price for multi-unit order.

    Computes the breakeven average price for filling from current position
    to target position. Ensures the algorithm doesn't realize losses
    while scaling out (reducing position as market approaches forecast).

    The limit price is the average of inv_price across each unit being
    added or removed.

    Args:
        t_pos: Target position size.
        pos: Current position size.
        f: Forecasted price.
        w: Width coefficient.
        max_pos: Maximum position size.

    Returns:
        Average limit price for the order.

    Reference:
        AFML Snippet 10.4 — limitPrice function
    """
    sgn = 1 if t_pos >= pos else -1
    l_p = 0.0
    for j in range(abs(pos + sgn), abs(t_pos) + 1):
        l_p += inv_price(f, w, j / float(max_pos))
    l_p /= t_pos - pos
    return l_p


def get_w(x: float, m: float) -> float:
    """Calibrate width coefficient for desired divergence-to-size mapping.

    Given a user-defined price divergence x and desired bet size m* at
    that divergence, solves for the width parameter omega that produces
    this behavior.

    Derivation from m = x / sqrt(w + x^2):
        w = x^2 * (m^{-2} - 1)

    Args:
        x: Reference price divergence (f - mP) for calibration.
        m: Desired target bet size at divergence x, must satisfy 0 < |m| < 1.

    Returns:
        Width coefficient omega.

    Reference:
        AFML Snippet 10.4 — getW function

    Example:
        >>> # At divergence of 10, want 95% of max position
        >>> w = get_w(x=10, m=0.95)
        >>> # Now use w in get_target_pos for actual sizing
    """
    return x**2 * (m ** (-2) - 1)
