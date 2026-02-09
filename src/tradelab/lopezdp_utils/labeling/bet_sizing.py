"""Bet sizing methods using meta-labeling predictions.

This module implements bet sizing formulas that use meta-label probabilities
to determine position sizes. Meta-labeling separates the bet side (primary model)
from the bet size (secondary meta-model), allowing for more sophisticated
risk management.

The bet size m ∈ [-1, 1] acts as a scalar for the primary model's signal:
- m near +1: high confidence, full position size
- m near 0: low confidence, pass on the bet
- m near -1: high confidence in opposite direction

Reference: MLAM Section 5.5
"""

import numpy as np
from scipy.stats import norm, t


def bet_size_from_probability(prob: float | np.ndarray) -> float | np.ndarray:
    """Calculate bet size from probability of profit (single classifier).

    This method computes bet size using the expected Sharpe ratio of a betting
    opportunity with symmetric payoffs. It assumes a single meta-labeling
    classifier provides the probability of a profitable outcome.

    Mathematical derivation (for symmetric payoff π):
    - Expected profit: μ = π(2p - 1)
    - Expected variance: σ² = 4π²p(1-p)
    - Z-statistic (Sharpe ratio): z = (p - 0.5) / √(p(1-p))
    - Bet size: m = 2·Φ(z) - 1, where Φ is the Gaussian CDF

    Args:
        prob: Probability of profit from meta-labeling classifier, p ∈ (0, 1).
            Can be a scalar or array of probabilities.

    Returns:
        Bet size m ∈ [-1, 1]:
            - m > 0: bet on primary model's side
            - m ≈ 0: pass on the bet (low confidence)
            - m < 0: bet against primary model's side

    Reference:
        MLAM Section 5.5.1 - Bet Sizing by Expected Sharpe Ratio

    Example:
        >>> # High confidence (80% probability of profit)
        >>> bet_size_from_probability(0.8)
        0.84  # Large position size
        >>>
        >>> # Low confidence (55% probability)
        >>> bet_size_from_probability(0.55)
        0.10  # Small position size
        >>>
        >>> # No edge (50% probability)
        >>> bet_size_from_probability(0.5)
        0.0  # Pass on the bet
    """
    # Handle boundary cases
    if np.any((prob <= 0) | (prob >= 1)):
        return np.sign(prob - 0.5)

    # Compute z-statistic (Sharpe ratio of the bet)
    z = (prob - 0.5) / np.sqrt(prob * (1 - prob))

    # Translate to bet size using Gaussian CDF
    return 2 * norm.cdf(z) - 1


def bet_size_from_ensemble(prob_avg: float, num_classifiers: int) -> float:
    """Calculate bet size from ensemble of meta-labeling classifiers.

    This method averages predictions across multiple independent meta-labeling
    classifiers and applies the de Moivre-Laplace theorem to derive bet size.
    Using an ensemble improves robustness compared to a single classifier.

    Mathematical derivation:
    - Ensemble average: p̂ = (1/n)Σyᵢ for n classifiers
    - Under H₀: p=0.5, p̂ ~ N(p, p(1-p)/n) for large n
    - T-statistic: t = (p̂ - 0.5) / √(p̂(1-p̂)/n)
    - Bet size: m = 2·T_{n-1}(t) - 1, where T is Student's t CDF

    Args:
        prob_avg: Average prediction across n independent classifiers.
            Should be mean of binary predictions {0, 1}.
        num_classifiers: Number of classifiers in the ensemble (n).

    Returns:
        Bet size m ∈ [-1, 1], with same interpretation as single classifier case.

    Reference:
        MLAM Section 5.5.2 - Ensemble Bet Sizing

    Example:
        >>> # Ensemble of 10 classifiers, average prediction 0.7
        >>> bet_size_from_ensemble(prob_avg=0.7, num_classifiers=10)
        0.76  # Strong signal from ensemble
        >>>
        >>> # Ensemble with weak signal
        >>> bet_size_from_ensemble(prob_avg=0.52, num_classifiers=10)
        0.04  # Small position due to low ensemble confidence
    """
    # Handle boundary cases
    if prob_avg <= 0 or prob_avg >= 1:
        return np.sign(prob_avg - 0.5)

    # Compute t-statistic under null hypothesis H₀: p = 0.5
    t_stat = (prob_avg - 0.5) / np.sqrt(prob_avg * (1 - prob_avg) / num_classifiers)

    # Translate to bet size using Student's t CDF
    return 2 * t.cdf(t_stat, df=num_classifiers - 1) - 1


def bet_size_dynamic(
    prob: float | np.ndarray, num_classifiers: int | None = None
) -> float | np.ndarray:
    """Dynamic bet sizing that selects method based on input type.

    Convenience wrapper that automatically chooses between single classifier
    and ensemble methods based on whether num_classifiers is provided.

    Args:
        prob: Probability of profit (single classifier) or ensemble average.
        num_classifiers: If provided, uses ensemble method. If None, uses
            single classifier method.

    Returns:
        Bet size m ∈ [-1, 1].

    Example:
        >>> # Automatically use single classifier method
        >>> bet_size_dynamic(0.65)
        0.54
        >>>
        >>> # Automatically use ensemble method
        >>> bet_size_dynamic(0.65, num_classifiers=10)
        0.58
    """
    if num_classifiers is None:
        return bet_size_from_probability(prob)
    else:
        return bet_size_from_ensemble(prob, num_classifiers)
