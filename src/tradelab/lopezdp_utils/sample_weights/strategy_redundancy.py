"""
Strategy-Level Redundancy and Multiple Testing Corrections

This module implements strategy-level redundancy detection and multiple testing
corrections from ML for Asset Managers Chapter 8. While AFML Chapter 4 focuses on
individual observation weighting, MLAM extends these concepts to the strategy level.

The core problem: After running thousands of backtests, how many are truly independent?
And how do we avoid false discoveries due to selection bias?

Main functionalities:
1. False Strategy Theorem: Expected max Sharpe under multiple testing
2. FWER: Familywise error rate (probability of false positives)
3. Type II Error: Probability of missing true strategies
4. Minimum variance weights: Aggregate redundant strategies within clusters

These functions work with ONC clustering (AFML Chapter 4) to estimate the effective
number of independent trials (K) from a correlation matrix of backtest returns.

References:
    - ML for Asset Managers, Chapter 8: Strategy Risk
    - MLAM Snippets 8.1, 8.3, 8.4
    - MLAM Sections 8.7.1-8.7.2
"""

import numpy as np
import pandas as pd
import scipy.stats as ss
from scipy.stats import norm


def estimate_independent_trials(
    backtest_returns: pd.DataFrame,
) -> tuple[int, dict]:
    """
    Estimate the number of effectively independent trials using ONC clustering.

    This is a placeholder function that should be implemented using the ONC
    clustering algorithm from AFML Chapter 4 (or directly from MLAM Section 4).

    The key insight: N backtests may contain only K << N independent strategies
    due to high correlation. All multiple testing corrections must use K, not N.

    Args:
        backtest_returns: DataFrame of shape (T, N) where T is time periods
            and N is number of backtested strategies. Each column contains
            returns for one strategy.

    Returns:
        Tuple of (K, clusters) where:
            - K: Number of effectively independent trials/clusters
            - clusters: Dictionary mapping cluster_id -> list of strategy names

    Example:
        >>> returns = pd.DataFrame(...)  # 6385 strategy columns
        >>> k, clusters = estimate_independent_trials(returns)
        >>> print(f"Effective K = {k} from {returns.shape[1]} backtests")
        Effective K = 4 from 6385 backtests

    Note:
        This function requires the ONC algorithm implementation. For now, it's
        a placeholder. The actual ONC clustering should:
        1. Compute correlation matrix of returns
        2. Apply hierarchical clustering
        3. Find optimal number of clusters
        4. Return cluster assignments

    Reference:
        MLAM Section 8.7.1, referencing AFML/MLAM Section 4 (ONC algorithm)
    """
    # TODO: Implement ONC clustering from AFML Chapter 4
    # For now, return placeholder that should be replaced with actual ONC
    raise NotImplementedError(
        "This function requires ONC clustering implementation from AFML Chapter 4. "
        "See AFML Snippets 4.1-4.2 for the optimal number of clusters algorithm."
    )


def min_variance_cluster_weights(cov: np.ndarray, mu: np.ndarray | None = None) -> np.ndarray:
    """
    Compute minimum variance portfolio weights for aggregating cluster strategies.

    Used to aggregate redundant strategies within a cluster into a single
    "cluster strategy" for estimating cross-trial variance.

    Args:
        cov: Covariance matrix of shape (n, n) for n strategies in cluster
        mu: Expected returns vector of length n. If None, defaults to
            minimum variance (equal weight in inverse-vol space)

    Returns:
        Weights vector of length n, summing to 1.0

    Mathematical Formula:
        w = (Σ^(-1) × μ) / (1' × Σ^(-1) × μ)

        When μ = None:
        w = (Σ^(-1) × 1) / (1' × Σ^(-1) × 1)

    Example:
        >>> # Cluster contains strategies bt_0, bt_5, bt_12
        >>> cluster_returns = backtest_returns[['bt_0', 'bt_5', 'bt_12']]
        >>> cov_cluster = cluster_returns.cov().values
        >>> weights = min_variance_cluster_weights(cov_cluster)
        >>> # Aggregate: cluster_strategy = weights @ cluster_returns.T

    Reference:
        MLAM Section 8.7.2
    """
    inv_cov = np.linalg.inv(cov)
    ones = np.ones(shape=(inv_cov.shape[0], 1))

    if mu is None:
        mu = ones  # Default: minimum variance

    # Numerator: Σ^(-1) × μ
    w = np.dot(inv_cov, mu)

    # Denominator: 1' × Σ^(-1) × μ (ensure weights sum to 1)
    w /= np.dot(ones.T, w)[0, 0]

    return w.flatten()


def false_strategy_theorem(
    n_trials: int,
    mean_sr: float = 0.0,
    std_sr: float = 1.0,
) -> float:
    """
    Compute expected maximum Sharpe ratio under multiple testing (False Strategy Theorem).

    Establishes the "hurdle" that the observed max SR must beat to claim statistical
    significance after K independent trials. If observed max SR < expected max SR,
    the discovery is likely a false positive from selection bias.

    Args:
        n_trials: Number of independent trials (K). Should be from ONC clustering,
            not the raw number of backtests.
        mean_sr: Expected value of SR distribution (default: 0, assumes true SR=0)
        std_sr: Standard deviation of SR distribution across trials (default: 1.0).
            In practice, compute as np.std(sharpe_ratios_across_trials).

    Returns:
        Expected maximum Sharpe ratio under selection bias from K trials

    Mathematical Formula:
        E[max{SR_k}] ≈ V[{SR_k}] × [(1-γ)×Z^(-1)(1-1/K) + γ×Z^(-1)(1-1/(K×e))]

        Where:
        - γ = Euler-Mascheroni constant ≈ 0.5772156649
        - Z^(-1)[·] = Inverse standard normal CDF
        - e = Euler's number ≈ 2.71828

    Example:
        >>> # After ONC: K=4 independent strategies, std_sr=0.3
        >>> hurdle = false_strategy_theorem(n_trials=4, std_sr=0.3)
        >>> observed_max_sr = 1.2
        >>> print(f"Hurdle: {hurdle:.3f}, Observed: {observed_max_sr:.3f}")
        >>> if observed_max_sr > hurdle:
        ...     print("Likely a true discovery")

    Note:
        Output is non-annualized (same frequency as input data)

    Reference:
        MLAM Snippet 8.1
    """
    # Euler-Mascheroni constant (high precision)
    emc = 0.577215664901532860606512090082402431042159336

    # Formula: (1-γ)×Z^(-1)(1-1/K) + γ×Z^(-1)(1-1/(K×e))
    sr0 = (1 - emc) * norm.ppf(1 - 1.0 / n_trials) + emc * norm.ppf(1 - 1.0 / (n_trials * np.e))

    # Scale by standard deviation and add mean
    sr0 = mean_sr + std_sr * sr0
    return sr0


def familywise_error_rate(
    sr: float,
    t: int,
    k: int = 1,
    sr_benchmark: float = 0.0,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> float:
    """
    Calculate Familywise Error Rate (FWER) using Šidàk's correction.

    FWER is the probability of at least one false positive among K independent trials.
    This accounts for multiple testing and non-Normal return characteristics.

    Args:
        sr: Estimated non-annualized Sharpe ratio
        t: Number of observations (sample size)
        k: Number of independent trials/clusters (default: 1 for single test)
        sr_benchmark: Benchmark/null Sharpe ratio (default: 0)
        skew: Skewness of returns (default: 0 for Normal)
        kurt: Kurtosis of returns (default: 3 for Normal)

    Returns:
        α_K = Familywise Error Rate, probability of at least one Type I error.
        Value in [0, 1].

    Mathematical Formulas:
        Adjusted z-statistic:
        z = [(SR - SR*) × sqrt(T-1)] / sqrt(1 - γ_3×SR + (γ_4-1)/4 × SR²)

        Šidàk's correction:
        α_K = 1 - (1 - α)^K

        where α = Φ(-z) and Φ is the standard normal CDF

    Example:
        >>> # Test strategy with SR=1.5, 252 observations, K=4 clusters
        >>> sr_nonannual = 1.5 / np.sqrt(252)
        >>> alpha_k = familywise_error_rate(
        ...     sr=sr_nonannual, t=252, k=4, skew=-0.1, kurt=4.5
        ... )
        >>> print(f"FWER α_K = {alpha_k:.4f}")
        >>> if alpha_k < 0.05:
        ...     print("Significant at 5% level")

    Note:
        All SR inputs (sr, sr_benchmark) must be in same frequency (non-annualized)

    Reference:
        MLAM Snippet 8.3
    """
    # Step 1: Compute adjusted z-statistic
    z = (sr - sr_benchmark) * np.sqrt(t - 1)
    denominator = np.sqrt(1 - skew * sr + (kurt - 1) / 4.0 * sr**2)
    z /= denominator

    # Step 2: Individual Type I error probability
    alpha = ss.norm.cdf(-z)

    # Step 3: Familywise error rate (Šidàk correction)
    alpha_k = 1 - (1 - alpha) ** k

    return alpha_k


def type_ii_error_prob(
    alpha_k: float,
    k: int,
    sr_true: float,
    t: int,
    sr_estimated: float = 0.0,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> tuple[float, float]:
    """
    Calculate Type II Error probability - probability of missing a true strategy.

    Returns both the individual trial error (β) and the probability that ALL K
    trials miss the strategy (β_K = β^K). The power of the test is 1 - β_K.

    Args:
        alpha_k: Target Familywise Error Rate (e.g., 0.05)
        k: Number of independent trials/clusters
        sr_true: True Sharpe ratio to detect (alternative hypothesis)
        t: Number of observations (sample size)
        sr_estimated: Estimated Sharpe ratio (default: 0)
        skew: Skewness of returns (default: 0)
        kurt: Kurtosis of returns (default: 3)

    Returns:
        Tuple of (β, β_K) where:
            - β: Individual probability of a single trial missing the strategy
            - β_K: Probability ALL K trials miss it (β_K = β^K)
        Power = 1 - β_K

    Mathematical Formulas:
        Non-centrality parameter:
        θ = [SR* × sqrt(T-1)] / sqrt(1 - γ_3×SR + (γ_4-1)/4 × SR²)

        Type II error:
        β = Φ[Φ^(-1)[1 - (1 - α_K)^(1/K)] - θ]
        β_K = β^K

    Example:
        >>> # Detect SR*=0.5 with 252 obs, K=4 clusters at α_K=0.05
        >>> sr_true_nonannual = 0.5 / np.sqrt(252)
        >>> beta, beta_k = type_ii_error_prob(
        ...     alpha_k=0.05, k=4, sr_true=sr_true_nonannual,
        ...     t=252, skew=-0.1, kurt=4.5
        ... )
        >>> power = 1 - beta_k
        >>> print(f"Type II Error β = {beta:.4f}, β_K = {beta_k:.4f}")
        >>> print(f"Power = {power:.4f}")

    Note:
        Higher θ (stronger signal) → lower β → higher power
        Higher K (more trials) → higher β (harder to detect with Šidàk correction)

    Reference:
        MLAM Snippet 8.4
    """
    # Step 1: Calculate non-centrality parameter (effect size)
    theta = sr_true * np.sqrt(t - 1)
    denominator = np.sqrt(1 - skew * sr_estimated + (kurt - 1) / 4.0 * sr_estimated**2)
    theta /= denominator

    # Step 2: Inverse Šidàk's correction - get critical z for single trial
    z_critical = ss.norm.ppf((1 - alpha_k) ** (1.0 / k))

    # Step 3: Type II error for single trial
    beta = ss.norm.cdf(z_critical - theta)

    # Step 4: Type II error for all K trials
    beta_k = beta**k

    return beta, beta_k
