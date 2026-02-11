"""Strategy discovery precision and recall from ML for Asset Managers.

Provides precision and recall metrics for strategy research pipelines,
based on the universe odds ratio θ (ratio of true to false strategies).
These metrics quantify the validity of the discovery process itself,
not the strategy's ML classification performance.

Reference: MLAM Section 8
"""


def strategy_precision(alpha: float, beta: float, theta: float) -> float:
    """Compute precision of strategy discovery process.

    Precision is the probability that a strategy flagged as significant
    is genuinely profitable, given the universe odds ratio θ.

    In finance θ is typically very small, so even a 5% false positive
    rate (alpha=0.05) can yield >80% false discovery rate.

    Args:
        alpha: Significance level (Type I error probability).
        beta: Type II error probability (probability of missing a true strategy).
        theta: Universe odds ratio s_T / s_F (true strategies / false strategies).

    Returns:
        Precision in [0, 1].

    Reference:
        MLAM Section 8
    """
    return ((1 - beta) * theta) / ((1 - beta) * theta + alpha)


def strategy_recall(beta: float) -> float:
    """Compute recall of strategy discovery process.

    Recall is the probability of detecting a true strategy (statistical power).

    Args:
        beta: Type II error probability.

    Returns:
        Recall (1 - β) in [0, 1].

    Reference:
        MLAM Section 8
    """
    return 1 - beta


def multi_test_precision_recall(
    alpha: float, beta: float, theta: float, k: int
) -> tuple[float, float]:
    """Extend precision/recall to K independent trials with Šidàk correction.

    Under K trials, the familywise Type I error increases (alpha_K) while
    familywise Type II error decreases (beta_K), creating a trade-off.

    Args:
        alpha: Per-trial significance level.
        beta: Per-trial Type II error probability.
        theta: Universe odds ratio s_T / s_F.
        k: Number of independent trials.

    Returns:
        Tuple of (precision_K, recall_K) adjusted for multiple testing.

    Reference:
        MLAM Section 8
    """
    alpha_k = 1 - (1 - alpha) ** k  # Familywise Type I error (Šidàk)
    beta_k = beta**k  # Familywise Type II error
    precision_k = ((1 - beta_k) * theta) / ((1 - beta_k) * theta + alpha_k)
    recall_k = 1 - beta_k
    return precision_k, recall_k
