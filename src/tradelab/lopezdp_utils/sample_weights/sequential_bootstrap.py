"""
Sequential Bootstrap Functions

This module implements the sequential bootstrap algorithm, a specialized resampling method
for financial data that accounts for informational redundancy caused by overlapping labels.

Standard bootstrap in finance causes Random Forests to overfit because in-bag and out-of-bag
samples are nearly identical due to overlaps. Sequential bootstrap ensures drawn samples
approach IID conditions by dynamically adjusting selection probabilities based on uniqueness.

The algorithm works in three steps:
1. Build indicator matrix tracking which bars influence each label (getIndMatrix)
2. Compute average uniqueness of observations (getAvgUniqueness)
3. Draw samples with probability proportional to uniqueness (seqBootstrap)

References:
    - AFML Chapter 4, Snippets 4.3-4.5
    - Section 4.5: The Sequential Bootstrap
"""

import numpy as np
import pandas as pd


def get_ind_matrix(bar_ix: pd.Index, t1: pd.Series) -> pd.DataFrame:
    """
    Build a binary indicator matrix representing the lifespan of labels.

    In financial ML, an observation's label is determined over a time window [t0, t1],
    during which the label "depends on" all price bars in that range. This function
    creates a matrix showing these dependencies.

    Args:
        bar_ix: Index of all price bars (e.g., datetime index). Rows of output matrix.
        t1: Series where index=observation start time, values=end time (first barrier
            touch or end of observation window)

    Returns:
        Binary indicator matrix of shape (num_bars, num_observations).
        Entry (t, i) is 1.0 if observation i depends on bar t, 0 otherwise.

    Mathematical Concept:
        Let I = number of observations (labels)
        Let T = number of bars (time periods)
        The matrix indM[t, i] = 1 if bar t falls within [t0_i, t1_i]
                              = 0 otherwise
        where [t0_i, t1_i] is the time window of observation i.

    Example:
        >>> bar_ix = pd.date_range('2023-01-01', periods=100, freq='H')
        >>> t1 = pd.Series(
        ...     index=pd.to_datetime(['2023-01-01 02:00', '2023-01-01 05:00']),
        ...     data=pd.to_datetime(['2023-01-01 04:00', '2023-01-01 08:00'])
        ... )
        >>> ind_m = get_ind_matrix(bar_ix, t1)
        >>> print(ind_m.shape)  # (100, 2)

    Reference:
        AFML Snippet 4.3
    """
    ind_m = pd.DataFrame(0, index=bar_ix, columns=range(t1.shape[0]))
    for i, (t0, t1_val) in enumerate(t1.items()):
        ind_m.loc[t0:t1_val, i] = 1.0
    return ind_m


def get_avg_uniqueness(ind_m: pd.DataFrame) -> pd.Series:
    """
    Compute the average uniqueness of each observation from an indicator matrix.

    Uniqueness measures how much an observation's information is NOT shared with
    other observations. If many labels overlap at a single bar, each label's
    uniqueness at that bar is reduced.

    Args:
        ind_m: Indicator matrix from get_ind_matrix, shape (num_bars, num_observations)

    Returns:
        Series of average uniqueness scores for each observation.
        Index = observation indices (column names from ind_m)
        Values in [0, 1], where 1 = completely unique, 0 = completely redundant

    Mathematical Concept:
        Step 1: Concurrency (c_t)
            c_t = sum over i of ind_m[t, i]
            = number of labels active at bar t

        Step 2: Uniqueness at each bar (u_t,i)
            u_t,i = ind_m[t, i] / c_t
            = label i's fractional "ownership" of bar t's information
            = 1 / c_t if label i is active at bar t
            = 0 if label i is not active at bar t

        Step 3: Average uniqueness for label i (avgU_i)
            avgU_i = mean(u_t,i) over all bars t where ind_m[t, i] = 1
                   = sum(u_t,i) / sum(ind_m[t, i])

    Intuition:
        - If a label is alone at all its bars: avgU = 1
        - If a label always shares bars with one other label: avgU = 0.5
        - If a label is completely redundant: avgU → 0

    Example:
        >>> avg_u = get_avg_uniqueness(ind_m)
        >>> print(avg_u)
        0    0.85
        1    0.50
        dtype: float64

    Reference:
        AFML Snippet 4.4
    """
    c = ind_m.sum(axis=1)  # Concurrency: count of active labels per bar
    u = ind_m.div(c, axis=0)  # Uniqueness: 1/c_t for each active label
    avg_u = u[u > 0].mean()  # Average uniqueness over each label's lifespan
    return avg_u


def seq_bootstrap(ind_m: pd.DataFrame, s_length: int | None = None) -> list:
    """
    Generate a sample index via the sequential bootstrap method.

    Unlike standard bootstrap (uniform selection probability), sequential bootstrap
    adjusts selection probabilities after each draw based on the redundancy of the
    remaining candidates. This produces a training set much closer to IID.

    The algorithm:
    1. For each draw, evaluate average uniqueness of all candidate observations
    2. Convert uniqueness to selection probability (higher uniqueness = higher prob)
    3. Draw one observation based on these probabilities
    4. Repeat until desired sample size is reached

    Args:
        ind_m: Indicator matrix from get_ind_matrix, shape (num_bars, num_observations)
        s_length: Number of draws to perform. Defaults to ind_m.shape[1]
                  (same as number of observations)

    Returns:
        List of sampled observation indices, length = s_length.
        Indices correspond to column labels in ind_m.
        Can have duplicates (bootstrap with replacement).

    Why It Works:
        1. Observations with higher average uniqueness have higher selection prob
        2. As we add more samples, redundant candidates become less likely to be
           selected because including them lowers the average uniqueness
        3. After s_length draws, result contains a mix minimizing redundancy
        4. This produces a training set where in-bag and out-of-bag samples are
           more different, reducing overfitting in tree-based models

    Example:
        >>> sampled_indices = seq_bootstrap(ind_m, s_length=5)
        >>> print(sampled_indices)  # [2, 0, 2, 4, 1]
        >>> train_subset = original_data.iloc[sampled_indices]

    Reference:
        AFML Snippet 4.5
    """
    if s_length is None:
        s_length = ind_m.shape[1]  # Default: resample to match original size

    phi = []  # List of selected observation indices

    while len(phi) < s_length:
        avg_u = pd.Series(dtype=float)  # Re-initialized each iteration

        for i in ind_m.columns:  # Iterate through all observation indices
            # Create subset with current selections + candidate i
            ind_m_ = ind_m[phi + [i]]
            # Get the average uniqueness of the LAST column added (observation i)
            avg_u.loc[i] = get_avg_uniqueness(ind_m_).iloc[-1]

        # Convert uniqueness scores to selection probabilities
        prob = avg_u / avg_u.sum()

        # Draw one observation with probability proportional to its uniqueness
        phi += [np.random.choice(ind_m.columns, p=prob)]

    return phi
