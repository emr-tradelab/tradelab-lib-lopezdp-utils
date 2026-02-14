"""Workload partitioning utilities from AFML Chapter 20.

Functions for dividing atoms (tasks) into segments suitable for
multiprocessing. Two strategies are provided:
- Linear partitioning: approximately equal-sized segments
- Nested partitioning: segments balanced for upper-triangular workloads

References:
    - AFML Chapter 20, Snippets 20.1-20.2
"""

import numpy as np


def lin_parts(num_atoms: int, num_threads: int) -> np.ndarray:
    """Partition atoms into approximately equal linear segments.

    Divides a range of atoms [0, num_atoms) into num_threads segments
    of approximately equal size. Each segment's start index is returned.

    Use this when each atom takes roughly the same amount of work
    (e.g., applying a function to each row independently).

    Args:
        num_atoms: Total number of atoms (tasks) to partition.
        num_threads: Number of threads/processes to partition across.

    Returns:
        Array of (num_threads + 1) partition boundaries. Segment i
        processes atoms from parts[i] to parts[i+1].

    Reference:
        AFML Snippet 20.1
    """
    parts = np.linspace(0, num_atoms, min(num_threads, num_atoms) + 1)
    parts = np.ceil(parts).astype(int)
    return parts


def nested_parts(num_atoms: int, num_threads: int, upper_triangle: bool = False) -> np.ndarray:
    """Partition atoms into nested (balanced) segments for matrix tasks.

    When computing an upper-triangular matrix (e.g., distance matrices,
    covariance subsets), atoms near the start of the range have more work
    than atoms near the end. Nested partitioning equalizes total workload
    across threads by making earlier segments smaller.

    Args:
        num_atoms: Total number of atoms (tasks) to partition.
        num_threads: Number of threads/processes to partition across.
        upper_triangle: If True, balance for upper-triangular workload
            where atom i processes (num_atoms - i) pairs. If False,
            falls back to linear partitioning.

    Returns:
        Array of (num_threads + 1) partition boundaries. Segment i
        processes atoms from parts[i] to parts[i+1].

    Reference:
        AFML Snippet 20.2
    """
    parts = [0]
    num_threads_ = min(num_threads, num_atoms)
    for _ in range(num_threads_):
        part = 1 + 4 * (parts[-1] ** 2 + parts[-1] + num_atoms * (num_atoms + 1.0) / num_threads_)
        part = (-1 + part**0.5) / 2.0
        parts.append(part)
    parts = np.round(parts).astype(int)
    if upper_triangle:
        # Reverse so that segments with more work are assigned first
        parts = np.cumsum(np.diff(parts)[::-1])
        parts = np.append(np.array([0]), parts)
    return parts
