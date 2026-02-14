"""High-Performance Computing Recipes — AFML Chapter 20.

Multiprocessing and vectorization utilities for parallelizing
financial ML computations. Provides:
- Workload partitioning (linear and nested)
- pandas-aware multiprocessing engine
- General-purpose job dispatching

Reference:
    López de Prado, "Advances in Financial Machine Learning", Chapter 20
"""

from tradelab.lopezdp_utils.hpc.multiprocessing import (
    expand_call,
    mp_job_list,
    mp_pandas_obj,
    process_jobs,
    process_jobs_redux,
    report_progress,
    single_thread_dispatch,
)
from tradelab.lopezdp_utils.hpc.partitioning import (
    lin_parts,
    nested_parts,
)

__all__ = [
    "expand_call",
    "lin_parts",
    "mp_job_list",
    "mp_pandas_obj",
    "nested_parts",
    "process_jobs",
    "process_jobs_redux",
    "report_progress",
    "single_thread_dispatch",
]
