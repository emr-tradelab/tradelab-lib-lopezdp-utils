"""Multiprocessing engine from AFML Chapter 20.

Core multiprocessing utilities for parallelizing pandas operations
and arbitrary job lists. These functions underpin the parallel
computations in Chapters 3, 4, 10, and others.

References:
    - AFML Chapter 20, Snippets 20.3-20.9
"""

from __future__ import annotations

import datetime as dt
import logging
import multiprocessing as mp
import sys
import time
from typing import Any, Callable

import pandas as pd

from tradelab.lopezdp_utils.hpc.partitioning import lin_parts, nested_parts

logger = logging.getLogger(__name__)


def expand_call(kargs: dict[str, Any]) -> Any:
    """Expand keyword arguments dictionary into a function call.

    Used as a wrapper to unpack arguments for multiprocessing dispatch.
    Extracts the 'func' key and calls it with the remaining kwargs.

    Args:
        kargs: Dictionary containing 'func' (the callable) and all
            keyword arguments to pass to it.

    Returns:
        Result of calling func(**remaining_kwargs).

    Reference:
        AFML Snippet 20.9
    """
    func = kargs["func"]
    del kargs["func"]
    return func(**kargs)


def process_jobs_redux(jobs: list[dict[str, Any]]) -> list[Any]:
    """Process jobs sequentially with error logging.

    Unwraps each job's callback via expand_call. Failed jobs are logged
    and skipped rather than raising exceptions.

    Args:
        jobs: List of job dictionaries, each containing 'func' and its
            keyword arguments.

    Returns:
        List of results from successful jobs.

    Reference:
        AFML Snippet 20.4
    """
    out = []
    for i, job in enumerate(jobs):
        try:
            out.append(expand_call(job))
        except Exception:
            logger.exception("process_jobs_redux: job %d failed", i)
    return out


def report_progress(job_num: int, num_jobs: int, time0: float, task: str = "") -> None:
    """Report asynchronous progress for long-running multiprocessing tasks.

    Logs completion percentage and estimated remaining time based on
    elapsed time and fraction of jobs completed.

    Args:
        job_num: Index of the job just completed (0-based).
        num_jobs: Total number of jobs.
        time0: Start time from time.time().
        task: Optional task description for the log message.

    Reference:
        AFML Snippet 20.6
    """
    msg_parts = [float(job_num) / num_jobs, (time.time() - time0) / 60.0]
    msg_parts.append(msg_parts[1] * (1 / msg_parts[0] - 1))
    time_stamp = str(dt.datetime.fromtimestamp(time.time()))
    msg = f"{time_stamp} {task} {msg_parts[0]:.2%} done after {msg_parts[1]:.2f} min. "
    msg += f"Remaining {msg_parts[2]:.2f} min."
    if job_num < num_jobs:
        sys.stderr.write(msg + "\r")
    else:
        sys.stderr.write(msg + "\n")


def process_jobs(
    jobs: list[dict[str, Any]],
    task: str = "",
    num_threads: int = 24,
) -> list[Any]:
    """Execute jobs using multiprocessing Pool with imap_unordered.

    Dispatches jobs to a process pool and collects results as they
    complete. Reports progress asynchronously via stderr.

    Args:
        jobs: List of job dictionaries, each containing 'func' and its
            keyword arguments.
        task: Optional task description for progress reporting.
        num_threads: Maximum number of worker processes.

    Returns:
        List of results in completion order.

    Reference:
        AFML Snippet 20.5
    """
    if num_threads <= 1:
        return process_jobs_redux(jobs)

    pool = mp.Pool(processes=num_threads)
    outputs = pool.imap_unordered(expand_call, jobs)
    out = []
    time0 = time.time()
    # Process outputs
    for i, out_ in enumerate(outputs, 1):
        out.append(out_)
        report_progress(i, len(jobs), time0, task)
    pool.close()
    pool.join()
    return out


def mp_job_list(
    func: Callable,
    arg_list: list[dict[str, Any]],
    num_threads: int = 24,
    linmols: bool = True,
    redux: bool = False,
    task: str = "",
    **kwargs: Any,
) -> list[Any]:
    """General-purpose multiprocessing dispatcher for arbitrary job lists.

    Partitions a list of argument dictionaries across threads and
    dispatches them for parallel execution.

    Args:
        func: The function to call for each job.
        arg_list: List of dictionaries, each containing keyword
            arguments for one call to func.
        num_threads: Maximum number of worker processes.
        linmols: If True, use linear partitioning; if False, use
            nested partitioning.
        redux: If True, use sequential processing (process_jobs_redux);
            if False, use multiprocessing pool.
        task: Optional task description for progress reporting.
        **kwargs: Additional keyword arguments appended to every job.

    Returns:
        List of results from all jobs.

    Reference:
        AFML Snippet 20.7
    """
    # Build job list
    jobs = []
    for arg in arg_list:
        job = {"func": func}
        job.update(arg)
        job.update(kwargs)
        jobs.append(job)

    if redux:
        out = process_jobs_redux(jobs)
    else:
        out = process_jobs(jobs, task=task, num_threads=num_threads)
    return out


def mp_pandas_obj(
    func: Callable,
    pd_obj: tuple[str, pd.DataFrame | pd.Series],
    num_threads: int = 24,
    mp_batches: int = 1,
    lin_mols: bool = True,
    **kwargs: Any,
) -> pd.DataFrame | pd.Series:
    """Main multiprocessing engine for pandas objects.

    Partitions a pandas DataFrame or Series into segments, dispatches
    each segment to a callback function in parallel, and concatenates
    the results. This is the primary parallelization utility used
    throughout AFML.

    Args:
        func: Callback function that accepts a pandas object (via the
            molecule keyword) and returns a pandas DataFrame or Series.
        pd_obj: Tuple of (argument_name, pandas_object). The argument_name
            is the keyword argument name under which molecule indices
            are passed to func. The pandas_object provides the index
            to partition.
        num_threads: Maximum number of worker processes. Set to 1 for
            single-threaded debugging.
        mp_batches: Multiplier for number of partitions. Total partitions
            = num_threads * mp_batches. Values > 1 help balance workload
            when individual tasks vary in duration.
        lin_mols: If True, use linear partitioning (equal segments).
            If False, use nested partitioning (balanced for upper-triangle).
        **kwargs: Additional keyword arguments passed to func.

    Returns:
        Concatenated pandas DataFrame or Series from all parallel results.

    Reference:
        AFML Snippet 20.3

    Example:
        >>> result = mp_pandas_obj(
        ...     func=my_callback,
        ...     pd_obj=("molecule", my_dataframe),
        ...     num_threads=4,
        ...     other_arg=42,
        ... )
    """
    if lin_mols:
        parts = lin_parts(len(pd_obj[1]), num_threads * mp_batches)
    else:
        parts = nested_parts(len(pd_obj[1]), num_threads * mp_batches)

    jobs = []
    for i in range(1, len(parts)):
        job = {pd_obj[0]: pd_obj[1].index[parts[i - 1] : parts[i]], "func": func}
        job.update(kwargs)
        jobs.append(job)

    if num_threads == 1:
        out = process_jobs_redux(jobs)
    else:
        out = process_jobs(jobs, task=func.__name__, num_threads=num_threads)

    if isinstance(out[0], pd.DataFrame):
        df0 = pd.DataFrame()
    elif isinstance(out[0], pd.Series):
        df0 = pd.Series(dtype=float)
    else:
        return out

    for i in out:
        df0 = pd.concat([df0, i])

    df0 = df0.sort_index()
    return df0


def single_thread_dispatch(
    func: Callable,
    pd_obj: tuple[str, pd.DataFrame | pd.Series],
    **kwargs: Any,
) -> pd.DataFrame | pd.Series:
    """Single-threaded fallback for debugging multiprocessing pipelines.

    Calls mp_pandas_obj with num_threads=1 so that all work executes
    sequentially in the main process, making debugging straightforward.

    Args:
        func: Callback function (same signature as mp_pandas_obj).
        pd_obj: Tuple of (argument_name, pandas_object).
        **kwargs: Additional keyword arguments passed to func.

    Returns:
        Concatenated pandas DataFrame or Series from sequential execution.

    Reference:
        AFML Section 20.5 (debugging recommendation)
    """
    return mp_pandas_obj(func=func, pd_obj=pd_obj, num_threads=1, **kwargs)
