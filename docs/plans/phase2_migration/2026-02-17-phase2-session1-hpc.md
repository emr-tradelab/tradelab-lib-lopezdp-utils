# Phase 2 Session 1: `_hpc` — Internal Parallel Dispatch

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Migrate `hpc/` (3 files, 390 lines) into a single private `_hpc.py` module with tests.

**Architecture:** Merge `partitioning.py` + `multiprocessing.py` into `_hpc.py` (underscore = internal, not in public API). Keep pandas interface for now since downstream modules still use pandas — Polars migration happens when each consumer module is migrated. Add Pydantic validation for public-facing parameters.

**Tech Stack:** pandas, numpy, multiprocessing (stdlib), pydantic, pytest

---

### Task 1: Create branch and test skeleton

**Files:**
- Create: `tests/test_hpc.py`

**Step 1: Create feature branch**

Run: `git checkout -b phase2/hpc`

**Step 2: Create test file with partitioning tests**

```python
"""Tests for _hpc internal module."""

import numpy as np
import pandas as pd
import pytest


class TestLinParts:
    """Tests for lin_parts partitioning."""

    def test_equal_split(self):
        """10 atoms across 2 threads -> [0, 5, 10]."""
        from tradelab.lopezdp_utils._hpc import lin_parts

        result = lin_parts(10, 2)
        assert len(result) == 3
        assert result[0] == 0
        assert result[-1] == 10

    def test_more_threads_than_atoms(self):
        """Threads capped at num_atoms."""
        from tradelab.lopezdp_utils._hpc import lin_parts

        result = lin_parts(3, 10)
        assert len(result) == 4  # min(10, 3) + 1
        assert result[-1] == 3

    def test_single_thread(self):
        """One thread -> [0, num_atoms]."""
        from tradelab.lopezdp_utils._hpc import lin_parts

        result = lin_parts(100, 1)
        assert len(result) == 2
        assert result[0] == 0
        assert result[-1] == 100

    def test_partitions_cover_all_atoms(self):
        """Every atom is covered exactly once across segments."""
        from tradelab.lopezdp_utils._hpc import lin_parts

        parts = lin_parts(17, 4)
        total = sum(parts[i + 1] - parts[i] for i in range(len(parts) - 1))
        assert total == 17


class TestNestedParts:
    """Tests for nested_parts partitioning."""

    def test_lower_triangle_default(self):
        """Default (upper_triangle=False) produces balanced segments."""
        from tradelab.lopezdp_utils._hpc import nested_parts

        result = nested_parts(10, 3)
        assert result[0] == 0
        assert result[-1] == 10

    def test_upper_triangle(self):
        """Upper triangle reverses segment sizes."""
        from tradelab.lopezdp_utils._hpc import nested_parts

        result = nested_parts(10, 3, upper_triangle=True)
        assert result[0] == 0
        assert result[-1] == 10

    def test_partitions_cover_all_atoms(self):
        from tradelab.lopezdp_utils._hpc import nested_parts

        parts = nested_parts(20, 5, upper_triangle=True)
        total = sum(parts[i + 1] - parts[i] for i in range(len(parts) - 1))
        assert total == 20


class TestExpandCall:
    """Tests for expand_call argument unpacker."""

    def test_basic_call(self):
        from tradelab.lopezdp_utils._hpc import expand_call

        def add(a, b):
            return a + b

        result = expand_call({"func": add, "a": 2, "b": 3})
        assert result == 5


class TestProcessJobsRedux:
    """Tests for sequential job processing."""

    def test_processes_all_jobs(self):
        from tradelab.lopezdp_utils._hpc import process_jobs_redux

        def double(x):
            return x * 2

        jobs = [{"func": double, "x": i} for i in range(5)]
        results = process_jobs_redux(jobs)
        assert results == [0, 2, 4, 6, 8]

    def test_skips_failed_jobs(self):
        from tradelab.lopezdp_utils._hpc import process_jobs_redux

        def fail_on_two(x):
            if x == 2:
                raise ValueError("boom")
            return x

        jobs = [{"func": fail_on_two, "x": i} for i in range(4)]
        results = process_jobs_redux(jobs)
        assert results == [0, 1, 3]


class TestProcessJobs:
    """Tests for multiprocessing job execution."""

    def test_single_thread_fallback(self):
        """num_threads=1 falls back to sequential."""
        from tradelab.lopezdp_utils._hpc import process_jobs

        def square(x):
            return x**2

        jobs = [{"func": square, "x": i} for i in range(4)]
        results = process_jobs(jobs, num_threads=1)
        assert sorted(results) == [0, 1, 4, 9]


class TestMpPandasObj:
    """Tests for the main pandas multiprocessing engine."""

    def test_single_thread_series(self):
        """Process a Series with 1 thread (sequential)."""
        from tradelab.lopezdp_utils._hpc import mp_pandas_obj

        series = pd.Series(range(10), index=pd.date_range("2020-01-01", periods=10))

        def callback(molecule):
            return series.loc[molecule] * 2

        result = mp_pandas_obj(
            func=callback,
            pd_obj=("molecule", series),
            num_threads=1,
        )
        assert isinstance(result, pd.Series)
        assert len(result) == 10
        assert result.iloc[0] == 0
        assert result.iloc[5] == 10

    def test_single_thread_dataframe(self):
        """Process a DataFrame with 1 thread."""
        from tradelab.lopezdp_utils._hpc import mp_pandas_obj

        df = pd.DataFrame(
            {"a": range(10), "b": range(10, 20)},
            index=pd.date_range("2020-01-01", periods=10),
        )

        def callback(molecule):
            return df.loc[molecule] * 2

        result = mp_pandas_obj(
            func=callback,
            pd_obj=("molecule", df),
            num_threads=1,
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 10

    def test_result_sorted_by_index(self):
        """Output index is sorted regardless of processing order."""
        from tradelab.lopezdp_utils._hpc import mp_pandas_obj

        idx = pd.date_range("2020-01-01", periods=20)
        series = pd.Series(range(20), index=idx)

        def callback(molecule):
            return series.loc[molecule]

        result = mp_pandas_obj(
            func=callback,
            pd_obj=("molecule", series),
            num_threads=1,
            mp_batches=4,
        )
        assert list(result.index) == list(idx)


class TestMpJobList:
    """Tests for general-purpose job dispatcher."""

    def test_basic_dispatch(self):
        from tradelab.lopezdp_utils._hpc import mp_job_list

        def multiply(x, factor):
            return x * factor

        arg_list = [{"x": i} for i in range(5)]
        results = mp_job_list(multiply, arg_list, num_threads=1, redux=True, factor=3)
        assert sorted(results) == [0, 3, 6, 9, 12]


class TestSingleThreadDispatch:
    """Tests for debugging fallback."""

    def test_delegates_to_mp_pandas_obj(self):
        from tradelab.lopezdp_utils._hpc import single_thread_dispatch

        series = pd.Series([1, 2, 3], index=pd.date_range("2020-01-01", periods=3))

        def callback(molecule):
            return series.loc[molecule] + 10

        result = single_thread_dispatch(func=callback, pd_obj=("molecule", series))
        assert list(result.values) == [11, 12, 13]
```

**Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_hpc.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tradelab.lopezdp_utils._hpc'`

**Step 4: Commit**

```bash
git add tests/test_hpc.py
git commit -m "test(hpc): add test suite for _hpc module"
```

---

### Task 2: Create `_hpc.py` module

**Files:**
- Create: `src/tradelab/lopezdp_utils/_hpc.py`
- Delete (later): `src/tradelab/lopezdp_utils/hpc/` directory

**Step 1: Create `_hpc.py` by merging partitioning + multiprocessing**

Merge the contents of `hpc/partitioning.py` and `hpc/multiprocessing.py` into a single file `_hpc.py`. Keep all functions, docstrings, and type hints intact. The only changes:
- Single file instead of two
- Internal imports become local references (no cross-file import)
- Add module docstring noting this is internal

```python
"""Internal multiprocessing utilities — AFML Chapter 20.

This is a private module (_hpc) providing workload partitioning and
parallel dispatch for pandas objects. Not part of the public API.

Reference:
    López de Prado, "Advances in Financial Machine Learning", Chapter 20
"""

from __future__ import annotations

import datetime as dt
import logging
import multiprocessing as mp
import sys
import time
from typing import Any, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Workload Partitioning (AFML Snippets 20.1-20.2)
# ---------------------------------------------------------------------------


def lin_parts(num_atoms: int, num_threads: int) -> np.ndarray:
    # ... exact copy from partitioning.py ...


def nested_parts(num_atoms: int, num_threads: int, upper_triangle: bool = False) -> np.ndarray:
    # ... exact copy from partitioning.py ...


# ---------------------------------------------------------------------------
# Job Dispatch (AFML Snippets 20.3-20.9)
# ---------------------------------------------------------------------------


def expand_call(kargs: dict[str, Any]) -> Any:
    # ... exact copy from multiprocessing.py ...


def process_jobs_redux(jobs: list[dict[str, Any]]) -> list[Any]:
    # ... exact copy ...


def report_progress(job_num: int, num_jobs: int, time0: float, task: str = "") -> None:
    # ... exact copy ...


def process_jobs(jobs: list[dict[str, Any]], task: str = "", num_threads: int = 24) -> list[Any]:
    # ... exact copy ...


def mp_job_list(func: Callable, arg_list: list[dict[str, Any]], num_threads: int = 24,
                linmols: bool = True, redux: bool = False, task: str = "", **kwargs: Any) -> list[Any]:
    # ... exact copy ...


def mp_pandas_obj(func: Callable, pd_obj: tuple[str, pd.DataFrame | pd.Series],
                  num_threads: int = 24, mp_batches: int = 1, lin_mols: bool = True,
                  **kwargs: Any) -> pd.DataFrame | pd.Series:
    # ... exact copy, but internal calls to lin_parts/nested_parts are now local ...


def single_thread_dispatch(func: Callable, pd_obj: tuple[str, pd.DataFrame | pd.Series],
                           **kwargs: Any) -> pd.DataFrame | pd.Series:
    # ... exact copy ...
```

Note: Copy the FULL function bodies from the existing files. The comment `# ... exact copy ...` above is shorthand for this plan — the actual implementation must include complete code.

**Step 2: Run tests**

Run: `uv run pytest tests/test_hpc.py -v`
Expected: ALL PASS

**Step 3: Commit**

```bash
git add src/tradelab/lopezdp_utils/_hpc.py
git commit -m "feat(hpc): create merged _hpc.py internal module"
```

---

### Task 3: Update internal consumers and delete old `hpc/`

**Files:**
- Modify: Any file importing from `tradelab.lopezdp_utils.hpc`
- Delete: `src/tradelab/lopezdp_utils/hpc/` (entire directory)

**Step 1: Find all internal consumers**

Run: `grep -r "from tradelab.lopezdp_utils.hpc" src/`

Based on codebase exploration, no other module imports from `hpc` directly — it's only used by end users. But verify this.

**Step 2: Delete old hpc directory**

Run: `rm -rf src/tradelab/lopezdp_utils/hpc/`

**Step 3: Verify imports still work**

Run: `uv run python -c "from tradelab.lopezdp_utils._hpc import lin_parts, mp_pandas_obj; print('OK')"`
Expected: `OK`

**Step 4: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: ALL PASS

**Step 5: Lint**

Run: `uvx ruff check --fix . && uvx ruff format .`

**Step 6: Commit**

```bash
git add -A
git commit -m "refactor(hpc): remove old hpc/ directory, replaced by _hpc.py"
```

---

### Task 4: Merge to main

**Step 1: Merge branch**

Run: `git checkout main && git merge phase2/hpc`

**Step 2: Verify**

Run: `uv run pytest tests/ -v`
Expected: ALL PASS