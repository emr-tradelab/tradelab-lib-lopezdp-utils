# Phase 2 Session 7: `allocation/` — HRP, Denoising, NCO, Simulation

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Migrate `ml_asset_allocation/` (4 files, 728 lines) into `allocation/` package (4 files + `__init__.py`). This covers AFML Chapter 16 (HRP) + MLAM Sections 2.4–2.9 (denoising) + MLAM Snippets 7.3–7.6 (NCO).

**Architecture:**
- `ml_asset_allocation/hrp.py` → `allocation/hrp.py` (rename package, keep content)
- `ml_asset_allocation/denoising.py` → `allocation/denoising.py` (rename package, keep content)
- `ml_asset_allocation/nco.py` → `allocation/nco.py` (rename package, **fix import** from old `feature_importance.clustering` to `features.importance`)
- `ml_asset_allocation/simulation.py` → `allocation/simulation.py` (rename package, update internal import)
- Delete old `hpc/` directory (empty — only `__pycache__`, replaced by `_hpc.py` in session 1)

**Tech Stack:** numpy, pandas, scipy (cluster.hierarchy, optimize), sklearn (KernelDensity), pytest

**Depends on:** Session 6 (`evaluation/` merged to main), Session 4 (`features/` on main — NCO imports `cluster_kmeans_base` from features)

---

## Pre-Session Checklist

- [ ] Session 6 merged to main
- [ ] Branch from main: `git checkout -b phase2/allocation`
- [ ] `uv sync --all-extras --dev`
- [ ] Verify `features/importance.py` exports `cluster_kmeans_top` (NCO dependency)

---

## File Mapping: Old → New

| Old file | New location | Action |
|----------|-------------|--------|
| `ml_asset_allocation/hrp.py` | `allocation/hrp.py` | Migrate |
| `ml_asset_allocation/denoising.py` | `allocation/denoising.py` | Migrate |
| `ml_asset_allocation/nco.py` | `allocation/nco.py` | Migrate + fix imports |
| `ml_asset_allocation/simulation.py` | `allocation/simulation.py` | Migrate + fix imports |
| `hpc/` (empty, only `__pycache__`) | — | Delete |

---

## Polars Migration Decisions (per function)

### allocation/hrp.py
| Function | Migration | Notes |
|----------|-----------|-------|
| `correl_dist(corr)` | **Keep pandas/numpy** | Accepts pandas corr matrix, scipy linkage requires ndarray |
| `tree_clustering(corr, method)` | **Keep pandas/numpy** | scipy.cluster.hierarchy.linkage |
| `get_quasi_diag(link)` | **Keep Python** | Pure tree traversal |
| `get_ivp(cov, **kwargs)` | **Keep numpy** | Diagonal inverse |
| `get_cluster_var(cov, items)` | **Keep numpy** | Delegates to `get_ivp` |
| `get_rec_bipart(cov, sort_ix)` | **Keep numpy** | Recursive bisection loop |
| `hrp_alloc(cov, corr)` | **Keep pandas** | Full pipeline, returns pd.Series of weights |

> **Rationale:** HRP operates on covariance/correlation matrices (square, dense). These are naturally numpy arrays. No benefit from Polars — no time series, no columnar operations.

### allocation/denoising.py
| Function | Migration | Notes |
|----------|-----------|-------|
| `mp_pdf(var, q, pts)` | **Keep numpy** | Marcenko-Pastur density, pure math |
| `find_max_eval(evals, q, bwidth)` | **Keep numpy/scipy** | KDE fitting + minimize |
| `denoised_corr(evals, evecs, n_facts)` | **Keep numpy** | Eigenvalue manipulation |
| `denoised_corr_shrinkage(evals, evecs, n_facts, alpha)` | **Keep numpy** | Targeted shrinkage |
| `denoise_cov(cov, q, bwidth)` | **Keep pandas/numpy** | Wrapper: accepts pd.DataFrame cov |
| `detone_corr(corr, n_remove)` | **Keep numpy** | Eigenvector removal |

> **Rationale:** All operations are on square covariance matrices. Purely linear algebra — numpy is the right tool.

### allocation/nco.py
| Function | Migration | Notes |
|----------|-----------|-------|
| `opt_port_nco(cov, mu, max_num_clusters, objective)` | **Keep pandas/numpy** | **Fix import:** `cluster_kmeans_base` → `cluster_kmeans_top` from `features.importance` |

> **Note:** The v1 code imports `cluster_kmeans_base` from the old `feature_importance.clustering` path. This was migrated to `features.importance.cluster_kmeans_top` in session 4. Update the import path.

### allocation/simulation.py
| Function | Migration | Notes |
|----------|-----------|-------|
| `generate_data(n_obs, size0, size1, sigma1)` | **Keep pandas/numpy** | Random covariance generation |
| `hrp_mc(n_iter, ...)` | **Keep pandas/numpy** | Monte Carlo loop, returns pd.DataFrame of variances |

---

## Pydantic Validation

No Pydantic configs needed — all functions take simple numeric parameters. Input validation at function boundaries:
- `hrp_alloc`: Validate `cov` and `corr` have matching dimensions
- `opt_port_nco`: Validate `max_num_clusters >= 2`

---

## Tasks

### Task 1: Create branch, package structure, and test fixtures

**Files:**
- Create: `src/tradelab/lopezdp_utils/allocation/__init__.py` (empty placeholder)
- Create: `tests/allocation/__init__.py`
- Create: `tests/allocation/conftest.py`

**Step 1: Create branch**

Run: `git checkout -b phase2/allocation`

**Step 2: Create directories**

Run: `mkdir -p tests/allocation`

**Step 3: Create shared fixtures**

```python
"""Shared fixtures for allocation package tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_cov() -> pd.DataFrame:
    """5x5 positive-definite covariance matrix."""
    np.random.seed(42)
    n_assets = 5
    n_obs = 500
    data = np.random.randn(n_obs, n_assets)
    # Add some correlation structure
    data[:, 1] = data[:, 0] * 0.8 + data[:, 1] * 0.6
    data[:, 3] = data[:, 2] * 0.7 + data[:, 3] * 0.7
    cov = pd.DataFrame(
        np.cov(data, rowvar=False),
        columns=[f"asset_{i}" for i in range(n_assets)],
        index=[f"asset_{i}" for i in range(n_assets)],
    )
    return cov


@pytest.fixture
def sample_corr(sample_cov) -> pd.DataFrame:
    """Correlation matrix derived from sample_cov."""
    std = np.sqrt(np.diag(sample_cov.values))
    corr = sample_cov.values / np.outer(std, std)
    return pd.DataFrame(corr, columns=sample_cov.columns, index=sample_cov.index)


@pytest.fixture
def large_cov() -> pd.DataFrame:
    """50x50 covariance matrix for denoising tests (high q = T/N ratio)."""
    np.random.seed(42)
    n_assets = 50
    n_obs = 500
    data = np.random.randn(n_obs, n_assets)
    # Add block correlation structure
    for i in range(0, n_assets, 10):
        block = data[:, i : i + 10]
        common = np.random.randn(n_obs, 1) * 0.5
        data[:, i : i + 10] = block + common
    cols = [f"asset_{i}" for i in range(n_assets)]
    return pd.DataFrame(np.cov(data, rowvar=False), columns=cols, index=cols)
```

**Step 4: Commit**

```bash
git add tests/allocation/ src/tradelab/lopezdp_utils/allocation/
git commit -m "test(allocation): add test skeleton and fixtures"
```

---

### Task 2: Migrate `hrp.py` — Hierarchical Risk Parity

**Files:**
- Create: `src/tradelab/lopezdp_utils/allocation/hrp.py`
- Create: `tests/allocation/test_hrp.py`

**Step 1: Write failing tests**

```python
"""Tests for allocation.hrp — Hierarchical Risk Parity."""

import numpy as np
import pandas as pd
import pytest


class TestCorrelDist:
    def test_returns_dataframe(self, sample_corr):
        from tradelab.lopezdp_utils.allocation.hrp import correl_dist

        result = correl_dist(sample_corr)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == sample_corr.shape

    def test_zero_diagonal(self, sample_corr):
        from tradelab.lopezdp_utils.allocation.hrp import correl_dist

        result = correl_dist(sample_corr)
        np.testing.assert_array_almost_equal(np.diag(result.values), 0.0)

    def test_perfect_corr_zero_dist(self):
        from tradelab.lopezdp_utils.allocation.hrp import correl_dist

        corr = pd.DataFrame(np.ones((3, 3)), columns=["a", "b", "c"], index=["a", "b", "c"])
        result = correl_dist(corr)
        assert (result.values == 0).all()


class TestTreeClustering:
    def test_returns_linkage_matrix(self, sample_corr):
        from tradelab.lopezdp_utils.allocation.hrp import correl_dist, tree_clustering

        dist = correl_dist(sample_corr)
        link = tree_clustering(dist)
        assert link.shape[1] == 4  # scipy linkage format
        assert link.shape[0] == len(sample_corr) - 1


class TestGetQuasiDiag:
    def test_returns_sorted_indices(self, sample_corr):
        from tradelab.lopezdp_utils.allocation.hrp import correl_dist, get_quasi_diag, tree_clustering

        dist = correl_dist(sample_corr)
        link = tree_clustering(dist)
        sort_ix = get_quasi_diag(link)
        assert len(sort_ix) == len(sample_corr)
        assert set(sort_ix) == set(range(len(sample_corr)))


class TestHRPAlloc:
    def test_weights_sum_to_one(self, sample_cov, sample_corr):
        from tradelab.lopezdp_utils.allocation.hrp import hrp_alloc

        weights = hrp_alloc(sample_cov, sample_corr)
        assert isinstance(weights, pd.Series)
        assert abs(weights.sum() - 1.0) < 1e-10

    def test_all_weights_positive(self, sample_cov, sample_corr):
        from tradelab.lopezdp_utils.allocation.hrp import hrp_alloc

        weights = hrp_alloc(sample_cov, sample_corr)
        assert (weights > 0).all()

    def test_weights_match_assets(self, sample_cov, sample_corr):
        from tradelab.lopezdp_utils.allocation.hrp import hrp_alloc

        weights = hrp_alloc(sample_cov, sample_corr)
        assert list(weights.index) == list(sample_cov.columns) or set(weights.index) == set(sample_cov.columns)
```

**Step 2: Run to verify fail**

Run: `uv run pytest tests/allocation/test_hrp.py -v`

**Step 3: Implement `allocation/hrp.py`**

Copy from `ml_asset_allocation/hrp.py`. Update module docstring. No API changes needed — keep pandas/numpy/scipy throughout.

**Step 4: Run tests**

Run: `uv run pytest tests/allocation/test_hrp.py -v`

**Step 5: Commit**

```bash
git add src/tradelab/lopezdp_utils/allocation/hrp.py tests/allocation/test_hrp.py
git commit -m "feat(allocation): migrate HRP from ml_asset_allocation"
```

---

### Task 3: Migrate `denoising.py` — RMT covariance cleaning

**Files:**
- Create: `src/tradelab/lopezdp_utils/allocation/denoising.py`
- Create: `tests/allocation/test_denoising.py`

**Step 1: Write failing tests**

```python
"""Tests for allocation.denoising — Random Matrix Theory covariance cleaning."""

import numpy as np
import pandas as pd
import pytest


class TestMPPdf:
    def test_returns_series(self):
        from tradelab.lopezdp_utils.allocation.denoising import mp_pdf

        result = mp_pdf(var=1.0, q=10.0, pts=100)
        assert isinstance(result, pd.Series)
        assert len(result) == 100

    def test_non_negative(self):
        from tradelab.lopezdp_utils.allocation.denoising import mp_pdf

        result = mp_pdf(var=1.0, q=10.0, pts=100)
        assert (result >= 0).all()


class TestFindMaxEval:
    def test_returns_reasonable_value(self, large_cov):
        from tradelab.lopezdp_utils.allocation.denoising import find_max_eval

        corr = large_cov.values / np.outer(np.sqrt(np.diag(large_cov.values)), np.sqrt(np.diag(large_cov.values)))
        evals = np.linalg.eigvalsh(corr)
        q = large_cov.shape[0] / 500  # T/N
        result, var = find_max_eval(np.diag(evals), q, bwidth=0.01)
        assert result > 0
        assert var > 0


class TestDenoiseCov:
    def test_returns_positive_definite(self, large_cov):
        from tradelab.lopezdp_utils.allocation.denoising import denoise_cov

        q = large_cov.shape[0] / 500
        result = denoise_cov(large_cov, q, bwidth=0.01)
        assert isinstance(result, pd.DataFrame)
        evals = np.linalg.eigvalsh(result.values)
        assert (evals > -1e-10).all()  # positive semi-definite

    def test_shape_preserved(self, large_cov):
        from tradelab.lopezdp_utils.allocation.denoising import denoise_cov

        q = large_cov.shape[0] / 500
        result = denoise_cov(large_cov, q, bwidth=0.01)
        assert result.shape == large_cov.shape


class TestDetoneCorr:
    def test_removes_market_component(self, large_cov):
        from tradelab.lopezdp_utils.allocation.denoising import detone_corr

        std = np.sqrt(np.diag(large_cov.values))
        corr = large_cov.values / np.outer(std, std)
        result = detone_corr(corr, n_remove=1)
        assert result.shape == corr.shape
        # First eigenvector contribution removed — average correlation should decrease
        assert np.mean(np.abs(result)) < np.mean(np.abs(corr))
```

**Step 2: Run to verify fail**

Run: `uv run pytest tests/allocation/test_denoising.py -v`

**Step 3: Implement `allocation/denoising.py`**

Copy from `ml_asset_allocation/denoising.py`. Update module docstring. No API changes.

**Step 4: Run tests**

Run: `uv run pytest tests/allocation/test_denoising.py -v`

**Step 5: Commit**

```bash
git add src/tradelab/lopezdp_utils/allocation/denoising.py tests/allocation/test_denoising.py
git commit -m "feat(allocation): migrate RMT denoising from ml_asset_allocation"
```

---

### Task 4: Migrate `nco.py` — Nested Clustered Optimization

**Files:**
- Create: `src/tradelab/lopezdp_utils/allocation/nco.py`
- Create: `tests/allocation/test_nco.py`

**Step 1: Write failing tests**

```python
"""Tests for allocation.nco — Nested Clustered Optimization."""

import numpy as np
import pandas as pd
import pytest


class TestOptPortNCO:
    def test_min_var_weights_sum_to_one(self, sample_cov):
        from tradelab.lopezdp_utils.allocation.nco import opt_port_nco

        weights = opt_port_nco(sample_cov, max_num_clusters=3, objective="minVar")
        assert isinstance(weights, pd.DataFrame)
        assert abs(weights.values.sum() - 1.0) < 1e-6

    def test_max_sharpe_with_mu(self, sample_cov):
        from tradelab.lopezdp_utils.allocation.nco import opt_port_nco

        mu = pd.DataFrame(
            np.random.randn(len(sample_cov), 1) * 0.01,
            index=sample_cov.index,
        )
        weights = opt_port_nco(sample_cov, mu=mu, max_num_clusters=3, objective="maxSharpe")
        assert isinstance(weights, pd.DataFrame)

    def test_weights_shape(self, sample_cov):
        from tradelab.lopezdp_utils.allocation.nco import opt_port_nco

        weights = opt_port_nco(sample_cov, max_num_clusters=3)
        assert weights.shape[0] == len(sample_cov)
```

**Step 2: Run to verify fail**

Run: `uv run pytest tests/allocation/test_nco.py -v`

**Step 3: Implement `allocation/nco.py`**

Copy from `ml_asset_allocation/nco.py`. **Critical fix:** Update import:
```python
# OLD (broken):
from tradelab.lopezdp_utils.feature_importance.clustering import cluster_kmeans_base
# NEW:
from tradelab.lopezdp_utils.features.importance import cluster_kmeans_top
```

Update call sites from `cluster_kmeans_base` to `cluster_kmeans_top`. Also update the internal import of `_cov2corr` to point to `allocation.denoising._cov2corr`.

**Step 4: Run tests**

Run: `uv run pytest tests/allocation/test_nco.py -v`

**Step 5: Commit**

```bash
git add src/tradelab/lopezdp_utils/allocation/nco.py tests/allocation/test_nco.py
git commit -m "feat(allocation): migrate NCO with updated imports from features/"
```

---

### Task 5: Migrate `simulation.py` — Monte Carlo comparison

**Files:**
- Create: `src/tradelab/lopezdp_utils/allocation/simulation.py`
- Create: `tests/allocation/test_simulation.py`

**Step 1: Write failing tests**

```python
"""Tests for allocation.simulation — synthetic data and MC experiments."""

import numpy as np
import pandas as pd
import pytest


class TestGenerateData:
    def test_returns_correct_shape(self):
        from tradelab.lopezdp_utils.allocation.simulation import generate_data

        mu, cols = generate_data(n_obs=100, size0=3, size1=3, sigma1=0.25)
        assert mu.shape == (100, 6)  # size0 + size1*size0/size0... depends on implementation
        assert isinstance(cols, list)

    def test_reproducible_with_seed(self):
        from tradelab.lopezdp_utils.allocation.simulation import generate_data

        np.random.seed(42)
        mu1, _ = generate_data(n_obs=100, size0=3, size1=3, sigma1=0.25)
        np.random.seed(42)
        mu2, _ = generate_data(n_obs=100, size0=3, size1=3, sigma1=0.25)
        np.testing.assert_array_equal(mu1, mu2)


class TestHRPMC:
    def test_returns_dataframe(self):
        from tradelab.lopezdp_utils.allocation.simulation import hrp_mc

        result = hrp_mc(
            n_iter=5,
            n_obs=100,
            size0=3,
            size1=3,
            sigma1=0.25,
            min_var_portf=True,
        )
        assert isinstance(result, pd.DataFrame)
        assert "hrp" in result.columns or len(result.columns) >= 1

    def test_hrp_variance_reasonable(self):
        from tradelab.lopezdp_utils.allocation.simulation import hrp_mc

        result = hrp_mc(n_iter=10, n_obs=200, size0=3, size1=3, sigma1=0.25)
        # All variance values should be positive
        assert (result > 0).all().all()
```

**Step 2: Run to verify fail**

Run: `uv run pytest tests/allocation/test_simulation.py -v`

**Step 3: Implement `allocation/simulation.py`**

Copy from `ml_asset_allocation/simulation.py`. Update internal imports:
```python
# OLD:
from tradelab.lopezdp_utils.ml_asset_allocation.hrp import get_ivp, hrp_alloc
# NEW:
from tradelab.lopezdp_utils.allocation.hrp import get_ivp, hrp_alloc
```

**Step 4: Run tests**

Run: `uv run pytest tests/allocation/test_simulation.py -v`

**Step 5: Commit**

```bash
git add src/tradelab/lopezdp_utils/allocation/simulation.py tests/allocation/test_simulation.py
git commit -m "feat(allocation): migrate simulation MC harness"
```

---

### Task 6: Create `allocation/__init__.py` with public exports

```python
"""Portfolio allocation — HRP, denoising, NCO — AFML Ch. 16 + MLAM Sections 2, 7.

This package implements López de Prado's portfolio construction methods:
- Hierarchical Risk Parity (HRP): graph-theory-based allocation avoiding
  covariance matrix inversion
- Random Matrix Theory denoising: Marcenko-Pastur-based cleaning of empirical
  covariance matrices
- Nested Clustered Optimization (NCO): hierarchical optimization with ONC
  clustering from features/

Reference:
    López de Prado, "Advances in Financial Machine Learning", Chapter 16
    López de Prado, "Machine Learning for Asset Managers", Sections 2.4–2.9, 7.3–7.6
"""

from tradelab.lopezdp_utils.allocation.denoising import (
    denoise_cov,
    denoised_corr,
    denoised_corr_shrinkage,
    detone_corr,
    find_max_eval,
    mp_pdf,
)
from tradelab.lopezdp_utils.allocation.hrp import (
    correl_dist,
    get_cluster_var,
    get_ivp,
    get_quasi_diag,
    get_rec_bipart,
    hrp_alloc,
    tree_clustering,
)
from tradelab.lopezdp_utils.allocation.nco import opt_port_nco
from tradelab.lopezdp_utils.allocation.simulation import generate_data, hrp_mc

__all__ = [
    # HRP
    "correl_dist",
    "tree_clustering",
    "get_quasi_diag",
    "get_ivp",
    "get_cluster_var",
    "get_rec_bipart",
    "hrp_alloc",
    # Denoising
    "mp_pdf",
    "find_max_eval",
    "denoised_corr",
    "denoised_corr_shrinkage",
    "denoise_cov",
    "detone_corr",
    # NCO
    "opt_port_nco",
    # Simulation
    "generate_data",
    "hrp_mc",
]
```

---

### Task 7: Integration test — allocation pipeline

**Files:**
- Create: `tests/allocation/test_integration.py`

```python
"""Integration tests: allocation pipeline."""

import numpy as np
import pandas as pd
import pytest


class TestDenoiseThenHRP:
    """Denoise covariance → HRP allocation."""

    def test_denoised_hrp_weights_sum_to_one(self, large_cov):
        from tradelab.lopezdp_utils.allocation.denoising import denoise_cov
        from tradelab.lopezdp_utils.allocation.hrp import hrp_alloc

        q = large_cov.shape[0] / 500
        clean_cov = denoise_cov(large_cov, q, bwidth=0.01)
        std = np.sqrt(np.diag(clean_cov.values))
        clean_corr = pd.DataFrame(
            clean_cov.values / np.outer(std, std),
            columns=clean_cov.columns,
            index=clean_cov.index,
        )
        weights = hrp_alloc(clean_cov, clean_corr)
        assert abs(weights.sum() - 1.0) < 1e-10
        assert (weights > 0).all()


class TestDenoiseThenNCO:
    """Denoise covariance → NCO allocation."""

    def test_denoised_nco_weights(self, large_cov):
        from tradelab.lopezdp_utils.allocation.denoising import denoise_cov
        from tradelab.lopezdp_utils.allocation.nco import opt_port_nco

        q = large_cov.shape[0] / 500
        clean_cov = denoise_cov(large_cov, q, bwidth=0.01)
        weights = opt_port_nco(clean_cov, max_num_clusters=5, objective="minVar")
        assert abs(weights.values.sum() - 1.0) < 1e-6
```

**Steps: Write → Run → Verify pass → Commit**

```bash
git commit -m "test(allocation): add integration tests for allocation pipeline"
```

---

### Task 8: Delete old directories and verify

**Step 1: Delete old modules**

```bash
rm -rf src/tradelab/lopezdp_utils/ml_asset_allocation/
rm -rf src/tradelab/lopezdp_utils/hpc/
```

**Step 2: Verify imports**

Run: `uv run python -c "from tradelab.lopezdp_utils.allocation import hrp_alloc, denoise_cov, opt_port_nco, hrp_mc; print('OK')"`

**Step 3: Run full test suite**

Run: `uv run pytest tests/ -v`

**Step 4: Lint**

Run: `uvx ruff check --fix . && uvx ruff format .`

**Step 5: Commit**

```bash
git add -A
git commit -m "refactor(allocation): remove old ml_asset_allocation/ and hpc/"
```

---

### Task 9: Merge to main

Run: `git checkout main && git merge phase2/allocation`
Verify: `uv run pytest tests/ -v`
