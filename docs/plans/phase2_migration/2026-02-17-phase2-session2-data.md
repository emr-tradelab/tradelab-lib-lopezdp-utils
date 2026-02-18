# Phase 2 Session 2: `data/` — Data Layer

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Migrate `data_structures/` (9 files, 1888 lines) + `microstructure/` (5 files, 568 lines) into `data/` package (5 files). Polars for tabular ops, NumPy for math.

**Architecture:** Merge 3 bar files into `bars.py`, keep `sampling.py`/`futures.py`/`etf.py` as separate files, merge 4 microstructure files into `microstructure.py`. Move `discretization.py` and `pca.py` out to `features/` (session 4) — leave forwarding stubs or just delete them now. Public API accepts/returns Polars DataFrames.

**Tech Stack:** polars, numpy, scipy, sklearn (for microstructure regressions), pydantic, pytest

**Depends on:** Session 1 (`_hpc.py` merged to main)

---

## Pre-Session Checklist

- [ ] Session 1 merged to main
- [ ] Branch from main: `git checkout -b phase2/data`
- [ ] `uv sync --all-extras --dev`

---

## File Mapping: Old → New

| Old file | New location | Action |
|----------|-------------|--------|
| `data_structures/standard_bars.py` | `data/bars.py` | Merge |
| `data_structures/imbalance_bars.py` | `data/bars.py` | Merge |
| `data_structures/runs_bars.py` | `data/bars.py` | Merge |
| `data_structures/sampling.py` | `data/sampling.py` | Move + migrate |
| `data_structures/futures.py` | `data/futures.py` | Move + migrate |
| `data_structures/etf.py` | `data/etf.py` | Move (low-priority Polars) |
| `data_structures/discretization.py` | Deferred to session 4 (`features/entropy.py`) | Delete from here |
| `data_structures/pca.py` | Deferred to session 4 (`features/orthogonal.py`) | Delete from here |
| `microstructure/trade_classification.py` | `data/microstructure.py` | Merge |
| `microstructure/spread_estimators.py` | `data/microstructure.py` | Merge |
| `microstructure/price_impact.py` | `data/microstructure.py` | Merge |
| `microstructure/vpin.py` | `data/microstructure.py` | Merge |

---

## Polars Migration Decisions (per function)

### bars.py
| Function | Migration | Notes |
|----------|-----------|-------|
| `time_bars()` | **Polars** | `group_by_dynamic` + `agg` replaces `resample` |
| `tick_bars()` | **Polars output, Python loop** | Stateful accumulation loop stays Python; output as `pl.DataFrame` |
| `volume_bars()` | Same as tick_bars | |
| `dollar_bars()` | Same as tick_bars | |
| `tick_imbalance_bars()` | **Fix EWMA antipattern**, Polars output | Replace per-iteration `pd.Series().ewm()` with incremental EWMA formula |
| `volume_imbalance_bars()` | Same | |
| `dollar_imbalance_bars()` | Same | |
| `tick_runs_bars()` | Same pattern as imbalance | |
| `volume_runs_bars()` | Same | |
| `dollar_runs_bars()` | Same | |
| `_compute_tick_rule()` (private) | **Keep NumPy** | `np.sign` on list of floats |
| `_aggregate_bar()` (private) | **Keep Python** | Pure dict construction |

### sampling.py
| Function | Migration | Notes |
|----------|-----------|-------|
| `get_t_events()` | **Polars I/O, NumPy loop** | CUSUM loop is inherently sequential; accept/return Polars |
| `sampling_linspace()` | **Full Polars** | Trivial `df[::step]` equivalent |
| `sampling_uniform()` | **Polars output, NumPy RNG** | |

### futures.py
| Function | Migration | Notes |
|----------|-----------|-------|
| `roll_gaps()` | **Polars** | `unique`, `cum_sum` map well |
| `roll_and_rebase()` | **Polars** | `diff`, `shift`, `cum_prod` all native |
| `get_rolled_series()` | **Keep pandas** | `pd.read_hdf` has no Polars equivalent; convert to Polars at boundary |

### etf.py
| Function | Migration | Notes |
|----------|-----------|-------|
| `etf_trick()` | **Keep pandas internally** | Complex multi-level column loop; low priority. Polars at boundary only |

### microstructure.py
| Function | Migration | Notes |
|----------|-----------|-------|
| `tick_rule()` | **Full Polars** | `diff`, `forward_fill`, `when/then` |
| `roll_model()` | **Keep pandas** | `autocorr()` has no Polars equivalent |
| `high_low_volatility()` | **Keep NumPy** | Pure math |
| `get_beta()`, `get_gamma()`, `get_alpha()` | **Polars** | `rolling_mean`, arithmetic |
| `corwin_schultz_spread()` | **Polars** | Calls above + concat |
| `becker_parkinson_volatility()` | **Polars** | Calls beta/gamma |
| `kyle_lambda()` | **Mixed** | Polars for Series ops, NumPy arrays for sklearn |
| `amihud_lambda()` | **Polars** | `log`, `shift`, `mean` |
| `hasbrouck_lambda()` | **Mixed** | Same as kyle |
| `volume_bucket()` | **Python loop, Polars output** | Stateful split loop |
| `vpin()` | **Polars** | Rolling sum at end |

---

## Pydantic Validation

Input validation at public function boundaries:

```python
from pydantic import BaseModel, field_validator

class BarConfig(BaseModel):
    """Configuration for bar construction."""
    threshold: float  # tick count, volume, or dollar amount
    expected_ticks_init: int = 100  # for imbalance/run bars
    ewma_span: int = 20  # for imbalance/run bars

class CusumConfig(BaseModel):
    """Configuration for CUSUM event filter."""
    threshold: float  # h parameter
```

Functions validate that input DataFrames have required columns (e.g., `price`, `volume` for bars; `close` for CUSUM).

---

## Tasks

### Task 1: Create branch, package structure, and test skeleton

**Files:**
- Create: `src/tradelab/lopezdp_utils/data/__init__.py`
- Create: `tests/data/__init__.py`
- Create: `tests/data/conftest.py` (shared fixtures)
- Create: `tests/data/test_bars.py`
- Create: `tests/data/test_sampling.py`
- Create: `tests/data/test_microstructure.py`

**Step 1: Create branch**

Run: `git checkout -b phase2/data`

**Step 2: Create package directory**

Run: `mkdir -p src/tradelab/lopezdp_utils/data tests/data`

**Step 3: Create shared test fixtures in `tests/data/conftest.py`**

```python
"""Shared fixtures for data package tests."""

import polars as pl
import numpy as np
import pytest


@pytest.fixture
def ohlcv_1min() -> pl.DataFrame:
    """Synthetic 1-minute OHLCV data (100 bars)."""
    np.random.seed(42)
    n = 100
    timestamps = pl.datetime_range(
        pl.datetime(2024, 1, 1),
        pl.datetime(2024, 1, 1, 1, 39),  # 100 minutes
        interval="1m",
        eager=True,
    )
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.2)
    low = close - np.abs(np.random.randn(n) * 0.2)
    open_ = close + np.random.randn(n) * 0.1
    volume = np.random.randint(100, 10000, size=n).astype(float)

    return pl.DataFrame({
        "timestamp": timestamps,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


@pytest.fixture
def tick_data() -> pl.DataFrame:
    """Synthetic tick data (1000 ticks) for bar construction."""
    np.random.seed(42)
    n = 1000
    timestamps = pl.datetime_range(
        pl.datetime(2024, 1, 1),
        pl.datetime(2024, 1, 1, 0, 16, 39),  # ~1000 seconds
        interval="1s",
        eager=True,
    )
    prices = 100.0 + np.cumsum(np.random.randn(n) * 0.1)
    volumes = np.random.randint(1, 100, size=n).astype(float)

    return pl.DataFrame({
        "timestamp": timestamps,
        "price": prices,
        "volume": volumes,
    })


@pytest.fixture
def close_series() -> pl.DataFrame:
    """Close price series as a Polars DataFrame with timestamp index."""
    np.random.seed(42)
    n = 500
    timestamps = pl.datetime_range(
        pl.datetime(2024, 1, 1),
        pl.datetime(2024, 1, 1, 8, 19),  # 500 minutes
        interval="1m",
        eager=True,
    )
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.3)

    return pl.DataFrame({
        "timestamp": timestamps,
        "close": close,
    })
```

**Step 4: Create `tests/data/__init__.py`**

Empty file.

**Step 5: Run to verify fixtures load**

Run: `uv run pytest tests/data/ --collect-only`
Expected: No errors, 0 tests collected (no test functions yet)

**Step 6: Commit**

```bash
git add tests/data/ src/tradelab/lopezdp_utils/data/
git commit -m "test(data): add test skeleton and fixtures for data package"
```

---

### Task 2: Migrate `sampling.py` (CUSUM filter)

This is the most critical function for the user's 1-min time bar workflow.

**Files:**
- Create: `src/tradelab/lopezdp_utils/data/sampling.py`
- Create: `tests/data/test_sampling.py`

**Step 1: Write failing tests**

```python
"""Tests for data.sampling — CUSUM filter and sampling utilities."""

import numpy as np
import polars as pl
import pytest

from tests.data.conftest import *  # noqa: fixtures


class TestGetTEvents:
    """Tests for CUSUM event filter."""

    def test_returns_polars_series(self, close_series):
        from tradelab.lopezdp_utils.data.sampling import get_t_events

        events = get_t_events(close_series, threshold=1.0)
        assert isinstance(events, pl.Series)
        assert events.dtype == pl.Datetime

    def test_no_events_when_threshold_high(self, close_series):
        from tradelab.lopezdp_utils.data.sampling import get_t_events

        events = get_t_events(close_series, threshold=1000.0)
        assert len(events) == 0

    def test_more_events_with_lower_threshold(self, close_series):
        from tradelab.lopezdp_utils.data.sampling import get_t_events

        events_high = get_t_events(close_series, threshold=2.0)
        events_low = get_t_events(close_series, threshold=0.5)
        assert len(events_low) >= len(events_high)

    def test_events_are_subset_of_timestamps(self, close_series):
        from tradelab.lopezdp_utils.data.sampling import get_t_events

        events = get_t_events(close_series, threshold=1.0)
        all_ts = close_series["timestamp"]
        for ts in events:
            assert ts in all_ts

    def test_known_cusum_detection(self):
        """A price series with a known jump should trigger exactly at the jump."""
        from tradelab.lopezdp_utils.data.sampling import get_t_events

        # Flat at 100, then jump to 105 at index 50
        n = 100
        timestamps = pl.datetime_range(
            pl.datetime(2024, 1, 1),
            pl.datetime(2024, 1, 1, 1, 39),
            interval="1m",
            eager=True,
        )
        prices = np.full(n, 100.0)
        prices[50:] = 105.0
        df = pl.DataFrame({"timestamp": timestamps, "close": prices})

        events = get_t_events(df, threshold=3.0)
        assert len(events) >= 1
        # First event should be at or near the jump
        first_event = events[0]
        assert first_event == timestamps[50]


class TestSamplingLinspace:
    """Tests for linspace sampling."""

    def test_step_sampling(self, ohlcv_1min):
        from tradelab.lopezdp_utils.data.sampling import sampling_linspace

        result = sampling_linspace(ohlcv_1min, step=10)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 10  # 100 / 10

    def test_num_samples(self, ohlcv_1min):
        from tradelab.lopezdp_utils.data.sampling import sampling_linspace

        result = sampling_linspace(ohlcv_1min, num_samples=20)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 20


class TestSamplingUniform:
    """Tests for uniform random sampling."""

    def test_returns_correct_count(self, ohlcv_1min):
        from tradelab.lopezdp_utils.data.sampling import sampling_uniform

        result = sampling_uniform(ohlcv_1min, num_samples=25, random_state=42)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 25

    def test_reproducible_with_seed(self, ohlcv_1min):
        from tradelab.lopezdp_utils.data.sampling import sampling_uniform

        r1 = sampling_uniform(ohlcv_1min, num_samples=10, random_state=42)
        r2 = sampling_uniform(ohlcv_1min, num_samples=10, random_state=42)
        assert r1.frame_equal(r2)
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/data/test_sampling.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement `data/sampling.py`**

Migrate the three functions from `data_structures/sampling.py`:
- `get_t_events`: Accept `pl.DataFrame` with `timestamp` + `close` columns. CUSUM loop stays Python/NumPy (inherently sequential). Return `pl.Series` of detected event timestamps.
- `sampling_linspace`: Full Polars — `df.gather_every(step)` or equivalent.
- `sampling_uniform`: NumPy RNG for index selection, Polars for output.

Key implementation notes:
- `get_t_events` should extract close prices as NumPy array internally for the loop, then convert result back to `pl.Series`.
- Add `# TODO(numba): evaluate JIT for CUSUM loop` comment.

**Step 4: Run tests**

Run: `uv run pytest tests/data/test_sampling.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/tradelab/lopezdp_utils/data/sampling.py tests/data/test_sampling.py
git commit -m "feat(data): migrate sampling.py with CUSUM filter to Polars"
```

---

### Task 3: Migrate `bars.py` — standard bars

**Files:**
- Create: `src/tradelab/lopezdp_utils/data/bars.py`
- Create: `tests/data/test_bars.py`

**Step 1: Write failing tests for standard bars**

```python
"""Tests for data.bars — bar construction from tick/time data."""

import numpy as np
import polars as pl
import pytest


class TestTimeBars:
    """Tests for time bar aggregation."""

    def test_returns_polars_dataframe(self, tick_data):
        from tradelab.lopezdp_utils.data.bars import time_bars

        result = time_bars(tick_data, frequency="1m")
        assert isinstance(result, pl.DataFrame)

    def test_has_ohlcv_columns(self, tick_data):
        from tradelab.lopezdp_utils.data.bars import time_bars

        result = time_bars(tick_data, frequency="1m")
        for col in ["timestamp", "open", "high", "low", "close", "volume"]:
            assert col in result.columns

    def test_high_ge_low(self, tick_data):
        from tradelab.lopezdp_utils.data.bars import time_bars

        result = time_bars(tick_data, frequency="1m")
        assert (result["high"] >= result["low"]).all()

    def test_volume_positive(self, tick_data):
        from tradelab.lopezdp_utils.data.bars import time_bars

        result = time_bars(tick_data, frequency="1m")
        assert (result["volume"] > 0).all()


class TestTickBars:
    """Tests for tick bars."""

    def test_returns_polars_dataframe(self, tick_data):
        from tradelab.lopezdp_utils.data.bars import tick_bars

        result = tick_bars(tick_data, threshold=50)
        assert isinstance(result, pl.DataFrame)

    def test_bar_count(self, tick_data):
        """1000 ticks / 50 per bar = ~20 bars."""
        from tradelab.lopezdp_utils.data.bars import tick_bars

        result = tick_bars(tick_data, threshold=50)
        assert 15 <= len(result) <= 25

    def test_has_ohlcv_columns(self, tick_data):
        from tradelab.lopezdp_utils.data.bars import tick_bars

        result = tick_bars(tick_data, threshold=50)
        for col in ["timestamp", "open", "high", "low", "close", "volume"]:
            assert col in result.columns


class TestVolumeBars:
    """Tests for volume bars."""

    def test_returns_polars_dataframe(self, tick_data):
        from tradelab.lopezdp_utils.data.bars import volume_bars

        result = volume_bars(tick_data, threshold=2500.0)
        assert isinstance(result, pl.DataFrame)

    def test_lower_threshold_more_bars(self, tick_data):
        from tradelab.lopezdp_utils.data.bars import volume_bars

        bars_high = volume_bars(tick_data, threshold=5000.0)
        bars_low = volume_bars(tick_data, threshold=1000.0)
        assert len(bars_low) > len(bars_high)


class TestDollarBars:
    """Tests for dollar bars."""

    def test_returns_polars_dataframe(self, tick_data):
        from tradelab.lopezdp_utils.data.bars import dollar_bars

        result = dollar_bars(tick_data, threshold=250000.0)
        assert isinstance(result, pl.DataFrame)

    def test_has_ohlcv_columns(self, tick_data):
        from tradelab.lopezdp_utils.data.bars import dollar_bars

        result = dollar_bars(tick_data, threshold=250000.0)
        for col in ["timestamp", "open", "high", "low", "close", "volume"]:
            assert col in result.columns
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/data/test_bars.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement standard bars in `data/bars.py`**

Implementation notes:
- `time_bars()`: Use Polars `group_by_dynamic("timestamp", every=frequency)` with `agg` for OHLCV. This is the cleanest Polars migration.
- `tick_bars()`, `volume_bars()`, `dollar_bars()`: The accumulation loop stays Python. Accept `pl.DataFrame`, convert to rows iterator or NumPy arrays internally, build bars in a list, return `pl.DataFrame`.
- `_aggregate_bar()`: Private helper, pure Python dict construction, keep as-is.
- Input validation: require `price` and `volume` columns (or `timestamp`+`price`+`volume`).

**Step 4: Run tests**

Run: `uv run pytest tests/data/test_bars.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/tradelab/lopezdp_utils/data/bars.py tests/data/test_bars.py
git commit -m "feat(data): migrate standard bars (time, tick, volume, dollar) to Polars"
```

---

### Task 4: Add imbalance and run bars to `bars.py`

**Files:**
- Modify: `src/tradelab/lopezdp_utils/data/bars.py`
- Modify: `tests/data/test_bars.py`

**Step 1: Write failing tests for imbalance bars**

Add to `tests/data/test_bars.py`:

```python
class TestTickImbalanceBars:
    """Tests for tick imbalance bars."""

    def test_returns_polars_dataframe(self, tick_data):
        from tradelab.lopezdp_utils.data.bars import tick_imbalance_bars

        result = tick_imbalance_bars(tick_data, expected_ticks_init=50)
        assert isinstance(result, pl.DataFrame)

    def test_has_ohlcv_columns(self, tick_data):
        from tradelab.lopezdp_utils.data.bars import tick_imbalance_bars

        result = tick_imbalance_bars(tick_data, expected_ticks_init=50)
        for col in ["timestamp", "open", "high", "low", "close", "volume"]:
            assert col in result.columns

    def test_produces_bars(self, tick_data):
        from tradelab.lopezdp_utils.data.bars import tick_imbalance_bars

        result = tick_imbalance_bars(tick_data, expected_ticks_init=50)
        assert len(result) > 0


class TestVolumeImbalanceBars:
    def test_returns_polars_dataframe(self, tick_data):
        from tradelab.lopezdp_utils.data.bars import volume_imbalance_bars

        result = volume_imbalance_bars(tick_data, expected_ticks_init=50)
        assert isinstance(result, pl.DataFrame)

    def test_produces_bars(self, tick_data):
        from tradelab.lopezdp_utils.data.bars import volume_imbalance_bars

        result = volume_imbalance_bars(tick_data, expected_ticks_init=50)
        assert len(result) > 0


class TestTickRunsBars:
    def test_returns_polars_dataframe(self, tick_data):
        from tradelab.lopezdp_utils.data.bars import tick_runs_bars

        result = tick_runs_bars(tick_data, expected_ticks_init=50)
        assert isinstance(result, pl.DataFrame)

    def test_produces_bars(self, tick_data):
        from tradelab.lopezdp_utils.data.bars import tick_runs_bars

        result = tick_runs_bars(tick_data, expected_ticks_init=50)
        assert len(result) > 0
```

**Step 2: Run to verify they fail**

Run: `uv run pytest tests/data/test_bars.py::TestTickImbalanceBars -v`
Expected: FAIL

**Step 3: Implement imbalance and run bars**

Key refactoring:
- **Fix the EWMA antipattern**: Replace `pd.Series([...]).ewm(span=...).mean().iloc[-1]` with an incremental formula: `ewma = alpha * new_val + (1 - alpha) * ewma` where `alpha = 2 / (span + 1)`. This is a significant performance improvement.
- Deduplicate `_compute_tick_rule()` — appears in both `imbalance_bars.py` and `runs_bars.py`. Single private function.
- Accept `pl.DataFrame`, extract NumPy arrays for the loop, return `pl.DataFrame`.
- Add `# TODO(numba): evaluate JIT for imbalance bar inner loop`.

**Step 4: Run all bar tests**

Run: `uv run pytest tests/data/test_bars.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/tradelab/lopezdp_utils/data/bars.py tests/data/test_bars.py
git commit -m "feat(data): add imbalance and run bars with fixed EWMA"
```

---

### Task 5: Migrate `futures.py`

**Files:**
- Create: `src/tradelab/lopezdp_utils/data/futures.py`
- Create: `tests/data/test_futures.py`

**Step 1: Write failing tests**

```python
"""Tests for data.futures — roll-adjusted continuous contracts."""

import numpy as np
import polars as pl
import pytest


class TestRollGaps:
    def test_returns_polars_series(self):
        from tradelab.lopezdp_utils.data.futures import roll_gaps

        # Simulate two contracts overlapping
        df = pl.DataFrame({
            "timestamp": pl.datetime_range(
                pl.datetime(2024, 1, 1), pl.datetime(2024, 1, 10),
                interval="1d", eager=True,
            ),
            "close": [100.0, 101.0, 102.0, 103.0, 104.0,
                      204.0, 205.0, 206.0, 207.0, 208.0],
        })
        result = roll_gaps(df)
        assert isinstance(result, pl.Series)


class TestRollAndRebase:
    def test_returns_polars_dataframe(self):
        from tradelab.lopezdp_utils.data.futures import roll_and_rebase

        df = pl.DataFrame({
            "timestamp": pl.datetime_range(
                pl.datetime(2024, 1, 1), pl.datetime(2024, 1, 5),
                interval="1d", eager=True,
            ),
            "close": [100.0, 101.0, 102.0, 103.0, 104.0],
        })
        result = roll_and_rebase(df)
        assert isinstance(result, pl.DataFrame)
```

**Step 2: Run to verify fail, Step 3: Implement, Step 4: Run tests, Step 5: Commit**

Notes:
- `roll_gaps`: Polars `unique`, `cum_sum`. Keep `get_rolled_series` as pandas-based (HDF5 reader) with conversion to Polars at output.
- `roll_and_rebase`: Polars `diff`, `shift`, `cum_prod`.

```bash
git commit -m "feat(data): migrate futures.py with roll adjustment to Polars"
```

---

### Task 6: Migrate `etf.py`

**Files:**
- Create: `src/tradelab/lopezdp_utils/data/etf.py`
- Create: `tests/data/test_etf.py`

Low-priority migration. Keep pandas internally due to complex multi-level column operations. Accept Polars at boundary, convert internally.

**Step 1: Write minimal test**

```python
"""Tests for data.etf — ETF trick."""

import polars as pl
import pytest


class TestEtfTrick:
    def test_returns_polars_series(self):
        from tradelab.lopezdp_utils.data.etf import etf_trick

        # Minimal smoke test with 2 assets, 5 days
        prices = pl.DataFrame({
            "timestamp": pl.datetime_range(
                pl.datetime(2024, 1, 1), pl.datetime(2024, 1, 5),
                interval="1d", eager=True,
            ),
            "asset_a": [100.0, 101.0, 102.0, 103.0, 104.0],
            "asset_b": [50.0, 50.5, 51.0, 50.5, 51.5],
        })
        weights = pl.DataFrame({
            "timestamp": pl.datetime_range(
                pl.datetime(2024, 1, 1), pl.datetime(2024, 1, 5),
                interval="1d", eager=True,
            ),
            "asset_a": [0.6, 0.6, 0.6, 0.6, 0.6],
            "asset_b": [0.4, 0.4, 0.4, 0.4, 0.4],
        })
        result = etf_trick(prices, weights)
        assert isinstance(result, pl.Series)
        assert len(result) == 5
```

**Steps 2-5: Standard TDD cycle + commit**

```bash
git commit -m "feat(data): migrate etf.py with Polars boundaries"
```

---

### Task 7: Migrate `microstructure.py`

**Files:**
- Create: `src/tradelab/lopezdp_utils/data/microstructure.py`
- Create: `tests/data/test_microstructure.py`

**Step 1: Write failing tests**

```python
"""Tests for data.microstructure — spread estimators, price impact, VPIN."""

import numpy as np
import polars as pl
import pytest


class TestTickRule:
    def test_returns_polars_series(self, ohlcv_1min):
        from tradelab.lopezdp_utils.data.microstructure import tick_rule

        result = tick_rule(ohlcv_1min["close"])
        assert isinstance(result, pl.Series)
        assert len(result) == len(ohlcv_1min)

    def test_values_are_plus_minus_one(self, ohlcv_1min):
        from tradelab.lopezdp_utils.data.microstructure import tick_rule

        result = tick_rule(ohlcv_1min["close"])
        unique_vals = result.unique().sort().to_list()
        assert all(v in [-1, 1] for v in unique_vals)

    def test_known_sequence(self):
        from tradelab.lopezdp_utils.data.microstructure import tick_rule

        prices = pl.Series("close", [100.0, 101.0, 101.0, 99.0, 100.0])
        result = tick_rule(prices)
        # 101>100 → +1, 101==101 → +1 (ffill), 99<101 → -1, 100>99 → +1
        expected = [1, 1, 1, -1, 1]
        assert result.to_list() == expected


class TestCorwinSchultzSpread:
    def test_returns_polars_dataframe(self, ohlcv_1min):
        from tradelab.lopezdp_utils.data.microstructure import corwin_schultz_spread

        result = corwin_schultz_spread(ohlcv_1min)
        assert isinstance(result, pl.DataFrame)
        assert "spread" in result.columns

    def test_spread_non_negative(self, ohlcv_1min):
        from tradelab.lopezdp_utils.data.microstructure import corwin_schultz_spread

        result = corwin_schultz_spread(ohlcv_1min)
        # After clipping, spreads should be >= 0
        assert (result["spread"].drop_nulls() >= 0).all()


class TestRollModel:
    def test_returns_dict(self, ohlcv_1min):
        from tradelab.lopezdp_utils.data.microstructure import roll_model

        result = roll_model(ohlcv_1min["close"])
        assert isinstance(result, dict)
        assert "spread" in result


class TestKyleLambda:
    def test_returns_dict(self, ohlcv_1min):
        from tradelab.lopezdp_utils.data.microstructure import kyle_lambda

        signs = pl.Series("signs", np.random.choice([-1, 1], size=len(ohlcv_1min)))
        result = kyle_lambda(
            ohlcv_1min["close"],
            ohlcv_1min["volume"],
            signs,
        )
        assert isinstance(result, dict)
        assert "lambda" in result


class TestAmihudLambda:
    def test_returns_float(self, ohlcv_1min):
        from tradelab.lopezdp_utils.data.microstructure import amihud_lambda

        dollar_vol = ohlcv_1min["close"] * ohlcv_1min["volume"]
        result = amihud_lambda(ohlcv_1min["close"], dollar_vol)
        assert isinstance(result, float)
        assert result >= 0


class TestVolumeBucket:
    def test_returns_polars_dataframe(self, ohlcv_1min):
        from tradelab.lopezdp_utils.data.microstructure import volume_bucket

        result = volume_bucket(
            ohlcv_1min["close"],
            ohlcv_1min["volume"],
            bucket_size=50000.0,
        )
        assert isinstance(result, pl.DataFrame)
        assert len(result) > 0


class TestVpin:
    def test_returns_polars_series(self, ohlcv_1min):
        from tradelab.lopezdp_utils.data.microstructure import vpin

        result = vpin(
            ohlcv_1min["close"],
            ohlcv_1min["volume"],
            bucket_size=50000.0,
            n_buckets=5,
        )
        assert isinstance(result, pl.Series)

    def test_vpin_between_zero_and_one(self, ohlcv_1min):
        from tradelab.lopezdp_utils.data.microstructure import vpin

        result = vpin(
            ohlcv_1min["close"],
            ohlcv_1min["volume"],
            bucket_size=50000.0,
            n_buckets=5,
        )
        non_null = result.drop_nulls()
        if len(non_null) > 0:
            assert (non_null >= 0).all()
            assert (non_null <= 1).all()
```

**Steps 2-5: Standard TDD cycle**

Implementation notes:
- `tick_rule()`: Full Polars — `diff`, `when/then/otherwise`, `forward_fill`.
- `get_beta/get_gamma/get_alpha/corwin_schultz_spread`: Polars rolling operations.
- `roll_model`: Keep pandas internally (needs `autocorr()`). Accept Polars, convert.
- `kyle_lambda/hasbrouck_lambda`: Mixed — Polars for Series prep, NumPy for sklearn regression.
- `amihud_lambda`: Full Polars — `log`, `shift`, `mean`.
- `volume_bucket`: Python loop (stateful), Polars output.
- `vpin`: Polars rolling sum at end.

```bash
git commit -m "feat(data): migrate microstructure (spreads, impact, VPIN) to Polars"
```

---

### Task 8: Create `data/__init__.py` with public exports

**Files:**
- Create: `src/tradelab/lopezdp_utils/data/__init__.py`

```python
"""Data layer — bar construction, sampling, and microstructure features.

This package handles the first stage of López de Prado's pipeline:
raw market data → structured bars → event-driven sampling.

Reference:
    AFML Chapters 2, 19
"""

from tradelab.lopezdp_utils.data.bars import (
    dollar_bars,
    dollar_imbalance_bars,
    dollar_runs_bars,
    tick_bars,
    tick_imbalance_bars,
    tick_runs_bars,
    time_bars,
    volume_bars,
    volume_imbalance_bars,
    volume_runs_bars,
)
from tradelab.lopezdp_utils.data.sampling import (
    get_t_events,
    sampling_linspace,
    sampling_uniform,
)

__all__ = [
    # Standard bars
    "time_bars",
    "tick_bars",
    "volume_bars",
    "dollar_bars",
    # Information-driven bars
    "tick_imbalance_bars",
    "volume_imbalance_bars",
    "dollar_imbalance_bars",
    "tick_runs_bars",
    "volume_runs_bars",
    "dollar_runs_bars",
    # Sampling
    "get_t_events",
    "sampling_linspace",
    "sampling_uniform",
]
```

Note: `futures`, `etf`, and `microstructure` are available via direct import (`from tradelab.lopezdp_utils.data.futures import ...`) but not re-exported at the package level — they are secondary utilities.

---

### Task 9: Delete old directories and verify

**Step 1: Delete old modules**

Run:
```bash
rm -rf src/tradelab/lopezdp_utils/data_structures/
rm -rf src/tradelab/lopezdp_utils/microstructure/
```

Note: `discretization.py` and `pca.py` contents will be needed in session 4. Before deleting, ensure their code is captured in the git history (it is — Phase 1 commits).

**Step 2: Verify all imports work**

Run: `uv run python -c "from tradelab.lopezdp_utils.data import time_bars, get_t_events; print('OK')"`

**Step 3: Run full test suite**

Run: `uv run pytest tests/ -v`

**Step 4: Lint**

Run: `uvx ruff check --fix . && uvx ruff format .`

**Step 5: Commit**

```bash
git add -A
git commit -m "refactor(data): remove old data_structures/ and microstructure/ directories"
```

---

### Task 10: Merge to main

Run: `git checkout main && git merge phase2/data`
Verify: `uv run pytest tests/ -v`
