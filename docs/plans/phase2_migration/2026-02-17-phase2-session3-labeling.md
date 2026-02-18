# Phase 2 Session 3: `labeling/` — Labeling + Sample Weights

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Migrate `labeling/` (8 files, 1088 lines) + `sample_weights/` (5 files, 922 lines) into `labeling/` package (4 files). This is the heart of López de Prado's pipeline — t1 timestamps are first-class citizens.

**Architecture:** Merge threshold/barrier/fixed-horizon/trend-scanning into `triple_barrier.py`. Merge all sample weight functions into `sample_weights.py`. Move `bet_sizing.py` to evaluation (session 6). Move `strategy_redundancy.py` to evaluation (session 6). Enforce t1 non-null validation at all boundaries.

**Tech Stack:** polars, numpy, scipy.stats, statsmodels (OLS for trend scanning), sklearn (class weights), pydantic, pytest

**Depends on:** Session 2 (`data/` merged to main — provides `get_t_events`, `daily_volatility` concepts)

---

## Pre-Session Checklist

- [ ] Session 2 merged to main
- [ ] Branch from main: `git checkout -b phase2/labeling`
- [ ] `uv sync --all-extras --dev`

---

## File Mapping: Old → New

| Old file | New location | Action |
|----------|-------------|--------|
| `labeling/triple_barrier.py` | `labeling/triple_barrier.py` | Migrate + merge |
| `labeling/meta_labeling.py` | `labeling/meta_labeling.py` | Migrate |
| `labeling/thresholds.py` | `labeling/triple_barrier.py` | Merge into |
| `labeling/barriers.py` | `labeling/triple_barrier.py` | Merge into |
| `labeling/fixed_horizon.py` | `labeling/triple_barrier.py` | Merge into |
| `labeling/trend_scanning.py` | `labeling/triple_barrier.py` | Merge into |
| `labeling/class_balance.py` | `labeling/class_balance.py` | Migrate |
| `labeling/bet_sizing.py` | Deferred to session 6 (`evaluation/bet_sizing.py`) | Delete from here |
| `sample_weights/concurrency.py` | `labeling/sample_weights.py` | Merge |
| `sample_weights/sequential_bootstrap.py` | `labeling/sample_weights.py` | Merge |
| `sample_weights/return_attribution.py` | `labeling/sample_weights.py` | Merge |
| `sample_weights/class_weights.py` | `labeling/class_balance.py` | Merge |
| `sample_weights/strategy_redundancy.py` | Deferred to session 6 (`evaluation/overfitting.py`) | Delete from here |

---

## Polars Migration Decisions (per function)

### triple_barrier.py
| Function | Migration | Notes |
|----------|-----------|-------|
| `daily_volatility()` | **Polars** | `ewm_std` + temporal join for day-prior lookup |
| `add_vertical_barrier()` | **Polars** | Index arithmetic, produces t1 |
| `fixed_time_horizon()` | **Full Polars** | `shift`, `when/then/otherwise` |
| `get_events()` | **Polars frame setup, Python loop core** | Frame construction is Polars; `apply_pt_sl_on_t1` loop stays Python |
| `apply_pt_sl_on_t1()` | **Keep Python loop** | Path-dependent per-event slicing (`close[loc:t1]`) is inherently sequential. `# TODO(numba): evaluate JIT for barrier loop` |
| `get_bins()` | **Polars** | Return computation from t1 lookups |
| `triple_barrier_labels()` | **Polars** | Wrapper orchestrating the above |
| `t_value_linear_trend()` | **Keep NumPy/statsmodels** | Pure OLS regression |
| `trend_scanning_labels()` | **Python loop, Polars output** | OLS per event per horizon — sequential. Outputs t1 column |

### meta_labeling.py
| Function | Migration | Notes |
|----------|-----------|-------|
| `get_events_meta()` | **Same as get_events** | Shares `apply_pt_sl_on_t1` |
| `get_bins_meta()` | **Polars** | Return + side computation |

### sample_weights.py
| Function | Migration | Notes |
|----------|-----------|-------|
| `mp_num_co_events()` | **Python loop, Polars I/O** | Range-accumulation pattern (`loc[t_in:t_out] += 1`) is hard to vectorize. Accept/return Polars. |
| `mp_sample_tw()` | **Python loop, Polars I/O** | Per-event mean over time window |
| `get_ind_matrix()` | **Keep NumPy** | Builds wide binary matrix — array construction, not tabular |
| `get_avg_uniqueness()` | **Polars or NumPy** | `sum(axis=1)`, `div`, `mean` — straightforward either way. Keep NumPy since input is already a matrix |
| `seq_bootstrap()` | **Keep Python** | Inherently sequential by design. `# TODO(numba): evaluate JIT for sequential bootstrap probability update` |
| `mp_sample_w()` | **Python loop, Polars I/O** | Per-event window sum of attributed returns |
| `get_time_decay()` | **Polars** | `cum_sum`, arithmetic, `clip` |

### class_balance.py
| Function | Migration | Notes |
|----------|-----------|-------|
| `drop_labels()` | **Full Polars** | `value_counts`, filter |
| `get_class_weights()` | **Keep sklearn** | Pass-through to `compute_class_weight` |

---

## Pydantic Validation

```python
from pydantic import BaseModel, field_validator
import polars as pl


class TripleBarrierConfig(BaseModel):
    """Configuration for triple-barrier labeling."""
    model_config = {"arbitrary_types_allowed": True}

    profit_taking: float  # multiplier on volatility for upper barrier
    stop_loss: float  # multiplier on volatility for lower barrier (positive value)
    max_holding_bars: int  # vertical barrier in number of bars
    vol_span: int = 100  # EWMA span for volatility estimation
    min_ret: float = 0.0  # minimum return threshold for labeling

    @field_validator("profit_taking", "stop_loss")
    @classmethod
    def must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("Barrier multipliers must be positive")
        return v


class SampleWeightConfig(BaseModel):
    """Configuration for sample weight computation."""
    decay_factor: float = 1.0  # clf_last_w parameter for time decay (1.0 = no decay)
    method: str = "uniqueness"  # "uniqueness", "return_attribution", or "both"
```

**Critical validation at function boundaries:**
- Any function accepting `events: pl.DataFrame` must validate that `t1` column exists and has no nulls
- Any function accepting `close` must validate it's sorted by timestamp
- `get_events` must validate that `t_events` timestamps are a subset of `close` timestamps

---

## Tasks

### Task 1: Create branch, package structure, and test fixtures

**Files:**
- Create: `src/tradelab/lopezdp_utils/labeling/__init__.py` (new version)
- Create: `tests/labeling/__init__.py`
- Create: `tests/labeling/conftest.py`

**Step 1: Create branch**

Run: `git checkout -b phase2/labeling`

**Step 2: Create directories**

Run: `mkdir -p tests/labeling`

**Step 3: Create shared fixtures**

```python
"""Shared fixtures for labeling package tests."""

import numpy as np
import polars as pl
import pytest


@pytest.fixture
def close_prices() -> pl.DataFrame:
    """500-bar close price series with realistic random walk."""
    np.random.seed(42)
    n = 500
    timestamps = pl.datetime_range(
        pl.datetime(2024, 1, 1),
        pl.datetime(2024, 1, 1, 8, 19),
        interval="1m",
        eager=True,
    )
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.3)
    return pl.DataFrame({"timestamp": timestamps, "close": close})


@pytest.fixture
def close_with_trend() -> pl.DataFrame:
    """Price series with a clear uptrend then downtrend for labeling tests."""
    np.random.seed(42)
    n = 200
    timestamps = pl.datetime_range(
        pl.datetime(2024, 1, 1),
        pl.datetime(2024, 1, 1, 3, 19),
        interval="1m",
        eager=True,
    )
    # Up for 100 bars, down for 100 bars
    up = np.linspace(100, 110, 100) + np.random.randn(100) * 0.1
    down = np.linspace(110, 95, 100) + np.random.randn(100) * 0.1
    close = np.concatenate([up, down])
    return pl.DataFrame({"timestamp": timestamps, "close": close})


@pytest.fixture
def events_with_t1(close_prices) -> pl.DataFrame:
    """Pre-built events DataFrame with t1 column for weight tests."""
    np.random.seed(42)
    # Sample 50 events from the close prices
    event_indices = np.sort(np.random.choice(400, size=50, replace=False))
    timestamps = close_prices["timestamp"]
    t0 = timestamps.gather(event_indices)
    # t1 = t0 + random 5-20 bars
    offsets = np.random.randint(5, 20, size=50)
    t1_indices = np.minimum(event_indices + offsets, 499)
    t1 = timestamps.gather(t1_indices)
    labels = np.random.choice([-1, 0, 1], size=50)

    return pl.DataFrame({
        "timestamp": t0,
        "t1": t1,
        "label": labels,
    })
```

**Step 4: Commit**

```bash
git add tests/labeling/ src/tradelab/lopezdp_utils/labeling/
git commit -m "test(labeling): add test skeleton and fixtures"
```

---

### Task 2: Migrate `triple_barrier.py` — volatility + barriers + fixed horizon

Start with the simpler utility functions before tackling the main triple-barrier loop.

**Files:**
- Create: `src/tradelab/lopezdp_utils/labeling/triple_barrier.py`
- Create: `tests/labeling/test_triple_barrier.py`

**Step 1: Write failing tests for utility functions**

```python
"""Tests for labeling.triple_barrier."""

import numpy as np
import polars as pl
import pytest


class TestDailyVolatility:
    def test_returns_polars_series(self, close_prices):
        from tradelab.lopezdp_utils.labeling.triple_barrier import daily_volatility

        result = daily_volatility(close_prices, span=50)
        assert isinstance(result, pl.DataFrame)
        assert "volatility" in result.columns

    def test_volatility_positive(self, close_prices):
        from tradelab.lopezdp_utils.labeling.triple_barrier import daily_volatility

        result = daily_volatility(close_prices, span=50)
        non_null = result["volatility"].drop_nulls()
        assert (non_null > 0).all()


class TestAddVerticalBarrier:
    def test_returns_t1_series(self, close_prices):
        from tradelab.lopezdp_utils.labeling.triple_barrier import add_vertical_barrier

        t_events = close_prices["timestamp"].gather([0, 50, 100, 200])
        result = add_vertical_barrier(t_events, close_prices, num_bars=10)
        assert isinstance(result, pl.DataFrame)
        assert "t1" in result.columns
        assert len(result) == 4

    def test_t1_after_t0(self, close_prices):
        from tradelab.lopezdp_utils.labeling.triple_barrier import add_vertical_barrier

        t_events = close_prices["timestamp"].gather([0, 50, 100])
        result = add_vertical_barrier(t_events, close_prices, num_bars=10)
        for row in result.iter_rows(named=True):
            assert row["t1"] >= row["timestamp"]


class TestFixedTimeHorizon:
    def test_returns_polars_dataframe(self, close_prices):
        from tradelab.lopezdp_utils.labeling.triple_barrier import fixed_time_horizon

        result = fixed_time_horizon(close_prices, horizon=10, threshold=0.0)
        assert isinstance(result, pl.DataFrame)
        assert "label" in result.columns

    def test_labels_are_valid(self, close_prices):
        from tradelab.lopezdp_utils.labeling.triple_barrier import fixed_time_horizon

        result = fixed_time_horizon(close_prices, horizon=10, threshold=0.005)
        labels = result["label"].unique().sort().to_list()
        assert all(l in [-1, 0, 1] for l in labels)


class TestTripleBarrierLabels:
    """Integration test for the full pipeline."""

    def test_returns_events_with_t1(self, close_prices):
        from tradelab.lopezdp_utils.labeling.triple_barrier import triple_barrier_labels

        # Use every 10th bar as an event
        t_events = close_prices["timestamp"].gather(list(range(0, 400, 10)))
        result = triple_barrier_labels(
            close=close_prices,
            t_events=t_events,
            pt_sl=1.0,
            num_bars=20,
            vol_span=50,
        )
        assert isinstance(result, pl.DataFrame)
        assert "t1" in result.columns
        assert "label" in result.columns
        assert "ret" in result.columns
        # t1 must not be null (safety guardrail)
        assert result["t1"].null_count() == 0

    def test_labels_valid_values(self, close_prices):
        from tradelab.lopezdp_utils.labeling.triple_barrier import triple_barrier_labels

        t_events = close_prices["timestamp"].gather(list(range(0, 400, 10)))
        result = triple_barrier_labels(
            close=close_prices,
            t_events=t_events,
            pt_sl=1.0,
            num_bars=20,
            vol_span=50,
        )
        labels = result["label"].unique().sort().to_list()
        assert all(l in [-1, 0, 1] for l in labels)

    def test_t1_validates_not_null(self, close_prices):
        """The t1 column must never contain nulls — this is a safety guardrail."""
        from tradelab.lopezdp_utils.labeling.triple_barrier import triple_barrier_labels

        t_events = close_prices["timestamp"].gather(list(range(0, 450, 10)))
        result = triple_barrier_labels(
            close=close_prices,
            t_events=t_events,
            pt_sl=1.0,
            num_bars=20,
            vol_span=50,
        )
        assert result["t1"].null_count() == 0

    def test_with_trend_data(self, close_with_trend):
        """Uptrend should produce more +1 labels, downtrend more -1."""
        from tradelab.lopezdp_utils.labeling.triple_barrier import triple_barrier_labels

        t_events = close_with_trend["timestamp"].gather(list(range(0, 180, 5)))
        result = triple_barrier_labels(
            close=close_with_trend,
            t_events=t_events,
            pt_sl=1.0,
            num_bars=15,
            vol_span=30,
        )
        # Just verify it runs and produces labels — exact distribution
        # depends on volatility scaling
        assert len(result) > 0
        assert "label" in result.columns


class TestTrendScanningLabels:
    def test_returns_with_t1(self, close_prices):
        from tradelab.lopezdp_utils.labeling.triple_barrier import trend_scanning_labels

        t_events = close_prices["timestamp"].gather(list(range(50, 400, 20)))
        result = trend_scanning_labels(
            close=close_prices,
            t_events=t_events,
            span=range(5, 20),
        )
        assert isinstance(result, pl.DataFrame)
        assert "t1" in result.columns
        assert "label" in result.columns
```

**Step 2: Run to verify fail**

Run: `uv run pytest tests/labeling/test_triple_barrier.py -v`
Expected: FAIL

**Step 3: Implement triple_barrier.py**

This is the largest file. Implementation strategy:
1. Start with utility functions: `daily_volatility`, `add_vertical_barrier`, `fixed_time_horizon`
2. Then the core: `apply_pt_sl_on_t1` (keep Python loop), `get_events`, `get_bins`
3. Then the wrapper: `triple_barrier_labels`
4. Then trend scanning: `t_value_linear_trend` (NumPy), `trend_scanning_labels`

Key Polars patterns:
- `daily_volatility`: Use `pl.col("close").pct_change().ewm_std(span=span)` with a temporal self-join for day-prior alignment.
- `add_vertical_barrier`: Use `search_sorted` equivalent or join with offset.
- `get_events` frame setup: Build Polars DataFrame with `t1`, `trgt`, `side` columns.
- `apply_pt_sl_on_t1`: Extract close prices as NumPy array. Loop over events. For each event, slice the NumPy array between t0 and t1 indices, compute path touches. Return Polars DataFrame. Add `# TODO(numba): evaluate JIT for barrier touch loop`.
- `get_bins`: Polars join/filter to compute returns from close at t1 vs t0.

**Critical safety validation:**
```python
def _validate_t1(events: pl.DataFrame) -> None:
    """Validate t1 column exists and has no nulls."""
    if "t1" not in events.columns:
        raise ValueError("events DataFrame must contain 't1' column")
    if events["t1"].null_count() > 0:
        raise ValueError("t1 column must not contain null values")
```

Call this at the entry point of every function that consumes t1.

**Step 4: Run tests**

Run: `uv run pytest tests/labeling/test_triple_barrier.py -v`

**Step 5: Commit**

```bash
git add src/tradelab/lopezdp_utils/labeling/triple_barrier.py tests/labeling/test_triple_barrier.py
git commit -m "feat(labeling): migrate triple_barrier with t1 validation to Polars"
```

---

### Task 3: Migrate `meta_labeling.py`

**Files:**
- Create: `src/tradelab/lopezdp_utils/labeling/meta_labeling.py`
- Create: `tests/labeling/test_meta_labeling.py`

**Step 1: Write failing tests**

```python
"""Tests for labeling.meta_labeling."""

import numpy as np
import polars as pl
import pytest


class TestGetEventsMeta:
    def test_returns_events_with_t1_and_side(self, close_prices):
        from tradelab.lopezdp_utils.labeling.meta_labeling import get_events_meta
        from tradelab.lopezdp_utils.labeling.triple_barrier import daily_volatility

        t_events = close_prices["timestamp"].gather(list(range(0, 400, 10)))
        vol = daily_volatility(close_prices, span=50)
        # Simulate a primary model's side predictions
        side = pl.Series("side", np.random.choice([-1, 1], size=len(t_events)))

        result = get_events_meta(
            close=close_prices,
            t_events=t_events,
            pt_sl=[1.0, 1.0],
            trgt=vol,
            min_ret=0.0,
            side=side,
        )
        assert isinstance(result, pl.DataFrame)
        assert "t1" in result.columns
        assert "side" in result.columns
        assert result["t1"].null_count() == 0


class TestGetBinsMeta:
    def test_returns_binary_labels(self, close_prices):
        from tradelab.lopezdp_utils.labeling.meta_labeling import (
            get_bins_meta,
            get_events_meta,
        )
        from tradelab.lopezdp_utils.labeling.triple_barrier import daily_volatility

        t_events = close_prices["timestamp"].gather(list(range(0, 400, 10)))
        vol = daily_volatility(close_prices, span=50)
        side = pl.Series("side", np.random.choice([-1, 1], size=len(t_events)))

        events = get_events_meta(
            close=close_prices,
            t_events=t_events,
            pt_sl=[1.0, 1.0],
            trgt=vol,
            min_ret=0.0,
            side=side,
        )
        result = get_bins_meta(events, close_prices)
        assert isinstance(result, pl.DataFrame)
        assert "label" in result.columns
        # Meta-labels are binary: 0 (don't act) or 1 (act)
        labels = result["label"].unique().sort().to_list()
        assert all(l in [0, 1] for l in labels)
```

**Steps 2-5: Standard TDD cycle**

Implementation notes:
- `get_events_meta` shares `apply_pt_sl_on_t1` from `triple_barrier.py` — import it.
- Asymmetric barriers: when `side` is provided, only the barrier matching the side direction is active.
- `get_bins_meta` maps directional returns to binary {0, 1}: 1 if the trade was profitable given the side, 0 otherwise.

```bash
git commit -m "feat(labeling): migrate meta_labeling with Polars and t1 validation"
```

---

### Task 4: Migrate `sample_weights.py`

**Files:**
- Create: `src/tradelab/lopezdp_utils/labeling/sample_weights.py`
- Create: `tests/labeling/test_sample_weights.py`

**Step 1: Write failing tests**

```python
"""Tests for labeling.sample_weights — concurrency, uniqueness, sequential bootstrap."""

import numpy as np
import polars as pl
import pytest


class TestMpNumCoEvents:
    """Tests for co-event counting."""

    def test_returns_polars_series(self, close_prices, events_with_t1):
        from tradelab.lopezdp_utils.labeling.sample_weights import mp_num_co_events

        result = mp_num_co_events(
            close_idx=close_prices["timestamp"],
            t1=events_with_t1,
        )
        assert isinstance(result, pl.DataFrame)
        assert "num_co_events" in result.columns

    def test_co_events_positive(self, close_prices, events_with_t1):
        from tradelab.lopezdp_utils.labeling.sample_weights import mp_num_co_events

        result = mp_num_co_events(
            close_idx=close_prices["timestamp"],
            t1=events_with_t1,
        )
        non_null = result["num_co_events"].drop_nulls()
        assert (non_null >= 0).all()

    def test_non_overlapping_events_have_count_one(self):
        """Events that don't overlap should have co-event count = 1."""
        from tradelab.lopezdp_utils.labeling.sample_weights import mp_num_co_events

        timestamps = pl.datetime_range(
            pl.datetime(2024, 1, 1),
            pl.datetime(2024, 1, 1, 1, 39),
            interval="1m",
            eager=True,
        )
        # Two non-overlapping events: [0, 10] and [20, 30]
        events = pl.DataFrame({
            "timestamp": [timestamps[0], timestamps[20]],
            "t1": [timestamps[10], timestamps[30]],
        })
        result = mp_num_co_events(close_idx=timestamps, t1=events)
        # Within each event window, co-event count should be 1
        # (only one event active at any time)
        assert result.filter(
            pl.col("timestamp").is_between(timestamps[0], timestamps[10])
        )["num_co_events"].max() == 1


class TestMpSampleTw:
    """Tests for average uniqueness."""

    def test_returns_polars_dataframe(self, close_prices, events_with_t1):
        from tradelab.lopezdp_utils.labeling.sample_weights import (
            mp_num_co_events,
            mp_sample_tw,
        )

        co_events = mp_num_co_events(
            close_idx=close_prices["timestamp"],
            t1=events_with_t1,
        )
        result = mp_sample_tw(t1=events_with_t1, num_co_events=co_events)
        assert isinstance(result, pl.DataFrame)
        assert "uniqueness" in result.columns

    def test_uniqueness_between_zero_and_one(self, close_prices, events_with_t1):
        from tradelab.lopezdp_utils.labeling.sample_weights import (
            mp_num_co_events,
            mp_sample_tw,
        )

        co_events = mp_num_co_events(
            close_idx=close_prices["timestamp"],
            t1=events_with_t1,
        )
        result = mp_sample_tw(t1=events_with_t1, num_co_events=co_events)
        u = result["uniqueness"].drop_nulls()
        assert (u > 0).all()
        assert (u <= 1).all()


class TestGetIndMatrix:
    """Tests for indicator matrix construction."""

    def test_shape(self):
        from tradelab.lopezdp_utils.labeling.sample_weights import get_ind_matrix

        timestamps = pl.datetime_range(
            pl.datetime(2024, 1, 1),
            pl.datetime(2024, 1, 1, 0, 9),
            interval="1m",
            eager=True,
        )
        events = pl.DataFrame({
            "timestamp": [timestamps[0], timestamps[3]],
            "t1": [timestamps[5], timestamps[8]],
        })
        result = get_ind_matrix(bar_idx=timestamps, t1=events)
        assert isinstance(result, np.ndarray)
        assert result.shape == (10, 2)  # 10 bars x 2 events

    def test_binary_values(self):
        from tradelab.lopezdp_utils.labeling.sample_weights import get_ind_matrix

        timestamps = pl.datetime_range(
            pl.datetime(2024, 1, 1),
            pl.datetime(2024, 1, 1, 0, 9),
            interval="1m",
            eager=True,
        )
        events = pl.DataFrame({
            "timestamp": [timestamps[0]],
            "t1": [timestamps[5]],
        })
        result = get_ind_matrix(bar_idx=timestamps, t1=events)
        assert set(np.unique(result)).issubset({0.0, 1.0})
        # Bars 0-5 should be 1, bars 6-9 should be 0
        assert result[:6, 0].sum() == 6
        assert result[6:, 0].sum() == 0


class TestGetAvgUniqueness:
    def test_single_event_full_uniqueness(self):
        """A single non-overlapping event has uniqueness = 1.0."""
        from tradelab.lopezdp_utils.labeling.sample_weights import get_avg_uniqueness

        # 1 event active for 5 bars
        ind_m = np.zeros((10, 1))
        ind_m[:5, 0] = 1.0
        result = get_avg_uniqueness(ind_m)
        assert abs(result[0] - 1.0) < 1e-10

    def test_overlapping_events_lower_uniqueness(self):
        """Two fully overlapping events should have uniqueness = 0.5."""
        from tradelab.lopezdp_utils.labeling.sample_weights import get_avg_uniqueness

        ind_m = np.zeros((10, 2))
        ind_m[:5, 0] = 1.0
        ind_m[:5, 1] = 1.0  # same window
        result = get_avg_uniqueness(ind_m)
        assert abs(result[0] - 0.5) < 1e-10
        assert abs(result[1] - 0.5) < 1e-10


class TestSeqBootstrap:
    def test_returns_list(self):
        from tradelab.lopezdp_utils.labeling.sample_weights import seq_bootstrap

        ind_m = np.zeros((20, 5))
        for i in range(5):
            ind_m[i * 4 : (i + 1) * 4, i] = 1.0
        result = seq_bootstrap(ind_m, s_length=5)
        assert isinstance(result, list)
        assert len(result) == 5

    def test_favors_unique_samples(self):
        """Sequential bootstrap should prefer non-overlapping samples."""
        from tradelab.lopezdp_utils.labeling.sample_weights import seq_bootstrap

        np.random.seed(42)
        # 3 non-overlapping events + 1 overlapping with event 0
        ind_m = np.zeros((20, 4))
        ind_m[0:5, 0] = 1.0   # event 0: bars 0-4
        ind_m[0:5, 3] = 1.0   # event 3: overlaps event 0
        ind_m[5:10, 1] = 1.0  # event 1: bars 5-9
        ind_m[10:15, 2] = 1.0 # event 2: bars 10-14

        # Run many bootstraps and check event 3 is less frequent
        counts = np.zeros(4)
        for _ in range(200):
            sample = seq_bootstrap(ind_m, s_length=3)
            for s in sample:
                counts[s] += 1
        # Event 3 (overlapping) should be selected less than events 1, 2
        assert counts[3] < counts[1]
        assert counts[3] < counts[2]


class TestGetTimeDecay:
    def test_returns_polars_dataframe(self, close_prices, events_with_t1):
        from tradelab.lopezdp_utils.labeling.sample_weights import (
            mp_num_co_events,
            mp_sample_tw,
            get_time_decay,
        )

        co_events = mp_num_co_events(close_prices["timestamp"], events_with_t1)
        tw = mp_sample_tw(events_with_t1, co_events)
        result = get_time_decay(tw, clf_last_w=0.5)
        assert isinstance(result, pl.DataFrame)
        assert "weight" in result.columns

    def test_no_decay_when_clf_last_w_one(self, close_prices, events_with_t1):
        """clf_last_w=1.0 means no decay — all weights should be 1.0."""
        from tradelab.lopezdp_utils.labeling.sample_weights import (
            mp_num_co_events,
            mp_sample_tw,
            get_time_decay,
        )

        co_events = mp_num_co_events(close_prices["timestamp"], events_with_t1)
        tw = mp_sample_tw(events_with_t1, co_events)
        result = get_time_decay(tw, clf_last_w=1.0)
        assert (result["weight"] - 1.0).abs().max() < 1e-10


class TestValidateT1:
    """Tests for t1 validation guardrail."""

    def test_raises_on_missing_t1(self):
        from tradelab.lopezdp_utils.labeling.sample_weights import mp_num_co_events

        timestamps = pl.datetime_range(
            pl.datetime(2024, 1, 1),
            pl.datetime(2024, 1, 1, 0, 9),
            interval="1m",
            eager=True,
        )
        bad_events = pl.DataFrame({"timestamp": [timestamps[0]]})  # no t1!
        with pytest.raises(ValueError, match="t1"):
            mp_num_co_events(close_idx=timestamps, t1=bad_events)

    def test_raises_on_null_t1(self):
        from tradelab.lopezdp_utils.labeling.sample_weights import mp_num_co_events

        timestamps = pl.datetime_range(
            pl.datetime(2024, 1, 1),
            pl.datetime(2024, 1, 1, 0, 9),
            interval="1m",
            eager=True,
        )
        bad_events = pl.DataFrame({
            "timestamp": [timestamps[0]],
            "t1": [None],  # null t1!
        }).cast({"t1": pl.Datetime})
        with pytest.raises(ValueError, match="t1.*null"):
            mp_num_co_events(close_idx=timestamps, t1=bad_events)
```

**Step 2: Run to verify fail**

Run: `uv run pytest tests/labeling/test_sample_weights.py -v`

**Step 3: Implement `labeling/sample_weights.py`**

Implementation notes:
- `_validate_t1()` shared helper — called at entry of every public function.
- `mp_num_co_events`: Accept `pl.DataFrame` with `timestamp` + `t1`. Extract as NumPy for the range-accumulation loop. Return `pl.DataFrame` with `timestamp` + `num_co_events`.
- `mp_sample_tw`: Same pattern — Polars I/O, NumPy loop.
- `get_ind_matrix`: Accept `pl.Series` for bar_idx and `pl.DataFrame` for t1. Build NumPy matrix. Return `np.ndarray`. This stays NumPy because the downstream `get_avg_uniqueness` and `seq_bootstrap` operate on matrices.
- `get_avg_uniqueness`: Pure NumPy. Input/output are arrays.
- `seq_bootstrap`: Pure Python + NumPy. Add `# TODO(numba): evaluate JIT for sequential bootstrap probability update`.
- `mp_sample_w`: Polars I/O, NumPy loop for per-event window sum.
- `get_time_decay`: Full Polars — `cum_sum`, arithmetic, `clip_min`.

**Step 4: Run tests**

Run: `uv run pytest tests/labeling/test_sample_weights.py -v`

**Step 5: Commit**

```bash
git add src/tradelab/lopezdp_utils/labeling/sample_weights.py tests/labeling/test_sample_weights.py
git commit -m "feat(labeling): migrate sample_weights with t1 validation to Polars"
```

---

### Task 5: Migrate `class_balance.py`

**Files:**
- Create: `src/tradelab/lopezdp_utils/labeling/class_balance.py`
- Create: `tests/labeling/test_class_balance.py`

**Step 1: Write failing tests**

```python
"""Tests for labeling.class_balance."""

import numpy as np
import polars as pl
import pytest


class TestDropLabels:
    def test_drops_rare_class(self):
        from tradelab.lopezdp_utils.labeling.class_balance import drop_labels

        events = pl.DataFrame({
            "label": [1] * 90 + [0] * 5 + [-1] * 5,
            "ret": np.random.randn(100),
        })
        result = drop_labels(events, min_pct=0.08)
        # Classes with < 8% should be dropped
        remaining = result["label"].unique().to_list()
        assert 1 in remaining
        # 0 and -1 are each 5% < 8%, should be dropped
        assert 0 not in remaining
        assert -1 not in remaining

    def test_keeps_all_when_above_threshold(self):
        from tradelab.lopezdp_utils.labeling.class_balance import drop_labels

        events = pl.DataFrame({
            "label": [1] * 40 + [0] * 30 + [-1] * 30,
        })
        result = drop_labels(events, min_pct=0.05)
        assert len(result["label"].unique()) == 3


class TestGetClassWeights:
    def test_returns_dict(self):
        from tradelab.lopezdp_utils.labeling.class_balance import get_class_weights

        labels = pl.Series("label", [1, 1, 1, 0, 0, -1])
        result = get_class_weights(labels)
        assert isinstance(result, dict)
        assert -1 in result
        assert result[-1] > result[1]  # minority class gets higher weight
```

**Steps 2-5: Standard TDD cycle**

```bash
git commit -m "feat(labeling): migrate class_balance with Polars"
```

---

### Task 6: Create `labeling/__init__.py` with public exports

```python
"""Labeling and sample weighting — AFML Chapters 3-4.

This package handles the second and third stages of López de Prado's pipeline:
price series → labeled observations (with t1 metadata) → sample weights.

The t1 timestamp (first barrier touch time) is the critical metadata that
connects labeling to weighting, purging, and embargoing downstream.

Reference:
    López de Prado, "Advances in Financial Machine Learning", Chapters 3-4
    López de Prado, "Machine Learning for Asset Managers", Chapter 5
"""

from tradelab.lopezdp_utils.labeling.class_balance import (
    drop_labels,
    get_class_weights,
)
from tradelab.lopezdp_utils.labeling.meta_labeling import (
    get_bins_meta,
    get_events_meta,
)
from tradelab.lopezdp_utils.labeling.sample_weights import (
    get_avg_uniqueness,
    get_ind_matrix,
    get_time_decay,
    mp_num_co_events,
    mp_sample_tw,
    mp_sample_w,
    seq_bootstrap,
)
from tradelab.lopezdp_utils.labeling.triple_barrier import (
    add_vertical_barrier,
    daily_volatility,
    fixed_time_horizon,
    get_bins,
    get_events,
    trend_scanning_labels,
    triple_barrier_labels,
)

__all__ = [
    # Triple-barrier labeling
    "daily_volatility",
    "add_vertical_barrier",
    "fixed_time_horizon",
    "get_events",
    "get_bins",
    "triple_barrier_labels",
    "trend_scanning_labels",
    # Meta-labeling
    "get_events_meta",
    "get_bins_meta",
    # Sample weights
    "mp_num_co_events",
    "mp_sample_tw",
    "get_ind_matrix",
    "get_avg_uniqueness",
    "seq_bootstrap",
    "mp_sample_w",
    "get_time_decay",
    # Class balance
    "drop_labels",
    "get_class_weights",
]
```

---

### Task 7: Integration test — labeling → weights pipeline

**Files:**
- Create: `tests/labeling/test_integration.py`

This test verifies the full pipeline from close prices to weighted labels.

```python
"""Integration tests: labeling → sample weights pipeline."""

import numpy as np
import polars as pl
import pytest


class TestLabelingToWeightsPipeline:
    """End-to-end: close prices → triple-barrier labels → sample weights."""

    def test_full_pipeline(self, close_prices):
        from tradelab.lopezdp_utils.labeling import (
            daily_volatility,
            get_ind_matrix,
            get_avg_uniqueness,
            mp_num_co_events,
            mp_sample_tw,
            get_time_decay,
            triple_barrier_labels,
        )

        # Step 1: Generate labels with t1
        t_events = close_prices["timestamp"].gather(list(range(10, 400, 10)))
        labels = triple_barrier_labels(
            close=close_prices,
            t_events=t_events,
            pt_sl=1.0,
            num_bars=20,
            vol_span=50,
        )

        # Verify t1 exists and is valid
        assert "t1" in labels.columns
        assert labels["t1"].null_count() == 0

        # Step 2: Compute co-events
        co_events = mp_num_co_events(
            close_idx=close_prices["timestamp"],
            t1=labels,
        )
        assert "num_co_events" in co_events.columns

        # Step 3: Compute uniqueness
        tw = mp_sample_tw(t1=labels, num_co_events=co_events)
        assert "uniqueness" in tw.columns
        u = tw["uniqueness"].drop_nulls()
        assert (u > 0).all()
        assert (u <= 1).all()

        # Step 4: Apply time decay
        decayed = get_time_decay(tw, clf_last_w=0.5)
        assert "weight" in decayed.columns
        assert (decayed["weight"].drop_nulls() > 0).all()

        # Step 5: Alternative — indicator matrix path
        ind_m = get_ind_matrix(
            bar_idx=close_prices["timestamp"],
            t1=labels,
        )
        avg_u = get_avg_uniqueness(ind_m)
        assert len(avg_u) == len(labels)
        assert all(0 < u <= 1 for u in avg_u)

    def test_meta_labeling_pipeline(self, close_prices):
        from tradelab.lopezdp_utils.labeling import (
            daily_volatility,
            get_events_meta,
            get_bins_meta,
            mp_num_co_events,
            mp_sample_tw,
        )

        t_events = close_prices["timestamp"].gather(list(range(10, 400, 10)))
        vol = daily_volatility(close_prices, span=50)

        # Simulate primary model sides
        side = pl.Series("side", np.random.choice([-1, 1], size=len(t_events)))

        events = get_events_meta(
            close=close_prices,
            t_events=t_events,
            pt_sl=[1.0, 1.0],
            trgt=vol,
            min_ret=0.0,
            side=side,
        )

        bins = get_bins_meta(events, close_prices)
        assert "label" in bins.columns
        labels = bins["label"].unique().sort().to_list()
        assert all(l in [0, 1] for l in labels)

        # Weights should still work with meta-labeling events
        co_events = mp_num_co_events(
            close_idx=close_prices["timestamp"],
            t1=events,
        )
        tw = mp_sample_tw(t1=events, num_co_events=co_events)
        assert "uniqueness" in tw.columns
```

**Steps: Write → Run → Verify pass → Commit**

```bash
git commit -m "test(labeling): add integration tests for labeling → weights pipeline"
```

---

### Task 8: Delete old directories and verify

**Step 1: Delete old modules**

```bash
rm -rf src/tradelab/lopezdp_utils/labeling/  # old version (will be replaced by new)
rm -rf src/tradelab/lopezdp_utils/sample_weights/
```

Wait — the new `labeling/` package needs to be in place first. The implementation steps above create the new files. This task deletes any leftover old files that weren't overwritten.

Specifically:
- Delete `labeling/bet_sizing.py` (moved to session 6)
- Delete `sample_weights/` directory entirely
- Verify the new `labeling/` package has only the 4 new files + `__init__.py`

**Step 2: Verify imports**

Run: `uv run python -c "from tradelab.lopezdp_utils.labeling import triple_barrier_labels, seq_bootstrap; print('OK')"`

**Step 3: Run full test suite**

Run: `uv run pytest tests/ -v`

**Step 4: Lint**

Run: `uvx ruff check --fix . && uvx ruff format .`

**Step 5: Commit**

```bash
git add -A
git commit -m "refactor(labeling): remove old labeling/ and sample_weights/ directories"
```

---

### Task 9: Merge to main

Run: `git checkout main && git merge phase2/labeling`
Verify: `uv run pytest tests/ -v`
