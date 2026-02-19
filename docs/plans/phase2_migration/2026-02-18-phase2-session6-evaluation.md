# Phase 2 Session 6: `evaluation/` — Backtest Statistics, CPCV, Overfitting, Synthetic, Strategy Risk, Bet Sizing

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Migrate `backtest_statistics/` (5 files, 369 lines) + `backtest_cv/` (1 file, 331 lines) + `backtesting_dangers/` (1 file, 150 lines) + `backtest_synthetic/` (2 files, 206 lines) + `strategy_risk/` (1 file, 185 lines) + `bet_sizing/` (2 files, 301 lines) + deferred `labeling/bet_sizing.py` + deferred `sample_weights/strategy_redundancy.py` into `evaluation/` package (6 files). This covers AFML Chapters 10-15 + MLAM Section 8.

**Architecture:**
- Merge `backtest_statistics/sharpe.py` + `drawdown.py` + `concentration.py` + `bet_timing.py` + `strategy_metrics.py` → `evaluation/statistics.py`
- Migrate `backtest_cv/cpcv.py` → `evaluation/cpcv.py`
- Merge `backtesting_dangers/cscv.py` + deferred `sample_weights/strategy_redundancy.py` → `evaluation/overfitting.py`
- Merge `backtest_synthetic/otr.py` + `ou_process.py` → `evaluation/synthetic.py`
- Migrate `strategy_risk/binomial_model.py` → `evaluation/strategy_risk.py`
- Merge `bet_sizing/signals.py` + `bet_sizing/dynamic_sizing.py` + deferred `labeling/bet_sizing.py` → `evaluation/bet_sizing.py`

**Tech Stack:** polars, numpy, pandas, scipy.stats, sklearn (KFold), pydantic, pytest

**Depends on:** Session 5 (`modeling/` merged to main — `cpcv.py` imports `get_train_times`, `get_embargo_times` from `modeling/cross_validation`)

---

## Pre-Session Checklist

- [ ] Session 5 merged to main
- [ ] Branch from main: `git checkout -b phase2/evaluation`
- [ ] `uv sync --all-extras --dev`
- [ ] Verify deferred files still exist in v1 locations (they should have been kept during sessions 3-4)

---

## File Mapping: Old → New

| Old file | New location | Action |
|----------|-------------|--------|
| `backtest_statistics/sharpe.py` | `evaluation/statistics.py` | Merge into |
| `backtest_statistics/drawdown.py` | `evaluation/statistics.py` | Merge into |
| `backtest_statistics/concentration.py` | `evaluation/statistics.py` | Merge into |
| `backtest_statistics/bet_timing.py` | `evaluation/statistics.py` | Merge into |
| `backtest_statistics/strategy_metrics.py` | `evaluation/statistics.py` | Merge into |
| `backtest_cv/cpcv.py` | `evaluation/cpcv.py` | Migrate |
| `backtesting_dangers/cscv.py` | `evaluation/overfitting.py` | Merge into |
| `sample_weights/strategy_redundancy.py` | `evaluation/overfitting.py` | Merge into (deferred from session 3) |
| `backtest_synthetic/otr.py` | `evaluation/synthetic.py` | Merge into |
| `backtest_synthetic/ou_process.py` | `evaluation/synthetic.py` | Merge into |
| `strategy_risk/binomial_model.py` | `evaluation/strategy_risk.py` | Migrate |
| `bet_sizing/signals.py` | `evaluation/bet_sizing.py` | Merge into |
| `bet_sizing/dynamic_sizing.py` | `evaluation/bet_sizing.py` | Merge into |
| `labeling/bet_sizing.py` | `evaluation/bet_sizing.py` | Merge into (deferred from session 3) |

---

## Polars Migration Decisions (per function)

### evaluation/statistics.py
| Function | Migration | Notes |
|----------|-----------|-------|
| `sharpe_ratio(returns, periods_per_year)` | **Polars I/O** | Accept `pl.Series`, compute mean/std via Polars, return float |
| `probabilistic_sharpe_ratio(...)` | **Keep NumPy/scipy** | Pure math + `norm.cdf` |
| `deflated_sharpe_ratio(...)` | **Keep NumPy/scipy** | Pure math — requires explicit `num_trials` parameter (no default) |
| `compute_dd_tuw(series, dollars)` | **Polars** | Accept `pl.DataFrame` with timestamp + pnl. High-watermark, drawdown, TuW all expressible in Polars (`cum_max`, `group_by`) |
| `get_hhi(bet_ret)` | **Polars** | Accept `pl.Series`, `value_counts` + sum of squares |
| `get_bet_timing(t_pos)` | **Polars** | Accept `pl.DataFrame` with timestamp + position. Detect flattenings/flips via `shift` + `sign` |
| `get_holding_period(t_pos)` | **Polars I/O** | Accept `pl.DataFrame`, weighted-average entry time loop stays Python |
| `strategy_precision(...)` | **Keep Python** | Pure math |
| `strategy_recall(...)` | **Keep Python** | Pure math |
| `multi_test_precision_recall(...)` | **Keep Python** | Pure math |

### evaluation/cpcv.py
| Function | Migration | Notes |
|----------|-----------|-------|
| `get_num_splits(n, k)` | **Keep Python** | `math.comb` |
| `get_num_backtest_paths(n, k)` | **Keep Python** | Arithmetic on `comb` |
| `CombinatorialPurgedKFold` | **Keep pandas/sklearn** | Extends `KFold`, uses purging/embargo from `modeling/` |
| `assemble_backtest_paths(...)` | **Keep pandas** | Complex index manipulation for path assembly |

### evaluation/overfitting.py
| Function | Migration | Notes |
|----------|-----------|-------|
| `probability_of_backtest_overfitting(...)` | **Polars I/O** | Accept `pl.DataFrame` of trial returns. Internal combinatorial loop stays NumPy. Return dict with PBO + diagnostics |
| `_compute_metric(data, metric)` | **Keep NumPy** | Internal helper |

> **Note on `strategy_redundancy.py`:** Check if this file still exists in the v1 `sample_weights/` directory. If it was deleted during session 3, recover it from git history (`git show HEAD~N:src/.../sample_weights/strategy_redundancy.py`). Merge its functions into `overfitting.py`.

### evaluation/synthetic.py
| Function | Migration | Notes |
|----------|-----------|-------|
| `ou_half_life(phi)` | **Keep Python** | Pure math |
| `ou_fit(prices, forecast)` | **Polars I/O** | Accept `pl.Series` of prices, OLS stays NumPy, return dict |
| `otr_batch(...)` | **Keep NumPy** | Monte Carlo simulation loop |
| `otr_main(...)` | **Keep NumPy** | Cartesian product driver over `otr_batch`. Return dict of `pl.DataFrame` |

### evaluation/strategy_risk.py
| Function | Migration | Notes |
|----------|-----------|-------|
| `sharpe_ratio_symmetric(p, n)` | **Keep Python** | Pure math |
| `implied_precision_symmetric(n, target_sr)` | **Keep Python** | Delegates to `bin_hr` |
| `sharpe_ratio_asymmetric(p, n, sl, pt)` | **Keep Python** | Pure math |
| `bin_hr(sl, pt, freq, target_sr)` | **Keep Python** | Quadratic formula |
| `bin_freq(sl, pt, p, target_sr)` | **Keep Python** | Quadratic formula |
| `mix_gaussians(...)` | **Keep NumPy** | Random sampling |
| `prob_failure(ret, freq, target_sr)` | **Polars I/O** | Accept `pl.Series` of returns, derive empirical stats, return float |

### evaluation/bet_sizing.py
| Function | Migration | Notes |
|----------|-----------|-------|
| `get_signal(events, step_size, prob, pred, num_classes)` | **Polars** | Accept `pl.DataFrame` events with t1 + side. Convert prob → z-stat → signal via Polars expressions. Call `avg_active_signals` + `discrete_signal` |
| `avg_active_signals(signals)` | **Polars** | Accept `pl.DataFrame` with timestamp + t1 + signal. Temporal self-join to average active signals at each time point |
| `discrete_signal(signal0, step_size)` | **Polars** | `round` + `clip` on Series |
| `bet_size(w, x)` | **Keep Python** | Pure math (sigmoid) |
| `get_target_pos(w, f, m_p, max_pos)` | **Keep Python** | Arithmetic |
| `inv_price(f, w, m)` | **Keep Python** | Algebraic inverse |
| `limit_price(t_pos, pos, f, w, max_pos)` | **Keep Python** | Loop over unit increments |
| `get_w(x, m)` | **Keep Python** | Calibration formula |

---

## Pydantic Validation

```python
from pydantic import BaseModel, field_validator


class DSRConfig(BaseModel):
    """Configuration for Deflated Sharpe Ratio.

    num_trials is MANDATORY — López de Prado insists on explicit
    accounting for the number of strategy variations tested.
    """
    num_trials: int  # no default — must be explicit
    observed_sr: float
    n_obs: int
    skew: float = 0.0
    kurtosis: float = 3.0

    @field_validator("num_trials")
    @classmethod
    def must_be_positive(cls, v):
        if v < 1:
            raise ValueError("num_trials must be >= 1")
        return v


class BetSizingConfig(BaseModel):
    """Configuration for signal → bet size conversion."""
    step_size: float = 0.0  # 0.0 = continuous, > 0 = discrete
    max_pos: int = 1

    @field_validator("step_size")
    @classmethod
    def non_negative(cls, v):
        if v < 0:
            raise ValueError("step_size must be >= 0")
        return v
```

**Critical validation at function boundaries:**
- `deflated_sharpe_ratio` must require explicit `num_trials` (no default that hides multiple testing)
- `get_signal` must validate that `events` has `t1` column and it has no nulls
- `probability_of_backtest_overfitting` must validate `n_partitions` is even
- `CombinatorialPurgedKFold` must validate `t1` is provided

---

## Tasks

### Task 1: Create branch, package structure, and test fixtures

**Files:**
- Create: `src/tradelab/lopezdp_utils/evaluation/__init__.py`
- Create: `tests/evaluation/__init__.py`
- Create: `tests/evaluation/conftest.py`

**Step 1: Create branch**

Run: `git checkout -b phase2/evaluation`

**Step 2: Create directories**

Run: `mkdir -p tests/evaluation`

**Step 3: Create shared fixtures**

```python
"""Shared fixtures for evaluation package tests."""

import numpy as np
import pandas as pd
import polars as pl
import pytest


@pytest.fixture
def daily_returns() -> pl.Series:
    """252-day return series for Sharpe ratio / drawdown tests."""
    np.random.seed(42)
    returns = np.random.randn(252) * 0.01 + 0.0003  # slight positive drift
    return pl.Series("returns", returns)


@pytest.fixture
def pnl_series() -> pl.DataFrame:
    """Cumulative PnL series with a drawdown episode."""
    np.random.seed(42)
    n = 252
    timestamps = pl.datetime_range(
        pl.datetime(2024, 1, 1),
        pl.datetime(2024, 9, 8),
        interval="1d",
        eager=True,
    )[:n]
    # Up for 150 days, down for 50, recover for 52
    up = np.linspace(100, 130, 150) + np.random.randn(150) * 0.5
    down = np.linspace(130, 115, 50) + np.random.randn(50) * 0.5
    recover = np.linspace(115, 135, 52) + np.random.randn(52) * 0.5
    pnl = np.concatenate([up, down, recover])
    return pl.DataFrame({"timestamp": timestamps, "pnl": pnl})


@pytest.fixture
def position_series() -> pl.DataFrame:
    """Target position series with flips and flattenings for bet timing tests."""
    np.random.seed(42)
    n = 100
    timestamps = pl.datetime_range(
        pl.datetime(2024, 1, 1),
        pl.datetime(2024, 4, 9),
        interval="1d",
        eager=True,
    )[:n]
    # Long, flat, short, flat, long
    pos = np.concatenate([
        np.ones(20),      # long
        np.zeros(10),     # flat
        -np.ones(25),     # short
        np.zeros(10),     # flat
        np.ones(35),      # long
    ])
    return pl.DataFrame({"timestamp": timestamps, "position": pos})


@pytest.fixture
def trial_returns() -> pl.DataFrame:
    """Multiple strategy trial return series for CSCV/PBO tests."""
    np.random.seed(42)
    n_obs = 200
    n_trials = 10
    data = {"timestamp": pl.datetime_range(
        pl.datetime(2024, 1, 1),
        pl.datetime(2024, 7, 18),
        interval="1d",
        eager=True,
    )[:n_obs]}
    for i in range(n_trials):
        data[f"trial_{i}"] = np.random.randn(n_obs) * 0.01
    return pl.DataFrame(data)


@pytest.fixture
def price_series_ou() -> pl.Series:
    """Mean-reverting price series for O-U tests."""
    np.random.seed(42)
    n = 500
    phi = 0.95
    mu = 100.0
    sigma = 0.5
    prices = np.zeros(n)
    prices[0] = mu
    for t in range(1, n):
        prices[t] = mu + phi * (prices[t - 1] - mu) + np.random.randn() * sigma
    return pl.Series("price", prices)
```

**Step 4: Commit**

```bash
git add tests/evaluation/ src/tradelab/lopezdp_utils/evaluation/
git commit -m "test(evaluation): add test skeleton and fixtures"
```

---

### Task 2: Migrate `statistics.py` — Sharpe, PSR, DSR, drawdown, HHI, bet timing, strategy metrics

**Files:**
- Create: `src/tradelab/lopezdp_utils/evaluation/statistics.py`
- Create: `tests/evaluation/test_statistics.py`

**Step 1: Write failing tests**

```python
"""Tests for evaluation.statistics."""

import numpy as np
import polars as pl
import pytest


class TestSharpeRatio:
    def test_returns_float(self, daily_returns):
        from tradelab.lopezdp_utils.evaluation.statistics import sharpe_ratio

        result = sharpe_ratio(daily_returns, periods_per_year=252)
        assert isinstance(result, float)

    def test_positive_for_positive_drift(self, daily_returns):
        from tradelab.lopezdp_utils.evaluation.statistics import sharpe_ratio

        result = sharpe_ratio(daily_returns, periods_per_year=252)
        assert result > 0

    def test_zero_returns_zero_sharpe(self):
        from tradelab.lopezdp_utils.evaluation.statistics import sharpe_ratio

        returns = pl.Series("returns", [0.0] * 100)
        result = sharpe_ratio(returns, periods_per_year=252)
        assert result == 0.0


class TestProbabilisticSharpeRatio:
    def test_high_sr_high_psr(self):
        from tradelab.lopezdp_utils.evaluation.statistics import probabilistic_sharpe_ratio

        result = probabilistic_sharpe_ratio(
            observed_sr=2.0, benchmark_sr=0.0, n_obs=252, skew=0.0, kurtosis=3.0,
        )
        assert result > 0.95

    def test_low_sr_low_psr(self):
        from tradelab.lopezdp_utils.evaluation.statistics import probabilistic_sharpe_ratio

        result = probabilistic_sharpe_ratio(
            observed_sr=0.1, benchmark_sr=1.0, n_obs=50, skew=0.0, kurtosis=3.0,
        )
        assert result < 0.5


class TestDeflatedSharpeRatio:
    def test_requires_num_trials(self):
        """DSR must not have a default num_trials — enforce explicit accounting."""
        from tradelab.lopezdp_utils.evaluation.statistics import deflated_sharpe_ratio

        # Should work with explicit sr_estimates list
        result = deflated_sharpe_ratio(
            observed_sr=1.5,
            sr_estimates=[0.5, 0.8, 1.0, 1.2, 1.5],
            n_obs=252,
            skew=0.0,
            kurtosis=3.0,
        )
        assert 0 <= result <= 1

    def test_many_trials_deflates_sr(self):
        from tradelab.lopezdp_utils.evaluation.statistics import deflated_sharpe_ratio

        few = deflated_sharpe_ratio(
            observed_sr=1.5, sr_estimates=[0.5, 1.5], n_obs=252,
        )
        many = deflated_sharpe_ratio(
            observed_sr=1.5, sr_estimates=[0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.5] * 10,
            n_obs=252,
        )
        assert many < few  # more trials → more deflation


class TestComputeDdTuw:
    def test_returns_polars_dataframe(self, pnl_series):
        from tradelab.lopezdp_utils.evaluation.statistics import compute_dd_tuw

        dd, tuw = compute_dd_tuw(pnl_series, dollars=False)
        assert isinstance(dd, pl.DataFrame)
        assert isinstance(tuw, pl.DataFrame)

    def test_drawdown_negative(self, pnl_series):
        from tradelab.lopezdp_utils.evaluation.statistics import compute_dd_tuw

        dd, _ = compute_dd_tuw(pnl_series, dollars=False)
        assert (dd["drawdown"].drop_nulls() <= 0).all()


class TestGetHHI:
    def test_concentrated_returns(self):
        from tradelab.lopezdp_utils.evaluation.statistics import get_hhi

        # One huge return, rest tiny
        returns = pl.Series("ret", [10.0] + [0.01] * 99)
        result = get_hhi(returns)
        assert result > 0.5  # concentrated

    def test_uniform_returns_low_hhi(self):
        from tradelab.lopezdp_utils.evaluation.statistics import get_hhi

        returns = pl.Series("ret", [1.0] * 100)
        result = get_hhi(returns)
        assert result < 0.05  # diversified


class TestGetBetTiming:
    def test_detects_flattenings_and_flips(self, position_series):
        from tradelab.lopezdp_utils.evaluation.statistics import get_bet_timing

        result = get_bet_timing(position_series)
        assert isinstance(result, pl.DataFrame)
        assert "timestamp" in result.columns
        assert len(result) >= 3  # at least 3 bet boundaries


class TestGetHoldingPeriod:
    def test_returns_float(self, position_series):
        from tradelab.lopezdp_utils.evaluation.statistics import get_holding_period

        result = get_holding_period(position_series)
        assert isinstance(result, float)
        assert result > 0


class TestStrategyPrecision:
    def test_perfect_test_high_precision(self):
        from tradelab.lopezdp_utils.evaluation.statistics import strategy_precision

        result = strategy_precision(alpha=0.01, beta=0.01, theta=0.1)
        assert result > 0.5

    def test_bad_test_low_precision(self):
        from tradelab.lopezdp_utils.evaluation.statistics import strategy_precision

        result = strategy_precision(alpha=0.5, beta=0.5, theta=0.01)
        assert result < 0.05


class TestMultiTestPrecisionRecall:
    def test_returns_tuple(self):
        from tradelab.lopezdp_utils.evaluation.statistics import multi_test_precision_recall

        prec, recall = multi_test_precision_recall(
            alpha=0.05, beta=0.2, theta=0.1, k=10,
        )
        assert 0 <= prec <= 1
        assert 0 <= recall <= 1

    def test_more_trials_lower_precision(self):
        from tradelab.lopezdp_utils.evaluation.statistics import multi_test_precision_recall

        prec_few, _ = multi_test_precision_recall(alpha=0.05, beta=0.2, theta=0.1, k=2)
        prec_many, _ = multi_test_precision_recall(alpha=0.05, beta=0.2, theta=0.1, k=100)
        assert prec_many < prec_few
```

**Step 2: Run to verify fail**

Run: `uv run pytest tests/evaluation/test_statistics.py -v`

**Step 3: Implement `evaluation/statistics.py`**

Implementation notes:
- `sharpe_ratio`: Accept `pl.Series`, use `series.mean()` / `series.std()` * sqrt(periods). Return float.
- `probabilistic_sharpe_ratio`, `deflated_sharpe_ratio`: Pure math with scipy `norm.cdf`. DSR takes `sr_estimates: list[float]` (the list of all trial SRs). No default — caller must provide.
- `compute_dd_tuw`: Accept `pl.DataFrame` with `timestamp` + `pnl`. Use `pl.col("pnl").cum_max()` for high-watermarks, compute DD via arithmetic. Group by HWM episode for TuW. Return `(dd: pl.DataFrame, tuw: pl.DataFrame)`.
- `get_hhi`: Accept `pl.Series`. Normalized HHI = `(sum(w_i^2) - 1/N) / (1 - 1/N)` where `w_i = |r_i| / sum(|r|)`.
- `get_bet_timing`: Accept `pl.DataFrame` with `timestamp` + `position`. Detect flips via `shift` + `sign` changes. Return `pl.DataFrame` of bet boundary timestamps.
- `get_holding_period`: Accept `pl.DataFrame`, Python loop for weighted-average entry time. Return float (days).
- `strategy_precision`, `strategy_recall`, `multi_test_precision_recall`: Pure math, no changes.

**Step 4: Run tests**

Run: `uv run pytest tests/evaluation/test_statistics.py -v`

**Step 5: Commit**

```bash
git add src/tradelab/lopezdp_utils/evaluation/statistics.py tests/evaluation/test_statistics.py
git commit -m "feat(evaluation): migrate statistics with Polars (SR, DSR, drawdown, HHI, bet timing)"
```

---

### Task 3: Migrate `cpcv.py` — Combinatorial Purged Cross-Validation

**Files:**
- Create: `src/tradelab/lopezdp_utils/evaluation/cpcv.py`
- Create: `tests/evaluation/test_cpcv.py`

**Step 1: Write failing tests**

```python
"""Tests for evaluation.cpcv — Combinatorial Purged Cross-Validation."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier


class TestGetNumSplits:
    def test_known_values(self):
        from tradelab.lopezdp_utils.evaluation.cpcv import get_num_splits

        assert get_num_splits(n_groups=6, k_test_groups=2) == 15  # C(6,2)
        assert get_num_splits(n_groups=10, k_test_groups=2) == 45  # C(10,2)


class TestGetNumBacktestPaths:
    def test_known_values(self):
        from tradelab.lopezdp_utils.evaluation.cpcv import get_num_backtest_paths

        # φ[N,k] = k/N * C(N,k)
        assert get_num_backtest_paths(n_groups=6, k_test_groups=2) == 5  # 2/6 * 15
        assert get_num_backtest_paths(n_groups=10, k_test_groups=2) == 9  # 2/10 * 45


class TestCombinatorialPurgedKFold:
    def test_correct_number_of_splits(self):
        from tradelab.lopezdp_utils.evaluation.cpcv import CombinatorialPurgedKFold

        np.random.seed(42)
        n = 200
        X = pd.DataFrame(np.random.randn(n, 5), columns=[f"f{i}" for i in range(5)])
        idx = pd.date_range("2024-01-01", periods=n, freq="min")
        X.index = idx
        t1 = pd.Series(
            pd.date_range("2024-01-01 00:05", periods=n, freq="min").clip(upper=idx[-1]),
            index=idx,
        )

        cpkf = CombinatorialPurgedKFold(
            n_splits=6, k_test_groups=2, t1=t1, pct_embargo=0.01,
        )
        splits = list(cpkf.split(X))
        assert len(splits) == 15  # C(6,2)

    def test_no_leakage(self):
        from tradelab.lopezdp_utils.evaluation.cpcv import CombinatorialPurgedKFold

        np.random.seed(42)
        n = 120
        X = pd.DataFrame(np.random.randn(n, 3), columns=["a", "b", "c"])
        idx = pd.date_range("2024-01-01", periods=n, freq="min")
        X.index = idx
        t1 = pd.Series(
            pd.date_range("2024-01-01 00:03", periods=n, freq="min").clip(upper=idx[-1]),
            index=idx,
        )

        cpkf = CombinatorialPurgedKFold(
            n_splits=6, k_test_groups=2, t1=t1, pct_embargo=0.01,
        )
        for train_idx, test_idx in cpkf.split(X):
            train_t1 = t1.iloc[train_idx]
            test_start = X.index[test_idx].min()
            test_end = X.index[test_idx].max()
            # No train t1 should overlap with test window
            overlap = train_t1[(train_t1 >= test_start) & (train_t1.index <= test_end)]
            assert len(overlap) == 0

    def test_all_indices_appear_in_test(self):
        from tradelab.lopezdp_utils.evaluation.cpcv import CombinatorialPurgedKFold

        np.random.seed(42)
        n = 120
        X = pd.DataFrame(np.random.randn(n, 3), columns=["a", "b", "c"])
        idx = pd.date_range("2024-01-01", periods=n, freq="min")
        X.index = idx
        t1 = pd.Series(
            pd.date_range("2024-01-01 00:03", periods=n, freq="min").clip(upper=idx[-1]),
            index=idx,
        )

        cpkf = CombinatorialPurgedKFold(
            n_splits=6, k_test_groups=2, t1=t1, pct_embargo=0.0,
        )
        all_test = set()
        for _, test_idx in cpkf.split(X):
            all_test.update(test_idx.tolist())
        # Every observation should appear in at least one test fold
        assert all_test == set(range(n))


class TestAssembleBacktestPaths:
    def test_returns_correct_shape(self):
        from tradelab.lopezdp_utils.evaluation.cpcv import (
            CombinatorialPurgedKFold,
            assemble_backtest_paths,
            get_num_backtest_paths,
        )

        np.random.seed(42)
        n = 120
        n_groups = 6
        k_test = 2
        idx = pd.date_range("2024-01-01", periods=n, freq="min")
        t1 = pd.Series(
            pd.date_range("2024-01-01 00:03", periods=n, freq="min").clip(upper=idx[-1]),
            index=idx,
        )

        # Simulate predictions from all splits
        predictions = {}
        cpkf = CombinatorialPurgedKFold(
            n_splits=n_groups, k_test_groups=k_test, t1=t1,
        )
        X = pd.DataFrame(np.random.randn(n, 3), columns=["a", "b", "c"], index=idx)
        y = pd.Series(np.random.randint(0, 2, n), index=idx)

        for i, (train_idx, test_idx) in enumerate(cpkf.split(X)):
            clf = RandomForestClassifier(n_estimators=10, random_state=42)
            clf.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = clf.predict_proba(X.iloc[test_idx])
            predictions[i] = pd.DataFrame(preds, index=X.index[test_idx])

        paths = assemble_backtest_paths(predictions, n_groups, k_test, n)
        expected_n_paths = get_num_backtest_paths(n_groups, k_test)
        assert len(paths) == expected_n_paths
```

**Step 2: Run to verify fail**

Run: `uv run pytest tests/evaluation/test_cpcv.py -v`

**Step 3: Implement `evaluation/cpcv.py`**

Implementation notes:
- `get_num_splits`, `get_num_backtest_paths`: Pure math, copy from v1.
- `CombinatorialPurgedKFold(KFold)`: Copy from v1. **Update import** of `get_train_times` and `get_embargo_times` to point to `modeling.cross_validation` instead of `cross_validation.purging`. Add t1 validation.
- `assemble_backtest_paths`: Copy from v1 — complex index manipulation, keep pandas.

**Step 4: Run tests**

Run: `uv run pytest tests/evaluation/test_cpcv.py -v`

**Step 5: Commit**

```bash
git add src/tradelab/lopezdp_utils/evaluation/cpcv.py tests/evaluation/test_cpcv.py
git commit -m "feat(evaluation): migrate CPCV with purging from modeling/"
```

---

### Task 4: Migrate `overfitting.py` — CSCV/PBO + strategy redundancy

**Files:**
- Create: `src/tradelab/lopezdp_utils/evaluation/overfitting.py`
- Create: `tests/evaluation/test_overfitting.py`

**Step 1: Recover deferred file if needed**

Check if `sample_weights/strategy_redundancy.py` still exists. If deleted during session 3:
```bash
git show HEAD~N:src/tradelab/lopezdp_utils/sample_weights/strategy_redundancy.py
```

**Step 2: Write failing tests**

```python
"""Tests for evaluation.overfitting — CSCV, PBO."""

import numpy as np
import polars as pl
import pytest


class TestProbabilityOfBacktestOverfitting:
    def test_returns_dict(self, trial_returns):
        from tradelab.lopezdp_utils.evaluation.overfitting import (
            probability_of_backtest_overfitting,
        )

        result = probability_of_backtest_overfitting(
            trial_returns, n_partitions=6, metric="sharpe",
        )
        assert isinstance(result, dict)
        assert "pbo" in result
        assert "logits" in result

    def test_pbo_between_zero_and_one(self, trial_returns):
        from tradelab.lopezdp_utils.evaluation.overfitting import (
            probability_of_backtest_overfitting,
        )

        result = probability_of_backtest_overfitting(
            trial_returns, n_partitions=6, metric="sharpe",
        )
        assert 0 <= result["pbo"] <= 1

    def test_validates_even_partitions(self, trial_returns):
        from tradelab.lopezdp_utils.evaluation.overfitting import (
            probability_of_backtest_overfitting,
        )

        with pytest.raises(ValueError, match="even"):
            probability_of_backtest_overfitting(
                trial_returns, n_partitions=5, metric="sharpe",
            )

    def test_random_trials_high_pbo(self):
        """Random strategies should show high probability of overfitting."""
        from tradelab.lopezdp_utils.evaluation.overfitting import (
            probability_of_backtest_overfitting,
        )

        np.random.seed(42)
        n_obs = 500
        data = {"timestamp": pl.datetime_range(
            pl.datetime(2024, 1, 1),
            pl.datetime(2025, 5, 15),
            interval="1d",
            eager=True,
        )[:n_obs]}
        for i in range(20):
            data[f"trial_{i}"] = np.random.randn(n_obs) * 0.01
        trials = pl.DataFrame(data)

        result = probability_of_backtest_overfitting(
            trials, n_partitions=10, metric="sharpe",
        )
        # With 20 random strategies, PBO should be high
        assert result["pbo"] > 0.3
```

**Step 3: Implement `evaluation/overfitting.py`**

Implementation notes:
- `probability_of_backtest_overfitting`: Accept `pl.DataFrame` with timestamp + trial columns. Validate `n_partitions` is even. Convert to NumPy internally for the combinatorial loop. Compute metric (Sharpe/total_return/mean_return) per partition per trial. Return dict with `pbo`, `logits`, `ranks`, `n_combinations`.
- `_compute_metric`: Internal helper, NumPy.
- If `strategy_redundancy.py` has functions, merge them here with appropriate naming.

**Step 4: Run tests**

Run: `uv run pytest tests/evaluation/test_overfitting.py -v`

**Step 5: Commit**

```bash
git add src/tradelab/lopezdp_utils/evaluation/overfitting.py tests/evaluation/test_overfitting.py
git commit -m "feat(evaluation): migrate overfitting with CSCV/PBO"
```

---

### Task 5: Migrate `synthetic.py` — O-U process + Optimal Trading Rule

**Files:**
- Create: `src/tradelab/lopezdp_utils/evaluation/synthetic.py`
- Create: `tests/evaluation/test_synthetic.py`

**Step 1: Write failing tests**

```python
"""Tests for evaluation.synthetic — O-U process and Optimal Trading Rule."""

import numpy as np
import polars as pl
import pytest


class TestOUHalfLife:
    def test_known_value(self):
        from tradelab.lopezdp_utils.evaluation.synthetic import ou_half_life

        # phi=0.5 → half-life = -log(2)/log(0.5) = 1.0
        result = ou_half_life(phi=0.5)
        assert abs(result - 1.0) < 1e-10

    def test_high_phi_long_half_life(self):
        from tradelab.lopezdp_utils.evaluation.synthetic import ou_half_life

        result = ou_half_life(phi=0.99)
        assert result > 50  # slow mean reversion

    def test_invalid_phi_raises(self):
        from tradelab.lopezdp_utils.evaluation.synthetic import ou_half_life

        with pytest.raises(ValueError):
            ou_half_life(phi=1.0)
        with pytest.raises(ValueError):
            ou_half_life(phi=0.0)


class TestOUFit:
    def test_returns_dict(self, price_series_ou):
        from tradelab.lopezdp_utils.evaluation.synthetic import ou_fit

        result = ou_fit(price_series_ou, forecast=100.0)
        assert isinstance(result, dict)
        assert "phi" in result
        assert "sigma" in result
        assert "half_life" in result

    def test_estimates_phi_close_to_true(self, price_series_ou):
        from tradelab.lopezdp_utils.evaluation.synthetic import ou_fit

        result = ou_fit(price_series_ou, forecast=100.0)
        # True phi = 0.95
        assert 0.85 < result["phi"] < 1.0


class TestOTRBatch:
    def test_returns_polars_dataframe(self):
        from tradelab.lopezdp_utils.evaluation.synthetic import otr_batch

        result = otr_batch(
            coeffs={"phi": 0.95, "forecast": 100.0, "sigma": 0.5},
            n_iter=100,
            max_hp=50,
            r_pt=np.linspace(0.5, 2.0, 4),
            r_slm=np.linspace(0.5, 2.0, 4),
            seed=42,
        )
        assert isinstance(result, pl.DataFrame)
        assert "r_pt" in result.columns
        assert "r_slm" in result.columns
        assert "sharpe" in result.columns

    def test_result_shape(self):
        from tradelab.lopezdp_utils.evaluation.synthetic import otr_batch

        r_pt = np.linspace(0.5, 2.0, 3)
        r_slm = np.linspace(0.5, 2.0, 4)
        result = otr_batch(
            coeffs={"phi": 0.95, "forecast": 100.0, "sigma": 0.5},
            n_iter=50,
            max_hp=30,
            r_pt=r_pt,
            r_slm=r_slm,
            seed=42,
        )
        assert len(result) == 12  # 3 * 4


class TestOTRMain:
    def test_returns_dict_of_dataframes(self):
        from tradelab.lopezdp_utils.evaluation.synthetic import otr_main

        result = otr_main(
            forecasts=[100.0, 105.0],
            half_lives=[10.0],
            sigma=0.5,
            n_iter=50,
            max_hp=30,
        )
        assert isinstance(result, dict)
        assert len(result) == 2  # 2 forecasts * 1 half_life
        for key, df in result.items():
            assert isinstance(df, pl.DataFrame)
```

**Step 2: Run to verify fail**

Run: `uv run pytest tests/evaluation/test_synthetic.py -v`

**Step 3: Implement `evaluation/synthetic.py`**

Implementation notes:
- `ou_half_life`: Pure math. Validate 0 < phi < 1.
- `ou_fit`: Accept `pl.Series` of prices. Extract to NumPy for OLS. Return dict.
- `otr_batch`: Accept dict of O-U coefficients + threshold arrays. Monte Carlo loop stays Python/NumPy. Return `pl.DataFrame` with `r_pt`, `r_slm`, `mean_pnl`, `std_pnl`, `sharpe`.
- `otr_main`: Driver over `otr_batch`. Accept lists of forecasts/half_lives. Return `dict[(forecast, half_life), pl.DataFrame]`.

**Step 4: Run tests**

Run: `uv run pytest tests/evaluation/test_synthetic.py -v`

**Step 5: Commit**

```bash
git add src/tradelab/lopezdp_utils/evaluation/synthetic.py tests/evaluation/test_synthetic.py
git commit -m "feat(evaluation): migrate synthetic O-U and OTR with Polars output"
```

---

### Task 6: Migrate `strategy_risk.py` — binomial model

**Files:**
- Create: `src/tradelab/lopezdp_utils/evaluation/strategy_risk.py`
- Create: `tests/evaluation/test_strategy_risk.py`

**Step 1: Write failing tests**

```python
"""Tests for evaluation.strategy_risk — binomial model for strategy viability."""

import numpy as np
import polars as pl
import pytest


class TestSharpeRatioSymmetric:
    def test_coin_flip_zero(self):
        from tradelab.lopezdp_utils.evaluation.strategy_risk import sharpe_ratio_symmetric

        result = sharpe_ratio_symmetric(p=0.5, n=252)
        assert abs(result) < 1e-10

    def test_perfect_classifier(self):
        from tradelab.lopezdp_utils.evaluation.strategy_risk import sharpe_ratio_symmetric

        result = sharpe_ratio_symmetric(p=1.0, n=252)
        assert result > 10  # very high


class TestImpliedPrecisionSymmetric:
    def test_known_sr_target(self):
        from tradelab.lopezdp_utils.evaluation.strategy_risk import implied_precision_symmetric

        result = implied_precision_symmetric(n=252, target_sr=1.0)
        assert 0.5 < result < 1.0


class TestSharpeRatioAsymmetric:
    def test_symmetric_case_matches(self):
        from tradelab.lopezdp_utils.evaluation.strategy_risk import (
            sharpe_ratio_asymmetric,
            sharpe_ratio_symmetric,
        )

        sr_sym = sharpe_ratio_symmetric(p=0.6, n=252)
        sr_asym = sharpe_ratio_asymmetric(p=0.6, n=252, sl=-1.0, pt=1.0)
        assert abs(sr_sym - sr_asym) < 0.1


class TestBinHR:
    def test_returns_float(self):
        from tradelab.lopezdp_utils.evaluation.strategy_risk import bin_hr

        result = bin_hr(sl=-1.0, pt=1.0, freq=252, target_sr=1.0)
        assert isinstance(result, float)
        assert 0.5 < result < 1.0


class TestBinFreq:
    def test_returns_float(self):
        from tradelab.lopezdp_utils.evaluation.strategy_risk import bin_freq

        result = bin_freq(sl=-1.0, pt=1.0, p=0.6, target_sr=1.0)
        assert isinstance(result, float)
        assert result > 0


class TestMixGaussians:
    def test_returns_correct_length(self):
        from tradelab.lopezdp_utils.evaluation.strategy_risk import mix_gaussians

        result = mix_gaussians(
            mu1=0.01, mu2=-0.01, sigma1=0.02, sigma2=0.03, prob1=0.6, n_obs=1000,
        )
        assert len(result) == 1000


class TestProbFailure:
    def test_returns_float(self):
        from tradelab.lopezdp_utils.evaluation.strategy_risk import prob_failure

        np.random.seed(42)
        ret = pl.Series("ret", np.random.randn(252) * 0.01 + 0.001)
        result = prob_failure(ret, freq=252, target_sr=1.0)
        assert isinstance(result, float)
        assert 0 <= result <= 1
```

**Step 2: Run to verify fail**

Run: `uv run pytest tests/evaluation/test_strategy_risk.py -v`

**Step 3: Implement `evaluation/strategy_risk.py`**

Implementation notes:
- Mostly pure math — copy from v1 with cleanup.
- `prob_failure`: Accept `pl.Series` of returns, extract to NumPy for empirical stats, return float.
- `mix_gaussians`: Keep NumPy random sampling.

**Step 4: Run tests**

Run: `uv run pytest tests/evaluation/test_strategy_risk.py -v`

**Step 5: Commit**

```bash
git add src/tradelab/lopezdp_utils/evaluation/strategy_risk.py tests/evaluation/test_strategy_risk.py
git commit -m "feat(evaluation): migrate strategy_risk binomial model"
```

---

### Task 7: Migrate `bet_sizing.py` — signals + dynamic sizing

**Files:**
- Create: `src/tradelab/lopezdp_utils/evaluation/bet_sizing.py`
- Create: `tests/evaluation/test_bet_sizing.py`

**Step 1: Recover deferred file if needed**

Check if `labeling/bet_sizing.py` was preserved or deleted during session 3. If deleted, recover from git.

**Step 2: Write failing tests**

```python
"""Tests for evaluation.bet_sizing — signal → position sizing."""

import numpy as np
import polars as pl
import pytest


class TestBetSize:
    def test_zero_divergence_zero_size(self):
        from tradelab.lopezdp_utils.evaluation.bet_sizing import bet_size

        assert abs(bet_size(w=1.0, x=0.0)) < 1e-10

    def test_large_divergence_near_one(self):
        from tradelab.lopezdp_utils.evaluation.bet_sizing import bet_size

        result = bet_size(w=1.0, x=100.0)
        assert abs(result) > 0.99

    def test_output_bounded(self):
        from tradelab.lopezdp_utils.evaluation.bet_sizing import bet_size

        for x in np.linspace(-10, 10, 50):
            assert -1 < bet_size(w=1.0, x=x) < 1


class TestGetTargetPos:
    def test_returns_int(self):
        from tradelab.lopezdp_utils.evaluation.bet_sizing import get_target_pos

        result = get_target_pos(w=1.0, f=105.0, m_p=100.0, max_pos=10)
        assert isinstance(result, (int, np.integer))

    def test_max_pos_clipped(self):
        from tradelab.lopezdp_utils.evaluation.bet_sizing import get_target_pos

        result = get_target_pos(w=0.01, f=200.0, m_p=100.0, max_pos=5)
        assert abs(result) <= 5


class TestInvPrice:
    def test_roundtrip(self):
        from tradelab.lopezdp_utils.evaluation.bet_sizing import bet_size, inv_price

        w = 2.0
        f = 100.0
        m_p = 95.0
        x = f - m_p
        m = bet_size(w, x)
        recovered_mp = inv_price(f, w, m)
        assert abs(recovered_mp - m_p) < 1e-6


class TestLimitPrice:
    def test_returns_float(self):
        from tradelab.lopezdp_utils.evaluation.bet_sizing import limit_price

        result = limit_price(t_pos=3, pos=0, f=100.0, w=1.0, max_pos=10)
        assert isinstance(result, float)

    def test_symmetric_for_opposite_directions(self):
        from tradelab.lopezdp_utils.evaluation.bet_sizing import limit_price

        lp_long = limit_price(t_pos=3, pos=0, f=100.0, w=1.0, max_pos=10)
        lp_short = limit_price(t_pos=-3, pos=0, f=100.0, w=1.0, max_pos=10)
        # Symmetric around f=100: long entry should be below 100, short above
        assert lp_long < 100.0
        assert lp_short > 100.0


class TestGetW:
    def test_calibration(self):
        from tradelab.lopezdp_utils.evaluation.bet_sizing import bet_size, get_w

        # Calibrate w so that divergence x=5 → bet size m=0.5
        w = get_w(x=5.0, m=0.5)
        result = bet_size(w, x=5.0)
        assert abs(result - 0.5) < 1e-6


class TestGetSignal:
    def test_returns_polars_dataframe(self):
        from tradelab.lopezdp_utils.evaluation.bet_sizing import get_signal

        np.random.seed(42)
        n = 50
        timestamps = pl.datetime_range(
            pl.datetime(2024, 1, 1),
            pl.datetime(2024, 1, 1, 0, 49),
            interval="1m",
            eager=True,
        )
        events = pl.DataFrame({
            "timestamp": timestamps,
            "t1": timestamps.shift(-5, fill_value=timestamps[-1]),
            "side": np.random.choice([-1, 1], n),
        })
        prob = pl.Series("prob", np.random.uniform(0.5, 0.9, n))

        result = get_signal(events, step_size=0.1, prob=prob, num_classes=2)
        assert isinstance(result, pl.DataFrame)
        assert "signal" in result.columns

    def test_signals_bounded(self):
        from tradelab.lopezdp_utils.evaluation.bet_sizing import get_signal

        np.random.seed(42)
        n = 50
        timestamps = pl.datetime_range(
            pl.datetime(2024, 1, 1),
            pl.datetime(2024, 1, 1, 0, 49),
            interval="1m",
            eager=True,
        )
        events = pl.DataFrame({
            "timestamp": timestamps,
            "t1": timestamps.shift(-5, fill_value=timestamps[-1]),
            "side": np.random.choice([-1, 1], n),
        })
        prob = pl.Series("prob", np.random.uniform(0.5, 0.9, n))

        result = get_signal(events, step_size=0.0, prob=prob, num_classes=2)
        signals = result["signal"].drop_nulls()
        assert (signals.abs() <= 1.0).all()

    def test_validates_t1(self):
        from tradelab.lopezdp_utils.evaluation.bet_sizing import get_signal

        events = pl.DataFrame({"timestamp": [pl.datetime(2024, 1, 1)]})  # no t1
        prob = pl.Series("prob", [0.7])
        with pytest.raises(ValueError, match="t1"):
            get_signal(events, step_size=0.1, prob=prob, num_classes=2)


class TestDiscreteSignal:
    def test_discretization(self):
        from tradelab.lopezdp_utils.evaluation.bet_sizing import discrete_signal

        signal = pl.Series("signal", [0.37, -0.62, 0.15, -0.08])
        result = discrete_signal(signal, step_size=0.2)
        # 0.37 → 0.4, -0.62 → -0.6, 0.15 → 0.2, -0.08 → -0.0
        expected = [0.4, -0.6, 0.2, 0.0]
        for r, e in zip(result.to_list(), expected):
            assert abs(r - e) < 0.01

    def test_clipped_to_bounds(self):
        from tradelab.lopezdp_utils.evaluation.bet_sizing import discrete_signal

        signal = pl.Series("signal", [1.5, -1.5])
        result = discrete_signal(signal, step_size=0.1)
        assert result.max() <= 1.0
        assert result.min() >= -1.0


class TestAvgActiveSignals:
    def test_returns_polars_dataframe(self):
        from tradelab.lopezdp_utils.evaluation.bet_sizing import avg_active_signals

        timestamps = pl.datetime_range(
            pl.datetime(2024, 1, 1),
            pl.datetime(2024, 1, 1, 0, 9),
            interval="1m",
            eager=True,
        )
        signals = pl.DataFrame({
            "timestamp": [timestamps[0], timestamps[2], timestamps[5]],
            "t1": [timestamps[4], timestamps[6], timestamps[9]],
            "signal": [0.5, -0.3, 0.8],
        })
        result = avg_active_signals(signals)
        assert isinstance(result, pl.DataFrame)
        assert "signal" in result.columns
```

**Step 3: Implement `evaluation/bet_sizing.py`**

Implementation notes:
- **Dynamic sizing** (`bet_size`, `get_target_pos`, `inv_price`, `limit_price`, `get_w`): Pure math, copy from v1.
- **Signals** (`get_signal`, `avg_active_signals`, `discrete_signal`):
  - `get_signal`: Accept `pl.DataFrame` events with `timestamp`, `t1`, `side` columns. Accept `pl.Series` prob. Validate t1 exists and non-null. z-stat → Normal CDF → signal via Polars expressions. Call `avg_active_signals` then `discrete_signal`.
  - `avg_active_signals`: Accept `pl.DataFrame` with `timestamp`, `t1`, `signal`. At each unique timestamp, average signals where that timestamp is between the signal's start and t1. Use Polars cross-join + filter (or Python loop if too complex).
  - `discrete_signal`: Accept `pl.Series`, round to `step_size`, clip to `[-1, 1]`.
- If deferred `labeling/bet_sizing.py` has additional functions not covered by the above, merge them in.

**Step 4: Run tests**

Run: `uv run pytest tests/evaluation/test_bet_sizing.py -v`

**Step 5: Commit**

```bash
git add src/tradelab/lopezdp_utils/evaluation/bet_sizing.py tests/evaluation/test_bet_sizing.py
git commit -m "feat(evaluation): migrate bet_sizing with Polars signals"
```

---

### Task 8: Create `evaluation/__init__.py` with public exports

```python
"""Backtest evaluation, risk assessment, and bet sizing — AFML Chapters 10-15.

This package covers the final stage of López de Prado's pipeline:
model predictions → bet sizing → backtest evaluation → overfitting detection.

Key safety guardrails:
- DSR requires explicit num_trials (no default that hides multiple testing)
- CPCV generates complete OOS backtest paths for unbiased evaluation
- PBO quantifies the probability that in-sample performance is spurious
- Backtesting is a final sanity check, never a research tool

Reference:
    López de Prado, "Advances in Financial Machine Learning", Chapters 10-15
    López de Prado, "Machine Learning for Asset Managers", Section 8
"""

from tradelab.lopezdp_utils.evaluation.bet_sizing import (
    avg_active_signals,
    bet_size,
    discrete_signal,
    get_signal,
    get_target_pos,
    get_w,
    inv_price,
    limit_price,
)
from tradelab.lopezdp_utils.evaluation.cpcv import (
    CombinatorialPurgedKFold,
    assemble_backtest_paths,
    get_num_backtest_paths,
    get_num_splits,
)
from tradelab.lopezdp_utils.evaluation.overfitting import (
    probability_of_backtest_overfitting,
)
from tradelab.lopezdp_utils.evaluation.statistics import (
    compute_dd_tuw,
    deflated_sharpe_ratio,
    get_bet_timing,
    get_hhi,
    get_holding_period,
    multi_test_precision_recall,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
    strategy_precision,
    strategy_recall,
)
from tradelab.lopezdp_utils.evaluation.strategy_risk import (
    bin_freq,
    bin_hr,
    implied_precision_symmetric,
    mix_gaussians,
    prob_failure,
    sharpe_ratio_asymmetric,
    sharpe_ratio_symmetric,
)
from tradelab.lopezdp_utils.evaluation.synthetic import (
    otr_batch,
    otr_main,
    ou_fit,
    ou_half_life,
)

__all__ = [
    # Statistics
    "sharpe_ratio",
    "probabilistic_sharpe_ratio",
    "deflated_sharpe_ratio",
    "compute_dd_tuw",
    "get_hhi",
    "get_bet_timing",
    "get_holding_period",
    "strategy_precision",
    "strategy_recall",
    "multi_test_precision_recall",
    # CPCV
    "CombinatorialPurgedKFold",
    "assemble_backtest_paths",
    "get_num_splits",
    "get_num_backtest_paths",
    # Overfitting
    "probability_of_backtest_overfitting",
    # Synthetic
    "ou_half_life",
    "ou_fit",
    "otr_batch",
    "otr_main",
    # Strategy risk
    "sharpe_ratio_symmetric",
    "sharpe_ratio_asymmetric",
    "implied_precision_symmetric",
    "bin_hr",
    "bin_freq",
    "mix_gaussians",
    "prob_failure",
    # Bet sizing
    "get_signal",
    "avg_active_signals",
    "discrete_signal",
    "bet_size",
    "get_target_pos",
    "inv_price",
    "limit_price",
    "get_w",
]
```

---

### Task 9: Integration test — full evaluation pipeline

**Files:**
- Create: `tests/evaluation/test_integration.py`

```python
"""Integration tests: evaluation pipeline."""

import numpy as np
import polars as pl
import pytest


class TestSharpeToDeflatedPipeline:
    """SR → PSR → DSR with explicit trial accounting."""

    def test_sr_to_dsr(self, daily_returns):
        from tradelab.lopezdp_utils.evaluation.statistics import (
            deflated_sharpe_ratio,
            probabilistic_sharpe_ratio,
            sharpe_ratio,
        )

        sr = sharpe_ratio(daily_returns, periods_per_year=252)
        psr = probabilistic_sharpe_ratio(
            observed_sr=sr, benchmark_sr=0.0, n_obs=len(daily_returns),
        )
        dsr = deflated_sharpe_ratio(
            observed_sr=sr,
            sr_estimates=[sr * 0.5, sr * 0.8, sr],  # 3 trials
            n_obs=len(daily_returns),
        )
        # DSR should be lower than PSR (accounts for multiple testing)
        assert dsr <= psr


class TestBetSizingPipeline:
    """Signal → discrete signal → position."""

    def test_signal_to_position(self):
        from tradelab.lopezdp_utils.evaluation.bet_sizing import (
            bet_size,
            discrete_signal,
            get_target_pos,
            get_w,
        )

        # Step 1: Calibrate w
        w = get_w(x=5.0, m=0.5)

        # Step 2: Compute raw signal
        raw = bet_size(w, x=3.0)
        assert -1 < raw < 1

        # Step 3: Discretize
        signal = discrete_signal(pl.Series("s", [raw]), step_size=0.1)
        assert len(signal) == 1

        # Step 4: Get target position
        pos = get_target_pos(w, f=105.0, m_p=102.0, max_pos=10)
        assert isinstance(pos, (int, np.integer))


class TestOUToOTRPipeline:
    """Fit O-U → estimate half-life → optimal trading rule."""

    def test_fit_then_otr(self, price_series_ou):
        from tradelab.lopezdp_utils.evaluation.synthetic import (
            ou_fit,
            ou_half_life,
            otr_batch,
        )

        # Step 1: Fit O-U model
        params = ou_fit(price_series_ou, forecast=100.0)
        assert 0 < params["phi"] < 1

        # Step 2: Verify half-life
        hl = ou_half_life(params["phi"])
        assert hl > 0

        # Step 3: Run OTR
        result = otr_batch(
            coeffs=params | {"forecast": 100.0},
            n_iter=100,
            max_hp=int(hl * 3),
            r_pt=np.linspace(0.5, 2.0, 3),
            r_slm=np.linspace(0.5, 2.0, 3),
            seed=42,
        )
        assert len(result) == 9
        assert "sharpe" in result.columns
```

**Steps: Write → Run → Verify pass → Commit**

```bash
git commit -m "test(evaluation): add integration tests for evaluation pipeline"
```

---

### Task 10: Delete old directories and verify

**Step 1: Delete old modules**

```bash
rm -rf src/tradelab/lopezdp_utils/backtest_statistics/
rm -rf src/tradelab/lopezdp_utils/backtest_cv/
rm -rf src/tradelab/lopezdp_utils/backtesting_dangers/
rm -rf src/tradelab/lopezdp_utils/backtest_synthetic/
rm -rf src/tradelab/lopezdp_utils/strategy_risk/
rm -rf src/tradelab/lopezdp_utils/bet_sizing/
```

> **Also check for any remaining deferred files:**
> - `sample_weights/strategy_redundancy.py` — should be gone (merged into `evaluation/overfitting.py`)
> - `labeling/bet_sizing.py` — should be gone (merged into `evaluation/bet_sizing.py`)

**Step 2: Verify imports**

Run: `uv run python -c "from tradelab.lopezdp_utils.evaluation import deflated_sharpe_ratio, CombinatorialPurgedKFold, probability_of_backtest_overfitting, otr_batch, prob_failure, get_signal; print('OK')"`

**Step 3: Run full test suite**

Run: `uv run pytest tests/ -v`

**Step 4: Lint**

Run: `uvx ruff check --fix . && uvx ruff format .`

**Step 5: Commit**

```bash
git add -A
git commit -m "refactor(evaluation): remove old backtest_statistics/, backtest_cv/, backtesting_dangers/, backtest_synthetic/, strategy_risk/, bet_sizing/"
```

---

### Task 11: Merge to main

Run: `git checkout main && git merge phase2/evaluation`
Verify: `uv run pytest tests/ -v`
