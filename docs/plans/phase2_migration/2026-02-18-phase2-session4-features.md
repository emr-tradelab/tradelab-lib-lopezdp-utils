# Phase 2 Session 4: `features/` — Feature Engineering

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Migrate `fractional_diff/` (2 files, 378 lines) + `entropy_features/` (4 files, 462 lines) + `structural_breaks/` (3 files, 573 lines) + `feature_importance/` (4 files, 844 lines) + deferred `data_structures/discretization.py` (192 lines) + `data_structures/pca.py` (64 lines) into `features/` package (5 files). This covers AFML Chapters 5, 8, 17-18 + MLAM Chapters 3-4, 6.

**Architecture:**
- Merge `fractional_diff/weights.py` + `fracdiff.py` → `features/fractional_diff.py`
- Merge `entropy_features/*` + `data_structures/discretization.py` → `features/entropy.py`
- Merge `structural_breaks/*` → `features/structural_breaks.py`
- Merge `feature_importance/importance.py` + `clustering.py` + `synthetic.py` → `features/importance.py`
- Merge `feature_importance/orthogonal.py` + `data_structures/pca.py` → `features/orthogonal.py`

**Tech Stack:** polars, numpy, scipy.stats, statsmodels (ADF), sklearn (KMeans, silhouette, mutual_info, make_classification, BaggingClassifier), matplotlib (optional plots), pydantic, pytest

**Depends on:** Session 3 (`labeling/` merged to main — `importance.py` imports `PurgedKFold` + `cv_score` from `cross_validation/`)

> **Note:** `feature_importance` depends on `cross_validation.PurgedKFold` (session 5). Since `PurgedKFold` is not yet migrated, keep the import as-is pointing to the v1 `cross_validation/` module. Session 5 will update the import path.

---

## Pre-Session Checklist

- [ ] Session 3 merged to main
- [ ] Branch from main: `git checkout -b phase2/features`
- [ ] `uv sync --all-extras --dev`

---

## File Mapping: Old → New

| Old file | New location | Action |
|----------|-------------|--------|
| `fractional_diff/weights.py` | `features/fractional_diff.py` | Merge into |
| `fractional_diff/fracdiff.py` | `features/fractional_diff.py` | Migrate |
| `entropy_features/estimators.py` | `features/entropy.py` | Merge into |
| `entropy_features/encoding.py` | `features/entropy.py` | Merge into |
| `entropy_features/applications.py` | `features/entropy.py` | Merge into |
| `entropy_features/information_theory.py` | `features/entropy.py` | Merge into |
| `data_structures/discretization.py` | `features/entropy.py` | Merge into |
| `structural_breaks/sadf.py` | `features/structural_breaks.py` | Merge into |
| `structural_breaks/cusum.py` | `features/structural_breaks.py` | Merge into |
| `structural_breaks/explosiveness.py` | `features/structural_breaks.py` | Merge into |
| `feature_importance/importance.py` | `features/importance.py` | Merge into |
| `feature_importance/clustering.py` | `features/importance.py` | Merge into |
| `feature_importance/synthetic.py` | `features/importance.py` | Merge into |
| `feature_importance/orthogonal.py` | `features/orthogonal.py` | Migrate |
| `data_structures/pca.py` | `features/orthogonal.py` | Merge into |

---

## Polars Migration Decisions (per function)

### features/fractional_diff.py
| Function | Migration | Notes |
|----------|-----------|-------|
| `get_weights(d, size)` | **Keep NumPy** | Pure array construction (iterative formula) |
| `get_weights_ffd(d, thres)` | **Keep NumPy** | Same — iterative weight generation |
| `frac_diff(series, d, thres)` | **Polars I/O, NumPy core** | Accept Polars DataFrame (timestamp + column), extract values to NumPy for convolution, return Polars. Expanding-window variant |
| `frac_diff_ffd(series, d, thres)` | **Polars I/O, NumPy core** | Same pattern. Fixed-window variant (recommended) |
| `plot_min_ffd(series, column, d_values, thres)` | **Polars I/O** | Accept Polars, delegate to `frac_diff_ffd` + `statsmodels.adfuller`. Return Polars DataFrame of ADF stats. Plot optional |
| `plot_weights(d_range, n_plots, size)` | **Keep as-is** | Pure matplotlib visualization, no data ops |

### features/entropy.py
| Function | Migration | Notes |
|----------|-----------|-------|
| `pmf1(msg, w)` | **Keep Python** | Pure string PMF construction |
| `plug_in(msg, w)` | **Keep Python** | Shannon entropy from string PMF |
| `lempel_ziv_lib(msg)` | **Keep Python** | String parsing. `# TODO(numba): evaluate JIT for LZ decomposition` |
| `match_length(msg, i, n)` | **Keep Python** | String search helper |
| `konto(msg, window)` | **Keep Python** | LZ entropy estimator on strings |
| `encode_binary(returns)` | **Polars I/O** | Accept Polars Series, encode to binary string |
| `encode_quantile(returns, num_letters)` | **Polars** | `qcut` → Polars equivalent or keep pandas for `pd.qcut` |
| `encode_sigma(returns, sigma_step)` | **Polars** | Bin arithmetic on Series |
| `market_efficiency_metric(returns)` | **Polars I/O** | Wraps encoding + konto |
| `portfolio_concentration(w, cov)` | **Keep NumPy** | Eigendecomposition of covariance matrix |
| `adverse_selection_feature(...)` | **Polars I/O** | Wraps encoding + konto |
| `kl_divergence(p, q)` | **Keep scipy** | `scipy.stats.entropy` passthrough |
| `cross_entropy(p, q)` | **Keep scipy** | Arithmetic on `kl_divergence` |
| `num_bins(n_obs, corr)` | **Keep NumPy** | Pure math formula |
| `discretize_optimal(x, n_bins)` | **Keep NumPy** | Binning with numpy |
| `variation_of_information(x, y)` | **Keep NumPy/sklearn** | Histogram + `mutual_info_score` |
| `mutual_information_optimal(x, y)` | **Keep NumPy/sklearn** | Same |

### features/structural_breaks.py
| Function | Migration | Notes |
|----------|-----------|-------|
| `lag_df(df0, lags)` | **Keep NumPy/pandas** | Builds lagged regression matrix — internal helper, not public |
| `get_y_x(series, constant, lags)` | **Keep NumPy/pandas** | Prepares ADF regression data — internal |
| `get_betas(y, x)` | **Keep NumPy** | OLS normal equations |
| `get_bsadf(log_p, min_sl, constant, lags)` | **Keep NumPy** | Inner SADF loop |
| `sadf_test(log_p, min_sl, constant, lags)` | **Polars I/O** | Accept Polars Series of log prices, return Polars DataFrame |
| `brown_durbin_evans_cusum(series, lags)` | **Polars I/O** | Accept Polars, internal OLS stays NumPy, return Polars |
| `chu_stinchcombe_white_cusum(series)` | **Polars I/O** | Accept Polars, return Polars |
| `chow_type_dickey_fuller(log_p, ...)` | **Polars I/O** | Accept/return Polars |
| `qadf_test(log_p, ...)` | **Polars I/O** | Accept/return Polars |
| `cadf_test(log_p, ...)` | **Polars I/O** | Accept/return Polars |

### features/importance.py
| Function | Migration | Notes |
|----------|-----------|-------|
| `feat_imp_mdi(fit, feat_names)` | **Keep pandas** | Extracts sklearn `feature_importances_` — sklearn returns arrays/DataFrames |
| `feat_imp_mda(clf, X, y, ...)` | **Keep pandas** | Uses `PurgedKFold`, sklearn CV pipeline — all pandas/numpy internally |
| `feat_imp_sfi(feat_names, clf, X, ...)` | **Keep pandas** | Single-feature CV loop |
| `cluster_kmeans_base(corr0, ...)` | **Keep pandas/sklearn** | KMeans + silhouette on distance matrix |
| `cluster_kmeans_top(corr0, ...)` | **Keep pandas/sklearn** | Recursive clustering |
| `feat_imp_mdi_clustered(fit, feat_names, clstrs)` | **Keep pandas** | Cluster-level MDI aggregation |
| `feat_imp_mda_clustered(clf, X, y, clstrs, ...)` | **Keep pandas** | Cluster-level MDA |
| `get_test_data(...)` | **Keep pandas** | `make_classification` wrapper |
| `feat_importance(...)` | **Keep pandas** | Unified MDI/MDA/SFI dispatcher |
| `test_func(...)` | **Keep as-is** | Pipeline test helper |
| `plot_feat_importance(...)` | **Keep matplotlib** | Visualization |

> **Rationale:** `importance.py` is deeply coupled to sklearn's cross-validation and ensemble APIs, which all operate on pandas/numpy. Converting I/O to Polars here adds complexity with no benefit — the real consumers are sklearn classifiers. Leave as pandas/numpy; users convert at the boundary.

### features/orthogonal.py
| Function | Migration | Notes |
|----------|-----------|-------|
| `pca_weights(cov, ...)` | **Keep NumPy** | Eigendecomposition of covariance matrix |
| `get_e_vec(dot, var_thres)` | **Keep NumPy** | Eigenvalue decomposition |
| `get_ortho_feats(dfX, var_thres)` | **Keep NumPy** | PCA projection — operates on arrays |
| `weighted_kendall_tau(feat_imp, pc_rank)` | **Keep scipy** | `weightedtau` wrapper |

---

## Tasks

### Task 1: Create branch, package structure, and test fixtures

**Files:**
- Create: `src/tradelab/lopezdp_utils/features/__init__.py`
- Create: `tests/features/__init__.py`
- Create: `tests/features/conftest.py`

**Step 1: Create branch**

Run: `git checkout -b phase2/features`

**Step 2: Create directories**

Run: `mkdir -p tests/features`

**Step 3: Create shared fixtures**

```python
"""Shared fixtures for features package tests."""

import numpy as np
import polars as pl
import pytest


@pytest.fixture
def price_series() -> pl.DataFrame:
    """500-bar close price series with realistic random walk for fracdiff/structural tests."""
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
def log_price_series(price_series) -> pl.Series:
    """Log prices for structural break tests."""
    return price_series.select(pl.col("close").log().alias("log_close"))["log_close"]


@pytest.fixture
def return_series() -> pl.Series:
    """1000-observation return series for entropy tests."""
    np.random.seed(42)
    returns = np.random.randn(1000) * 0.01
    return pl.Series("returns", returns)


@pytest.fixture
def explosive_series() -> pl.DataFrame:
    """Price series with an explosive (bubble) regime for SADF tests."""
    np.random.seed(42)
    n = 300
    timestamps = pl.datetime_range(
        pl.datetime(2024, 1, 1),
        pl.datetime(2024, 1, 1, 4, 59),
        interval="1m",
        eager=True,
    )
    # Random walk for 200 bars, then exponential growth for 100
    rw = 100.0 + np.cumsum(np.random.randn(200) * 0.3)
    bubble = rw[-1] * np.exp(np.linspace(0, 0.5, 100))
    close = np.concatenate([rw, bubble])
    return pl.DataFrame({"timestamp": timestamps, "close": close})


@pytest.fixture
def synthetic_features() -> tuple:
    """Synthetic classification data for feature importance tests."""
    from sklearn.datasets import make_classification

    np.random.seed(42)
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=3,
        n_redundant=2,
        n_clusters_per_class=1,
        random_state=42,
    )
    import pandas as pd

    feat_names = [f"f_{i}" for i in range(10)]
    X_df = pd.DataFrame(X, columns=feat_names)
    return X_df, pd.Series(y, name="label"), feat_names
```

**Step 4: Commit**

```bash
git add tests/features/ src/tradelab/lopezdp_utils/features/
git commit -m "test(features): add test skeleton and fixtures"
```

---

### Task 2: Migrate `fractional_diff.py` — weights + fracdiff + FFD + min_ffd

**Files:**
- Create: `src/tradelab/lopezdp_utils/features/fractional_diff.py`
- Create: `tests/features/test_fractional_diff.py`

**Step 1: Write failing tests**

```python
"""Tests for features.fractional_diff."""

import numpy as np
import polars as pl
import pytest


class TestGetWeights:
    def test_returns_ndarray(self):
        from tradelab.lopezdp_utils.features.fractional_diff import get_weights

        result = get_weights(d=0.5, size=10)
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 10

    def test_first_weight_is_one(self):
        from tradelab.lopezdp_utils.features.fractional_diff import get_weights

        result = get_weights(d=0.5, size=10)
        assert abs(result[0, 0] - 1.0) < 1e-10

    def test_weights_alternate_sign(self):
        """For 0 < d < 1, weights alternate in sign after the first."""
        from tradelab.lopezdp_utils.features.fractional_diff import get_weights

        result = get_weights(d=0.5, size=10).flatten()
        # First weight positive, second negative, etc.
        assert result[0] > 0
        assert result[1] < 0


class TestGetWeightsFFD:
    def test_returns_ndarray(self):
        from tradelab.lopezdp_utils.features.fractional_diff import get_weights_ffd

        result = get_weights_ffd(d=0.5, thres=1e-5)
        assert isinstance(result, np.ndarray)
        assert len(result) > 0

    def test_all_weights_above_threshold(self):
        from tradelab.lopezdp_utils.features.fractional_diff import get_weights_ffd

        thres = 1e-4
        result = get_weights_ffd(d=0.5, thres=thres)
        assert np.all(np.abs(result) >= thres)


class TestFracDiffFFD:
    def test_returns_polars_dataframe(self, price_series):
        from tradelab.lopezdp_utils.features.fractional_diff import frac_diff_ffd

        result = frac_diff_ffd(price_series, column="close", d=0.5, thres=1e-4)
        assert isinstance(result, pl.DataFrame)
        assert "close_ffd" in result.columns
        assert "timestamp" in result.columns

    def test_output_shorter_than_input(self, price_series):
        """FFD drops initial rows where the weight window doesn't fit."""
        from tradelab.lopezdp_utils.features.fractional_diff import frac_diff_ffd

        result = frac_diff_ffd(price_series, column="close", d=0.5, thres=1e-4)
        assert len(result) <= len(price_series)
        assert len(result) > 0

    def test_d_zero_returns_original(self, price_series):
        """d=0 means no differentiation — output should equal input."""
        from tradelab.lopezdp_utils.features.fractional_diff import frac_diff_ffd

        result = frac_diff_ffd(price_series, column="close", d=0.0, thres=1e-4)
        original = price_series["close"][:len(result)]
        np.testing.assert_allclose(
            result["close_ffd"].to_numpy(),
            original.to_numpy(),
            atol=1e-8,
        )

    def test_d_one_returns_first_diff(self, price_series):
        """d=1 should approximate first differences."""
        from tradelab.lopezdp_utils.features.fractional_diff import frac_diff_ffd

        result = frac_diff_ffd(price_series, column="close", d=1.0, thres=1e-4)
        # Compare with manual first difference (approximate due to window truncation)
        manual_diff = price_series["close"].diff().drop_nulls()
        # Should be highly correlated
        corr = np.corrcoef(
            result["close_ffd"].to_numpy()[-100:],
            manual_diff.to_numpy()[-100:],
        )[0, 1]
        assert corr > 0.99


class TestPlotMinFFD:
    def test_returns_polars_dataframe(self, price_series):
        from tradelab.lopezdp_utils.features.fractional_diff import plot_min_ffd

        result = plot_min_ffd(
            price_series, column="close", d_values=np.linspace(0.0, 1.0, 5), thres=1e-4
        )
        assert isinstance(result, pl.DataFrame)
        assert "d" in result.columns
        assert "adf_stat" in result.columns
        assert "p_value" in result.columns
```

**Step 2: Run to verify fail**

Run: `uv run pytest tests/features/test_fractional_diff.py -v`
Expected: FAIL

**Step 3: Implement `features/fractional_diff.py`**

Implementation notes:
- `get_weights` / `get_weights_ffd`: Pure NumPy, copy from v1 with snake_case cleanup.
- `frac_diff_ffd`: Accept `pl.DataFrame` with timestamp + value column. Extract NumPy array, apply FFD convolution with the fixed weight vector. Return `pl.DataFrame` with `timestamp` + `{column}_ffd`. Add `# TODO(numba): evaluate JIT for FFD convolution loop`.
- `frac_diff`: Same pattern as FFD but expanding window. Keep for completeness but docstring should note FFD is preferred.
- `plot_min_ffd`: Accept Polars, iterate d-values calling `frac_diff_ffd` + `statsmodels.adfuller`. Return `pl.DataFrame` with columns: `d`, `adf_stat`, `p_value`, `lags`, `n_obs`, `confidence_95pct`, `corr_with_original`. Plotting is optional (only if matplotlib available).
- `plot_weights`: Keep as pure matplotlib visualization — no migration needed.

**Step 4: Run tests**

Run: `uv run pytest tests/features/test_fractional_diff.py -v`

**Step 5: Commit**

```bash
git add src/tradelab/lopezdp_utils/features/fractional_diff.py tests/features/test_fractional_diff.py
git commit -m "feat(features): migrate fractional_diff with Polars I/O"
```

---

### Task 3: Migrate `entropy.py` — estimators + encoding + applications + discretization + information theory

**Files:**
- Create: `src/tradelab/lopezdp_utils/features/entropy.py`
- Create: `tests/features/test_entropy.py`

**Step 1: Write failing tests**

```python
"""Tests for features.entropy."""

import numpy as np
import polars as pl
import pytest


class TestPlugIn:
    def test_uniform_distribution_max_entropy(self):
        """A message with all unique substrings should have high entropy."""
        from tradelab.lopezdp_utils.features.entropy import plug_in

        msg = "0123456789" * 10
        entropy, _ = plug_in(msg, w=1)
        assert entropy > 0

    def test_constant_message_zero_entropy(self):
        """A constant message should have zero entropy."""
        from tradelab.lopezdp_utils.features.entropy import plug_in

        msg = "0" * 100
        entropy, _ = plug_in(msg, w=1)
        assert abs(entropy) < 1e-10


class TestLempelZivLib:
    def test_returns_list(self):
        from tradelab.lopezdp_utils.features.entropy import lempel_ziv_lib

        result = lempel_ziv_lib("1011010100010")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_constant_message_small_library(self):
        from tradelab.lopezdp_utils.features.entropy import lempel_ziv_lib

        result = lempel_ziv_lib("0" * 100)
        assert len(result) <= 3  # very compressible


class TestKonto:
    def test_returns_dict(self):
        from tradelab.lopezdp_utils.features.entropy import konto

        np.random.seed(42)
        msg = "".join(str(x) for x in np.random.randint(0, 2, 200))
        result = konto(msg, window=None)
        assert "h" in result
        assert "r" in result

    def test_random_message_high_entropy(self):
        from tradelab.lopezdp_utils.features.entropy import konto

        np.random.seed(42)
        msg = "".join(str(x) for x in np.random.randint(0, 2, 200))
        result = konto(msg, window=None)
        assert result["h"] > 0.5  # should be close to 1.0 for random binary


class TestEncodeBinary:
    def test_returns_string(self, return_series):
        from tradelab.lopezdp_utils.features.entropy import encode_binary

        result = encode_binary(return_series)
        assert isinstance(result, str)
        assert set(result).issubset({"0", "1"})

    def test_length(self, return_series):
        from tradelab.lopezdp_utils.features.entropy import encode_binary

        result = encode_binary(return_series)
        # Zeros are removed, so length <= input length
        assert len(result) <= len(return_series)


class TestEncodeQuantile:
    def test_returns_string(self, return_series):
        from tradelab.lopezdp_utils.features.entropy import encode_quantile

        result = encode_quantile(return_series, num_letters=10)
        assert isinstance(result, str)
        assert len(result) == len(return_series)


class TestMarketEfficiencyMetric:
    def test_returns_dict(self, return_series):
        from tradelab.lopezdp_utils.features.entropy import market_efficiency_metric

        result = market_efficiency_metric(return_series, encoding="binary")
        assert isinstance(result, dict)
        assert "entropy_rate" in result
        assert "redundancy" in result

    def test_redundancy_between_zero_and_one(self, return_series):
        from tradelab.lopezdp_utils.features.entropy import market_efficiency_metric

        result = market_efficiency_metric(return_series, encoding="binary")
        assert 0 <= result["redundancy"] <= 1


class TestNumBins:
    def test_returns_int(self):
        from tradelab.lopezdp_utils.features.entropy import num_bins

        result = num_bins(n_obs=1000, corr=None)
        assert isinstance(result, int)
        assert result > 0

    def test_bivariate_adjusts_for_correlation(self):
        from tradelab.lopezdp_utils.features.entropy import num_bins

        bins_uncorr = num_bins(n_obs=1000, corr=0.0)
        bins_corr = num_bins(n_obs=1000, corr=0.9)
        # High correlation should yield fewer bins
        assert bins_corr <= bins_uncorr


class TestVariationOfInformation:
    def test_identical_variables_zero_vi(self):
        from tradelab.lopezdp_utils.features.entropy import variation_of_information

        np.random.seed(42)
        x = np.random.randn(500)
        result = variation_of_information(x, x, normalize=True)
        assert abs(result) < 0.1  # should be close to 0


class TestMutualInformationOptimal:
    def test_independent_variables_low_mi(self):
        from tradelab.lopezdp_utils.features.entropy import mutual_information_optimal

        np.random.seed(42)
        x = np.random.randn(500)
        y = np.random.randn(500)
        result = mutual_information_optimal(x, y)
        assert result < 0.1  # nearly independent

    def test_dependent_variables_high_mi(self):
        from tradelab.lopezdp_utils.features.entropy import mutual_information_optimal

        np.random.seed(42)
        x = np.random.randn(500)
        y = x + np.random.randn(500) * 0.1  # highly correlated
        result = mutual_information_optimal(x, y)
        assert result > 0.5
```

**Step 2: Run to verify fail**

Run: `uv run pytest tests/features/test_entropy.py -v`

**Step 3: Implement `features/entropy.py`**

Implementation notes:
- **Estimators** (`pmf1`, `plug_in`, `lempel_ziv_lib`, `match_length`, `konto`): Keep as pure Python string operations. Add `# TODO(numba): evaluate JIT for LZ decomposition`.
- **Encoding** (`encode_binary`, `encode_quantile`, `encode_sigma`): Accept `pl.Series`. `encode_binary`: filter > 0 → "1", else "0". `encode_quantile`: use `pandas.qcut` internally (no clean Polars equivalent for equal-frequency binning). `encode_sigma`: Polars arithmetic.
- **Applications** (`market_efficiency_metric`, `portfolio_concentration`, `adverse_selection_feature`): `market_efficiency_metric` and `adverse_selection_feature` accept `pl.Series`, delegate to encoding + konto. `portfolio_concentration` stays NumPy (eigendecomposition).
- **Information theory** (`kl_divergence`, `cross_entropy`): Keep as scipy wrappers.
- **Discretization** (`num_bins`, `discretize_optimal`, `variation_of_information`, `mutual_information_optimal`): Keep as NumPy/sklearn — these operate on arrays, not DataFrames.

**Step 4: Run tests**

Run: `uv run pytest tests/features/test_entropy.py -v`

**Step 5: Commit**

```bash
git add src/tradelab/lopezdp_utils/features/entropy.py tests/features/test_entropy.py
git commit -m "feat(features): migrate entropy with Polars encoding and discretization"
```

---

### Task 4: Migrate `structural_breaks.py` — SADF, CUSUM tests, explosiveness

**Files:**
- Create: `src/tradelab/lopezdp_utils/features/structural_breaks.py`
- Create: `tests/features/test_structural_breaks.py`

**Step 1: Write failing tests**

```python
"""Tests for features.structural_breaks."""

import numpy as np
import polars as pl
import pytest


class TestSADFTest:
    def test_returns_polars_dataframe(self, explosive_series):
        from tradelab.lopezdp_utils.features.structural_breaks import sadf_test

        log_p = explosive_series.select(pl.col("close").log().alias("log_close"))["log_close"]
        result = sadf_test(log_p, min_sl=50, constant="c", lags=1)
        assert isinstance(result, pl.DataFrame)
        assert "sadf" in result.columns

    def test_detects_bubble(self, explosive_series):
        """SADF should spike during the explosive regime."""
        from tradelab.lopezdp_utils.features.structural_breaks import sadf_test

        log_p = explosive_series.select(pl.col("close").log().alias("log_close"))["log_close"]
        result = sadf_test(log_p, min_sl=50, constant="c", lags=1)
        # The maximum SADF value should be in the bubble period (last 100 bars)
        max_sadf = result["sadf"].max()
        assert max_sadf > 0  # explosive series should produce positive SADF

    def test_random_walk_no_explosion(self, price_series):
        """A random walk should not trigger large SADF values."""
        from tradelab.lopezdp_utils.features.structural_breaks import sadf_test

        log_p = price_series.select(pl.col("close").log().alias("log_close"))["log_close"]
        result = sadf_test(log_p, min_sl=50, constant="c", lags=1)
        # SADF should be moderate for a random walk
        max_sadf = result["sadf"].max()
        assert max_sadf < 5  # not explosive


class TestBrownDurbinEvansCUSUM:
    def test_returns_polars_dataframe(self, price_series):
        from tradelab.lopezdp_utils.features.structural_breaks import brown_durbin_evans_cusum

        result = brown_durbin_evans_cusum(price_series["close"], lags=5)
        assert isinstance(result, pl.DataFrame)
        assert "s_t" in result.columns
        assert "upper" in result.columns
        assert "lower" in result.columns


class TestChuStinchcombeWhiteCUSUM:
    def test_returns_polars_dataframe(self, price_series):
        from tradelab.lopezdp_utils.features.structural_breaks import (
            chu_stinchcombe_white_cusum,
        )

        result = chu_stinchcombe_white_cusum(price_series["close"], critical_value=4.6)
        assert isinstance(result, pl.DataFrame)
        assert "s_t" in result.columns
        assert "critical" in result.columns


class TestChowTypeDickeyFuller:
    def test_returns_polars_dataframe(self, explosive_series):
        from tradelab.lopezdp_utils.features.structural_breaks import chow_type_dickey_fuller

        log_p = explosive_series.select(pl.col("close").log().alias("log_close"))["log_close"]
        result = chow_type_dickey_fuller(log_p, min_sl=50, constant="c", lags=1)
        assert isinstance(result, pl.DataFrame)
        assert "dfc" in result.columns


class TestQADFTest:
    def test_returns_polars_dataframe(self, explosive_series):
        from tradelab.lopezdp_utils.features.structural_breaks import qadf_test

        log_p = explosive_series.select(pl.col("close").log().alias("log_close"))["log_close"]
        result = qadf_test(log_p, min_sl=50, q=0.95, constant="c", lags=1)
        assert isinstance(result, pl.DataFrame)
        assert "qadf" in result.columns


class TestGetBetas:
    def test_returns_correct_shape(self):
        """OLS on simple data should return correct coefficient shape."""
        from tradelab.lopezdp_utils.features.structural_breaks import get_betas

        np.random.seed(42)
        n = 100
        x = np.column_stack([np.ones(n), np.random.randn(n)])
        y = 2.0 + 3.0 * x[:, 1] + np.random.randn(n) * 0.1
        b_mean, b_var = get_betas(y, x)
        assert len(b_mean) == 2
        assert b_var.shape == (2, 2)
        assert abs(b_mean[0] - 2.0) < 0.5
        assert abs(b_mean[1] - 3.0) < 0.5
```

**Step 2: Run to verify fail**

Run: `uv run pytest tests/features/test_structural_breaks.py -v`

**Step 3: Implement `features/structural_breaks.py`**

Implementation notes:
- **Internal helpers** (`lag_df`, `get_y_x`, `get_betas`, `get_bsadf`): Prefix with `_` (private). Keep as NumPy/pandas internally — these build regression matrices for OLS.
- **Public functions** (`sadf_test`, `brown_durbin_evans_cusum`, `chu_stinchcombe_white_cusum`, `chow_type_dickey_fuller`, `qadf_test`, `cadf_test`): Accept `pl.Series` of prices/log-prices. Convert to pandas internally for the regression loops. Return `pl.DataFrame` with results.
- `get_betas` stays public (useful for custom ADF regressions).
- Column naming: use snake_case (`s_t`, `sadf`, `dfc`, `qadf`, `cadf`).

**Step 4: Run tests**

Run: `uv run pytest tests/features/test_structural_breaks.py -v`

**Step 5: Commit**

```bash
git add src/tradelab/lopezdp_utils/features/structural_breaks.py tests/features/test_structural_breaks.py
git commit -m "feat(features): migrate structural_breaks with Polars I/O"
```

---

### Task 5: Migrate `importance.py` — MDI, MDA, SFI, clustering, synthetic

**Files:**
- Create: `src/tradelab/lopezdp_utils/features/importance.py`
- Create: `tests/features/test_importance.py`

**Step 1: Write failing tests**

```python
"""Tests for features.importance — MDI, MDA, SFI, clustering."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier


class TestFeatImpMDI:
    def test_returns_dataframe(self, synthetic_features):
        from tradelab.lopezdp_utils.features.importance import feat_imp_mdi

        X, y, feat_names = synthetic_features
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X, y)
        result = feat_imp_mdi(clf, feat_names)
        assert isinstance(result, pd.DataFrame)
        assert "mean" in result.columns
        assert len(result) == len(feat_names)

    def test_importances_sum_to_one(self, synthetic_features):
        from tradelab.lopezdp_utils.features.importance import feat_imp_mdi

        X, y, feat_names = synthetic_features
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X, y)
        result = feat_imp_mdi(clf, feat_names)
        assert abs(result["mean"].sum() - 1.0) < 0.05


class TestFeatImpMDA:
    def test_returns_dataframe_and_score(self, synthetic_features):
        from tradelab.lopezdp_utils.features.importance import feat_imp_mda
        from sklearn.model_selection import KFold

        X, y, feat_names = synthetic_features
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        cv = KFold(n_splits=3)
        result, score = feat_imp_mda(
            clf, X, y, cv=cv, scoring="accuracy",
        )
        assert isinstance(result, pd.DataFrame)
        assert "mean" in result.columns
        assert isinstance(score, float)


class TestClusterKMeansBase:
    def test_returns_clusters(self):
        from tradelab.lopezdp_utils.features.importance import cluster_kmeans_base

        np.random.seed(42)
        # Build a correlation matrix with 2 clear clusters
        n = 100
        x1 = np.random.randn(n)
        x2 = x1 + np.random.randn(n) * 0.1  # correlated with x1
        x3 = np.random.randn(n)  # independent
        x4 = x3 + np.random.randn(n) * 0.1  # correlated with x3
        data = pd.DataFrame({"a": x1, "b": x2, "c": x3, "d": x4})
        corr = data.corr()
        _, clstrs, _ = cluster_kmeans_base(corr, max_num_clusters=4)
        assert isinstance(clstrs, dict)
        assert len(clstrs) >= 2


class TestGetTestData:
    def test_returns_expected_shape(self):
        from tradelab.lopezdp_utils.features.importance import get_test_data

        trns_x, cont = get_test_data(
            n_features=10, n_informative=3, n_redundant=2, n_samples=200,
        )
        assert isinstance(trns_x, pd.DataFrame)
        assert trns_x.shape == (200, 10)
        assert "bin" in cont.columns
        assert "t1" in cont.columns
```

**Step 2: Run to verify fail**

Run: `uv run pytest tests/features/test_importance.py -v`

**Step 3: Implement `features/importance.py`**

Implementation notes:
- This module stays **pandas/numpy/sklearn** throughout — see rationale in migration decisions above.
- Copy functions from v1 with snake_case cleanup and consolidated imports.
- `feat_imp_mdi`, `feat_imp_mda`, `feat_imp_sfi`: Keep as-is with pandas I/O.
- `cluster_kmeans_base`, `cluster_kmeans_top`, `feat_imp_mdi_clustered`, `feat_imp_mda_clustered`: Keep as-is.
- `get_test_data`, `feat_importance`, `test_func`, `plot_feat_importance`: Keep as-is.
- Import `PurgedKFold` and `cv_score` from `cross_validation` (v1 path for now — updated in session 5).

**Step 4: Run tests**

Run: `uv run pytest tests/features/test_importance.py -v`

**Step 5: Commit**

```bash
git add src/tradelab/lopezdp_utils/features/importance.py tests/features/test_importance.py
git commit -m "feat(features): migrate importance with clustering (pandas/sklearn)"
```

---

### Task 6: Migrate `orthogonal.py` — PCA features + portfolio PCA weights

**Files:**
- Create: `src/tradelab/lopezdp_utils/features/orthogonal.py`
- Create: `tests/features/test_orthogonal.py`

**Step 1: Write failing tests**

```python
"""Tests for features.orthogonal — PCA orthogonalization and portfolio weights."""

import numpy as np
import pandas as pd
import pytest


class TestGetOrthoFeats:
    def test_returns_array(self):
        from tradelab.lopezdp_utils.features.orthogonal import get_ortho_feats

        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 5), columns=[f"f{i}" for i in range(5)])
        result = get_ortho_feats(X, var_thres=0.95)
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 100

    def test_orthogonality(self):
        """Output PCs should be uncorrelated."""
        from tradelab.lopezdp_utils.features.orthogonal import get_ortho_feats

        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(200, 5), columns=[f"f{i}" for i in range(5)])
        result = get_ortho_feats(X, var_thres=0.95)
        corr = np.corrcoef(result.T)
        # Off-diagonal elements should be near zero
        np.fill_diagonal(corr, 0)
        assert np.max(np.abs(corr)) < 0.1


class TestPCAWeights:
    def test_returns_ndarray(self):
        from tradelab.lopezdp_utils.features.orthogonal import pca_weights

        np.random.seed(42)
        cov = np.eye(3) + np.random.randn(3, 3) * 0.01
        cov = (cov + cov.T) / 2  # ensure symmetric
        result = pca_weights(cov)
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 3

    def test_weights_not_all_zero(self):
        from tradelab.lopezdp_utils.features.orthogonal import pca_weights

        cov = np.eye(3)
        result = pca_weights(cov)
        assert np.sum(np.abs(result)) > 0


class TestWeightedKendallTau:
    def test_returns_correlation(self):
        from tradelab.lopezdp_utils.features.orthogonal import weighted_kendall_tau

        feat_imp = pd.Series([0.5, 0.3, 0.1, 0.05, 0.05])
        pc_rank = pd.Series([1, 2, 3, 4, 5])
        result = weighted_kendall_tau(feat_imp, pc_rank)
        assert isinstance(result, float)
        assert -1 <= result <= 1
```

**Step 2: Run to verify fail**

Run: `uv run pytest tests/features/test_orthogonal.py -v`

**Step 3: Implement `features/orthogonal.py`**

Implementation notes:
- `pca_weights` from `data_structures/pca.py`: Pure NumPy eigendecomposition. Keep as-is.
- `get_e_vec`, `get_ortho_feats`: Pure NumPy/pandas. Keep as-is (accept pandas DataFrame since sklearn pipeline uses pandas).
- `weighted_kendall_tau`: scipy wrapper. Keep as-is.

**Step 4: Run tests**

Run: `uv run pytest tests/features/test_orthogonal.py -v`

**Step 5: Commit**

```bash
git add src/tradelab/lopezdp_utils/features/orthogonal.py tests/features/test_orthogonal.py
git commit -m "feat(features): migrate orthogonal with PCA weights"
```

---

### Task 7: Create `features/__init__.py` with public exports

```python
"""Feature engineering — AFML Chapters 5, 8, 17-18 + MLAM Chapters 3-4, 6.

This package covers the fourth stage of López de Prado's pipeline:
labeled data → engineered features (stationarity, entropy, structural breaks,
importance analysis, orthogonalization).

Reference:
    López de Prado, "Advances in Financial Machine Learning", Chapters 5, 8, 17-18
    López de Prado, "Machine Learning for Asset Managers", Chapters 3-4, 6
"""

from tradelab.lopezdp_utils.features.entropy import (
    adverse_selection_feature,
    cross_entropy,
    discretize_optimal,
    encode_binary,
    encode_quantile,
    encode_sigma,
    kl_divergence,
    konto,
    lempel_ziv_lib,
    market_efficiency_metric,
    mutual_information_optimal,
    num_bins,
    plug_in,
    portfolio_concentration,
    variation_of_information,
)
from tradelab.lopezdp_utils.features.fractional_diff import (
    frac_diff,
    frac_diff_ffd,
    get_weights,
    get_weights_ffd,
    plot_min_ffd,
)
from tradelab.lopezdp_utils.features.importance import (
    cluster_kmeans_base,
    cluster_kmeans_top,
    feat_imp_mda,
    feat_imp_mda_clustered,
    feat_imp_mdi,
    feat_imp_mdi_clustered,
    feat_imp_sfi,
    feat_importance,
    get_test_data,
)
from tradelab.lopezdp_utils.features.orthogonal import (
    get_ortho_feats,
    pca_weights,
    weighted_kendall_tau,
)
from tradelab.lopezdp_utils.features.structural_breaks import (
    brown_durbin_evans_cusum,
    cadf_test,
    chow_type_dickey_fuller,
    chu_stinchcombe_white_cusum,
    get_betas,
    qadf_test,
    sadf_test,
)

__all__ = [
    # Fractional differentiation
    "get_weights",
    "get_weights_ffd",
    "frac_diff",
    "frac_diff_ffd",
    "plot_min_ffd",
    # Entropy & information theory
    "plug_in",
    "lempel_ziv_lib",
    "konto",
    "encode_binary",
    "encode_quantile",
    "encode_sigma",
    "market_efficiency_metric",
    "portfolio_concentration",
    "adverse_selection_feature",
    "kl_divergence",
    "cross_entropy",
    "num_bins",
    "discretize_optimal",
    "variation_of_information",
    "mutual_information_optimal",
    # Structural breaks
    "sadf_test",
    "brown_durbin_evans_cusum",
    "chu_stinchcombe_white_cusum",
    "chow_type_dickey_fuller",
    "qadf_test",
    "cadf_test",
    "get_betas",
    # Feature importance
    "feat_imp_mdi",
    "feat_imp_mda",
    "feat_imp_sfi",
    "cluster_kmeans_base",
    "cluster_kmeans_top",
    "feat_imp_mdi_clustered",
    "feat_imp_mda_clustered",
    "get_test_data",
    "feat_importance",
    # Orthogonal features
    "get_ortho_feats",
    "pca_weights",
    "weighted_kendall_tau",
]
```

---

### Task 8: Integration test — fracdiff → structural breaks pipeline

**Files:**
- Create: `tests/features/test_integration.py`

```python
"""Integration tests: features pipeline."""

import numpy as np
import polars as pl
import pytest


class TestFracdiffToStationarity:
    """FFD should produce stationary series from non-stationary prices."""

    def test_ffd_series_is_more_stationary(self, price_series):
        from statsmodels.tsa.stattools import adfuller

        from tradelab.lopezdp_utils.features.fractional_diff import frac_diff_ffd

        # Original series: likely non-stationary (random walk)
        original = price_series["close"].to_numpy()
        adf_original = adfuller(original, maxlag=10)[1]  # p-value

        # FFD with d=0.5: should be more stationary
        ffd = frac_diff_ffd(price_series, column="close", d=0.5, thres=1e-4)
        adf_ffd = adfuller(ffd["close_ffd"].to_numpy(), maxlag=10)[1]

        # FFD p-value should be lower (more stationary)
        assert adf_ffd < adf_original


class TestEntropyPipeline:
    """Encoding → entropy estimation pipeline."""

    def test_encode_then_estimate(self, return_series):
        from tradelab.lopezdp_utils.features.entropy import (
            encode_binary,
            konto,
            plug_in,
        )

        msg = encode_binary(return_series)
        assert len(msg) > 0

        # Both estimators should produce entropy > 0
        h_plugin, _ = plug_in(msg, w=1)
        h_konto = konto(msg, window=None)
        assert h_plugin > 0
        assert h_konto["h"] > 0


class TestSADFOnExplosive:
    """SADF should detect the bubble in the explosive series."""

    def test_sadf_detects_regime_change(self, explosive_series):
        from tradelab.lopezdp_utils.features.structural_breaks import sadf_test

        log_p = explosive_series.select(
            pl.col("close").log().alias("log_close")
        )["log_close"]
        result = sadf_test(log_p, min_sl=50, constant="c", lags=1)
        assert len(result) > 0
        assert result["sadf"].max() > 0
```

**Steps: Write → Run → Verify pass → Commit**

```bash
git commit -m "test(features): add integration tests for features pipeline"
```

---

### Task 9: Delete old directories and verify

**Step 1: Delete old modules**

```bash
rm -rf src/tradelab/lopezdp_utils/fractional_diff/
rm -rf src/tradelab/lopezdp_utils/entropy_features/
rm -rf src/tradelab/lopezdp_utils/structural_breaks/
rm -rf src/tradelab/lopezdp_utils/feature_importance/
rm -rf src/tradelab/lopezdp_utils/data_structures/  # discretization.py and pca.py are the last files
```

> **Warning:** `data_structures/` deletion — verify that `data/__init__.py` does NOT import from `data_structures/` anymore (it shouldn't — session 2 already migrated bars/sampling).

**Step 2: Verify imports**

Run: `uv run python -c "from tradelab.lopezdp_utils.features import frac_diff_ffd, sadf_test, feat_imp_mdi, pca_weights; print('OK')"`

**Step 3: Run full test suite**

Run: `uv run pytest tests/ -v`

**Step 4: Lint**

Run: `uvx ruff check --fix . && uvx ruff format .`

**Step 5: Commit**

```bash
git add -A
git commit -m "refactor(features): remove old fractional_diff/, entropy_features/, structural_breaks/, feature_importance/, data_structures/"
```

---

### Task 10: Merge to main

Run: `git checkout main && git merge phase2/features`
Verify: `uv run pytest tests/ -v`
