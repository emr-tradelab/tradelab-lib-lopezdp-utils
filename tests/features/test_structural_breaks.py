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
