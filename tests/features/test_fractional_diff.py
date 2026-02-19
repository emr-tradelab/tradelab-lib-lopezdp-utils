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

    def test_most_recent_weight_is_one(self):
        from tradelab.lopezdp_utils.features.fractional_diff import get_weights

        # Weights are in reverse chronological order: last element = most recent = omega_0 = 1
        result = get_weights(d=0.5, size=10)
        assert abs(result[-1, 0] - 1.0) < 1e-10

    def test_weights_alternate_sign(self):
        """For 0 < d < 1, weights alternate in sign."""
        from tradelab.lopezdp_utils.features.fractional_diff import get_weights

        result = get_weights(d=0.5, size=10).flatten()
        # Most recent weight (omega_0) positive, next (omega_1) negative
        assert result[-1] > 0
        assert result[-2] < 0


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
        original = price_series["close"][: len(result)]
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


class TestFracDiff:
    def test_returns_polars_dataframe(self, price_series):
        from tradelab.lopezdp_utils.features.fractional_diff import frac_diff

        result = frac_diff(price_series, column="close", d=0.5, thres=0.01)
        assert isinstance(result, pl.DataFrame)
        assert "close_ffd" in result.columns
        assert "timestamp" in result.columns

    def test_output_shorter_than_input(self, price_series):
        from tradelab.lopezdp_utils.features.fractional_diff import frac_diff

        result = frac_diff(price_series, column="close", d=0.5, thres=0.01)
        assert len(result) <= len(price_series)
        assert len(result) > 0


class TestPlotMinFFD:
    def test_returns_polars_dataframe(self, price_series):
        from tradelab.lopezdp_utils.features.fractional_diff import plot_min_ffd

        result = plot_min_ffd(
            price_series,
            column="close",
            d_values=np.linspace(0.0, 1.0, 5),
            thres=1e-4,
        )
        assert isinstance(result, pl.DataFrame)
        assert "d" in result.columns
        assert "adf_stat" in result.columns
        assert "p_value" in result.columns
