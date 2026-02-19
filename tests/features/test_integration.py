"""Integration tests: features pipeline."""

import polars as pl


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
