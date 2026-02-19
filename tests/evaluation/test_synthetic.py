"""Tests for evaluation.synthetic — O-U process and Optimal Trading Rule."""

import numpy as np
import polars as pl
import pytest


class TestOUHalfLife:
    def test_known_value(self):
        from tradelab.lopezdp_utils.evaluation.synthetic import ou_half_life

        result = ou_half_life(phi=0.5)
        assert abs(result - 1.0) < 1e-10

    def test_high_phi_long_half_life(self):
        from tradelab.lopezdp_utils.evaluation.synthetic import ou_half_life

        result = ou_half_life(phi=0.99)
        assert result > 50

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
        assert 0.85 < result["phi"] < 1.0


class TestOTRBatch:
    def test_returns_polars_dataframe(self):
        from tradelab.lopezdp_utils.evaluation.synthetic import otr_batch

        result = otr_batch(
            coeffs={"forecast": 100.0, "hl": 10.0, "sigma": 0.5},
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
            coeffs={"forecast": 100.0, "hl": 10.0, "sigma": 0.5},
            n_iter=50,
            max_hp=30,
            r_pt=r_pt,
            r_slm=r_slm,
            seed=42,
        )
        assert len(result) == 12


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
        assert len(result) == 2
        for key, df in result.items():
            assert isinstance(df, pl.DataFrame)
