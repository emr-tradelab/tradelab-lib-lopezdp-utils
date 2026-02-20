"""Tests for evaluation.strategy_risk — binomial model for strategy viability."""

import numpy as np
import polars as pl


class TestSharpeRatioSymmetric:
    def test_coin_flip_zero(self):
        from tradelab.lopezdp_utils.evaluation.strategy_risk import sharpe_ratio_symmetric

        result = sharpe_ratio_symmetric(p=0.5, n=252)
        assert abs(result) < 1e-10

    def test_high_precision(self):
        from tradelab.lopezdp_utils.evaluation.strategy_risk import sharpe_ratio_symmetric

        result = sharpe_ratio_symmetric(p=0.99, n=252)
        assert result > 10


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
            mu1=0.01,
            mu2=-0.01,
            sigma1=0.02,
            sigma2=0.03,
            prob1=0.6,
            n_obs=1000,
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

    def test_all_positive_returns_nan(self):
        from tradelab.lopezdp_utils.evaluation.strategy_risk import prob_failure

        ret = pl.Series("ret", [0.01] * 100)
        result = prob_failure(ret, freq=252, target_sr=1.0)
        assert np.isnan(result)

    def test_all_negative_returns_nan(self):
        from tradelab.lopezdp_utils.evaluation.strategy_risk import prob_failure

        ret = pl.Series("ret", [-0.01] * 100)
        result = prob_failure(ret, freq=252, target_sr=1.0)
        assert np.isnan(result)


class TestMixGaussiansReproducibility:
    def test_seed_reproducible(self):
        from tradelab.lopezdp_utils.evaluation.strategy_risk import mix_gaussians

        r1 = mix_gaussians(0, 1, 0.5, 0.5, 0.5, 100, seed=42)
        r2 = mix_gaussians(0, 1, 0.5, 0.5, 0.5, 100, seed=42)
        np.testing.assert_array_equal(r1, r2)
