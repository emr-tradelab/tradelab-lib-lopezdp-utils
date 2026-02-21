"""Tests for allocation.denoising — Random Matrix Theory covariance cleaning."""

import numpy as np
import pandas as pd


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

        std = np.sqrt(np.diag(large_cov.values))
        corr = large_cov.values / np.outer(std, std)
        evals = np.linalg.eigvalsh(corr)
        q = large_cov.shape[0] / 500  # T/N
        result, var = find_max_eval(evals, q, bwidth=0.01)
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
        result = detone_corr(corr, n_facts=1)
        assert result.shape == corr.shape
        # Detoned matrix should differ from original
        assert not np.allclose(result, corr)
        # Diagonal should be 1.0 (re-normalized)
        np.testing.assert_array_almost_equal(np.diag(result), 1.0)
