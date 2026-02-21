"""Integration tests: allocation pipeline."""

import numpy as np
import pandas as pd


class TestDenoiseThenHRP:
    """Denoise covariance -> HRP allocation."""

    def test_denoised_hrp_weights_sum_to_one(self, large_cov):
        from tradelab.lopezdp_utils.allocation.denoising import denoise_cov
        from tradelab.lopezdp_utils.allocation.hrp import hrp_alloc

        q = large_cov.shape[0] / 500
        clean_cov = denoise_cov(large_cov, q, bwidth=0.01)
        std = np.sqrt(np.diag(clean_cov.values))
        clean_corr = pd.DataFrame(
            clean_cov.values / np.outer(std, std),
            columns=clean_cov.columns,
            index=clean_cov.index,
        )
        weights = hrp_alloc(clean_cov, clean_corr)
        assert abs(weights.sum() - 1.0) < 1e-10
        assert (weights > 0).all()


class TestDenoiseThenNCO:
    """Denoise covariance -> NCO allocation."""

    def test_denoised_nco_weights(self, large_cov):
        from tradelab.lopezdp_utils.allocation.denoising import denoise_cov
        from tradelab.lopezdp_utils.allocation.nco import opt_port_nco

        q = large_cov.shape[0] / 500
        clean_cov = denoise_cov(large_cov, q, bwidth=0.01)
        weights = opt_port_nco(clean_cov, max_num_clusters=5)
        assert abs(weights.values.sum() - 1.0) < 1e-6
