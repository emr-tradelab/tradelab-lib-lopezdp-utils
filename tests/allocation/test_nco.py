"""Tests for allocation.nco — Nested Clustered Optimization."""

import numpy as np
import pandas as pd


class TestOptPortNCO:
    def test_min_var_weights_sum_to_one(self, sample_cov):
        from tradelab.lopezdp_utils.allocation.nco import opt_port_nco

        weights = opt_port_nco(sample_cov, max_num_clusters=3)
        assert isinstance(weights, pd.Series)
        assert abs(weights.values.sum() - 1.0) < 1e-6

    def test_max_sharpe_with_mu(self, sample_cov):
        from tradelab.lopezdp_utils.allocation.nco import opt_port_nco

        np.random.seed(99)
        mu = pd.Series(
            np.random.randn(len(sample_cov)) * 0.01,
            index=sample_cov.index,
        )
        weights = opt_port_nco(sample_cov, mu=mu, max_num_clusters=3)
        assert isinstance(weights, pd.Series)

    def test_weights_shape(self, sample_cov):
        from tradelab.lopezdp_utils.allocation.nco import opt_port_nco

        weights = opt_port_nco(sample_cov, max_num_clusters=3)
        assert len(weights) == len(sample_cov)
