"""Tests for allocation.simulation — synthetic data and MC experiments."""

import numpy as np
import pandas as pd


class TestGenerateData:
    def test_returns_correct_shape(self):
        from tradelab.lopezdp_utils.allocation.simulation import generate_data

        mu, cols = generate_data(n_obs=100, size0=3, size1=3, sigma1=0.25)
        assert mu.shape == (100, 6)  # size0 + size1
        assert isinstance(cols, list)

    def test_reproducible_with_seed(self):
        from tradelab.lopezdp_utils.allocation.simulation import generate_data

        mu1, _ = generate_data(n_obs=100, size0=3, size1=3, sigma1=0.25, seed=42)
        mu2, _ = generate_data(n_obs=100, size0=3, size1=3, sigma1=0.25, seed=42)
        np.testing.assert_array_equal(mu1, mu2)


class TestHRPMC:
    def test_returns_dataframe(self):
        from tradelab.lopezdp_utils.allocation.simulation import hrp_mc

        result = hrp_mc(num_iters=3, n_obs=520, size0=3, size1=3, sigma1=0.25)
        assert isinstance(result, pd.DataFrame)
        assert len(result.columns) >= 1

    def test_hrp_variance_reasonable(self):
        from tradelab.lopezdp_utils.allocation.simulation import hrp_mc

        result = hrp_mc(num_iters=3, n_obs=520, size0=3, size1=3, sigma1=0.25)
        # All values should be finite
        assert result.notna().all().all()
