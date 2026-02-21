"""Tests for allocation.hrp — Hierarchical Risk Parity."""

import numpy as np
import pandas as pd


class TestCorrelDist:
    def test_returns_dataframe(self, sample_corr):
        from tradelab.lopezdp_utils.allocation.hrp import correl_dist

        result = correl_dist(sample_corr)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == sample_corr.shape

    def test_zero_diagonal(self, sample_corr):
        from tradelab.lopezdp_utils.allocation.hrp import correl_dist

        result = correl_dist(sample_corr)
        np.testing.assert_array_almost_equal(np.diag(result.values), 0.0)

    def test_perfect_corr_zero_dist(self):
        from tradelab.lopezdp_utils.allocation.hrp import correl_dist

        corr = pd.DataFrame(np.ones((3, 3)), columns=["a", "b", "c"], index=["a", "b", "c"])
        result = correl_dist(corr)
        assert (result.values == 0).all()


class TestTreeClustering:
    def test_returns_linkage_matrix(self, sample_corr):
        from tradelab.lopezdp_utils.allocation.hrp import tree_clustering

        link = tree_clustering(sample_corr)
        assert link.shape[1] == 4  # scipy linkage format
        assert link.shape[0] == len(sample_corr) - 1


class TestGetQuasiDiag:
    def test_returns_sorted_indices(self, sample_corr):
        from tradelab.lopezdp_utils.allocation.hrp import get_quasi_diag, tree_clustering

        link = tree_clustering(sample_corr)
        sort_ix = get_quasi_diag(link)
        assert len(sort_ix) == len(sample_corr)
        assert set(sort_ix) == set(range(len(sample_corr)))


class TestHRPAlloc:
    def test_weights_sum_to_one(self, sample_cov, sample_corr):
        from tradelab.lopezdp_utils.allocation.hrp import hrp_alloc

        weights = hrp_alloc(sample_cov, sample_corr)
        assert isinstance(weights, pd.Series)
        assert abs(weights.sum() - 1.0) < 1e-10

    def test_all_weights_positive(self, sample_cov, sample_corr):
        from tradelab.lopezdp_utils.allocation.hrp import hrp_alloc

        weights = hrp_alloc(sample_cov, sample_corr)
        assert (weights > 0).all()

    def test_weights_match_assets(self, sample_cov, sample_corr):
        from tradelab.lopezdp_utils.allocation.hrp import hrp_alloc

        weights = hrp_alloc(sample_cov, sample_corr)
        assert set(weights.index) == set(sample_cov.columns)
