"""Tests for features.orthogonal — PCA orthogonalization and portfolio weights."""

import numpy as np
import pandas as pd
import pytest


class TestGetOrthoFeats:
    def test_returns_array(self):
        from tradelab.lopezdp_utils.features.orthogonal import get_ortho_feats

        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 5), columns=[f"f{i}" for i in range(5)])
        result = get_ortho_feats(X, var_thres=0.95)
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 100

    def test_orthogonality(self):
        """Output PCs should be uncorrelated."""
        from tradelab.lopezdp_utils.features.orthogonal import get_ortho_feats

        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(200, 5), columns=[f"f{i}" for i in range(5)])
        result = get_ortho_feats(X, var_thres=0.95)
        corr = np.corrcoef(result.T)
        # Off-diagonal elements should be near zero
        np.fill_diagonal(corr, 0)
        assert np.max(np.abs(corr)) < 0.1


class TestPCAWeights:
    def test_returns_ndarray(self):
        from tradelab.lopezdp_utils.features.orthogonal import pca_weights

        np.random.seed(42)
        cov = np.eye(3) + np.random.randn(3, 3) * 0.01
        cov = (cov + cov.T) / 2  # ensure symmetric
        result = pca_weights(cov)
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 3

    def test_weights_not_all_zero(self):
        from tradelab.lopezdp_utils.features.orthogonal import pca_weights

        cov = np.eye(3)
        result = pca_weights(cov)
        assert np.sum(np.abs(result)) > 0


class TestWeightedKendallTau:
    def test_returns_correlation(self):
        from tradelab.lopezdp_utils.features.orthogonal import weighted_kendall_tau

        feat_imp = pd.Series([0.5, 0.3, 0.1, 0.05, 0.05])
        pc_rank = pd.Series([1, 2, 3, 4, 5])
        result = weighted_kendall_tau(feat_imp, pc_rank)
        assert isinstance(result, float)
        assert -1 <= result <= 1
