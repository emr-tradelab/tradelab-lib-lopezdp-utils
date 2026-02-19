"""Tests for features.entropy."""

import numpy as np
import polars as pl
import pytest


class TestPlugIn:
    def test_uniform_distribution_max_entropy(self):
        """A message with all unique substrings should have high entropy."""
        from tradelab.lopezdp_utils.features.entropy import plug_in

        msg = "0123456789" * 10
        entropy, _ = plug_in(msg, w=1)
        assert entropy > 0

    def test_constant_message_zero_entropy(self):
        """A constant message should have zero entropy."""
        from tradelab.lopezdp_utils.features.entropy import plug_in

        msg = "0" * 100
        entropy, _ = plug_in(msg, w=1)
        assert abs(entropy) < 1e-10


class TestLempelZivLib:
    def test_returns_list(self):
        from tradelab.lopezdp_utils.features.entropy import lempel_ziv_lib

        result = lempel_ziv_lib("1011010100010")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_constant_message_small_library(self):
        """A constant message has a small library relative to its length."""
        from tradelab.lopezdp_utils.features.entropy import lempel_ziv_lib

        result = lempel_ziv_lib("0" * 100)
        # LZ on "000..." produces ["0", "00", "000", ...] — grows as O(sqrt(n))
        # Much smaller than a random message of same length
        random_result = lempel_ziv_lib("".join(str(x) for x in range(100)))
        assert len(result) < len(random_result)


class TestKonto:
    def test_returns_dict(self):
        from tradelab.lopezdp_utils.features.entropy import konto

        np.random.seed(42)
        msg = "".join(str(x) for x in np.random.randint(0, 2, 200))
        result = konto(msg, window=None)
        assert "h" in result
        assert "r" in result

    def test_random_message_high_entropy(self):
        from tradelab.lopezdp_utils.features.entropy import konto

        np.random.seed(42)
        msg = "".join(str(x) for x in np.random.randint(0, 2, 200))
        result = konto(msg, window=None)
        assert result["h"] > 0.5  # should be close to 1.0 for random binary


class TestEncodeBinary:
    def test_returns_string(self, return_series):
        from tradelab.lopezdp_utils.features.entropy import encode_binary

        result = encode_binary(return_series)
        assert isinstance(result, str)
        assert set(result).issubset({"0", "1"})

    def test_length(self, return_series):
        from tradelab.lopezdp_utils.features.entropy import encode_binary

        result = encode_binary(return_series)
        # Zeros are removed, so length <= input length
        assert len(result) <= len(return_series)


class TestEncodeQuantile:
    def test_returns_string(self, return_series):
        from tradelab.lopezdp_utils.features.entropy import encode_quantile

        result = encode_quantile(return_series, num_letters=10)
        assert isinstance(result, str)
        assert len(result) == len(return_series)


class TestEncodeSigma:
    def test_returns_string(self, return_series):
        from tradelab.lopezdp_utils.features.entropy import encode_sigma

        result = encode_sigma(return_series, sigma_step=0.005)
        assert isinstance(result, str)
        # Multi-digit bin indices make string longer than input length
        assert len(result) >= len(return_series)


class TestMarketEfficiencyMetric:
    def test_returns_dict(self, return_series):
        from tradelab.lopezdp_utils.features.entropy import market_efficiency_metric

        result = market_efficiency_metric(return_series, encoding="binary")
        assert isinstance(result, dict)
        assert "entropy_rate" in result
        assert "redundancy" in result

    def test_redundancy_between_zero_and_one(self, return_series):
        from tradelab.lopezdp_utils.features.entropy import market_efficiency_metric

        result = market_efficiency_metric(return_series, encoding="binary")
        assert 0 <= result["redundancy"] <= 1


class TestPortfolioConcentration:
    def test_returns_float(self):
        from tradelab.lopezdp_utils.features.entropy import portfolio_concentration

        w = np.array([0.5, 0.3, 0.2])
        cov = np.eye(3)
        result = portfolio_concentration(w, cov)
        assert isinstance(result, float)


class TestNumBins:
    def test_returns_int(self):
        from tradelab.lopezdp_utils.features.entropy import num_bins

        result = num_bins(n_obs=1000, corr=None)
        assert isinstance(result, int)
        assert result > 0

    def test_bivariate_returns_positive(self):
        from tradelab.lopezdp_utils.features.entropy import num_bins

        result = num_bins(n_obs=1000, corr=0.5)
        assert result > 0


class TestVariationOfInformation:
    def test_identical_variables_zero_vi(self):
        from tradelab.lopezdp_utils.features.entropy import variation_of_information

        np.random.seed(42)
        x = np.random.randn(500)
        result = variation_of_information(x, x, normalize=True)
        assert abs(result) < 0.1  # should be close to 0


class TestMutualInformationOptimal:
    def test_independent_variables_low_mi(self):
        from tradelab.lopezdp_utils.features.entropy import mutual_information_optimal

        np.random.seed(42)
        x = np.random.randn(500)
        y = np.random.randn(500)
        result = mutual_information_optimal(x, y)
        assert result < 0.1  # nearly independent

    def test_dependent_variables_high_mi(self):
        from tradelab.lopezdp_utils.features.entropy import mutual_information_optimal

        np.random.seed(42)
        x = np.random.randn(500)
        y = x + np.random.randn(500) * 0.1  # highly correlated
        result = mutual_information_optimal(x, y)
        assert result > 0.5


class TestKLDivergence:
    def test_identical_distributions_zero(self):
        from tradelab.lopezdp_utils.features.entropy import kl_divergence

        p = np.array([0.25, 0.25, 0.25, 0.25])
        result = kl_divergence(p, p)
        assert abs(result) < 1e-10

    def test_asymmetric(self):
        from tradelab.lopezdp_utils.features.entropy import kl_divergence

        p = np.array([0.5, 0.5])
        q = np.array([0.9, 0.1])
        d1 = kl_divergence(p, q)
        d2 = kl_divergence(q, p)
        assert d1 != d2  # KL divergence is asymmetric


class TestCrossEntropy:
    def test_cross_entropy_geq_entropy(self):
        from tradelab.lopezdp_utils.features.entropy import cross_entropy

        import scipy.stats as ss

        p = np.array([0.3, 0.7])
        q = np.array([0.5, 0.5])
        ce = cross_entropy(p, q)
        h_p = float(ss.entropy(p))
        assert ce >= h_p - 1e-10  # H_C(p||q) >= H(p)
