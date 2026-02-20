"""Tests for evaluation.statistics."""

import polars as pl


class TestSharpeRatio:
    def test_returns_float(self, daily_returns):
        from tradelab.lopezdp_utils.evaluation.statistics import sharpe_ratio

        result = sharpe_ratio(daily_returns, periods_per_year=252)
        assert isinstance(result, float)

    def test_positive_for_positive_drift(self, daily_returns):
        from tradelab.lopezdp_utils.evaluation.statistics import sharpe_ratio

        result = sharpe_ratio(daily_returns, periods_per_year=252)
        assert result > 0

    def test_zero_returns_zero_sharpe(self):
        from tradelab.lopezdp_utils.evaluation.statistics import sharpe_ratio

        returns = pl.Series("returns", [0.0] * 100)
        result = sharpe_ratio(returns, periods_per_year=252)
        assert result == 0.0


class TestProbabilisticSharpeRatio:
    def test_high_sr_high_psr(self):
        from tradelab.lopezdp_utils.evaluation.statistics import probabilistic_sharpe_ratio

        result = probabilistic_sharpe_ratio(
            observed_sr=2.0,
            benchmark_sr=0.0,
            n_obs=252,
            skew=0.0,
            kurtosis=3.0,
        )
        assert result > 0.95

    def test_low_sr_low_psr(self):
        from tradelab.lopezdp_utils.evaluation.statistics import probabilistic_sharpe_ratio

        result = probabilistic_sharpe_ratio(
            observed_sr=0.1,
            benchmark_sr=1.0,
            n_obs=50,
            skew=0.0,
            kurtosis=3.0,
        )
        assert result < 0.5


class TestDeflatedSharpeRatio:
    def test_requires_sr_estimates(self):
        from tradelab.lopezdp_utils.evaluation.statistics import deflated_sharpe_ratio

        result = deflated_sharpe_ratio(
            observed_sr=1.5,
            sr_estimates=[0.5, 0.8, 1.0, 1.2, 1.5],
            n_obs=252,
            skew=0.0,
            kurtosis=3.0,
        )
        assert 0 <= result <= 1

    def test_many_trials_deflates_sr(self):
        from tradelab.lopezdp_utils.evaluation.statistics import deflated_sharpe_ratio

        few = deflated_sharpe_ratio(
            observed_sr=1.5,
            sr_estimates=[0.5, 1.5],
            n_obs=252,
        )
        many = deflated_sharpe_ratio(
            observed_sr=1.5,
            sr_estimates=[0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.5] * 10,
            n_obs=252,
        )
        assert many < few


class TestComputeDdTuw:
    def test_returns_polars_dataframe(self, pnl_series):
        from tradelab.lopezdp_utils.evaluation.statistics import compute_dd_tuw

        dd, tuw = compute_dd_tuw(pnl_series, dollars=False)
        assert isinstance(dd, pl.DataFrame)
        assert isinstance(tuw, pl.DataFrame)

    def test_drawdown_negative(self, pnl_series):
        from tradelab.lopezdp_utils.evaluation.statistics import compute_dd_tuw

        dd, _ = compute_dd_tuw(pnl_series, dollars=False)
        # Percentage drawdowns should be non-negative (1 - min/hwm >= 0)
        assert (dd["drawdown"].drop_nulls() >= 0).all()


class TestGetHHI:
    def test_concentrated_returns(self):
        from tradelab.lopezdp_utils.evaluation.statistics import get_hhi

        returns = pl.Series("ret", [10.0] + [0.01] * 99)
        result = get_hhi(returns)
        assert result > 0.5

    def test_uniform_returns_low_hhi(self):
        from tradelab.lopezdp_utils.evaluation.statistics import get_hhi

        returns = pl.Series("ret", [1.0] * 100)
        result = get_hhi(returns)
        assert result < 0.05


class TestGetHHIDecomposed:
    def test_returns_dict_with_three_keys(self):
        from tradelab.lopezdp_utils.evaluation.statistics import get_hhi_decomposed

        returns = pl.Series("ret", [5.0, -2.0, 3.0, -1.0, 0.5] * 20)
        result = get_hhi_decomposed(returns)
        assert "hhi_positive" in result
        assert "hhi_negative" in result
        assert "hhi_total" in result

    def test_concentrated_positive_returns(self):
        from tradelab.lopezdp_utils.evaluation.statistics import get_hhi_decomposed

        returns = pl.Series("ret", [10.0] + [0.01] * 49 + [-1.0] * 50)
        result = get_hhi_decomposed(returns)
        assert result["hhi_positive"] > result["hhi_negative"]


class TestGetBetTiming:
    def test_detects_flattenings_and_flips(self, position_series):
        from tradelab.lopezdp_utils.evaluation.statistics import get_bet_timing

        result = get_bet_timing(position_series)
        assert isinstance(result, pl.DataFrame)
        assert "timestamp" in result.columns
        assert len(result) >= 3


class TestGetHoldingPeriod:
    def test_returns_float(self, position_series):
        from tradelab.lopezdp_utils.evaluation.statistics import get_holding_period

        result = get_holding_period(position_series)
        assert isinstance(result, float)
        assert result > 0


class TestStrategyPrecision:
    def test_perfect_test_high_precision(self):
        from tradelab.lopezdp_utils.evaluation.statistics import strategy_precision

        result = strategy_precision(alpha=0.01, beta=0.01, theta=0.1)
        assert result > 0.5

    def test_bad_test_low_precision(self):
        from tradelab.lopezdp_utils.evaluation.statistics import strategy_precision

        result = strategy_precision(alpha=0.5, beta=0.5, theta=0.01)
        assert result < 0.05


class TestMultiTestPrecisionRecall:
    def test_returns_tuple(self):
        from tradelab.lopezdp_utils.evaluation.statistics import multi_test_precision_recall

        prec, recall = multi_test_precision_recall(
            alpha=0.05,
            beta=0.2,
            theta=0.1,
            k=10,
        )
        assert 0 <= prec <= 1
        assert 0 <= recall <= 1

    def test_more_trials_lower_precision(self):
        from tradelab.lopezdp_utils.evaluation.statistics import multi_test_precision_recall

        prec_few, _ = multi_test_precision_recall(alpha=0.05, beta=0.2, theta=0.1, k=2)
        prec_many, _ = multi_test_precision_recall(alpha=0.05, beta=0.2, theta=0.1, k=100)
        assert prec_many < prec_few


# ---------------------------------------------------------------------------
# Analytical correctness tests
# ---------------------------------------------------------------------------


class TestPSRAnalytical:
    """Verify PSR against hand-computed values."""

    def test_known_value(self):
        from tradelab.lopezdp_utils.evaluation.statistics import probabilistic_sharpe_ratio

        # SR=0.1, n=50, skew=0, kurtosis=3 → PSR ≈ 0.7575
        result = probabilistic_sharpe_ratio(
            observed_sr=0.1,
            benchmark_sr=0.0,
            n_obs=50,
            skew=0.0,
            kurtosis=3.0,
        )
        assert abs(result - 0.7575) < 0.001

    def test_gaussian_defaults_match(self):
        from tradelab.lopezdp_utils.evaluation.statistics import probabilistic_sharpe_ratio

        # With default skew=0, kurtosis=3, high SR and many obs → PSR ≈ 1.0
        result = probabilistic_sharpe_ratio(
            observed_sr=1.0,
            benchmark_sr=0.0,
            n_obs=252,
        )
        assert result > 0.999


class TestDSRAnalytical:
    """Verify DSR against hand-computed values."""

    def test_known_value(self):
        from tradelab.lopezdp_utils.evaluation.statistics import deflated_sharpe_ratio

        # 3 trials [0.05, 0.08, 0.1], observed=0.1, n=50 → DSR ≈ 0.7083
        result = deflated_sharpe_ratio(
            observed_sr=0.1,
            sr_estimates=[0.05, 0.08, 0.1],
            n_obs=50,
        )
        assert abs(result - 0.7083) < 0.001

    def test_dsr_always_leq_psr(self):
        from tradelab.lopezdp_utils.evaluation.statistics import (
            deflated_sharpe_ratio,
            probabilistic_sharpe_ratio,
        )

        sr = 0.15
        psr = probabilistic_sharpe_ratio(observed_sr=sr, benchmark_sr=0.0, n_obs=100)
        dsr = deflated_sharpe_ratio(observed_sr=sr, sr_estimates=[0.05, 0.10, 0.15], n_obs=100)
        assert dsr <= psr


class TestDdTuwAnalytical:
    """Verify DD/TUW with a hand-constructed equity curve."""

    def test_known_drawdown(self):
        from tradelab.lopezdp_utils.evaluation.statistics import compute_dd_tuw

        # Equity: 100, 110, 105, 108, 115 (one drawdown episode: 110→105)
        ts = pl.datetime_range(
            pl.datetime(2024, 1, 1),
            pl.datetime(2024, 1, 5),
            interval="1d",
            eager=True,
        )
        pnl = [100.0, 110.0, 105.0, 108.0, 115.0]
        series = pl.DataFrame({"timestamp": ts, "pnl": pnl})

        dd, _tuw = compute_dd_tuw(series, dollars=True)

        # At index 2 (105), DD from HWM 110 = 5.0
        dd_vals = dd["drawdown"].to_list()
        assert dd_vals[0] == 0.0  # 100 is initial HWM
        assert dd_vals[1] == 0.0  # 110 is new HWM
        assert abs(dd_vals[2] - 5.0) < 1e-10  # 110 - 105
        assert abs(dd_vals[3] - 2.0) < 1e-10  # 110 - 108
        assert dd_vals[4] == 0.0  # 115 is new HWM

    def test_tuw_recovery_days(self):
        from tradelab.lopezdp_utils.evaluation.statistics import compute_dd_tuw

        # Peak at day 1 (110), recovers at day 4 (115) → 3 days TUW
        ts = pl.datetime_range(
            pl.datetime(2024, 1, 1),
            pl.datetime(2024, 1, 5),
            interval="1d",
            eager=True,
        )
        pnl = [100.0, 110.0, 105.0, 108.0, 115.0]
        series = pl.DataFrame({"timestamp": ts, "pnl": pnl})

        _, tuw = compute_dd_tuw(series, dollars=True)
        assert len(tuw) == 1
        assert abs(tuw["tuw_days"][0] - 3.0) < 1e-10
