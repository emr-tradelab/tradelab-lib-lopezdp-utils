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
