"""Tests for evaluation.bet_sizing — signal and position sizing."""

import numpy as np
import polars as pl
import pytest


class TestBetSize:
    def test_zero_divergence_zero_size(self):
        from tradelab.lopezdp_utils.evaluation.bet_sizing import bet_size

        assert abs(bet_size(w=1.0, x=0.0)) < 1e-10

    def test_large_divergence_near_one(self):
        from tradelab.lopezdp_utils.evaluation.bet_sizing import bet_size

        result = bet_size(w=1.0, x=100.0)
        assert abs(result) > 0.99

    def test_output_bounded(self):
        from tradelab.lopezdp_utils.evaluation.bet_sizing import bet_size

        for x in np.linspace(-10, 10, 50):
            assert -1 < bet_size(w=1.0, x=x) < 1


class TestGetTargetPos:
    def test_returns_int(self):
        from tradelab.lopezdp_utils.evaluation.bet_sizing import get_target_pos

        result = get_target_pos(w=1.0, f=105.0, m_p=100.0, max_pos=10)
        assert isinstance(result, (int, np.integer))

    def test_max_pos_clipped(self):
        from tradelab.lopezdp_utils.evaluation.bet_sizing import get_target_pos

        result = get_target_pos(w=0.01, f=200.0, m_p=100.0, max_pos=5)
        assert abs(result) <= 5


class TestInvPrice:
    def test_roundtrip(self):
        from tradelab.lopezdp_utils.evaluation.bet_sizing import bet_size, inv_price

        w = 2.0
        f = 100.0
        m_p = 95.0
        x = f - m_p
        m = bet_size(w, x)
        recovered_mp = inv_price(f, w, m)
        assert abs(recovered_mp - m_p) < 1e-6


class TestLimitPrice:
    def test_returns_float(self):
        from tradelab.lopezdp_utils.evaluation.bet_sizing import limit_price

        result = limit_price(t_pos=3, pos=0, f=100.0, w=1.0, max_pos=10)
        assert isinstance(result, float)

    def test_long_entry_below_forecast(self):
        from tradelab.lopezdp_utils.evaluation.bet_sizing import limit_price

        lp_long = limit_price(t_pos=3, pos=0, f=100.0, w=1.0, max_pos=10)
        assert lp_long < 100.0  # buy below forecast

    def test_same_position_returns_forecast(self):
        from tradelab.lopezdp_utils.evaluation.bet_sizing import limit_price

        result = limit_price(t_pos=5, pos=5, f=100.0, w=1.0, max_pos=10)
        assert result == 100.0


class TestGetW:
    def test_calibration(self):
        from tradelab.lopezdp_utils.evaluation.bet_sizing import bet_size, get_w

        w = get_w(x=5.0, m=0.5)
        result = bet_size(w, x=5.0)
        assert abs(result - 0.5) < 1e-6


class TestGetSignal:
    def test_returns_polars_dataframe(self):
        from tradelab.lopezdp_utils.evaluation.bet_sizing import get_signal

        np.random.seed(42)
        n = 50
        timestamps = pl.datetime_range(
            pl.datetime(2024, 1, 1),
            pl.datetime(2024, 1, 1, 0, 49),
            interval="1m",
            eager=True,
        )
        events = pl.DataFrame(
            {
                "timestamp": timestamps,
                "t1": timestamps.shift(-5, fill_value=timestamps[-1]),
                "side": np.random.choice([-1, 1], n),
            }
        )
        prob = pl.Series("prob", np.random.uniform(0.5, 0.9, n))

        result = get_signal(events, step_size=0.1, prob=prob, num_classes=2)
        assert isinstance(result, pl.DataFrame)
        assert "signal" in result.columns

    def test_signals_bounded(self):
        from tradelab.lopezdp_utils.evaluation.bet_sizing import get_signal

        np.random.seed(42)
        n = 50
        timestamps = pl.datetime_range(
            pl.datetime(2024, 1, 1),
            pl.datetime(2024, 1, 1, 0, 49),
            interval="1m",
            eager=True,
        )
        events = pl.DataFrame(
            {
                "timestamp": timestamps,
                "t1": timestamps.shift(-5, fill_value=timestamps[-1]),
                "side": np.random.choice([-1, 1], n),
            }
        )
        prob = pl.Series("prob", np.random.uniform(0.5, 0.9, n))

        result = get_signal(events, step_size=0.0, prob=prob, num_classes=2)
        signals = result["signal"].drop_nulls()
        assert (signals.abs() <= 1.0).all()

    def test_validates_t1(self):
        from tradelab.lopezdp_utils.evaluation.bet_sizing import get_signal

        events = pl.DataFrame({"timestamp": [pl.datetime(2024, 1, 1)]})
        prob = pl.Series("prob", [0.7])
        with pytest.raises(ValueError, match="t1"):
            get_signal(events, step_size=0.1, prob=prob, num_classes=2)


class TestDiscreteSignal:
    def test_discretization(self):
        from tradelab.lopezdp_utils.evaluation.bet_sizing import discrete_signal

        signal = pl.Series("signal", [0.37, -0.62, 0.15, -0.08])
        result = discrete_signal(signal, step_size=0.2)
        expected = [0.4, -0.6, 0.2, 0.0]
        for r, e in zip(result.to_list(), expected):
            assert abs(r - e) < 0.01

    def test_clipped_to_bounds(self):
        from tradelab.lopezdp_utils.evaluation.bet_sizing import discrete_signal

        signal = pl.Series("signal", [1.5, -1.5])
        result = discrete_signal(signal, step_size=0.1)
        assert result.max() <= 1.0
        assert result.min() >= -1.0


class TestAvgActiveSignals:
    def test_returns_polars_dataframe(self):
        from tradelab.lopezdp_utils.evaluation.bet_sizing import avg_active_signals

        timestamps = pl.datetime_range(
            pl.datetime(2024, 1, 1),
            pl.datetime(2024, 1, 1, 0, 9),
            interval="1m",
            eager=True,
        )
        signals = pl.DataFrame(
            {
                "timestamp": [timestamps[0], timestamps[2], timestamps[5]],
                "t1": [timestamps[4], timestamps[6], timestamps[9]],
                "signal": [0.5, -0.3, 0.8],
            }
        )
        result = avg_active_signals(signals)
        assert isinstance(result, pl.DataFrame)
        assert "signal" in result.columns


class TestBetSizeMixture:
    def test_returns_polars_series(self):
        from tradelab.lopezdp_utils.evaluation.bet_sizing import bet_size_mixture

        counts = pl.Series("c", [5.0, -3.0, 0.0, 2.0, -1.0])
        result = bet_size_mixture(counts, mu1=1.0, mu2=-1.0, sigma1=2.0, sigma2=2.0, prob1=0.5)
        assert isinstance(result, pl.Series)
        assert len(result) == 5

    def test_output_bounded(self):
        from tradelab.lopezdp_utils.evaluation.bet_sizing import bet_size_mixture

        counts = pl.Series("c", np.linspace(-10, 10, 50))
        result = bet_size_mixture(counts, mu1=0.0, mu2=0.0, sigma1=3.0, sigma2=3.0, prob1=0.5)
        assert result.min() >= -1.0
        assert result.max() <= 1.0

    def test_zero_count_near_zero_size(self):
        from tradelab.lopezdp_utils.evaluation.bet_sizing import bet_size_mixture

        counts = pl.Series("c", [0.0])
        result = bet_size_mixture(counts, mu1=0.0, mu2=0.0, sigma1=1.0, sigma2=1.0, prob1=0.5)
        assert abs(result[0]) < 0.01


class TestBetSizeEnsemble:
    def test_high_prob_positive_size(self):
        from tradelab.lopezdp_utils.evaluation.bet_sizing import bet_size_ensemble

        result = bet_size_ensemble(avg_prob=0.9, n_classifiers=10)
        assert result > 0

    def test_low_prob_negative_size(self):
        from tradelab.lopezdp_utils.evaluation.bet_sizing import bet_size_ensemble

        result = bet_size_ensemble(avg_prob=0.1, n_classifiers=10)
        assert result < 0

    def test_half_prob_zero_size(self):
        from tradelab.lopezdp_utils.evaluation.bet_sizing import bet_size_ensemble

        result = bet_size_ensemble(avg_prob=0.5, n_classifiers=10)
        assert abs(result) < 1e-10

    def test_raises_with_single_classifier(self):
        from tradelab.lopezdp_utils.evaluation.bet_sizing import bet_size_ensemble

        with pytest.raises(ValueError):
            bet_size_ensemble(avg_prob=0.7, n_classifiers=1)
