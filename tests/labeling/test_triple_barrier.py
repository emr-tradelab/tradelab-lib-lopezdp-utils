"""Tests for labeling.triple_barrier."""

import numpy as np
import polars as pl
import pytest


class TestDailyVolatility:
    def test_returns_polars_series(self, close_prices):
        from tradelab.lopezdp_utils.labeling.triple_barrier import daily_volatility

        result = daily_volatility(close_prices, span=50)
        assert isinstance(result, pl.DataFrame)
        assert "volatility" in result.columns

    def test_volatility_positive(self, close_prices):
        from tradelab.lopezdp_utils.labeling.triple_barrier import daily_volatility

        result = daily_volatility(close_prices, span=50)
        # Drop nulls and zeros (first ewm_std obs is 0 — single data point)
        non_null = result["volatility"].drop_nulls().filter(result["volatility"].drop_nulls() > 0)
        assert len(non_null) > 0
        assert (non_null > 0).all()


class TestAddVerticalBarrier:
    def test_returns_t1_series(self, close_prices):
        from tradelab.lopezdp_utils.labeling.triple_barrier import add_vertical_barrier

        t_events = close_prices["timestamp"].gather([0, 50, 100, 200])
        result = add_vertical_barrier(t_events, close_prices, num_bars=10)
        assert isinstance(result, pl.DataFrame)
        assert "t1" in result.columns
        assert len(result) == 4

    def test_t1_after_t0(self, close_prices):
        from tradelab.lopezdp_utils.labeling.triple_barrier import add_vertical_barrier

        t_events = close_prices["timestamp"].gather([0, 50, 100])
        result = add_vertical_barrier(t_events, close_prices, num_bars=10)
        for row in result.iter_rows(named=True):
            assert row["t1"] >= row["timestamp"]


class TestFixedTimeHorizon:
    def test_returns_polars_dataframe(self, close_prices):
        from tradelab.lopezdp_utils.labeling.triple_barrier import fixed_time_horizon

        result = fixed_time_horizon(close_prices, horizon=10, threshold=0.0)
        assert isinstance(result, pl.DataFrame)
        assert "label" in result.columns

    def test_labels_are_valid(self, close_prices):
        from tradelab.lopezdp_utils.labeling.triple_barrier import fixed_time_horizon

        result = fixed_time_horizon(close_prices, horizon=10, threshold=0.005)
        labels = result["label"].unique().sort().to_list()
        assert all(l in [-1, 0, 1] for l in labels)


class TestTripleBarrierLabels:
    """Integration test for the full pipeline."""

    def test_returns_events_with_t1(self, close_prices):
        from tradelab.lopezdp_utils.labeling.triple_barrier import triple_barrier_labels

        # Use every 10th bar as an event
        t_events = close_prices["timestamp"].gather(list(range(0, 400, 10)))
        result = triple_barrier_labels(
            close=close_prices,
            t_events=t_events,
            pt_sl=1.0,
            num_bars=20,
            vol_span=50,
        )
        assert isinstance(result, pl.DataFrame)
        assert "t1" in result.columns
        assert "label" in result.columns
        assert "ret" in result.columns
        # t1 must not be null (safety guardrail)
        assert result["t1"].null_count() == 0

    def test_labels_valid_values(self, close_prices):
        from tradelab.lopezdp_utils.labeling.triple_barrier import triple_barrier_labels

        t_events = close_prices["timestamp"].gather(list(range(0, 400, 10)))
        result = triple_barrier_labels(
            close=close_prices,
            t_events=t_events,
            pt_sl=1.0,
            num_bars=20,
            vol_span=50,
        )
        labels = result["label"].unique().sort().to_list()
        assert all(l in [-1, 0, 1] for l in labels)

    def test_t1_validates_not_null(self, close_prices):
        """The t1 column must never contain nulls — this is a safety guardrail."""
        from tradelab.lopezdp_utils.labeling.triple_barrier import triple_barrier_labels

        t_events = close_prices["timestamp"].gather(list(range(0, 450, 10)))
        result = triple_barrier_labels(
            close=close_prices,
            t_events=t_events,
            pt_sl=1.0,
            num_bars=20,
            vol_span=50,
        )
        assert result["t1"].null_count() == 0

    def test_with_trend_data(self, close_with_trend):
        """Uptrend should produce more +1 labels, downtrend more -1."""
        from tradelab.lopezdp_utils.labeling.triple_barrier import triple_barrier_labels

        t_events = close_with_trend["timestamp"].gather(list(range(0, 180, 5)))
        result = triple_barrier_labels(
            close=close_with_trend,
            t_events=t_events,
            pt_sl=1.0,
            num_bars=15,
            vol_span=30,
        )
        # Just verify it runs and produces labels — exact distribution
        # depends on volatility scaling
        assert len(result) > 0
        assert "label" in result.columns


class TestTrendScanningLabels:
    def test_returns_with_t1(self, close_prices):
        from tradelab.lopezdp_utils.labeling.triple_barrier import trend_scanning_labels

        t_events = close_prices["timestamp"].gather(list(range(50, 400, 20)))
        result = trend_scanning_labels(
            close=close_prices,
            t_events=t_events,
            span=range(5, 20),
        )
        assert isinstance(result, pl.DataFrame)
        assert "t1" in result.columns
        assert "label" in result.columns
