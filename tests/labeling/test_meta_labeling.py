"""Tests for labeling.meta_labeling."""

import numpy as np
import polars as pl
import pytest


class TestGetEventsMeta:
    def test_returns_events_with_t1_and_side(self, close_prices):
        from tradelab.lopezdp_utils.labeling.meta_labeling import get_events_meta
        from tradelab.lopezdp_utils.labeling.triple_barrier import daily_volatility

        t_events = close_prices["timestamp"].gather(list(range(0, 400, 10)))
        vol = daily_volatility(close_prices, span=50)
        # Simulate a primary model's side predictions
        side = pl.Series("side", np.random.choice([-1, 1], size=len(t_events)))

        result = get_events_meta(
            close=close_prices,
            t_events=t_events,
            pt_sl=[1.0, 1.0],
            trgt=vol,
            min_ret=0.0,
            side=side,
        )
        assert isinstance(result, pl.DataFrame)
        assert "t1" in result.columns
        assert "side" in result.columns
        assert result["t1"].null_count() == 0

    def test_symmetric_mode_no_side_column(self, close_prices):
        """When side is None, output should not have 'side' column."""
        from tradelab.lopezdp_utils.labeling.meta_labeling import get_events_meta
        from tradelab.lopezdp_utils.labeling.triple_barrier import daily_volatility

        t_events = close_prices["timestamp"].gather(list(range(0, 400, 10)))
        vol = daily_volatility(close_prices, span=50)

        result = get_events_meta(
            close=close_prices,
            t_events=t_events,
            pt_sl=1.0,
            trgt=vol,
            min_ret=0.0,
            side=None,
        )
        assert "t1" in result.columns
        assert "side" not in result.columns


class TestGetBinsMeta:
    def test_returns_binary_labels(self, close_prices):
        from tradelab.lopezdp_utils.labeling.meta_labeling import (
            get_bins_meta,
            get_events_meta,
        )
        from tradelab.lopezdp_utils.labeling.triple_barrier import daily_volatility

        t_events = close_prices["timestamp"].gather(list(range(0, 400, 10)))
        vol = daily_volatility(close_prices, span=50)
        side = pl.Series("side", np.random.choice([-1, 1], size=len(t_events)))

        events = get_events_meta(
            close=close_prices,
            t_events=t_events,
            pt_sl=[1.0, 1.0],
            trgt=vol,
            min_ret=0.0,
            side=side,
        )
        result = get_bins_meta(events, close_prices)
        assert isinstance(result, pl.DataFrame)
        assert "label" in result.columns
        # Meta-labels are binary: 0 (don't act) or 1 (act)
        labels = result["label"].unique().sort().to_list()
        assert all(l in [0, 1] for l in labels)

    def test_standard_mode_returns_directional_labels(self, close_prices):
        """Without side column, labels should be -1 or 1."""
        from tradelab.lopezdp_utils.labeling.meta_labeling import (
            get_bins_meta,
            get_events_meta,
        )
        from tradelab.lopezdp_utils.labeling.triple_barrier import daily_volatility

        t_events = close_prices["timestamp"].gather(list(range(0, 400, 10)))
        vol = daily_volatility(close_prices, span=50)

        events = get_events_meta(
            close=close_prices,
            t_events=t_events,
            pt_sl=1.0,
            trgt=vol,
            min_ret=0.0,
            side=None,
        )
        result = get_bins_meta(events, close_prices)
        labels = result["label"].unique().sort().to_list()
        assert all(l in [-1, 0, 1] for l in labels)
