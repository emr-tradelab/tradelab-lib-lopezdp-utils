"""Integration tests: labeling → sample weights pipeline."""

import numpy as np
import polars as pl


class TestLabelingToWeightsPipeline:
    """End-to-end: close prices → triple-barrier labels → sample weights."""

    def test_full_pipeline(self, close_prices):
        from tradelab.lopezdp_utils.labeling import (
            get_avg_uniqueness,
            get_ind_matrix,
            get_time_decay,
            mp_num_co_events,
            mp_sample_tw,
            triple_barrier_labels,
        )

        # Step 1: Generate labels with t1
        t_events = close_prices["timestamp"].gather(list(range(10, 400, 10)))
        labels = triple_barrier_labels(
            close=close_prices,
            t_events=t_events,
            pt_sl=1.0,
            num_bars=20,
            vol_span=50,
        )

        # Verify t1 exists and is valid
        assert "t1" in labels.columns
        assert labels["t1"].null_count() == 0

        # Step 2: Compute co-events
        co_events = mp_num_co_events(
            close_idx=close_prices["timestamp"],
            t1=labels,
        )
        assert "num_co_events" in co_events.columns

        # Step 3: Compute uniqueness
        tw = mp_sample_tw(t1=labels, num_co_events=co_events)
        assert "uniqueness" in tw.columns
        u = tw["uniqueness"].drop_nulls()
        assert (u > 0).all()
        assert (u <= 1).all()

        # Step 4: Apply time decay
        decayed = get_time_decay(tw, clf_last_w=0.5)
        assert "weight" in decayed.columns
        assert (decayed["weight"].drop_nulls() >= 0).all()

        # Step 5: Alternative — indicator matrix path
        ind_m = get_ind_matrix(
            bar_idx=close_prices["timestamp"],
            t1=labels,
        )
        avg_u = get_avg_uniqueness(ind_m)
        assert len(avg_u) == len(labels)
        assert all(u >= 0 for u in avg_u)

    def test_meta_labeling_pipeline(self, close_prices):
        from tradelab.lopezdp_utils.labeling import (
            daily_volatility,
            get_bins_meta,
            get_events_meta,
            mp_num_co_events,
            mp_sample_tw,
        )

        t_events = close_prices["timestamp"].gather(list(range(10, 400, 10)))
        vol = daily_volatility(close_prices, span=50)

        # Simulate primary model sides
        np.random.seed(42)
        side = pl.Series("side", np.random.choice([-1, 1], size=len(t_events)))

        events = get_events_meta(
            close=close_prices,
            t_events=t_events,
            pt_sl=[1.0, 1.0],
            trgt=vol,
            min_ret=0.0,
            side=side,
        )

        bins = get_bins_meta(events, close_prices)
        assert "label" in bins.columns
        labels = bins["label"].unique().sort().to_list()
        assert all(lbl in [0, 1] for lbl in labels)

        # Weights should still work with meta-labeling events
        co_events = mp_num_co_events(
            close_idx=close_prices["timestamp"],
            t1=events,
        )
        tw = mp_sample_tw(t1=events, num_co_events=co_events)
        assert "uniqueness" in tw.columns
