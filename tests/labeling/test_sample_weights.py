"""Tests for labeling.sample_weights — concurrency, uniqueness, sequential bootstrap."""

import numpy as np
import polars as pl
import pytest


class TestMpNumCoEvents:
    """Tests for co-event counting."""

    def test_returns_polars_series(self, close_prices, events_with_t1):
        from tradelab.lopezdp_utils.labeling.sample_weights import mp_num_co_events

        result = mp_num_co_events(
            close_idx=close_prices["timestamp"],
            t1=events_with_t1,
        )
        assert isinstance(result, pl.DataFrame)
        assert "num_co_events" in result.columns

    def test_co_events_positive(self, close_prices, events_with_t1):
        from tradelab.lopezdp_utils.labeling.sample_weights import mp_num_co_events

        result = mp_num_co_events(
            close_idx=close_prices["timestamp"],
            t1=events_with_t1,
        )
        non_null = result["num_co_events"].drop_nulls()
        assert (non_null >= 0).all()

    def test_non_overlapping_events_have_count_one(self):
        """Events that don't overlap should have co-event count = 1."""
        from tradelab.lopezdp_utils.labeling.sample_weights import mp_num_co_events

        timestamps = pl.datetime_range(
            pl.datetime(2024, 1, 1),
            pl.datetime(2024, 1, 1, 1, 39),
            interval="1m",
            eager=True,
        )
        # Two non-overlapping events: [0, 10] and [20, 30]
        events = pl.DataFrame({
            "timestamp": [timestamps[0], timestamps[20]],
            "t1": [timestamps[10], timestamps[30]],
        })
        result = mp_num_co_events(close_idx=timestamps, t1=events)
        # Within each event window, co-event count should be 1
        assert result.filter(
            pl.col("timestamp").is_between(timestamps[0], timestamps[10])
        )["num_co_events"].max() == 1


class TestMpSampleTw:
    """Tests for average uniqueness."""

    def test_returns_polars_dataframe(self, close_prices, events_with_t1):
        from tradelab.lopezdp_utils.labeling.sample_weights import (
            mp_num_co_events,
            mp_sample_tw,
        )

        co_events = mp_num_co_events(
            close_idx=close_prices["timestamp"],
            t1=events_with_t1,
        )
        result = mp_sample_tw(t1=events_with_t1, num_co_events=co_events)
        assert isinstance(result, pl.DataFrame)
        assert "uniqueness" in result.columns

    def test_uniqueness_between_zero_and_one(self, close_prices, events_with_t1):
        from tradelab.lopezdp_utils.labeling.sample_weights import (
            mp_num_co_events,
            mp_sample_tw,
        )

        co_events = mp_num_co_events(
            close_idx=close_prices["timestamp"],
            t1=events_with_t1,
        )
        result = mp_sample_tw(t1=events_with_t1, num_co_events=co_events)
        u = result["uniqueness"].drop_nulls()
        assert (u > 0).all()
        assert (u <= 1).all()


class TestGetIndMatrix:
    """Tests for indicator matrix construction."""

    def test_shape(self):
        from tradelab.lopezdp_utils.labeling.sample_weights import get_ind_matrix

        timestamps = pl.datetime_range(
            pl.datetime(2024, 1, 1),
            pl.datetime(2024, 1, 1, 0, 9),
            interval="1m",
            eager=True,
        )
        events = pl.DataFrame({
            "timestamp": [timestamps[0], timestamps[3]],
            "t1": [timestamps[5], timestamps[8]],
        })
        result = get_ind_matrix(bar_idx=timestamps, t1=events)
        assert isinstance(result, np.ndarray)
        assert result.shape == (10, 2)  # 10 bars x 2 events

    def test_binary_values(self):
        from tradelab.lopezdp_utils.labeling.sample_weights import get_ind_matrix

        timestamps = pl.datetime_range(
            pl.datetime(2024, 1, 1),
            pl.datetime(2024, 1, 1, 0, 9),
            interval="1m",
            eager=True,
        )
        events = pl.DataFrame({
            "timestamp": [timestamps[0]],
            "t1": [timestamps[5]],
        })
        result = get_ind_matrix(bar_idx=timestamps, t1=events)
        assert set(np.unique(result)).issubset({0.0, 1.0})
        # Bars 0-5 should be 1, bars 6-9 should be 0
        assert result[:6, 0].sum() == 6
        assert result[6:, 0].sum() == 0


class TestGetAvgUniqueness:
    def test_single_event_full_uniqueness(self):
        """A single non-overlapping event has uniqueness = 1.0."""
        from tradelab.lopezdp_utils.labeling.sample_weights import get_avg_uniqueness

        # 1 event active for 5 bars
        ind_m = np.zeros((10, 1))
        ind_m[:5, 0] = 1.0
        result = get_avg_uniqueness(ind_m)
        assert abs(result[0] - 1.0) < 1e-10

    def test_overlapping_events_lower_uniqueness(self):
        """Two fully overlapping events should have uniqueness = 0.5."""
        from tradelab.lopezdp_utils.labeling.sample_weights import get_avg_uniqueness

        ind_m = np.zeros((10, 2))
        ind_m[:5, 0] = 1.0
        ind_m[:5, 1] = 1.0  # same window
        result = get_avg_uniqueness(ind_m)
        assert abs(result[0] - 0.5) < 1e-10
        assert abs(result[1] - 0.5) < 1e-10


class TestSeqBootstrap:
    def test_returns_list(self):
        from tradelab.lopezdp_utils.labeling.sample_weights import seq_bootstrap

        ind_m = np.zeros((20, 5))
        for i in range(5):
            ind_m[i * 4 : (i + 1) * 4, i] = 1.0
        result = seq_bootstrap(ind_m, s_length=5)
        assert isinstance(result, list)
        assert len(result) == 5

    def test_favors_unique_samples(self):
        """Sequential bootstrap should prefer non-overlapping samples."""
        from tradelab.lopezdp_utils.labeling.sample_weights import seq_bootstrap

        np.random.seed(42)
        # 3 non-overlapping events + 1 overlapping with event 0
        ind_m = np.zeros((20, 4))
        ind_m[0:5, 0] = 1.0    # event 0: bars 0-4
        ind_m[0:5, 3] = 1.0    # event 3: overlaps event 0
        ind_m[5:10, 1] = 1.0   # event 1: bars 5-9
        ind_m[10:15, 2] = 1.0  # event 2: bars 10-14

        # Run many bootstraps and check event 3 is less frequent
        counts = np.zeros(4)
        for _ in range(1000):
            sample = seq_bootstrap(ind_m, s_length=3)
            for s in sample:
                counts[s] += 1
        # Event 3 (overlapping) should be selected less than events 1, 2
        # (statistically, not guaranteed for every seed, but holds over 1000 runs)
        assert counts[3] < counts[1]
        assert counts[3] < counts[2]


class TestGetTimeDecay:
    def test_returns_polars_dataframe(self, close_prices, events_with_t1):
        from tradelab.lopezdp_utils.labeling.sample_weights import (
            mp_num_co_events,
            mp_sample_tw,
            get_time_decay,
        )

        co_events = mp_num_co_events(close_prices["timestamp"], events_with_t1)
        tw = mp_sample_tw(events_with_t1, co_events)
        result = get_time_decay(tw, clf_last_w=0.5)
        assert isinstance(result, pl.DataFrame)
        assert "weight" in result.columns

    def test_no_decay_when_clf_last_w_one(self, close_prices, events_with_t1):
        """clf_last_w=1.0 means no decay — all weights should be 1.0."""
        from tradelab.lopezdp_utils.labeling.sample_weights import (
            mp_num_co_events,
            mp_sample_tw,
            get_time_decay,
        )

        co_events = mp_num_co_events(close_prices["timestamp"], events_with_t1)
        tw = mp_sample_tw(events_with_t1, co_events)
        result = get_time_decay(tw, clf_last_w=1.0)
        assert (result["weight"] - 1.0).abs().max() < 1e-10


class TestValidateT1:
    """Tests for t1 validation guardrail."""

    def test_raises_on_missing_t1(self):
        from tradelab.lopezdp_utils.labeling.sample_weights import mp_num_co_events

        timestamps = pl.datetime_range(
            pl.datetime(2024, 1, 1),
            pl.datetime(2024, 1, 1, 0, 9),
            interval="1m",
            eager=True,
        )
        bad_events = pl.DataFrame({"timestamp": [timestamps[0]]})  # no t1!
        with pytest.raises(ValueError, match="t1"):
            mp_num_co_events(close_idx=timestamps, t1=bad_events)

    def test_raises_on_null_t1(self):
        from tradelab.lopezdp_utils.labeling.sample_weights import mp_num_co_events

        timestamps = pl.datetime_range(
            pl.datetime(2024, 1, 1),
            pl.datetime(2024, 1, 1, 0, 9),
            interval="1m",
            eager=True,
        )
        bad_events = pl.DataFrame({
            "timestamp": [timestamps[0]],
            "t1": [None],
        }).cast({"t1": pl.Datetime})
        with pytest.raises(ValueError, match="t1.*null"):
            mp_num_co_events(close_idx=timestamps, t1=bad_events)
