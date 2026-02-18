"""Tests for data.sampling — CUSUM filter and sampling utilities."""

import numpy as np
import polars as pl


class TestGetTEvents:
    """Tests for CUSUM event filter."""

    def test_returns_polars_series(self, close_series):
        from tradelab.lopezdp_utils.data.sampling import get_t_events

        events = get_t_events(close_series, threshold=1.0)
        assert isinstance(events, pl.Series)
        assert events.dtype == pl.Datetime

    def test_no_events_when_threshold_high(self, close_series):
        from tradelab.lopezdp_utils.data.sampling import get_t_events

        events = get_t_events(close_series, threshold=1000.0)
        assert len(events) == 0

    def test_more_events_with_lower_threshold(self, close_series):
        from tradelab.lopezdp_utils.data.sampling import get_t_events

        events_high = get_t_events(close_series, threshold=2.0)
        events_low = get_t_events(close_series, threshold=0.5)
        assert len(events_low) >= len(events_high)

    def test_events_are_subset_of_timestamps(self, close_series):
        from tradelab.lopezdp_utils.data.sampling import get_t_events

        events = get_t_events(close_series, threshold=1.0)
        all_ts = close_series["timestamp"]
        for ts in events:
            assert ts in all_ts

    def test_known_cusum_detection(self):
        """A price series with a known jump should trigger exactly at the jump."""
        from tradelab.lopezdp_utils.data.sampling import get_t_events

        # Flat at 100, then jump to 105 at index 50
        n = 100
        timestamps = pl.datetime_range(
            pl.datetime(2024, 1, 1),
            pl.datetime(2024, 1, 1, 1, 39),
            interval="1m",
            eager=True,
        )
        prices = np.full(n, 100.0)
        prices[50:] = 105.0
        df = pl.DataFrame({"timestamp": timestamps, "close": prices})

        events = get_t_events(df, threshold=3.0)
        assert len(events) >= 1
        # First event should be at or near the jump
        first_event = events[0]
        assert first_event == timestamps[50]


class TestSamplingLinspace:
    """Tests for linspace sampling."""

    def test_step_sampling(self, ohlcv_1min):
        from tradelab.lopezdp_utils.data.sampling import sampling_linspace

        result = sampling_linspace(ohlcv_1min, step=10)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 10  # 100 / 10

    def test_num_samples(self, ohlcv_1min):
        from tradelab.lopezdp_utils.data.sampling import sampling_linspace

        result = sampling_linspace(ohlcv_1min, num_samples=20)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 20


class TestSamplingUniform:
    """Tests for uniform random sampling."""

    def test_returns_correct_count(self, ohlcv_1min):
        from tradelab.lopezdp_utils.data.sampling import sampling_uniform

        result = sampling_uniform(ohlcv_1min, num_samples=25, random_state=42)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 25

    def test_reproducible_with_seed(self, ohlcv_1min):
        from tradelab.lopezdp_utils.data.sampling import sampling_uniform

        r1 = sampling_uniform(ohlcv_1min, num_samples=10, random_state=42)
        r2 = sampling_uniform(ohlcv_1min, num_samples=10, random_state=42)
        assert r1.equals(r2)
