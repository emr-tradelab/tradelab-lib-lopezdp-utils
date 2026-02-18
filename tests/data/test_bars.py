"""Tests for data.bars — bar construction from tick/time data."""

import polars as pl


class TestTimeBars:
    """Tests for time bar aggregation."""

    def test_returns_polars_dataframe(self, tick_data):
        from tradelab.lopezdp_utils.data.bars import time_bars

        result = time_bars(tick_data, frequency="1m")
        assert isinstance(result, pl.DataFrame)

    def test_has_ohlcv_columns(self, tick_data):
        from tradelab.lopezdp_utils.data.bars import time_bars

        result = time_bars(tick_data, frequency="1m")
        for col in ["timestamp", "open", "high", "low", "close", "volume"]:
            assert col in result.columns

    def test_high_ge_low(self, tick_data):
        from tradelab.lopezdp_utils.data.bars import time_bars

        result = time_bars(tick_data, frequency="1m")
        assert (result["high"] >= result["low"]).all()

    def test_volume_positive(self, tick_data):
        from tradelab.lopezdp_utils.data.bars import time_bars

        result = time_bars(tick_data, frequency="1m")
        assert (result["volume"] > 0).all()


class TestTickBars:
    """Tests for tick bars."""

    def test_returns_polars_dataframe(self, tick_data):
        from tradelab.lopezdp_utils.data.bars import tick_bars

        result = tick_bars(tick_data, threshold=50)
        assert isinstance(result, pl.DataFrame)

    def test_bar_count(self, tick_data):
        """1000 ticks / 50 per bar = ~20 bars."""
        from tradelab.lopezdp_utils.data.bars import tick_bars

        result = tick_bars(tick_data, threshold=50)
        assert 15 <= len(result) <= 25

    def test_has_ohlcv_columns(self, tick_data):
        from tradelab.lopezdp_utils.data.bars import tick_bars

        result = tick_bars(tick_data, threshold=50)
        for col in ["timestamp", "open", "high", "low", "close", "volume"]:
            assert col in result.columns


class TestVolumeBars:
    """Tests for volume bars."""

    def test_returns_polars_dataframe(self, tick_data):
        from tradelab.lopezdp_utils.data.bars import volume_bars

        result = volume_bars(tick_data, threshold=2500.0)
        assert isinstance(result, pl.DataFrame)

    def test_lower_threshold_more_bars(self, tick_data):
        from tradelab.lopezdp_utils.data.bars import volume_bars

        bars_high = volume_bars(tick_data, threshold=5000.0)
        bars_low = volume_bars(tick_data, threshold=1000.0)
        assert len(bars_low) > len(bars_high)


class TestDollarBars:
    """Tests for dollar bars."""

    def test_returns_polars_dataframe(self, tick_data):
        from tradelab.lopezdp_utils.data.bars import dollar_bars

        result = dollar_bars(tick_data, threshold=250000.0)
        assert isinstance(result, pl.DataFrame)

    def test_has_ohlcv_columns(self, tick_data):
        from tradelab.lopezdp_utils.data.bars import dollar_bars

        result = dollar_bars(tick_data, threshold=250000.0)
        for col in ["timestamp", "open", "high", "low", "close", "volume"]:
            assert col in result.columns


class TestTickImbalanceBars:
    """Tests for tick imbalance bars."""

    def test_returns_polars_dataframe(self, tick_data):
        from tradelab.lopezdp_utils.data.bars import tick_imbalance_bars

        result = tick_imbalance_bars(tick_data, expected_ticks_init=50)
        assert isinstance(result, pl.DataFrame)

    def test_has_ohlcv_columns(self, tick_data):
        from tradelab.lopezdp_utils.data.bars import tick_imbalance_bars

        result = tick_imbalance_bars(tick_data, expected_ticks_init=50)
        for col in ["timestamp", "open", "high", "low", "close", "volume"]:
            assert col in result.columns

    def test_produces_bars(self, tick_data):
        from tradelab.lopezdp_utils.data.bars import tick_imbalance_bars

        result = tick_imbalance_bars(tick_data, expected_ticks_init=50)
        assert len(result) > 0


class TestVolumeImbalanceBars:
    def test_returns_polars_dataframe(self, tick_data):
        from tradelab.lopezdp_utils.data.bars import volume_imbalance_bars

        result = volume_imbalance_bars(tick_data, expected_ticks_init=50)
        assert isinstance(result, pl.DataFrame)

    def test_produces_bars(self, tick_data):
        from tradelab.lopezdp_utils.data.bars import volume_imbalance_bars

        result = volume_imbalance_bars(tick_data, expected_ticks_init=50)
        assert len(result) > 0


class TestTickRunsBars:
    def test_returns_polars_dataframe(self, tick_data):
        from tradelab.lopezdp_utils.data.bars import tick_runs_bars

        result = tick_runs_bars(tick_data, expected_ticks_init=50)
        assert isinstance(result, pl.DataFrame)

    def test_produces_bars(self, tick_data):
        from tradelab.lopezdp_utils.data.bars import tick_runs_bars

        result = tick_runs_bars(tick_data, expected_ticks_init=50)
        assert len(result) > 0
