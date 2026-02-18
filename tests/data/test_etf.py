"""Tests for data.etf — ETF trick."""

import polars as pl
import pytest


class TestEtfTrick:
    def test_returns_polars_series(self):
        from tradelab.lopezdp_utils.data.etf import etf_trick

        prices = pl.DataFrame(
            {
                "timestamp": pl.datetime_range(
                    pl.datetime(2024, 1, 1),
                    pl.datetime(2024, 1, 5),
                    interval="1d",
                    eager=True,
                ),
                "asset_a": [100.0, 101.0, 102.0, 103.0, 104.0],
                "asset_b": [50.0, 50.5, 51.0, 50.5, 51.5],
            }
        )
        weights = pl.DataFrame(
            {
                "timestamp": pl.datetime_range(
                    pl.datetime(2024, 1, 1),
                    pl.datetime(2024, 1, 5),
                    interval="1d",
                    eager=True,
                ),
                "asset_a": [0.6, 0.6, 0.6, 0.6, 0.6],
                "asset_b": [0.4, 0.4, 0.4, 0.4, 0.4],
            }
        )
        result = etf_trick(prices, weights)
        assert isinstance(result, pl.Series)
        assert len(result) == 5

    def test_starts_at_one(self):
        from tradelab.lopezdp_utils.data.etf import etf_trick

        prices = pl.DataFrame(
            {
                "timestamp": pl.datetime_range(
                    pl.datetime(2024, 1, 1),
                    pl.datetime(2024, 1, 5),
                    interval="1d",
                    eager=True,
                ),
                "asset_a": [100.0, 101.0, 102.0, 103.0, 104.0],
                "asset_b": [50.0, 50.5, 51.0, 50.5, 51.5],
            }
        )
        weights = pl.DataFrame(
            {
                "timestamp": pl.datetime_range(
                    pl.datetime(2024, 1, 1),
                    pl.datetime(2024, 1, 5),
                    interval="1d",
                    eager=True,
                ),
                "asset_a": [0.6, 0.6, 0.6, 0.6, 0.6],
                "asset_b": [0.4, 0.4, 0.4, 0.4, 0.4],
            }
        )
        result = etf_trick(prices, weights)
        assert result[0] == pytest.approx(1.0)
