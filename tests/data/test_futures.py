"""Tests for data.futures — roll-adjusted continuous contracts."""

import polars as pl


class TestRollGaps:
    def test_returns_polars_series(self):
        from tradelab.lopezdp_utils.data.futures import roll_gaps

        df = pl.DataFrame(
            {
                "timestamp": pl.datetime_range(
                    pl.datetime(2024, 1, 1),
                    pl.datetime(2024, 1, 10),
                    interval="1d",
                    eager=True,
                ),
                "instrument": ["A"] * 5 + ["B"] * 5,
                "open": [100.0, 101.0, 102.0, 103.0, 104.0, 204.0, 205.0, 206.0, 207.0, 208.0],
                "close": [100.5, 101.5, 102.5, 103.5, 104.5, 204.5, 205.5, 206.5, 207.5, 208.5],
            }
        )
        result = roll_gaps(df)
        assert isinstance(result, pl.Series)
        assert len(result) == len(df)

    def test_no_rolls_returns_zeros(self):
        from tradelab.lopezdp_utils.data.futures import roll_gaps

        df = pl.DataFrame(
            {
                "timestamp": pl.datetime_range(
                    pl.datetime(2024, 1, 1),
                    pl.datetime(2024, 1, 5),
                    interval="1d",
                    eager=True,
                ),
                "instrument": ["A"] * 5,
                "open": [100.0, 101.0, 102.0, 103.0, 104.0],
                "close": [100.5, 101.5, 102.5, 103.5, 104.5],
            }
        )
        result = roll_gaps(df)
        assert isinstance(result, pl.Series)
        assert result.abs().sum() == 0.0


class TestRollAndRebase:
    def test_returns_polars_dataframe(self):
        from tradelab.lopezdp_utils.data.futures import roll_and_rebase

        df = pl.DataFrame(
            {
                "timestamp": pl.datetime_range(
                    pl.datetime(2024, 1, 1),
                    pl.datetime(2024, 1, 5),
                    interval="1d",
                    eager=True,
                ),
                "instrument": ["A"] * 5,
                "open": [100.0, 101.0, 102.0, 103.0, 104.0],
                "close": [100.5, 101.5, 102.5, 103.5, 104.5],
            }
        )
        result = roll_and_rebase(df)
        assert isinstance(result, pl.DataFrame)

    def test_has_r_prices_column(self):
        from tradelab.lopezdp_utils.data.futures import roll_and_rebase

        df = pl.DataFrame(
            {
                "timestamp": pl.datetime_range(
                    pl.datetime(2024, 1, 1),
                    pl.datetime(2024, 1, 5),
                    interval="1d",
                    eager=True,
                ),
                "instrument": ["A"] * 5,
                "open": [100.0, 101.0, 102.0, 103.0, 104.0],
                "close": [100.5, 101.5, 102.5, 103.5, 104.5],
            }
        )
        result = roll_and_rebase(df)
        assert "r_prices" in result.columns

    def test_r_prices_positive(self):
        from tradelab.lopezdp_utils.data.futures import roll_and_rebase

        df = pl.DataFrame(
            {
                "timestamp": pl.datetime_range(
                    pl.datetime(2024, 1, 1),
                    pl.datetime(2024, 1, 5),
                    interval="1d",
                    eager=True,
                ),
                "instrument": ["A"] * 5,
                "open": [100.0, 101.0, 102.0, 103.0, 104.0],
                "close": [100.5, 101.5, 102.5, 103.5, 104.5],
            }
        )
        result = roll_and_rebase(df)
        assert (result["r_prices"].drop_nulls() > 0).all()
