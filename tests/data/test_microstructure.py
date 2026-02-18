"""Tests for data.microstructure — spread estimators, price impact, VPIN."""

import numpy as np
import polars as pl


class TestTickRule:
    def test_returns_polars_series(self, ohlcv_1min):
        from tradelab.lopezdp_utils.data.microstructure import tick_rule

        result = tick_rule(ohlcv_1min["close"])
        assert isinstance(result, pl.Series)
        assert len(result) == len(ohlcv_1min)

    def test_values_are_plus_minus_one(self, ohlcv_1min):
        from tradelab.lopezdp_utils.data.microstructure import tick_rule

        result = tick_rule(ohlcv_1min["close"])
        unique_vals = result.unique().sort().to_list()
        assert all(v in [-1, 1] for v in unique_vals)

    def test_known_sequence(self):
        from tradelab.lopezdp_utils.data.microstructure import tick_rule

        prices = pl.Series("close", [100.0, 101.0, 101.0, 99.0, 100.0])
        result = tick_rule(prices)
        # 101>100 → +1, 101==101 → +1 (ffill), 99<101 → -1, 100>99 → +1
        expected = [1, 1, 1, -1, 1]
        assert result.to_list() == expected


class TestCorwinSchultzSpread:
    def test_returns_polars_dataframe(self, ohlcv_1min):
        from tradelab.lopezdp_utils.data.microstructure import corwin_schultz_spread

        result = corwin_schultz_spread(ohlcv_1min)
        assert isinstance(result, pl.DataFrame)
        assert "spread" in result.columns

    def test_spread_non_negative(self, ohlcv_1min):
        from tradelab.lopezdp_utils.data.microstructure import corwin_schultz_spread

        result = corwin_schultz_spread(ohlcv_1min)
        assert (result["spread"].drop_nulls() >= 0).all()


class TestRollModel:
    def test_returns_dict(self, ohlcv_1min):
        from tradelab.lopezdp_utils.data.microstructure import roll_model

        result = roll_model(ohlcv_1min["close"])
        assert isinstance(result, dict)
        assert "spread" in result


class TestKyleLambda:
    def test_returns_dict(self, ohlcv_1min):
        from tradelab.lopezdp_utils.data.microstructure import kyle_lambda

        signs = pl.Series("signs", np.random.choice([-1, 1], size=len(ohlcv_1min)).astype(float))
        result = kyle_lambda(
            ohlcv_1min["close"],
            ohlcv_1min["volume"],
            signs,
        )
        assert isinstance(result, dict)
        assert "lambda" in result


class TestAmihudLambda:
    def test_returns_float(self, ohlcv_1min):
        from tradelab.lopezdp_utils.data.microstructure import amihud_lambda

        dollar_vol = ohlcv_1min["close"] * ohlcv_1min["volume"]
        result = amihud_lambda(ohlcv_1min["close"], dollar_vol)
        assert isinstance(result, float)
        assert result >= 0


class TestVolumeBucket:
    def test_returns_polars_dataframe(self, ohlcv_1min):
        from tradelab.lopezdp_utils.data.microstructure import volume_bucket

        result = volume_bucket(
            ohlcv_1min["close"],
            ohlcv_1min["volume"],
            bucket_size=50000.0,
        )
        assert isinstance(result, pl.DataFrame)
        assert len(result) > 0


class TestVpin:
    def test_returns_polars_series(self, ohlcv_1min):
        from tradelab.lopezdp_utils.data.microstructure import vpin

        result = vpin(
            ohlcv_1min["close"],
            ohlcv_1min["volume"],
            bucket_size=50000.0,
            n_buckets=5,
        )
        assert isinstance(result, pl.Series)

    def test_vpin_between_zero_and_one(self, ohlcv_1min):
        from tradelab.lopezdp_utils.data.microstructure import vpin

        result = vpin(
            ohlcv_1min["close"],
            ohlcv_1min["volume"],
            bucket_size=50000.0,
            n_buckets=5,
        )
        non_null = result.drop_nulls()
        if len(non_null) > 0:
            assert (non_null >= 0).all()
            assert (non_null <= 1).all()
