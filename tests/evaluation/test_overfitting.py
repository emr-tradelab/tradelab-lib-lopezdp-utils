"""Tests for evaluation.overfitting — CSCV, PBO."""

import numpy as np
import polars as pl
import pytest


class TestProbabilityOfBacktestOverfitting:
    def test_returns_dict(self, trial_returns):
        from tradelab.lopezdp_utils.evaluation.overfitting import (
            probability_of_backtest_overfitting,
        )

        result = probability_of_backtest_overfitting(
            trial_returns,
            n_partitions=6,
            metric="sharpe",
        )
        assert isinstance(result, dict)
        assert "pbo" in result
        assert "logits" in result

    def test_pbo_between_zero_and_one(self, trial_returns):
        from tradelab.lopezdp_utils.evaluation.overfitting import (
            probability_of_backtest_overfitting,
        )

        result = probability_of_backtest_overfitting(
            trial_returns,
            n_partitions=6,
            metric="sharpe",
        )
        assert 0 <= result["pbo"] <= 1

    def test_validates_even_partitions(self, trial_returns):
        from tradelab.lopezdp_utils.evaluation.overfitting import (
            probability_of_backtest_overfitting,
        )

        with pytest.raises(ValueError, match="even"):
            probability_of_backtest_overfitting(
                trial_returns,
                n_partitions=5,
                metric="sharpe",
            )

    def test_random_trials_high_pbo(self):
        from tradelab.lopezdp_utils.evaluation.overfitting import (
            probability_of_backtest_overfitting,
        )

        np.random.seed(42)
        n_obs = 500
        data = {
            "timestamp": pl.datetime_range(
                pl.datetime(2024, 1, 1),
                pl.datetime(2025, 5, 15),
                interval="1d",
                eager=True,
            )[:n_obs]
        }
        for i in range(20):
            data[f"trial_{i}"] = np.random.randn(n_obs) * 0.01
        trials = pl.DataFrame(data)

        result = probability_of_backtest_overfitting(
            trials,
            n_partitions=10,
            metric="sharpe",
        )
        assert result["pbo"] > 0.3
