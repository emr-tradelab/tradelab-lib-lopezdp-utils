"""Tests for evaluation.redundancy — ONC-based strategy redundancy."""

import numpy as np
import polars as pl
import pytest


class TestGetEffectiveTrials:
    def test_returns_dict_with_required_keys(self, trial_returns):
        from tradelab.lopezdp_utils.evaluation.redundancy import get_effective_trials

        result = get_effective_trials(trial_returns)
        assert isinstance(result, dict)
        assert "n_effective" in result
        assert "cluster_srs" in result
        assert "sr_variance" in result
        assert "clusters" in result

    def test_fewer_clusters_than_trials(self, trial_returns):
        from tradelab.lopezdp_utils.evaluation.redundancy import get_effective_trials

        n_trials = len([c for c in trial_returns.columns if c != "timestamp"])
        result = get_effective_trials(trial_returns)
        assert result["n_effective"] <= n_trials

    def test_correlated_trials_fewer_clusters(self):
        from tradelab.lopezdp_utils.evaluation.redundancy import get_effective_trials

        np.random.seed(42)
        n = 200
        base = np.random.randn(n) * 0.01
        data = {
            "timestamp": pl.datetime_range(
                pl.datetime(2024, 1, 1),
                pl.datetime(2024, 7, 18),
                interval="1d",
                eager=True,
            )[:n]
        }
        # 5 highly correlated trials + 5 independent
        for i in range(5):
            data[f"corr_{i}"] = base + np.random.randn(n) * 0.001
        for i in range(5):
            data[f"indep_{i}"] = np.random.randn(n) * 0.01
        df = pl.DataFrame(data)
        result = get_effective_trials(df)
        # Should find substantially fewer than 10 effective trials
        assert result["n_effective"] < 10

    def test_raises_with_single_trial(self):
        from tradelab.lopezdp_utils.evaluation.redundancy import get_effective_trials

        df = pl.DataFrame(
            {
                "timestamp": pl.datetime_range(
                    pl.datetime(2024, 1, 1),
                    pl.datetime(2024, 1, 10),
                    interval="1d",
                    eager=True,
                )[:10],
                "trial_0": np.random.randn(10),
            }
        )
        with pytest.raises(ValueError):
            get_effective_trials(df)


class TestDeflatedSharpeRatioClustered:
    def test_returns_float_between_0_and_1(self, trial_returns):
        from tradelab.lopezdp_utils.evaluation.redundancy import (
            deflated_sharpe_ratio_clustered,
        )

        result = deflated_sharpe_ratio_clustered(
            observed_sr=0.5,
            trial_returns=trial_returns,
            n_obs=200,
        )
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
