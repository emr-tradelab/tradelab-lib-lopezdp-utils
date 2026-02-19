"""Integration tests: evaluation pipeline."""

import numpy as np
import polars as pl


class TestSharpeToDeflatedPipeline:
    """SR -> PSR -> DSR with explicit trial accounting."""

    def test_sr_to_dsr(self, daily_returns):
        from tradelab.lopezdp_utils.evaluation.statistics import (
            deflated_sharpe_ratio,
            probabilistic_sharpe_ratio,
            sharpe_ratio,
        )

        sr = sharpe_ratio(daily_returns, periods_per_year=252)
        psr = probabilistic_sharpe_ratio(
            observed_sr=sr,
            benchmark_sr=0.0,
            n_obs=len(daily_returns),
        )
        dsr = deflated_sharpe_ratio(
            observed_sr=sr,
            sr_estimates=[sr * 0.5, sr * 0.8, sr],
            n_obs=len(daily_returns),
        )
        assert dsr <= psr


class TestBetSizingPipeline:
    """Signal -> discrete signal -> position."""

    def test_signal_to_position(self):
        from tradelab.lopezdp_utils.evaluation.bet_sizing import (
            bet_size,
            discrete_signal,
            get_target_pos,
            get_w,
        )

        w = get_w(x=5.0, m=0.5)
        raw = bet_size(w, x=3.0)
        assert -1 < raw < 1

        signal = discrete_signal(pl.Series("s", [raw]), step_size=0.1)
        assert len(signal) == 1

        pos = get_target_pos(w, f=105.0, m_p=102.0, max_pos=10)
        assert isinstance(pos, (int, np.integer))


class TestOUToOTRPipeline:
    """Fit O-U -> estimate half-life -> optimal trading rule."""

    def test_fit_then_otr(self, price_series_ou):
        from tradelab.lopezdp_utils.evaluation.synthetic import (
            otr_batch,
            ou_fit,
            ou_half_life,
        )

        params = ou_fit(price_series_ou, forecast=100.0)
        assert 0 < params["phi"] < 1

        hl = ou_half_life(params["phi"])
        assert hl > 0

        result = otr_batch(
            coeffs={"forecast": 100.0, "hl": hl, "sigma": params["sigma"]},
            n_iter=100,
            max_hp=int(hl * 3),
            r_pt=np.linspace(0.5, 2.0, 3),
            r_slm=np.linspace(0.5, 2.0, 3),
            seed=42,
        )
        assert len(result) == 9
        assert "sharpe" in result.columns
