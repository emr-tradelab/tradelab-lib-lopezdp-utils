# Quality Assessment — `data/` module

**Date:** 2026-02-21
**Status:** FIXED (2 P1 issues)
**Tests:** 45 passed, 285 total suite passed
**Reference:** AFML Chapters 2, 19

---

## Module Files Assessed

- `bars.py` — Standard and information-driven bar construction
- `sampling.py` — CUSUM filter, linspace/uniform sampling
- `futures.py` — Roll gap adjustment, continuous contracts
- `etf.py` — ETF trick for basket modeling
- `microstructure.py` — Spread estimators, price impact, VPIN

---

## Findings

### P1 Issues (Fixed)

#### 1. Imbalance bars EWMA tracked raw bar total instead of per-tick average

**Files:** `bars.py` — `tick_imbalance_bars`, `volume_imbalance_bars`, `dollar_imbalance_bars`

**Problem:** The EWMA update for the expected imbalance stored the raw cumulative bar imbalance (e.g., `sum(b_t)` for TIB) rather than the per-tick expected value (`E[b_t]`). Per the book, the threshold is `E_0[T] * |E[b_t]|`, where `E[b_t]` is the EWMA of individual tick signs. Storing the raw total made the threshold scale as `E[T]^2 * E[b_t]` — an extra factor of `E[T]`.

**Fix:** Changed EWMA update to divide by bar length: `ewma_expected_bt = alpha * (tick_imbalance / n) + (1 - alpha) * ewma_expected_bt`. Applied same fix to volume and dollar imbalance variants.

#### 2. Runs bars used current-bar buy probability instead of EWMA from prior bars

**Files:** `bars.py` — `tick_runs_bars`, `volume_runs_bars`, `dollar_runs_bars`

**Problem:** Tick runs bars computed `buy_prob = n_buy / n` within the current bar and used it directly in the threshold. Per the book, the threshold should use `P[b_t=1]` estimated as EWMA from prior bars. The `ewma_buy_prob` variable was computed but never used.

Additionally, the threshold comparison was `n >= expected_runs` (total ticks vs expected runs), but the book defines `theta_T = max(count_buys, count_sells)` — the max of independent buy/sell counts. Changed to `max(n_buy, n - n_buy) >= expected_runs`.

For volume/dollar runs bars, also restructured to track buy/sell volumes separately and use proper per-side conditional expectations: `E[T] * max(P[b=1]*E[v|b=1], P[b=-1]*E[v|b=-1])`.

### P2 Notes (Not Fixed — Enhancement)

- **ETF trick:** Implementation doesn't rebalance holdings when weights change over time. Only initial holdings are computed. Full rebalancing would be a feature enhancement, not a bug fix.

### Correct Implementations (No Issues)

| Component | Reference | Verdict |
|-----------|-----------|---------|
| `time_bars` | Ch. 2, §2.3.1 | PASS |
| `tick_bars` | Ch. 2, §2.3.2 | PASS |
| `volume_bars` | Ch. 2, §2.3.3 | PASS |
| `dollar_bars` | Ch. 2, §2.3.4 | PASS |
| `_compute_tick_rule` | Ch. 2, §2.4 | PASS |
| `get_t_events` (CUSUM) | Ch. 2, Snippet 2.4 | PASS |
| `sampling_linspace` | Ch. 2, §2.5.3.1 | PASS |
| `sampling_uniform` | Ch. 2, §2.5.3.1 | PASS |
| `roll_gaps` | Ch. 2, Snippet 2.2 | PASS |
| `roll_and_rebase` | Ch. 2, Snippet 2.3 | PASS |
| `get_rolled_series` | Ch. 2, Snippet 2.2 | PASS |
| `tick_rule` (microstructure) | Ch. 19, §19.3.1 | PASS |
| `corwin_schultz_spread` | Ch. 19, Snippet 19.1 | PASS |
| `becker_parkinson_volatility` | Ch. 19, Snippet 19.2 | PASS |
| `roll_model` | Ch. 19, §19.3.1 | PASS |
| `high_low_volatility` | Ch. 19, §19.3.1 | PASS |
| `kyle_lambda` | Ch. 19, §19.3.2 | PASS |
| `amihud_lambda` | Ch. 19, §19.3.2 | PASS |
| `hasbrouck_lambda` | Ch. 19, §19.3.2 | PASS |
| `volume_bucket` | Ch. 19, §19.3.3 | PASS |
| `vpin` | Ch. 19, §19.3.3 | PASS |

---

## Test Quality

Tests are functional but light — mostly type/shape checks with few numerical assertions. The existing tests correctly verify API contracts. Future improvement could add numerical accuracy tests for imbalance/runs bar thresholds.
