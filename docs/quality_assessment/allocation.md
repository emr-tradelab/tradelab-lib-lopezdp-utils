# Allocation — Quality Assessment

## Overall Verdict: FIXED — 1 P1 issue resolved

All allocation algorithms (HRP, denoising, detoning, NCO, simulation) are correctly
implemented per López de Prado's AFML Ch. 16 and MLAM Sections 2.4-2.9, Ch. 7.
One P1 correctness issue was found and fixed in NCO weight normalization.

## 1. Correctness Issues

### 1.1 NCO max-Sharpe weight normalization (P1 — FIXED)

`nco.py:41` — `_opt_port` normalized max-Sharpe weights by L1 norm (`np.abs(w).sum()`)
instead of by `ones.T @ w` (sum-to-one) as specified in the book's Snippet 2.10.

**Book (Snippet 2.10):** `w /= np.dot(ones.T, w)` for both min-var and max-Sharpe.
**Implementation:** `w /= np.abs(w).sum()` — L1 normalization, different behavior with
negative weights.

**Fix:** Changed to `w /= np.dot(ones.T, w)` matching the book exactly.

## 2. Missing Functionality (vs. the books)

### 2.1 CLA (Critical Line Algorithm) comparison in hrp_mc — Not implemented

AFML Snippet 16.5 compares HRP vs CLA vs IVP. CLA is omitted (requires external solver).
This is documented in the docstring and acceptable — CLA is a separate optimization
algorithm not part of this library's scope.

### 2.2 Experimental NCO validation (Snippets 7.7-7.9) — Not implemented

MLAM includes controlled experiments demonstrating NCO's ~45-55% RMSE reduction vs
Markowitz. These are research experiments, not reusable utilities. Acceptable omission.

## 3. Edge Cases & Robustness Issues

### 3.1 scipy linkage warning (P2 — documented, not fixed)

`hrp.py:55` — `sch.linkage(dist, method)` passes a full NxN distance matrix.
Scipy issues a `ClusterWarning` because it interprets this as an observation matrix
and computes Euclidean "distance of distances" internally. This matches the book's
intent (confirmed via NotebookLM) but triggers 22 warnings in tests.

Could be suppressed or converted to condensed form, but the behavior is correct.

### 3.2 _min_var_port is unused dead code (P2)

`nco.py:19-25` — `_min_var_port` is defined but never called; `_opt_port` handles
both min-var and max-Sharpe cases. Harmless but redundant.

## 4. Test Quality Assessment

- **23 tests, all passing** across 5 test files + conftest
- Tests are primarily smoke/property tests (shape, type, sum-to-one, positivity)
- No analytical correctness tests with hand-computed reference values
- Denoising tests check positive semi-definiteness — reasonable for RMT
- Integration tests verify denoise→HRP and denoise→NCO pipelines
- Simulation tests verify output format and finiteness

Test quality is adequate for the complexity of these algorithms (analytical reference
values are impractical for most allocation methods).

## 5. Summary: Priority Fixes

| Priority | Issue | Impact | Status |
|----------|-------|--------|--------|
| P1 | NCO max-Sharpe normalization uses L1 instead of sum-to-one | Wrong weights when mu has mixed signs | FIXED |
| P2 | scipy linkage ClusterWarning | Cosmetic warnings in tests | Documented |
| P2 | `_min_var_port` dead code | No impact | Documented |
