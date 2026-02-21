# Quality Assessment — `features/` module

**Date:** 2026-02-21
**Status:** PASS (zero P0/P1 issues)
**Tests:** 54 passed
**Reference:** AFML Chapters 5, 8, 17, 18; MLAM Chapters 3-6

---

## Module Files Assessed

- `fractional_diff.py` — Fractional differentiation (expanding + FFD)
- `entropy.py` — Plug-in, LZ, Kontoyiannis, encoding, MI, VI, KL divergence
- `structural_breaks.py` — SADF, BDE-CUSUM, CSW-CUSUM, Chow-DF, QADF
- `importance.py` — MDI, MDA, SFI, ONC clustering
- `orthogonal.py` — PCA weights, orthogonal features, weighted Kendall tau

---

## Findings

### No P0/P1 Issues

All implementations correctly match the book's algorithms and formulas.

### Correct Implementations

| Component | Reference | Verdict |
|-----------|-----------|---------|
| `get_weights` | Ch. 5, w_k = -w_{k-1}(d-k+1)/k | PASS |
| `get_weights_ffd` | Ch. 5, threshold-based cutoff | PASS |
| `frac_diff` | Ch. 5, expanding window | PASS |
| `frac_diff_ffd` | Ch. 5, fixed-width window | PASS |
| `plot_min_ffd` | Ch. 5, ADF sweep for d* | PASS |
| `plug_in` | Ch. 18, ML entropy rate | PASS |
| `lempel_ziv_lib` | Ch. 18, LZ77 parsing | PASS |
| `konto` | Ch. 18, Kontoyiannis estimator | PASS |
| `encode_binary/quantile/sigma` | Ch. 18, encoding schemes | PASS |
| `market_efficiency_metric` | Ch. 18, redundancy | PASS |
| `num_bins` | MLAM Ch. 3, optimal binning | PASS |
| `variation_of_information` | MLAM Ch. 3 | PASS |
| `mutual_information_optimal` | MLAM Ch. 3 | PASS |
| `kl_divergence` / `cross_entropy` | Information theory | PASS |
| `sadf_test` | Ch. 17, SADF | PASS |
| `brown_durbin_evans_cusum` | Ch. 17, BDE CUSUM | PASS |
| `chu_stinchcombe_white_cusum` | Ch. 17, CSW CUSUM | PASS |
| `chow_type_dickey_fuller` | Ch. 17, Chow-DF | PASS |
| `qadf_test` | Ch. 17, Quantile ADF | PASS |
| `feat_imp_mdi` | Ch. 8, MDI | PASS |
| `feat_imp_mda` | Ch. 8, MDA | PASS |
| `cluster_kmeans_base/top` | MLAM, ONC clustering | PASS |
| `get_ortho_feats` | Ch. 8 / MLAM, PCA orthogonalization | PASS |
| `pca_weights` | MLAM, eigenvalue-based allocation | PASS |
| `weighted_kendall_tau` | Ch. 8, confirmatory check | PASS |

### P2 Notes

- `structural_breaks.py:84` produces RuntimeWarnings for `sqrt` of negative values in some edge cases. This is expected behavior for unstable regressions with insufficient data and is clipped appropriately.
