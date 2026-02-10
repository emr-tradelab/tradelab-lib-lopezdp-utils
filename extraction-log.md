
---
## Extraction Run: 2026-02-10 02:56:14

Starting López de Prado extraction workflow...
/Users/ezequielmrivero/scripts/lopez-extraction.sh: line 22: claude: command not found

---
## Extraction Run: 2026-02-10 05:56:14

Starting López de Prado extraction workflow...
/Users/ezequielmrivero/scripts/lopez-extraction.sh: line 22: claude: command not found

---
## Extraction Run: 2026-02-10 08:25:07

Starting López de Prado extraction workflow...
/Users/ezequielmrivero/scripts/lopez-extraction.sh: line 22: claude: command not found

---
## Extraction Run: 2026-02-10 08:25:21

Starting López de Prado extraction workflow...
/Users/ezequielmrivero/scripts/lopez-extraction.sh: line 22: claude: command not found

---
## Extraction Run: 2026-02-10 08:26:59

Starting López de Prado extraction workflow...
**Chapter 6: Ensemble Methods — Complete and merged to main.**

Summary of what was extracted:

| Function | Source | Description |
|----------|--------|-------------|
| `bagging_accuracy()` | Snippet 6.1 | Theoretical majority-voting accuracy for N learners |
| `build_random_forest()` | Snippet 6.2 | Three RF configurations adapted for financial data (methods 0/1/2 with avgU sampling) |
| `bagging_classifier_factory()` | Section 6.6 | Generic BaggingClassifier wrapper for scalability |

MLAM had no new standalone algorithms for this chapter — ensemble applications to bet sizing (Ch.3), feature importance (Ch.8), and portfolio construction (Ch.16) will be covered in their respective chapters.

Next up would be **Chapter 7: Cross-Validation in Finance**.

### Git Status After Extraction
On branch main
Your branch is ahead of 'origin/main' by 2 commits.
  (use "git push" to publish your local commits)

Changes not staged for commit:
  (use "git add/rm <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
	deleted:    scripts/lopez-extraction.sh

Untracked files:
  (use "git add <file>..." to include in what will be committed)
	extraction-log.md

no changes added to commit (use "git add" and/or "git commit -a")
[main 2921c05] chore: automated extraction run 2026-02-10 08:34
 2 files changed, 58 insertions(+), 50 deletions(-)
 create mode 100644 extraction-log.md
 delete mode 100644 scripts/lopez-extraction.sh
To https://github.com/emr-tradelab/tradelab-lib-lopezdp-utils.git
   3cd4f84..2921c05  main -> main

✅ Changes committed and pushed

**Completed at:** 2026-02-10 08:34:29

---
## Extraction Run: 2026-02-10 11:34:29

Starting López de Prado extraction workflow...
Chapter 7 (Cross-Validation in Finance) is complete and merged to main. Here's what was extracted:

**AFML Snippets:**
- `get_train_times` (7.1) — Purging: removes training observations overlapping with test labels
- `get_embargo_times` (7.2) — Embargoing: defines wait period after test sets for serial correlation
- `PurgedKFold` (7.3) — scikit-learn KFold extension with purging + embargoing
- `cv_score` (7.4) — Robust cross_val_score fixing scikit-learn bugs

**MLAM Addition:**
- `probability_weighted_accuracy` (Section 6.4) — Scoring metric penalizing confident wrong predictions

**Note:** MLAM multiple-testing corrections (getZStat, type1Err, type2Err, clusterKMeansBase) were already extracted in Chapter 4 — no duplication.

### Git Status After Extraction
On branch main
Your branch is ahead of 'origin/main' by 2 commits.
  (use "git push" to publish your local commits)

Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
	modified:   extraction-log.md

no changes added to commit (use "git add" and/or "git commit -a")
[main 53ee84d] chore: automated extraction run 2026-02-10 11:43
 1 file changed, 39 insertions(+)
To https://github.com/emr-tradelab/tradelab-lib-lopezdp-utils.git
   2921c05..53ee84d  main -> main

✅ Changes committed and pushed

**Completed at:** 2026-02-10 11:49:06

---
## Extraction Run: 2026-02-10 14:49:06

Starting López de Prado extraction workflow...
**Chapter 8: Feature Importance** — extraction complete and merged to main.

**What was extracted:**

| File | Functions | Source |
|------|-----------|--------|
| `importance.py` | `feat_imp_mdi`, `feat_imp_mda`, `feat_imp_sfi` | AFML Snippets 8.2-8.4 |
| `orthogonal.py` | `get_ortho_feats`, `get_e_vec`, `weighted_kendall_tau` | AFML Snippets 8.5-8.6 |
| `synthetic.py` | `get_test_data`, `feat_importance`, `test_func`, `plot_feat_importance` | AFML Snippets 8.7-8.10 |
| `clustering.py` | `cluster_kmeans_base`, `cluster_kmeans_top`, `feat_imp_mdi_clustered`, `feat_imp_mda_clustered` | MLAM Snippets 4.1-4.2, 6.4-6.5 |

**MLAM note:** PWA scoring was already in Ch.7. The ONC clustering implementation also unblocks the placeholder in `sample_weights/strategy_redundancy.py`.

### Git Status After Extraction
On branch main
Your branch is ahead of 'origin/main' by 2 commits.
  (use "git push" to publish your local commits)

Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
	modified:   extraction-log.md

no changes added to commit (use "git add" and/or "git commit -a")
[main 417892d] chore: automated extraction run 2026-02-10 15:03
 1 file changed, 37 insertions(+)
failed to get: -60008
fatal: could not read Username for 'https://github.com': Device not configured

---
## Extraction Run: 2026-02-10 19:35:59

Starting López de Prado extraction workflow...
**Chapter 9 complete.** Extracted 3 utilities into `tradelab.lopezdp_utils.hyperparameter_tuning`:

| Function | Snippet | Purpose |
|----------|---------|---------|
| `clf_hyper_fit` | 9.1 & 9.3 | Grid/randomized search with PurgedKFold CV + optional bagging |
| `MyPipeline` | 9.2 | Pipeline fix for `sample_weight` routing |
| `log_uniform` | 9.4 | Log-uniform distribution for non-linear parameter search |

MLAM had no new complementary content — multiple-testing corrections were already extracted in Chapter 4. Branch merged to main and cleaned up. Next session: Chapter 10 (Bet Sizing).

### Git Status After Extraction
On branch main
Your branch is ahead of 'origin/main' by 2 commits.
  (use "git push" to publish your local commits)

Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
	modified:   extraction-log.md

no changes added to commit (use "git add" and/or "git commit -a")
