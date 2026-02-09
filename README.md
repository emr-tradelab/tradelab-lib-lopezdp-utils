# tradelab-lib-lopezdp-utils

A Python library of financial machine learning utilities extracted from:
- **"Advances in Financial Machine Learning"** by Marcos López de Prado (primary)
- **"Machine Learning for Asset Managers"** by Marcos López de Prado (complementary)

Part of the Tradelab algorithmic trading ecosystem.

## Usage

```python
from tradelab.lopezdp_utils.data_structures import ...
from tradelab.lopezdp_utils.labeling import ...
```

## Modules

Each submodule corresponds to a chapter/topic from the books:

### Part 1: Data Analysis
| Module | Chapter | Topic | Status |
|--------|---------|-------|--------|
| `data_structures` | Ch 2 | Financial Data Structures (bars, imbalance bars) | ✅ v1 Complete |
| `labeling` | Ch 3 | Triple-Barrier Method, Meta-Labeling | Planned |
| `sample_weights` | Ch 4 | Sample Weights & Uniqueness | Planned |
| `fractional_diff` | Ch 5 | Fractionally Differentiated Features | Planned |

### Part 2: Modelling
| Module | Chapter | Topic | Status |
|--------|---------|-------|--------|
| `ensemble_methods` | Ch 6 | Ensemble Methods | Planned |
| `cross_validation` | Ch 7 | Purged K-Fold Cross-Validation | Planned |
| `feature_importance` | Ch 8 | Feature Importance (MDA, MDI, SFI) | Planned |
| `hyperparameter_tuning` | Ch 9 | Hyper-Parameter Tuning | Planned |

### Part 3: Backtesting
| Module | Chapter | Topic | Status |
|--------|---------|-------|--------|
| `bet_sizing` | Ch 10 | Bet Sizing | Planned |
| `backtest_dangers` | Ch 11 | Backtesting Pitfalls | Planned |
| `backtest_cv` | Ch 12 | Backtesting through Cross-Validation | Planned |
| `backtest_synthetic` | Ch 13 | Backtesting on Synthetic Data | Planned |
| `backtest_statistics` | Ch 14 | Backtest Statistics | Planned |
| `strategy_risk` | Ch 15 | Understanding Strategy Risk | Planned |
| `ml_asset_allocation` | Ch 16 | Machine Learning Asset Allocation | Planned |

### Part 4: Useful Financial Features
| Module | Chapter | Topic | Status |
|--------|---------|-------|--------|
| `structural_breaks` | Ch 17 | Structural Breaks (CUSUM, etc.) | Planned |
| `entropy` | Ch 18 | Entropy Features | Planned |
| `microstructure` | Ch 19 | Market Microstructure Features | Planned |

### Part 5: High-Performance Computing
| Module | Chapter | Topic | Status |
|--------|---------|-------|--------|
| `hpc` | Ch 20 | Multiprocessing and Vectorization | Planned |

> See `TODO.md` for detailed progress tracking.