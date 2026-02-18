# Library Standards

> Auto-generated from Context7 documentation lookups.
> Last updated: 2026-02-18

## Polars Standards

### Version Notes
- Verified against py_1_32_3 / 0.20+

### Key API Patterns

#### Time-based grouping (OHLCV aggregation)
```python
# Correct: group_by_dynamic (not groupby_dynamic — deprecated)
df.group_by_dynamic("timestamp", every="1m").agg([
    pl.col("price").first().alias("open"),
    pl.col("price").max().alias("high"),
    pl.col("price").min().alias("low"),
    pl.col("price").last().alias("close"),
    pl.col("volume").sum().alias("volume"),
])
```

#### DataFrame equality
```python
# Correct: df.equals() (not frame_equal() — removed)
df1.equals(df2)
```

#### Every-nth-row sampling
```python
df.gather_every(step)  # replaces iloc[::step]
```

#### Datetime truncation
```python
pl.col("timestamp").dt.truncate("1m")  # string interval accepted
```

#### When/then/otherwise
```python
pl.when(condition).then(value).otherwise(other_value)
```

#### Polars Series with when/then/otherwise
```python
# pl.when() returns Expr — cannot call .to_numpy() on it directly
# When working with Series (not DataFrame), use NumPy instead:
arr = series.to_numpy()
result = np.where(condition, value_true, value_false)
# Then wrap back: pl.Series(name, result)
```

### Common Pitfalls
- `groupby_dynamic` is deprecated → use `group_by_dynamic`
- `frame_equal()` was removed → use `.equals()`
- Always call `.set_sorted()` on datetime columns before `group_by_dynamic` for performance
- `pl.when(series > 0).then(...)` on a bare Series context returns `Expr`, not `Series` — use NumPy for Series-level conditional ops
- `pyarrow` is not installed in this project — avoid `df.to_pandas()` which requires it
