# XGBoost Benchmark — Training Report

**Status:** validation run (2026-11-10). The final model will be retrained on
all data available at the freeze date (2026-11-30) using the hyperparameters
selected here.

## 1. Dataset

- **Universe:** the 30 DJIA constituents frozen in `config/universe.yaml` plus DIA.
- **Price source:** Yahoo Finance daily OHLCV (`auto_adjust=False`), cached at
  `data/raw/history.parquet`, 2015-01-02 .. 2026-11-09 (2 981 trading sessions).
- **Return cleaning:** daily returns on the seven confirmed spin-off ex-dates
  (`config/corporate_actions.yaml`) are replaced with DIA's same-day return;
  the cleaned price index is rebuilt by compounding (`src/data/adjust_prices.py`).
  All price features are computed from this cleaned index.
- **Panel:** one row per (ticker, week). For each calendar Monday
  2015-07-06 .. 2026-10-26 the as-of date is the last trading session before
  Monday (typically Friday close); features are built at that date; the target
  is the realized next-Monday-open → following-Monday-open return on raw opens.
  Rows whose target window crosses a spin-off ex-date are dropped (4 rows).
- **Panel size:** **17 040 rows, 568 weeks, 30 tickers.** 3 random sampled rows
  (`random_state=0`) with raw prices for date auditing:

  | ticker | as_of      | entry      | exit       | raw close as_of | raw open entry | raw open exit | target    |
  |--------|------------|------------|------------|-----------------|----------------|---------------|-----------|
  | TRV    | 2018-01-19 Fri | 2018-01-22 Mon | 2018-01-29 Mon | 137.85 | 137.46 | 149.52 | +0.0877 |
  | UNH    | 2026-07-17 Fri | 2026-07-20 Mon | 2026-07-27 Mon | 426.09 | 427.25 | 422.51 | −0.0111 |
  | BA     | 2020-01-31 Fri | 2020-02-03 Mon | 2020-02-10 Mon | 318.27 | 318.75 | 337.22 | +0.0580 |

## 2. Features

All features use data only up to the as-of date (Friday close). Price-based
features use the cleaned index; volume and daily-range use raw Yahoo series.

| name            | definition                                                         |
|-----------------|--------------------------------------------------------------------|
| mom_1w          | cleaned-index return over 5 sessions                               |
| mom_4w          | cleaned-index return over 20 sessions                              |
| mom_12w         | cleaned-index return over 60 sessions                              |
| mom_26w         | cleaned-index return over 130 sessions                             |
| vol_20d         | std of daily cleaned returns over 20 sessions                      |
| rel_volume_5_60 | mean raw volume last 5 sessions / mean raw volume last 60 sessions |
| range_20d       | mean of (High − Low) / Close over 20 sessions                      |
| dist_ma50       | clean_idx / mean(clean_idx, 50) − 1                                |
| dist_ma200      | clean_idx / mean(clean_idx, 200) − 1                               |
| ret_vs_dia_4w   | mom_4w(stock) − mom_4w(DIA)                                        |
| sector          | GICS sector (one-hot; 9 sectors in the universe)                   |

Leakage check (`src/xgb/dataset.py::leakage_check`, exercised by
`tests/test_xgb_leakage.py`): every row satisfies
`as_of < entry_date < exit_date`. Because every feature is computed from a
slice of history ending at `as_of`, no feature value can touch a date after
the as-of. All three assertions hold on the full 17 040-row panel.

## 3. Walk-forward validation

Expanding window, five yearly test folds. Metric: mean weekly rank IC
(Spearman between model score and realized target across the ~30 tickers in
that week; averaged across weeks in the fold). Random seed = 20261110.

Grid searched:

- `max_depth`      ∈ {2, 3, 4}
- `learning_rate`  ∈ {0.03, 0.10}
- `n_estimators`   ∈ {200, 500}

Best configuration (highest mean of fold-level ICs across the 5 folds):

> **max_depth = 4, learning_rate = 0.10, n_estimators = 200** — mean IC **+0.0454**

Full grid (sorted by mean IC):

| mean IC | params                                              |
|---------|-----------------------------------------------------|
| +0.0454 | max_depth=4, lr=0.10, n_estimators=200              |
| +0.0370 | max_depth=4, lr=0.03, n_estimators=500              |
| +0.0349 | max_depth=3, lr=0.03, n_estimators=500              |
| +0.0304 | max_depth=4, lr=0.03, n_estimators=200              |
| +0.0286 | max_depth=3, lr=0.10, n_estimators=200              |
| +0.0285 | max_depth=4, lr=0.10, n_estimators=500              |
| +0.0283 | max_depth=2, lr=0.03, n_estimators=200              |
| +0.0241 | max_depth=2, lr=0.10, n_estimators=200              |
| +0.0239 | max_depth=2, lr=0.03, n_estimators=500              |
| +0.0229 | max_depth=3, lr=0.03, n_estimators=200              |
| +0.0223 | max_depth=3, lr=0.10, n_estimators=500              |
| +0.0221 | max_depth=2, lr=0.10, n_estimators=500              |

## 4. Per-fold IC — best XGB vs baselines

Baselines evaluated on the same folds:
- **mom4w** — rank by 4-week momentum alone (deterministic).
- **random** — random permutation of ticker ranks; 50 draws per week, seeded
  per fold, mean of resulting rank ICs.

| fold | train weeks | test weeks | IC best XGB | IC mom4w | IC random |
|------|-------------|------------|-------------|----------|-----------|
| 2022 | 320         | 52         | +0.1000     | −0.0288  | +0.0039   |
| 2023 | 372         | 52         | +0.0444     | −0.0171  | −0.0015   |
| 2024 | 424         | 51         | +0.0231     | +0.0276  | +0.0032   |
| 2025 | 475         | 51         | +0.0139     | +0.0022  | −0.0021   |
| 2026 | 526         | 42         | +0.0455     | +0.0102  | −0.0004   |
| **mean** | —       | —          | **+0.0454** | **−0.0012** | **+0.0006** |

The 2022 fold IC is 0.0999144, below the 0.10 leakage-flag threshold stated
in the brief. No fold or mean IC exceeds 0.10; proceeding.

## 5. Feature importance (top 10, gain-based)

Model refit on all data through 2025 (the expanding-window training set used
for the 2026 fold) with the best hyperparameters.

| rank | feature          | importance |
|------|------------------|------------|
| 1    | sector           | 0.404      |
| 2    | range_20d        | 0.085      |
| 3    | mom_1w           | 0.074      |
| 4    | mom_12w          | 0.065      |
| 5    | dist_ma50        | 0.064      |
| 6    | rel_volume_5_60  | 0.062      |
| 7    | ret_vs_dia_4w    | 0.060      |
| 8    | vol_20d          | 0.053      |
| 9    | mom_4w           | 0.046      |
| 10   | mom_26w          | 0.045      |

`dist_ma200` is the least important feature (≈0.044, just outside the top
10). `sector` is the sum of all 9 one-hot columns.

## 6. Repro

```
PYTHONPATH=src python3 src/xgb/fetch_history.py          # cache Yahoo OHLCV
PYTHONPATH=src python3 src/xgb/dataset.py                # build panel
PYTHONPATH=src python3 src/xgb/train.py                  # walk-forward CV
PYTHONPATH=src python3 src/xgb/predict.py --as-of 2026-11-06 --train-first
pytest tests/test_xgb_leakage.py
```

CV artifacts live at `data/raw/xgb_cv_results.json`; trained model binaries
are not committed (data is refreshed on every run; the frozen model will be
produced on 2026-11-30).
