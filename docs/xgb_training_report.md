# XGBoost Benchmark — Training Report

**Status:** validation run (2026-11-10). The final model will be retrained on
all data available at the freeze date (2026-11-30) using the hyperparameters
selected here.

**Headline configuration:** 10 numeric features, **no sector column**
(decision 2026-11-10, `docs/zmiany_do_planu.md`). The with-sector variant is
kept as a comparison in §7.2.

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
The sector column defined in `features.py::CATEGORICAL_COL` is **not** passed
to the model (see §7.2 for the with-sector comparison and the reasoning).

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

> **max_depth = 4, learning_rate = 0.03, n_estimators = 200** — mean IC **+0.0356**

Median of the 12 grid-config mean ICs: **+0.0286**. Full grid (sorted):

| mean IC | params                                              |
|---------|-----------------------------------------------------|
| +0.0356 | max_depth=4, lr=0.03, n_estimators=200              |
| +0.0350 | max_depth=4, lr=0.03, n_estimators=500              |
| +0.0344 | max_depth=3, lr=0.03, n_estimators=500              |
| +0.0297 | max_depth=2, lr=0.10, n_estimators=500              |
| +0.0293 | max_depth=2, lr=0.03, n_estimators=200              |
| +0.0287 | max_depth=3, lr=0.10, n_estimators=200              |
| +0.0285 | max_depth=4, lr=0.10, n_estimators=200              |
| +0.0283 | max_depth=3, lr=0.10, n_estimators=500              |
| +0.0259 | max_depth=2, lr=0.10, n_estimators=200              |
| +0.0245 | max_depth=4, lr=0.10, n_estimators=500              |
| +0.0229 | max_depth=2, lr=0.03, n_estimators=500              |
| +0.0223 | max_depth=3, lr=0.03, n_estimators=200              |

## 4. Per-fold IC — best XGB vs baselines

Baselines evaluated on the same folds:
- **mom4w** — rank by 4-week momentum alone (deterministic).
- **random** — random permutation of ticker ranks; 50 draws per week, seeded
  per fold, mean of resulting rank ICs.

| fold | train weeks | test weeks | IC best XGB | IC mom4w | IC random |
|------|-------------|------------|-------------|----------|-----------|
| 2022 | 320         | 52         | +0.0628     | −0.0288  | +0.0039   |
| 2023 | 372         | 52         | +0.0214     | −0.0171  | −0.0015   |
| 2024 | 424         | 51         | +0.0156     | +0.0276  | +0.0032   |
| 2025 | 475         | 51         | +0.0203     | +0.0022  | −0.0021   |
| 2026 | 526         | 42         | +0.0580     | +0.0102  | −0.0004   |
| **mean** | —       | —          | **+0.0356** | **−0.0012** | **+0.0006** |

No fold or mean IC exceeds the 0.10 leakage-flag threshold; proceeding.

## 5. Feature importance (gain-based)

Model refit on all data through 2025 (the expanding-window training set used
for the 2026 fold) with the chosen hyperparameters.

| rank | feature          | importance |
|------|------------------|------------|
| 1    | range_20d        | 0.140      |
| 2    | mom_1w           | 0.132      |
| 3    | mom_12w          | 0.126      |
| 4    | dist_ma50        | 0.105      |
| 5    | ret_vs_dia_4w    | 0.101      |
| 6    | rel_volume_5_60  | 0.094      |
| 7    | vol_20d          | 0.086      |
| 8    | mom_4w           | 0.077      |
| 9    | mom_26w          | 0.075      |
| 10   | dist_ma200       | 0.065      |

Without sector, importance spreads more evenly across the 10 features;
short-horizon volatility-style signals (`range_20d`, `mom_1w`) sit on top.

## 6. Repro

```
PYTHONPATH=src python3 src/xgb/fetch_history.py          # cache Yahoo OHLCV
PYTHONPATH=src python3 src/xgb/dataset.py                # build panel
PYTHONPATH=src python3 src/xgb/train.py                  # walk-forward CV
PYTHONPATH=src python3 src/xgb/robustness.py             # section-7 checks
PYTHONPATH=src python3 src/xgb/predict.py --as-of 2026-11-06 --train-first
pytest tests/test_xgb_leakage.py
```

CV artifacts live at `data/raw/xgb_cv_results.json` and
`data/raw/xgb_robustness.json`; trained model binaries are not committed
(data is refreshed on every run; the frozen model will be produced on
2026-11-30).

## 7. Robustness

Three checks on the headline numbers; the chosen configuration is kept
unchanged.

### 7.1 Universe is ex post — survivorship bias is unavoidable here

The 30-ticker universe is the DJIA snapshot at 2026-11-04. **All 30 tickers
are ex-post survivors**: they passed the index committee's selection at
some point and remained listed. Stocks removed from the DJIA since 2015 are
absent from our data entirely — the committee replaced them when they
underperformed or were restructured. From the Wikipedia article
*"Historical components of the Dow Jones Industrial Average"* (fetched
2026-11-10), the 11 names dropped from the index during the validation
window are:

| removal date | ticker | company                                   |
|--------------|--------|-------------------------------------------|
| 2017-09-01   | DD     | DuPont (rolled into DowDuPont)            |
| 2018-06-26   | GE     | General Electric                          |
| 2019-04-02   | DWDP   | DowDuPont (replaced by Dow Inc.)          |
| 2020-04-06   | UTX    | United Technologies (merged into RTX)     |
| 2020-08-31   | RTX    | Raytheon Technologies                     |
| 2020-08-31   | XOM    | ExxonMobil                                |
| 2020-08-31   | PFE    | Pfizer                                    |
| 2024-02-26   | WBA    | Walgreens Boots Alliance                  |
| 2024-11-08   | INTC   | Intel                                     |
| 2024-11-08   | DOW    | Dow Inc.                                  |
| 2026-06-29   | VZ     | Verizon Communications                    |

Seven current constituents also joined only after 2015-07-01 (AMGN, CRM,
HON, AMZN, NVDA, SHW, GOOGL — see `src/xgb/robustness.py::NEW_ENTRANTS`);
including them in training over the full window is itself a look-ahead
selection by the index committee.

**Implication for the IC reported in §4:** the validation IC is computed on
an ex-post universe (survivors only, after-the-fact additions included) and
should be read as an **optimistic upper bound**. The clean evaluation is
the 28-week forward test, where the universe is frozen ex ante on
2026-12-04 and no committee change can rewrite history.

As a descriptive check we also score the model (trained on the full
universe) only on the 23 current tickers that were in the index for the
whole validation window. **This does not remove survivorship bias — the 23
are still ex-post survivors** — it merely shows that the bulk of the IC
does not come from the seven late-joiners alone.

| fold | IC best XGB (stable 23) | IC mom4w (stable 23) |
|------|-------------------------|----------------------|
| 2022 | +0.0834                 | −0.0321              |
| 2023 | +0.0301                 | +0.0119              |
| 2024 | −0.0024                 | +0.0034              |
| 2025 | +0.0412                 | +0.0077              |
| 2026 | +0.0634                 | +0.0300              |
| **mean** | **+0.0431**         | **+0.0041**          |

### 7.2 With-sector comparison

Rerunning the same grid with the 9 sector one-hots added to the feature
matrix (`src/xgb/robustness.py`, `_fit_xgb_with_sector`):

- **Best with-sector:** max_depth=4, lr=0.10, n_estimators=200;
  mean IC **+0.0454** (vs +0.0356 no-sector).

| fold | IC best XGB (with sector) |
|------|---------------------------|
| 2022 | +0.1000                   |
| 2023 | +0.0444                   |
| 2024 | +0.0231                   |
| 2025 | +0.0139                   |
| 2026 | +0.0455                   |
| **mean** | **+0.0454**           |

Including the sector column lifts the headline IC by roughly one
IC-point, but on an ex-post universe that improvement is not a clean
causal gain: the Materials sector is represented by a single ticker (SHW),
and other sectors have only two or three constituents — a tree node
splitting on `sector == Materials` then functions as a near-identity for
SHW. Combined with the survivorship issue in §7.1, we keep the model
sector-free so that no feature can act as a company ID on an ex-post
universe.

### 7.3 Selection bias (12 configs on the same folds)

The headline +0.0356 is the best of 12 grid configs evaluated on the same
walk-forward folds.

| statistic | value   |
|-----------|---------|
| best      | +0.0356 |
| median    | +0.0286 |
| min       | +0.0223 |
| max       | +0.0356 |
| n configs | 12      |

The gap between best and median (≈0.007 IC) is an upper bound on the
selection-bias premium baked into the headline. All 12 no-sector configs
still beat both baselines (mom4w −0.0012, random +0.0006); the choice
between them is tighter than the XGB-vs-baseline gap.

Per-fold IC of the chosen configuration (repeated for cross-reference
with §4):

| fold | IC best XGB |
|------|-------------|
| 2022 | +0.0628     |
| 2023 | +0.0214     |
| 2024 | +0.0156     |
| 2025 | +0.0203     |
| 2026 | +0.0580     |
