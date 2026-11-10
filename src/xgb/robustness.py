"""
Robustness checks for the XGBoost walk-forward benchmark (no new data).

1. Survivorship / stable subset: the 30-ticker universe is the 2026-11-04
   DJIA snapshot; 7 of those joined after 2015-07-01 (see STABLE_SUBSET
   below, verified against the Wikipedia "Historical components" article).
   Re-evaluate the best XGB config and the mom4w baseline on the 23 names
   that were in the index for the whole 2015-07..2026-10 period.
2. Without sector: rerun the walk-forward grid search dropping the sector
   feature. Report the best config and its per-fold IC.
3. Selection bias: report the median mean-IC across the 12 grid configs
   next to the best, and the best config's IC on each fold.

Output: data/raw/xgb_robustness.json; printed summary suitable for pasting
into docs/xgb_training_report.md.
"""

from __future__ import annotations

import json
import logging
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from xgb.features import CATEGORICAL_COL, FEATURE_COLS
from xgb.train import (
    FOLD_YEARS, GRID, SEED, _fit_xgb, _predict_xgb,
    _random_rank_ic, _weekly_rank_ic,
)

logger = logging.getLogger(__name__)
_ROOT = Path(__file__).parents[2]

# Verified against the Wikipedia article
# "Historical components of the Dow Jones Industrial Average"
# (fetched 2026-11-10). Tickers in the current universe that joined the
# DJIA after 2015-07-01 (i.e. absent for part of the 2015-07..2026-10
# validation window):
#   AMGN  (2020-08-31), CRM (2020-08-31), HON   (2020-08-31, re-added),
#   AMZN  (2024-02-26), NVDA (2024-11-08), SHW  (2024-11-08),
#   GOOGL (2026-06-29).
NEW_ENTRANTS = ["AMGN", "CRM", "HON", "AMZN", "NVDA", "SHW", "GOOGL"]


def _per_fold_ic_for_params(panel: pd.DataFrame, params: dict, sectors: list[str]) -> dict:
    """Train the config on each expanding window and score on the test fold."""
    out = {}
    for year in FOLD_YEARS:
        train = panel[panel["as_of"].dt.year <  year]
        test  = panel[panel["as_of"].dt.year == year]
        if len(train) == 0 or len(test) == 0:
            continue
        model, cols = _fit_xgb(train, params, sectors)
        pred = _predict_xgb(model, cols, test, sectors)
        tmp = test.copy()
        tmp["pred"] = pred
        out[year] = _weekly_rank_ic(tmp, "pred")
    return out


def _fit_xgb_nosector(train: pd.DataFrame, params: dict):
    """Fit using only the 10 numeric FEATURE_COLS (no sector one-hots)."""
    import xgboost as xgb

    X = train[FEATURE_COLS]
    y = train["target"].to_numpy()
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=SEED,
        n_jobs=1,
        **params,
    )
    model.fit(X, y)
    return model


def _per_fold_nosector(panel: pd.DataFrame, params: dict) -> dict:
    out = {}
    for year in FOLD_YEARS:
        train = panel[panel["as_of"].dt.year <  year]
        test  = panel[panel["as_of"].dt.year == year]
        if len(train) == 0 or len(test) == 0:
            continue
        model = _fit_xgb_nosector(train, params)
        pred  = model.predict(test[FEATURE_COLS])
        tmp = test.copy()
        tmp["pred"] = pred
        out[year] = _weekly_rank_ic(tmp, "pred")
    return out


def run(panel_path: Path | str, cv_path: Path | str) -> dict:
    panel = pd.read_parquet(panel_path)
    panel["as_of"] = pd.to_datetime(panel["as_of"])
    panel = panel.dropna(subset=FEATURE_COLS + ["target"]).copy()
    cv = json.loads(Path(cv_path).read_text())
    best = cv["best_params"]
    sectors = cv["sectors"]

    # ── 1. Survivorship (stable subset) ──────────────────────────────────
    stable_tickers = sorted(set(panel["ticker"]) - set(NEW_ENTRANTS))
    stable_panel   = panel[panel["ticker"].isin(stable_tickers)].copy()
    logger.info("Stable subset: %d tickers, %d rows",
                len(stable_tickers), len(stable_panel))

    stable_xgb: dict[int, float] = {}
    stable_mom: dict[int, float] = {}
    for year in FOLD_YEARS:
        # Train on the full universe (as in the headline run), score only on
        # stable-subset rows in the test fold. This isolates the survivorship
        # effect on the metric, not on the training signal.
        train = panel[panel["as_of"].dt.year <  year]
        test  = stable_panel[stable_panel["as_of"].dt.year == year]
        if len(train) == 0 or len(test) == 0:
            continue
        model, cols = _fit_xgb(train, best, sectors)
        pred = _predict_xgb(model, cols, test, sectors)
        tmp = test.copy()
        tmp["pred"] = pred
        stable_xgb[year] = _weekly_rank_ic(tmp, "pred")

        tmp2 = test.copy()
        tmp2["mom4w_pred"] = tmp2["mom_4w"]
        stable_mom[year] = _weekly_rank_ic(tmp2, "mom4w_pred")

    survivorship = {
        "stable_tickers":    stable_tickers,
        "new_entrants":      NEW_ENTRANTS,
        "rows_stable":       int(len(stable_panel)),
        "rows_full":         int(len(panel)),
        "folds": [
            {"year": y, "ic_xgb_best": stable_xgb[y], "ic_mom4w": stable_mom[y]}
            for y in stable_xgb
        ],
        "mean_ic_xgb_best":  float(np.mean(list(stable_xgb.values()))),
        "mean_ic_mom4w":     float(np.mean(list(stable_mom.values()))),
    }

    # ── 2. Without sector ───────────────────────────────────────────────
    param_combos = [
        dict(zip(GRID.keys(), vals)) for vals in product(*GRID.values())
    ]
    nosector_fold_ic: dict[str, dict] = {}
    nosector_mean_ic: dict[str, float] = {}
    for params in param_combos:
        folds = _per_fold_nosector(panel, params)
        nosector_fold_ic[json.dumps(params, sort_keys=True)] = folds
        nosector_mean_ic[json.dumps(params, sort_keys=True)] = float(np.mean(list(folds.values())))
        logger.info("no-sector %s mean IC %.4f", params, nosector_mean_ic[json.dumps(params, sort_keys=True)])

    best_ns_key = max(nosector_mean_ic, key=nosector_mean_ic.get)
    nosector = {
        "best_params":   json.loads(best_ns_key),
        "best_mean_ic":  nosector_mean_ic[best_ns_key],
        "best_folds":    nosector_fold_ic[best_ns_key],
        "all_mean_ic":   {k: v for k, v in sorted(nosector_mean_ic.items(),
                                                   key=lambda kv: -kv[1])},
    }

    # ── 3. Selection-bias summary ───────────────────────────────────────
    grid_ics = list(cv["grid_mean_ic"].values())
    best_fold_ic = {f["year"]: f["ic_best_xgb"] for f in cv["folds"]}
    selection = {
        "n_configs":            len(grid_ics),
        "best_mean_ic":         cv["best_mean_ic"],
        "median_mean_ic":       float(np.median(grid_ics)),
        "min_mean_ic":          float(np.min(grid_ics)),
        "max_mean_ic":          float(np.max(grid_ics)),
        "best_config":          cv["best_params"],
        "best_config_fold_ic":  best_fold_ic,
    }

    return {
        "survivorship": survivorship,
        "nosector":     nosector,
        "selection":    selection,
    }


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/raw/panel.parquet")
    ap.add_argument("--cv",    default="data/raw/xgb_cv_results.json")
    ap.add_argument("--out",   default="data/raw/xgb_robustness.json")
    args = ap.parse_args()

    out = run(args.panel, args.cv)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str))
