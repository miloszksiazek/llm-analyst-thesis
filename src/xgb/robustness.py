"""
Robustness checks for the XGBoost walk-forward benchmark (no new data).

1. Survivorship (descriptive only): the 30-ticker universe is the 2026-11-04
   DJIA snapshot. All 30 are ex-post survivors; names dropped from the index
   since 2015 are missing from the data entirely. We report scoring on the
   subset of current tickers that were in the index for the whole validation
   window (STABLE_SUBSET) as a descriptive check — this does NOT remove
   survivorship bias.
2. With-sector comparison: rerun the same walk-forward grid using the sector
   one-hot columns in addition to FEATURE_COLS. The chosen configuration is
   no-sector (docs/zmiany_do_planu.md 2026-11-10); the with-sector numbers
   are kept here for reference.
3. Selection bias: median mean-IC across the 12 grid configs next to the
   best, and the best config's IC on each fold.

Output: data/raw/xgb_robustness.json.
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
    FOLD_YEARS, GRID, SEED, _fit_xgb, _predict_xgb, _weekly_rank_ic,
)

logger = logging.getLogger(__name__)
_ROOT = Path(__file__).parents[2]

# Current universe tickers that joined DJIA after 2015-07-01, verified against
# the Wikipedia article "Historical components of the Dow Jones Industrial
# Average" (fetched 2026-11-10).
NEW_ENTRANTS = ["AMGN", "CRM", "HON", "AMZN", "NVDA", "SHW", "GOOGL"]

# Stocks REMOVED from DJIA since 2015-07-01 (and therefore absent from our
# universe – they are the other side of the ex-post selection).
REMOVED_SINCE_2015 = [
    ("DD",   "2017-09-01", "DuPont (merged into DowDuPont)"),
    ("GE",   "2018-06-26", "General Electric"),
    ("DWDP", "2019-04-02", "DowDuPont (replaced by Dow Inc.)"),
    ("UTX",  "2020-04-06", "United Technologies (merged into RTX)"),
    ("RTX",  "2020-08-31", "Raytheon Technologies"),
    ("XOM",  "2020-08-31", "ExxonMobil"),
    ("PFE",  "2020-08-31", "Pfizer"),
    ("WBA",  "2024-02-26", "Walgreens Boots Alliance"),
    ("INTC", "2024-11-08", "Intel"),
    ("DOW",  "2024-11-08", "Dow Inc."),
    ("VZ",   "2026-06-29", "Verizon Communications"),
]


# ── With-sector helpers (kept for the comparison section only) ─────────────

def _one_hot(df: pd.DataFrame, categories: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in categories:
        out[f"sector_{c}"] = (df[CATEGORICAL_COL] == c).astype(np.int8)
    return out


def _fit_xgb_with_sector(train: pd.DataFrame, params: dict, sectors: list[str]):
    import xgboost as xgb

    X = pd.concat([train[FEATURE_COLS], _one_hot(train, sectors)], axis=1)
    y = train["target"].to_numpy()
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=SEED,
        n_jobs=1,
        **params,
    )
    model.fit(X, y)
    return model, X.columns.tolist()


def _per_fold_with_sector(panel: pd.DataFrame, params: dict, sectors: list[str]) -> dict:
    out = {}
    for year in FOLD_YEARS:
        train = panel[panel["as_of"].dt.year <  year]
        test  = panel[panel["as_of"].dt.year == year]
        if len(train) == 0 or len(test) == 0:
            continue
        model, cols = _fit_xgb_with_sector(train, params, sectors)
        X_test = pd.concat([test[FEATURE_COLS], _one_hot(test, sectors)], axis=1)
        X_test = X_test.reindex(columns=cols, fill_value=0)
        tmp = test.copy()
        tmp["pred"] = model.predict(X_test)
        out[year] = _weekly_rank_ic(tmp, "pred")
    return out


# ── Main ────────────────────────────────────────────────────────────────────

def run(panel_path: Path | str, cv_path: Path | str) -> dict:
    panel = pd.read_parquet(panel_path)
    panel["as_of"] = pd.to_datetime(panel["as_of"])
    panel = panel.dropna(subset=FEATURE_COLS + ["target"]).copy()
    cv = json.loads(Path(cv_path).read_text())
    best = cv["best_params"]

    # ── 1. Survivorship (descriptive stable-subset) ─────────────────────
    stable_tickers = sorted(set(panel["ticker"]) - set(NEW_ENTRANTS))
    stable_panel   = panel[panel["ticker"].isin(stable_tickers)].copy()
    logger.info("Stable subset: %d tickers, %d rows",
                len(stable_tickers), len(stable_panel))

    stable_xgb: dict[int, float] = {}
    stable_mom: dict[int, float] = {}
    for year in FOLD_YEARS:
        train = panel[panel["as_of"].dt.year <  year]
        test  = stable_panel[stable_panel["as_of"].dt.year == year]
        if len(train) == 0 or len(test) == 0:
            continue
        model = _fit_xgb(train, best)
        tmp = test.copy()
        tmp["pred"] = _predict_xgb(model, test)
        stable_xgb[year] = _weekly_rank_ic(tmp, "pred")
        tmp2 = test.copy()
        tmp2["mom4w_pred"] = tmp2["mom_4w"]
        stable_mom[year]  = _weekly_rank_ic(tmp2, "mom4w_pred")

    survivorship = {
        "stable_tickers":    stable_tickers,
        "new_entrants":      NEW_ENTRANTS,
        "removed_since_2015": [
            {"ticker": t, "removal_date": d, "name": n}
            for t, d, n in REMOVED_SINCE_2015
        ],
        "rows_stable":       int(len(stable_panel)),
        "rows_full":         int(len(panel)),
        "folds": [
            {"year": y, "ic_xgb_best": stable_xgb[y], "ic_mom4w": stable_mom[y]}
            for y in stable_xgb
        ],
        "mean_ic_xgb_best":  float(np.mean(list(stable_xgb.values()))),
        "mean_ic_mom4w":     float(np.mean(list(stable_mom.values()))),
    }

    # ── 2. With-sector comparison ────────────────────────────────────────
    sectors = sorted(panel[CATEGORICAL_COL].dropna().unique().tolist())
    param_combos = [
        dict(zip(GRID.keys(), vals)) for vals in product(*GRID.values())
    ]
    ws_fold_ic: dict[str, dict] = {}
    ws_mean_ic: dict[str, float] = {}
    for params in param_combos:
        key = json.dumps(params, sort_keys=True)
        folds = _per_fold_with_sector(panel, params, sectors)
        ws_fold_ic[key] = folds
        ws_mean_ic[key] = float(np.mean(list(folds.values())))
        logger.info("with-sector %s mean IC %.4f", params, ws_mean_ic[key])

    best_ws_key = max(ws_mean_ic, key=ws_mean_ic.get)
    with_sector = {
        "best_params":   json.loads(best_ws_key),
        "best_mean_ic":  ws_mean_ic[best_ws_key],
        "best_folds":    ws_fold_ic[best_ws_key],
        "all_mean_ic":   dict(sorted(ws_mean_ic.items(), key=lambda kv: -kv[1])),
    }

    # ── 3. Selection-bias summary (on the chosen no-sector grid) ─────────
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
        "with_sector":  with_sector,
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
