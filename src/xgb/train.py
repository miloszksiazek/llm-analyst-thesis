"""
Walk-forward validation of the XGBoost benchmark.

Folds: expanding window with 5 yearly test slices (2022..2026). For each fold
we evaluate a small grid of XGB hyperparameters against two baselines on the
same weeks:
    - rank-by-4w-momentum  (deterministic)
    - random ranking       (seeded, mean of 50 draws per week)

Metric: mean weekly rank IC (Spearman correlation between the predicted
ranking and the realized next-week return across the ~30 tickers in that
week). Reported as a per-fold mean and an overall mean of fold means.

Entrypoint `cross_validate(panel)` returns:
    per_fold_df   (fold-level mean IC per config)
    best_config   (dict)
    summary       (dict with baselines + best XGB + top features)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from xgb.features import FEATURE_COLS, CATEGORICAL_COL

logger = logging.getLogger(__name__)

SEED = 20261110
GRID = {
    "max_depth":     [2, 3, 4],
    "learning_rate": [0.03, 0.1],
    "n_estimators":  [200, 500],
}
FOLD_YEARS = [2022, 2023, 2024, 2025, 2026]


def _one_hot(df: pd.DataFrame, categories: list[str]) -> pd.DataFrame:
    """Deterministic one-hot for `sector`. Categories supplied so train/test share schema."""
    out = pd.DataFrame(index=df.index)
    for c in categories:
        out[f"sector_{c}"] = (df[CATEGORICAL_COL] == c).astype(np.int8)
    return out


def _build_X(df: pd.DataFrame, sectors: list[str]) -> pd.DataFrame:
    return pd.concat([df[FEATURE_COLS], _one_hot(df, sectors)], axis=1)


def _weekly_rank_ic(df: pd.DataFrame, pred_col: str, y_col: str = "target") -> float:
    """Mean of weekly Spearman(pred, y) across weeks in df."""
    ics = []
    for _, grp in df.groupby("as_of"):
        if grp[pred_col].notna().sum() < 3 or grp[y_col].notna().sum() < 3:
            continue
        rho, _ = spearmanr(grp[pred_col], grp[y_col])
        if np.isfinite(rho):
            ics.append(rho)
    return float(np.mean(ics)) if ics else float("nan")


def _random_rank_ic(df: pd.DataFrame, n_draws: int = 50, seed: int = SEED) -> float:
    rng = np.random.default_rng(seed)
    ics = []
    for _, grp in df.groupby("as_of"):
        y = grp["target"].to_numpy()
        n = len(y)
        if n < 3:
            continue
        for _ in range(n_draws):
            pred = rng.permutation(n)
            rho, _ = spearmanr(pred, y)
            if np.isfinite(rho):
                ics.append(rho)
    return float(np.mean(ics)) if ics else float("nan")


@dataclass
class FoldResult:
    year: int
    train_weeks: int
    test_weeks: int
    results: dict[tuple, float] = field(default_factory=dict)  # (params) -> mean IC
    baseline_mom4w: float = float("nan")
    baseline_random: float = float("nan")
    feature_importance: dict[str, float] = field(default_factory=dict)


def _fit_xgb(train: pd.DataFrame, params: dict, sectors: list[str]):
    import xgboost as xgb

    X = _build_X(train, sectors)
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


def _predict_xgb(model, feature_cols: list[str], df: pd.DataFrame, sectors: list[str]) -> np.ndarray:
    X = _build_X(df, sectors).reindex(columns=feature_cols, fill_value=0)
    return model.predict(X)


def cross_validate(panel: pd.DataFrame) -> dict:
    sectors = sorted(panel[CATEGORICAL_COL].dropna().unique().tolist())
    panel = panel.dropna(subset=FEATURE_COLS + ["target"]).copy()

    param_combos = [
        dict(zip(GRID.keys(), vals)) for vals in product(*GRID.values())
    ]
    logger.info("Grid size: %d combos; folds: %s", len(param_combos), FOLD_YEARS)

    folds: list[FoldResult] = []
    for year in FOLD_YEARS:
        train = panel[panel["as_of"].dt.year <  year]
        test  = panel[panel["as_of"].dt.year == year]
        if len(train) == 0 or len(test) == 0:
            logger.warning("Fold %d empty – skipping", year)
            continue

        fr = FoldResult(
            year=year,
            train_weeks=train["as_of"].nunique(),
            test_weeks=test["as_of"].nunique(),
        )
        # Baselines
        tmp = test.copy()
        tmp["mom4w_pred"] = tmp["mom_4w"]
        fr.baseline_mom4w  = _weekly_rank_ic(tmp, "mom4w_pred")
        fr.baseline_random = _random_rank_ic(tmp, seed=SEED + year)

        # XGB grid
        for params in param_combos:
            model, cols = _fit_xgb(train, params, sectors)
            pred = _predict_xgb(model, cols, test, sectors)
            tmp2 = test.copy()
            tmp2["pred"] = pred
            ic = _weekly_rank_ic(tmp2, "pred")
            fr.results[tuple(sorted(params.items()))] = ic
            logger.info(
                "fold=%d params=%s IC=%.4f  mom4w=%.4f  random=%.4f",
                year, params, ic, fr.baseline_mom4w, fr.baseline_random,
            )
        folds.append(fr)

    # Mean IC per config across folds
    mean_ic = {}
    for key in folds[0].results:
        mean_ic[key] = float(np.mean([f.results[key] for f in folds]))
    best_key   = max(mean_ic, key=mean_ic.get)
    best_params = dict(best_key)
    best_mean  = mean_ic[best_key]

    # Fit best on ALL training folds combined (same as the final fold's training window)
    # to produce feature importance for the report.
    full_train = panel[panel["as_of"].dt.year <  FOLD_YEARS[-1]]
    model, cols = _fit_xgb(full_train, best_params, sectors)
    fi_raw = dict(zip(cols, model.feature_importances_.tolist()))
    # Aggregate sector_* into a single "sector" entry
    fi: dict[str, float] = {}
    sector_total = 0.0
    for k, v in fi_raw.items():
        if k.startswith("sector_"):
            sector_total += v
        else:
            fi[k] = v
    fi["sector"] = sector_total
    top10 = dict(sorted(fi.items(), key=lambda kv: kv[1], reverse=True)[:10])

    out = {
        "best_params":    best_params,
        "best_mean_ic":   best_mean,
        "grid_mean_ic":   {str(dict(k)): v for k, v in mean_ic.items()},
        "folds":          [
            {
                "year":            f.year,
                "train_weeks":     f.train_weeks,
                "test_weeks":      f.test_weeks,
                "ic_best_xgb":     f.results[best_key],
                "ic_baseline_mom": f.baseline_mom4w,
                "ic_baseline_rnd": f.baseline_random,
            }
            for f in folds
        ],
        "baseline_mom4w_mean":  float(np.mean([f.baseline_mom4w  for f in folds])),
        "baseline_random_mean": float(np.mean([f.baseline_random for f in folds])),
        "feature_importance_top10": top10,
        "sectors":        sectors,
        "feature_cols":   cols,
    }
    return out


if __name__ == "__main__":
    import argparse, json

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/raw/panel.parquet")
    ap.add_argument("--out",   default="data/raw/xgb_cv_results.json")
    args = ap.parse_args()

    panel = pd.read_parquet(args.panel)
    panel["as_of"] = pd.to_datetime(panel["as_of"])
    summary = cross_validate(panel)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
