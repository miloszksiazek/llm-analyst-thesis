"""
Weekly XGBoost ranking: load a saved model and produce a 1–30 ranking at `as_of`.

The model is trained with `train_final` on the 10 numeric features in
`FEATURE_COLS` (no sector; see docs/zmiany_do_planu.md 2026-11-10). Metadata
is stored next to the model binary as `<name>.meta.json`.

CLI:
    python src/xgb/predict.py --as-of 2026-11-06 [--model-path ... | --train-first]
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from xgb.features import FEATURE_COLS, build_asof, load_bundle

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parents[2]


def _meta_path(model_path: Path) -> Path:
    return model_path.with_name(model_path.stem + ".meta.json")


def train_final(
    panel: pd.DataFrame,
    params: dict,
    model_path: Path | str,
) -> dict:
    """
    Train on the full panel with `params` and save model + metadata.
    """
    import xgboost as xgb
    from xgb.train import SEED

    panel = panel.dropna(subset=FEATURE_COLS + ["target"]).copy()
    X = panel[FEATURE_COLS]
    y = panel["target"].to_numpy()

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=SEED,
        n_jobs=1,
        **params,
    )
    model.fit(X, y)

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))

    meta = {
        "params":          params,
        "feature_cols":    FEATURE_COLS,
        "train_rows":      int(len(panel)),
        "train_weeks":     int(panel["as_of"].nunique()),
        "trained_through": str(panel["as_of"].max().date()),
    }
    _meta_path(model_path).write_text(json.dumps(meta, indent=2))
    return meta


def load_model(model_path: Path | str):
    import xgboost as xgb

    model_path = Path(model_path)
    meta = json.loads(_meta_path(model_path).read_text())
    model = xgb.XGBRegressor()
    model.load_model(str(model_path))
    return model, meta


def rank(as_of: pd.Timestamp, model, meta: dict, bundle=None) -> pd.DataFrame:
    """Return a 1-based ranking of universe tickers for `as_of`."""
    if bundle is None:
        bundle = load_bundle()
    as_of = pd.Timestamp(as_of)
    feats = build_asof(bundle, as_of)
    feats = feats.dropna(subset=FEATURE_COLS)
    X = feats[meta["feature_cols"]]
    pred = model.predict(X)
    out = (
        pd.DataFrame({"ticker": feats.index, "score": pred})
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )
    out.index = np.arange(1, len(out) + 1)
    out.index.name = "rank"
    return out


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of",      required=True, help="YYYY-MM-DD trading date")
    ap.add_argument("--model-path", default="data/raw/xgb_latest.ubj")
    ap.add_argument("--train-first", action="store_true",
                    help="Train on data/raw/panel.parquet using params from "
                         "data/raw/xgb_cv_results.json before predicting.")
    args = ap.parse_args()

    model_path = Path(args.model_path)
    if args.train_first or not model_path.exists():
        cv = json.loads((_ROOT / "data" / "raw" / "xgb_cv_results.json").read_text())
        panel = pd.read_parquet(_ROOT / "data" / "raw" / "panel.parquet")
        panel["as_of"] = pd.to_datetime(panel["as_of"])
        logger.info("Training final model with %s on %d rows", cv["best_params"], len(panel))
        train_final(panel, cv["best_params"], model_path)

    model, meta = load_model(model_path)
    ranking = rank(pd.Timestamp(args.as_of), model, meta)
    print(f"Ranking for as_of={args.as_of} (model trained through {meta['trained_through']})")
    print(ranking.to_string())
