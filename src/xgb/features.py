"""
XGBoost features for the weekly cross-section.

All price-derived features are computed from the spin-off-cleaned index
(src/data/adjust_prices.build_clean_index). Volume features use raw Yahoo
volume. Daily-range features use raw High/Low/Close; event-day distortions
are small and leave the ordering intact.

Entry point: `build_asof(bundle, as_of)` returns a DataFrame indexed by
ticker with one row per universe stock and all features for that as_of.
`latest_date_used(feature_row)` gives the latest trading date touched by
any feature value — used for the leakage assertion.

Features (all numeric unless noted):
    mom_1w, mom_4w, mom_12w, mom_26w     cleaned-index returns
    vol_20d                               std of daily cleaned returns, 20 sessions
    rel_volume_5_60                       mean vol last 5 / mean vol last 60
    range_20d                             mean of (High-Low)/Close over last 20 days
    dist_ma50, dist_ma200                 clean_idx/MA - 1
    ret_vs_dia_4w                         mom_4w(stock) - mom_4w(DIA)
    sector                                categorical (str; dummified in dataset.py)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from data.adjust_prices import build_clean_index

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parents[2]
_UNIVERSE = _ROOT / "config" / "universe.yaml"

FEATURE_COLS = [
    "mom_1w", "mom_4w", "mom_12w", "mom_26w",
    "vol_20d", "rel_volume_5_60", "range_20d",
    "dist_ma50", "dist_ma200", "ret_vs_dia_4w",
]
CATEGORICAL_COL = "sector"


@dataclass
class HistoryBundle:
    """Pre-processed history used for feature building."""
    close: pd.DataFrame
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    volume: pd.DataFrame
    clean_idx: pd.DataFrame
    sectors: dict[str, str]

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.close.index


def load_sectors() -> dict[str, str]:
    with open(_UNIVERSE) as fh:
        u = yaml.safe_load(fh)
    return {s["ticker"]: s["sector"] for s in u["universe"]}


def load_bundle(parquet_path: Path | str | None = None) -> HistoryBundle:
    """Load the cached Yahoo history and pre-compute the cleaned index."""
    if parquet_path is None:
        parquet_path = _ROOT / "data" / "raw" / "history.parquet"
    df = pd.read_parquet(parquet_path)
    df.index = pd.to_datetime(df.index)

    close  = df["Close"].copy()
    open_  = df["Open"].copy()
    high   = df["High"].copy()
    low    = df["Low"].copy()
    volume = df["Volume"].copy()

    if "DIA" not in close.columns:
        raise RuntimeError("history parquet must include DIA")

    clean_idx = build_clean_index(close)

    return HistoryBundle(
        close=close, open=open_, high=high, low=low, volume=volume,
        clean_idx=clean_idx, sectors=load_sectors(),
    )


def _pos(index: pd.DatetimeIndex, as_of: pd.Timestamp) -> int:
    """Index position of as_of; raises if as_of is not a trading date."""
    if as_of not in index:
        raise ValueError(f"as_of {as_of.date()} is not a trading date in history")
    return index.get_loc(as_of)


def build_asof(bundle: HistoryBundle, as_of: pd.Timestamp) -> pd.DataFrame:
    """
    Build the feature cross-section at `as_of` (a trading date, typically Friday).

    Uses only data on or before `as_of`. Returns rows for the 30 universe
    tickers (DIA excluded) that have enough history for mom_26w.
    """
    as_of = pd.Timestamp(as_of)
    idx   = bundle.clean_idx.index
    p     = _pos(idx, as_of)

    if p < 200:
        raise ValueError(
            f"as_of {as_of.date()} has only {p+1} prior sessions – "
            f"need at least 200 for dist_ma200"
        )

    tickers = [t for t in bundle.sectors if t in bundle.clean_idx.columns]
    rows: list[dict] = []

    ci = bundle.clean_idx.iloc[: p + 1]
    rets = ci.pct_change()
    hlc_range = (bundle.high.iloc[: p + 1] - bundle.low.iloc[: p + 1]) / bundle.close.iloc[: p + 1]

    dia_mom_4w = ci["DIA"].iloc[-1] / ci["DIA"].iloc[-21] - 1 if p >= 20 else None

    for t in tickers:
        if t not in ci.columns:
            continue
        s_idx = ci[t]
        if s_idx.iloc[-1] != s_idx.iloc[-1]:  # NaN
            continue
        last = s_idx.iloc[-1]

        def mom(n: int) -> float | None:
            if p < n:
                return None
            base = s_idx.iloc[-(n + 1)]
            if pd.isna(base) or base == 0:
                return None
            return float(last / base - 1)

        mom_1w  = mom(5)
        mom_4w  = mom(20)
        mom_12w = mom(60)
        mom_26w = mom(130)

        # 20-day volatility of daily returns
        r20 = rets[t].iloc[-20:]
        vol_20d = float(r20.std()) if r20.notna().sum() >= 15 else None

        # Relative volume
        v = bundle.volume[t].iloc[: p + 1]
        v5  = v.iloc[-5:].mean()
        v60 = v.iloc[-60:].mean()
        rel_vol = float(v5 / v60) if v60 and pd.notna(v60) and v60 > 0 else None

        # Daily range (20d average)
        rng = hlc_range[t].iloc[-20:]
        range_20d = float(rng.mean()) if rng.notna().sum() >= 15 else None

        # Moving-average distance (clean index / clean-index MA - 1)
        ma50  = s_idx.iloc[-50:].mean()
        ma200 = s_idx.iloc[-200:].mean()
        dist_ma50  = float(last / ma50  - 1) if ma50  and ma50  > 0 else None
        dist_ma200 = float(last / ma200 - 1) if ma200 and ma200 > 0 else None

        ret_vs_dia_4w = None
        if mom_4w is not None and dia_mom_4w is not None:
            ret_vs_dia_4w = float(mom_4w - dia_mom_4w)

        rows.append({
            "ticker":          t,
            "as_of":           as_of,
            "mom_1w":          mom_1w,
            "mom_4w":          mom_4w,
            "mom_12w":         mom_12w,
            "mom_26w":         mom_26w,
            "vol_20d":         vol_20d,
            "rel_volume_5_60": rel_vol,
            "range_20d":       range_20d,
            "dist_ma50":       dist_ma50,
            "dist_ma200":      dist_ma200,
            "ret_vs_dia_4w":   ret_vs_dia_4w,
            "sector":          bundle.sectors[t],
        })

    out = pd.DataFrame(rows).set_index("ticker")
    return out


def latest_feature_date(bundle: HistoryBundle, as_of: pd.Timestamp) -> pd.Timestamp:
    """
    The most recent trading date any feature at `as_of` could have touched.

    For our feature set that's exactly `as_of`. Returned as a Timestamp so the
    leakage test can compare it to the target window.
    """
    as_of = pd.Timestamp(as_of)
    idx = bundle.clean_idx.index
    _pos(idx, as_of)  # raises if not a trading date
    return as_of
