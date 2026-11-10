"""
Build the (ticker, week) training panel for the XGBoost benchmark.

For each Monday M in the training window:
    as_of       = last trading session with date <  M      (typically Friday)
    entry_date  = first trading session with date >= M     (typically Monday)
    exit_date   = first trading session with date >= M + 7d (typically next Monday)
    target      = Open[exit_date] / Open[entry_date] - 1   (raw opens)

Rows whose [entry_date, exit_date) window crosses a spin-off ex-date are
dropped (spin-off cleaning adjusts close-based returns, not opens).

Features are computed from the cleaned index at `as_of` (features.build_asof).
No value in the row touches a date after `as_of`; the target uses only dates
strictly after `as_of`. See `leakage_check` and tests/test_xgb_leakage.py.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yaml

from xgb.features import HistoryBundle, build_asof, load_bundle

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parents[2]
_ACTIONS = _ROOT / "config" / "corporate_actions.yaml"


def _spinoff_dates() -> set[pd.Timestamp]:
    if not _ACTIONS.exists():
        return set()
    with open(_ACTIONS) as fh:
        events = yaml.safe_load(fh).get("events", []) or []
    out: set[pd.Timestamp] = set()
    for ev in events:
        for raw in ev.get("ex_dates") or []:
            out.add(pd.Timestamp(raw))
    return out


def _mondays(start: date, end: date) -> list[date]:
    out = []
    d = start
    while d.weekday() != 0:
        d += timedelta(days=1)
    while d <= end:
        out.append(d)
        d += timedelta(days=7)
    return out


def _next_session_on_or_after(idx: pd.DatetimeIndex, d: pd.Timestamp) -> pd.Timestamp | None:
    pos = idx.searchsorted(d, side="left")
    if pos >= len(idx):
        return None
    return idx[pos]


def _last_session_before(idx: pd.DatetimeIndex, d: pd.Timestamp) -> pd.Timestamp | None:
    pos = idx.searchsorted(d, side="left") - 1
    if pos < 0:
        return None
    return idx[pos]


def build_panel(
    bundle: HistoryBundle | None = None,
    start: date = date(2015, 7, 1),
    end:   date = date(2026, 10, 31),
) -> pd.DataFrame:
    """
    Build the weekly panel. Returns a DataFrame with:
        ticker, as_of, entry_date, exit_date, <features>, target

    `start` defaults to 2015-07-01 so every row has 200 prior sessions.
    `end` defaults to 2026-10-31 so the live pipeline's 2026-11 start week
    is held out (no leakage into the live test).
    """
    if bundle is None:
        bundle = load_bundle()

    idx = bundle.clean_idx.index
    ex_dates = _spinoff_dates()
    rows: list[dict] = []

    for m in _mondays(start, end):
        m_ts   = pd.Timestamp(m)
        as_of  = _last_session_before(idx, m_ts)
        entry  = _next_session_on_or_after(idx, m_ts)
        exit_  = _next_session_on_or_after(idx, m_ts + pd.Timedelta(days=7))
        if as_of is None or entry is None or exit_ is None:
            continue
        if exit_ <= entry:
            continue

        # Drop rows whose target window crosses a spin-off ex-date
        if any(entry < d <= exit_ for d in ex_dates):
            continue

        try:
            feats = build_asof(bundle, as_of)
        except ValueError:
            continue

        opens_entry = bundle.open.loc[entry]
        opens_exit  = bundle.open.loc[exit_]

        for ticker, frow in feats.iterrows():
            e = opens_entry.get(ticker)
            x = opens_exit.get(ticker)
            if pd.isna(e) or pd.isna(x) or e == 0:
                continue
            target = float(x / e - 1)
            row = {
                "ticker":     ticker,
                "as_of":      as_of,
                "entry_date": entry,
                "exit_date":  exit_,
                **{k: frow[k] for k in frow.index},
                "target":     target,
            }
            rows.append(row)

    panel = pd.DataFrame(rows)
    logger.info("Panel: %d rows across %d weeks", len(panel),
                panel["as_of"].nunique() if len(panel) else 0)
    return panel


def leakage_check(panel: pd.DataFrame) -> None:
    """
    Assert for every row:
      - as_of <  entry_date   (target window starts strictly after as_of)
      - as_of <  exit_date
      - entry_date < exit_date
    Features in this design only use data up to as_of (by construction in
    features.build_asof), so an as_of < entry_date check is sufficient.
    """
    if (panel["as_of"] >= panel["entry_date"]).any():
        bad = panel[panel["as_of"] >= panel["entry_date"]].head()
        raise AssertionError(f"Leakage: as_of >= entry_date\n{bad}")
    if (panel["as_of"] >= panel["exit_date"]).any():
        raise AssertionError("Leakage: as_of >= exit_date")
    if (panel["entry_date"] >= panel["exit_date"]).any():
        raise AssertionError("Bad window: entry_date >= exit_date")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/panel.parquet")
    ap.add_argument("--start", default="2015-07-01")
    ap.add_argument("--end",   default="2026-10-31")
    args = ap.parse_args()

    bundle = load_bundle()
    panel = build_panel(
        bundle,
        start=date.fromisoformat(args.start),
        end=date.fromisoformat(args.end),
    )
    leakage_check(panel)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out)
    print(f"Rows: {len(panel)}")
    print(f"Weeks: {panel['as_of'].nunique()}")
    print(f"Tickers: {panel['ticker'].nunique()}")
    print("\n3 random rows (seed=0):")
    print(panel.sample(3, random_state=0).to_string())
