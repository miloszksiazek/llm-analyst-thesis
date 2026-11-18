"""
Cash dividend history for the universe + DIA.

Weekly total returns for the portfolio ledger need the per-ticker schedule of
cash dividends (ex-date, amount). Yahoo serves this via yfinance's per-ticker
`.dividends` series. We cache the result to `data/raw/dividends.parquet` so
the Sunday pipeline does not re-hit Yahoo for history that never changes.

Output shape: DataFrame indexed by ex-date (DatetimeIndex, normalized), columns
are tickers (universe + DIA), values are cash dividend amount in USD on that
date (0 when there is no dividend). Only regular and special cash dividends
are included; stock splits are handled elsewhere via `Open`/`Close` already
split-adjusted by Yahoo.

CLI:
    python src/data/dividends.py [--force]
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import yaml
import yfinance as yf

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parents[2]
_UNIVERSE = _ROOT / "config" / "universe.yaml"
_CACHE = _ROOT / "data" / "raw" / "dividends.parquet"


def _tickers() -> list[str]:
    with open(_UNIVERSE) as fh:
        u = yaml.safe_load(fh)
    return [s["ticker"] for s in u["universe"]] + ["DIA"]


def fetch(force: bool = False) -> pd.DataFrame:
    """Download per-ticker cash dividend history and cache a wide DataFrame."""
    if _CACHE.exists() and not force:
        return load()

    out: dict[str, pd.Series] = {}
    for t in _tickers():
        try:
            s = yf.Ticker(t).dividends
        except Exception as exc:
            logger.warning("Dividends for %s failed: %s", t, exc)
            continue
        if s is None or len(s) == 0:
            continue
        s = s.copy()
        s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
        out[t] = s.groupby(level=0).sum()

    if not out:
        raise RuntimeError("No dividend data retrieved")

    df = pd.DataFrame(out).sort_index().fillna(0.0)
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_CACHE)
    logger.info("Saved %d dividend rows to %s", len(df), _CACHE)
    return df


def load() -> pd.DataFrame:
    """Load the cached dividend table, or return an empty frame."""
    if not _CACHE.exists():
        return pd.DataFrame()
    df = pd.read_parquet(_CACHE)
    df.index = pd.to_datetime(df.index)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = fetch(force="--force" in sys.argv)
    print(f"Shape: {df.shape}")
    print(f"Date range: {df.index.min().date()} .. {df.index.max().date()}")
    print(f"Tickers: {len(df.columns)}")
