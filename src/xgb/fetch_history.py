"""
Download daily OHLCV for the 30-stock universe + DIA, ~11 years of history,
and cache to data/raw/history.parquet.

Uses the shared Yahoo→Alpaca fallback in src/data/prices.py so the Sunday
pipeline keeps working when Yahoo blocks GitHub runner IPs. On fallback the
Alpaca source label is written to a sidecar file so we know which run used
which source.

CLI:
    python src/xgb/fetch_history.py [--force]
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

from data.prices import download_ohlcv

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parents[2]
_UNIVERSE = _ROOT / "config" / "universe.yaml"
_CACHE = _ROOT / "data" / "raw" / "history.parquet"
_SOURCE_LOG = _ROOT / "data" / "raw" / "history.source.json"


def load_tickers() -> list[str]:
    with open(_UNIVERSE) as fh:
        u = yaml.safe_load(fh)
    return [s["ticker"] for s in u["universe"]]


def fetch(start: str = "2015-01-01", end: str | None = None, force: bool = False) -> pd.DataFrame:
    if _CACHE.exists() and not force:
        logger.info("Using cached history at %s", _CACHE)
        return pd.read_parquet(_CACHE)

    if end is None:
        end = date.today().isoformat()

    tickers = load_tickers() + ["DIA"]
    logger.info("Downloading %d tickers %s … %s", len(tickers), start, end)
    df, source = download_ohlcv(tickers, date.fromisoformat(start), date.fromisoformat(end))
    logger.info("OHLCV source: %s", source)

    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_CACHE)
    _SOURCE_LOG.write_text(json.dumps({"source": source, "end": end}, indent=2))
    logger.info("Saved %d rows to %s", len(df), _CACHE)
    return df


def load() -> pd.DataFrame:
    if not _CACHE.exists():
        raise FileNotFoundError(
            f"No cached history at {_CACHE}. Run fetch_history.py first."
        )
    return pd.read_parquet(_CACHE)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    force = "--force" in sys.argv
    df = fetch(force=force)
    print(f"Shape: {df.shape}")
    print(f"Date range: {df.index.min().date()} .. {df.index.max().date()}")
