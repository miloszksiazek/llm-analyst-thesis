"""
Download daily OHLCV from Yahoo Finance for the 30-stock universe + DIA,
covering ~11 years. Cache to data/raw/history.parquet.

The file stores a MultiIndex column DataFrame with levels ("field", "ticker")
where field in {"Open", "High", "Low", "Close", "Volume"}.
Close is auto_adjust=False (raw close); we apply spin-off cleaning at feature
time (src/data/adjust_prices.py).
"""

from __future__ import annotations

import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import yaml
import yfinance as yf

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parents[2]
_UNIVERSE = _ROOT / "config" / "universe.yaml"
_CACHE = _ROOT / "data" / "raw" / "history.parquet"


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
    logger.info("Downloading %d tickers from %s to %s", len(tickers), start, end)
    df = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        group_by="column",
    )
    df = df.dropna(how="all")
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"

    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_CACHE)
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
    print(f"Columns (levels): {df.columns.names} — {len(df.columns)} columns")
