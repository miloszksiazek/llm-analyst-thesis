"""
Price-history download helpers used by both the weekly package and the
XGBoost history cache.

Tries Yahoo Finance first; on failure (empty frame or exception) falls back
to Alpaca's StockHistoricalDataClient. Returns the Close frame and a label
("yahoo" / "alpaca") so the caller can log the source used.

For the fallback, ALPACA_KEY and ALPACA_SECRET (data-only; the Sunday
workflow uses read-only access) must be in the environment. If they are
absent and Yahoo also fails, the caller gets a clear RuntimeError.
"""

from __future__ import annotations

import logging
import os
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parents[2]


def _yahoo_close(tickers: list[str], start: date, end: date) -> pd.DataFrame:
    df = yf.download(
        tickers, start=str(start), end=str(end),
        auto_adjust=False, progress=False,
    )
    if df.empty or "Close" not in df.columns.get_level_values(0):
        raise RuntimeError("Yahoo returned an empty frame")
    close = df["Close"].copy()
    close.index = pd.to_datetime(close.index)
    close = close.dropna(how="all")
    if close.empty:
        raise RuntimeError("Yahoo price frame has no non-null rows")
    return close


def _alpaca_close(tickers: list[str], start: date, end: date) -> pd.DataFrame:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests   import StockBarsRequest
    from alpaca.data.timeframe  import TimeFrame

    load_dotenv(_ROOT / ".env")
    key    = os.getenv("ALPACA_KEY")    or os.getenv("ALPACA_KEY_DATA")
    secret = os.getenv("ALPACA_SECRET") or os.getenv("ALPACA_SECRET_DATA")
    if not (key and secret):
        raise RuntimeError(
            "Alpaca price fallback needs ALPACA_KEY and ALPACA_SECRET in env"
        )

    client = StockHistoricalDataClient(api_key=key, secret_key=secret)
    req = StockBarsRequest(
        symbol_or_symbols=tickers, timeframe=TimeFrame.Day,
        start=datetime.combine(start, datetime.min.time()),
        end=datetime.combine(end,   datetime.min.time()),
        feed="iex",
    )
    bars = client.get_stock_bars(req).df
    if bars.empty:
        raise RuntimeError("Alpaca returned empty bars")
    close = (
        bars.reset_index()
            .pivot(index="timestamp", columns="symbol", values="close")
    )
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    close = close.dropna(how="all")
    return close


def download_close(tickers: list[str], start: date, end: date) -> tuple[pd.DataFrame, str]:
    """Return (close frame, source) for [start, end) daily closes.

    source is 'yahoo' or 'alpaca'. Raises RuntimeError if both fail.
    """
    try:
        return _yahoo_close(tickers, start, end), "yahoo"
    except Exception as yahoo_exc:
        logger.warning("Yahoo close failed (%s); trying Alpaca", yahoo_exc)
        return _alpaca_close(tickers, start, end), "alpaca"


def download_ohlcv_yahoo(tickers: list[str], start: date, end: date) -> pd.DataFrame:
    """Full OHLCV frame from Yahoo (used by xgb/fetch_history as the primary)."""
    df = yf.download(
        tickers, start=str(start), end=str(end),
        auto_adjust=False, progress=False, group_by="column",
    )
    if df.empty:
        raise RuntimeError("Yahoo returned an empty OHLCV frame")
    df = df.dropna(how="all")
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    return df


def download_ohlcv_alpaca(tickers: list[str], start: date, end: date) -> pd.DataFrame:
    """Fallback OHLCV frame from Alpaca formatted like Yahoo's multi-index columns.

    Returns a DataFrame with columns MultiIndex(level0 ∈ {Open, High, Low, Close,
    Volume}, level1 = ticker), matching what download_ohlcv_yahoo returns. Only
    the fields XGB features actually need.
    """
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests   import StockBarsRequest
    from alpaca.data.timeframe  import TimeFrame

    load_dotenv(_ROOT / ".env")
    key    = os.getenv("ALPACA_KEY")    or os.getenv("ALPACA_KEY_DATA")
    secret = os.getenv("ALPACA_SECRET") or os.getenv("ALPACA_SECRET_DATA")
    if not (key and secret):
        raise RuntimeError(
            "Alpaca OHLCV fallback needs ALPACA_KEY and ALPACA_SECRET in env"
        )

    client = StockHistoricalDataClient(api_key=key, secret_key=secret)
    req = StockBarsRequest(
        symbol_or_symbols=tickers, timeframe=TimeFrame.Day,
        start=datetime.combine(start, datetime.min.time()),
        end=datetime.combine(end,   datetime.min.time()),
        feed="iex",
    )
    bars = client.get_stock_bars(req).df
    if bars.empty:
        raise RuntimeError("Alpaca returned empty bars")
    bars = bars.reset_index()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"]).dt.tz_localize(None).dt.normalize()
    pieces = {}
    for col_in, col_out in [
        ("open", "Open"), ("high", "High"), ("low", "Low"),
        ("close", "Close"), ("volume", "Volume"),
    ]:
        pieces[col_out] = bars.pivot(index="timestamp", columns="symbol", values=col_in)
    out = pd.concat(pieces, axis=1)
    out.columns.names = ["Price", "Ticker"]
    out.index.name = "date"
    return out


def download_ohlcv(tickers: list[str], start: date, end: date) -> tuple[pd.DataFrame, str]:
    try:
        return download_ohlcv_yahoo(tickers, start, end), "yahoo"
    except Exception as yahoo_exc:
        logger.warning("Yahoo OHLCV failed (%s); trying Alpaca", yahoo_exc)
        return download_ohlcv_alpaca(tickers, start, end), "alpaca"
