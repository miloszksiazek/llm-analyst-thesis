"""
Build the weekly agent data package.

build_package(run_date, out_dir=None) writes two files:
    <out_dir>/<run_date>/package.json   – canonical UTF-8 JSON
    <out_dir>/<run_date>/package.sha256 – SHA-256 hex digest

out_dir defaults to data/weekly/ (production).
Tests and previews should pass out_dir=data/scratch/.

verify_package(path) re-reads package.json and checks the hash.

Rounding convention:
    prices      → 2 dp  (last_close, DIA close)
    percentages → 3 dp  (change_1w_pct etc.; stored as percentage, e.g. 1.234 means 1.234%)
    ratios      → 2 dp  (pe_trailing, dividend_yield)
    market_cap  → int   (no rounding)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf
import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

_ROOT     = Path(__file__).parents[2]
_UNIVERSE = _ROOT / "config" / "universe.yaml"


# ── Universe ──────────────────────────────────────────────────────────────────

def _load_universe() -> tuple[list[str], dict, dict]:
    with open(_UNIVERSE) as fh:
        u = yaml.safe_load(fh)
    stocks   = u["universe"]
    tickers  = [s["ticker"] for s in stocks]
    meta     = u.get("metadata", {})
    info_map = {s["ticker"]: s for s in stocks}
    return tickers, meta, info_map


# ── Trading calendar ──────────────────────────────────────────────────────────

def _alpaca_last_session(before: date) -> date:
    """
    Return the last US trading session date strictly before `before`,
    according to the Alpaca trading calendar.
    """
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetCalendarRequest

    load_dotenv()
    client = TradingClient(
        api_key=os.getenv("ALPACA_KEY_XGB"),
        secret_key=os.getenv("ALPACA_SECRET_XGB"),
        paper=True,
    )
    start = before - timedelta(days=10)
    end   = before - timedelta(days=1)
    cal   = client.get_calendar(GetCalendarRequest(start=start, end=end))
    dates = []
    for c in cal:
        d = c.date
        if hasattr(d, "date"):
            d = d.date()
        dates.append(d)
    if not dates:
        raise RuntimeError(f"No Alpaca trading sessions found before {before}")
    return max(dates)


# ── Price helpers ─────────────────────────────────────────────────────────────

def _pct(clean_idx: pd.DataFrame, ticker: str, pos: int, n: int) -> float | None:
    """% change over n sessions ending at pos in the cleaned price index."""
    if pos < n or ticker not in clean_idx.columns:
        return None
    p1 = float(clean_idx.iloc[pos - n][ticker])
    p2 = float(clean_idx.iloc[pos][ticker])
    if p1 == 0:
        return None
    return round((p2 / p1 - 1) * 100, 3)


def _fetch_fundamentals(tickers: list[str]) -> dict[str, dict]:
    """Return {ticker: {pe_trailing, dividend_yield, market_cap}} from Yahoo."""
    result = {}
    for tkr in tickers:
        try:
            info = yf.Ticker(tkr).info
            pe   = info.get("trailingPE")
            dy   = info.get("dividendYield")   # decimal in Yahoo; convert to %
            mc   = info.get("marketCap")
            result[tkr] = {
                "pe_trailing":    round(float(pe), 2)       if pe is not None else None,
                "dividend_yield": round(float(dy) * 100, 3) if dy is not None else None,
                "market_cap":     int(mc)                   if mc is not None else None,
            }
        except Exception as exc:
            logger.warning("%s: fundamentals failed – %s", tkr, exc)
            result[tkr] = {"pe_trailing": None, "dividend_yield": None, "market_cap": None}
    return result


# ── Serialization ─────────────────────────────────────────────────────────────

def _canonical_json(obj: object) -> str:
    """Sorted keys, stripped trailing whitespace, single trailing newline."""
    raw   = json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=2)
    lines = [line.rstrip() for line in raw.splitlines()]
    return "\n".join(lines) + "\n"


# ── Main build ────────────────────────────────────────────────────────────────

def build_package(run_date: date, out_dir: Path | None = None) -> Path:
    """
    Build the weekly data package for run_date.

    Parameters
    ----------
    run_date : date
        The Sunday the pipeline runs. as_of is set to the last US trading
        session before run_date per the Alpaca calendar. Raises RuntimeError
        if that session's close is not yet available in the price data.
    out_dir : Path | None
        Directory where <run_date>/package.json is written.
        Defaults to data/weekly/ (production).
        Pass data/scratch/ for tests and previews.

    Returns
    -------
    Path to the written package.json.
    """
    from data.adjust_prices import build_clean_index
    from data.earnings      import fetch_earnings
    from data.macro         import fetch_macro

    if out_dir is None:
        out_dir = _ROOT / "data" / "weekly"

    tickers, universe_meta, info_map = _load_universe()
    all_tickers = tickers + ["DIA"]

    # ── Strict as_of from Alpaca calendar ────────────────────────────────
    as_of = _alpaca_last_session(run_date)
    logger.info("run_date=%s  as_of=%s (%s)", run_date, as_of, as_of.strftime("%A"))

    # ── Download prices (need ≥63 sessions for 1-quarter change) ─────────
    start = run_date - timedelta(days=140)
    logger.info("Downloading prices %s … %s (exclusive)", start, run_date)
    prices = yf.download(
        all_tickers,
        start=str(start),
        end=str(run_date),    # exclusive: last row ≤ run_date − 1 trading day
        auto_adjust=False,
        progress=False,
    )["Close"]

    prices.index = pd.to_datetime(prices.index)
    prices = prices.dropna(how="all")

    # ── Strict availability check ─────────────────────────────────────────
    as_of_ts = pd.Timestamp(as_of)
    if as_of_ts not in prices.index:
        raise RuntimeError(f"close for {as_of} not available in price data")
    as_of_pos = prices.index.get_loc(as_of_ts)
    logger.info("as_of confirmed at index position %d / %d", as_of_pos, len(prices) - 1)

    # ── Cleaned price index ───────────────────────────────────────────────
    clean_idx = build_clean_index(prices.copy())

    # ── Fundamental data ──────────────────────────────────────────────────
    logger.info("Fetching fundamentals for %d tickers …", len(tickers))
    fundamentals = _fetch_fundamentals(tickers)

    retrieved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # ── DIA market summary ────────────────────────────────────────────────
    market = {
        "ticker":        "DIA",
        "last_close":    round(float(prices.iloc[as_of_pos]["DIA"]), 2),
        "change_1w_pct": _pct(clean_idx, "DIA", as_of_pos, 5),
        "change_1m_pct": _pct(clean_idx, "DIA", as_of_pos, 21),
        "change_1q_pct": _pct(clean_idx, "DIA", as_of_pos, 63),
    }

    # ── 30-stock table (sorted by ticker) ─────────────────────────────────
    stocks = []
    row = prices.iloc[as_of_pos]
    for tkr in sorted(tickers):
        raw_close  = row.get(tkr)
        last_close = round(float(raw_close), 2) if (raw_close is not None and pd.notna(raw_close)) else None
        f = fundamentals.get(tkr, {})
        stocks.append({
            "ticker":         tkr,
            "name":           info_map[tkr]["company"],
            "sector":         info_map[tkr]["sector"],
            "last_close":     last_close,
            "change_1w_pct":  _pct(clean_idx, tkr, as_of_pos, 5),
            "change_1m_pct":  _pct(clean_idx, tkr, as_of_pos, 21),
            "change_1q_pct":  _pct(clean_idx, tkr, as_of_pos, 63),
            "pe_trailing":    f.get("pe_trailing"),
            "dividend_yield": f.get("dividend_yield"),
            "market_cap":     f.get("market_cap"),
        })

    # ── Earnings & macro ──────────────────────────────────────────────────
    earnings_entries = fetch_earnings(run_date=run_date)
    macro_entries    = fetch_macro()

    # ── Assemble ──────────────────────────────────────────────────────────
    package = {
        "meta": {
            "run_date":          str(run_date),
            "as_of":             str(as_of),
            "universe_version":  universe_meta.get("status", "unknown"),
            "universe_tickers":  len(tickers),
            "sources":           [
                "Yahoo Finance (prices, fundamentals, earnings)",
                "FRED (macro)",
            ],
            "retrieved_at": retrieved_at,
        },
        "market":   market,
        "stocks":   stocks,
        "earnings": earnings_entries,
        "macro":    macro_entries,
    }

    # ── Write canonical JSON ──────────────────────────────────────────────
    pkg_dir  = out_dir / str(run_date)
    pkg_dir.mkdir(parents=True, exist_ok=True)
    pkg_path = pkg_dir / "package.json"

    content = _canonical_json(package)
    pkg_path.write_text(content, encoding="utf-8")
    logger.info("package.json: %d bytes (~%d tokens)", len(content.encode()), len(content) // 4)

    # ── SHA-256 ───────────────────────────────────────────────────────────
    sha = hashlib.sha256(pkg_path.read_bytes()).hexdigest()
    (pkg_dir / "package.sha256").write_text(sha + "\n", encoding="utf-8")
    logger.info("SHA-256: %s", sha)

    return pkg_path


# ── Verification ──────────────────────────────────────────────────────────────

def verify_package(path: Path | str) -> bool:
    """
    Verify SHA-256 of package.json against the sibling package.sha256.
    Returns True if hashes match; raises ValueError if not.
    """
    path     = Path(path)
    sha_path = path.parent / "package.sha256"
    if not sha_path.exists():
        raise FileNotFoundError(f"Hash file not found: {sha_path}")
    expected = sha_path.read_text(encoding="utf-8").strip()
    actual   = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(
            f"Hash mismatch:\n  expected: {expected}\n  actual:   {actual}"
        )
    return True
