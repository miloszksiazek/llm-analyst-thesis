"""
Build the weekly agent data package.

Output (under <out_dir>/<run_date>/):
    package.json       – canonical UTF-8 JSON, agent-visible, hashed.
                         No retrieval timestamps. No date later than as_of.
    package.sha256     – hex digest of package.json.
    package_meta.json  – retrieval timestamps, source URLs, HON-style notes.
                         Not part of the hashed package.

verify_package(path) re-reads package.json and checks the hash.

Rounding convention:
    prices          → 2 dp  (last_close, DIA close)
    percentages     → 3 dp  (change_1w_pct etc.)
    ratios          → 2 dp  (pe_trailing, dividend_yield — stored as PERCENT, e.g. 0.50 = 0.50%)
    market_cap      → int   (Yahoo's marketCap field; covers all share classes,
                             which close × sharesOutstanding misses for GOOGL/NKE)

Sanity checks that RAISE (unless noted):
    - dividend_yield  outside [0, 15]        → RuntimeError
    - pe_trailing     <= 0                   → RuntimeError (null is fine)
    - implied shares (= market_cap / close)  more than 20 % off vs previous
      week's package → RuntimeError. Skipped if no previous package exists.
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
_ACTIONS  = _ROOT / "config" / "corporate_actions.yaml"

DIVIDEND_YIELD_MAX_PCT    = 15.0
SHARES_DRIFT_TOLERANCE    = 0.20
SPINOFF_LOOKBACK_DAYS     = 365


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
    if pos < n or ticker not in clean_idx.columns:
        return None
    p1 = float(clean_idx.iloc[pos - n][ticker])
    p2 = float(clean_idx.iloc[pos][ticker])
    if p1 == 0:
        return None
    return round((p2 / p1 - 1) * 100, 3)


def _fetch_fundamentals(tickers: list[str]) -> dict[str, dict]:
    """
    Pull Yahoo's fundamentals.

    market_cap uses Yahoo's `marketCap` field (covers all share classes;
    close × sharesOutstanding understates GOOGL and NKE, which have several
    classes). sharesOutstanding is still logged into package_meta.json.
    """
    out: dict[str, dict] = {}
    for tkr in tickers:
        try:
            info = yf.Ticker(tkr).info
            pe   = info.get("trailingPE")
            dy   = info.get("dividendYield")   # yfinance 1.8 returns this as percent already
            so   = info.get("sharesOutstanding")
            mc   = info.get("marketCap")
            out[tkr] = {
                "pe_trailing":        round(float(pe), 2)  if pe is not None else None,
                "dividend_yield":     round(float(dy), 3)  if dy is not None else None,
                "shares_outstanding": int(so)              if so is not None else None,
                "market_cap":         int(mc)              if mc is not None else None,
            }
        except Exception as exc:
            logger.warning("%s: fundamentals failed – %s", tkr, exc)
            out[tkr] = {
                "pe_trailing": None, "dividend_yield": None,
                "shares_outstanding": None, "market_cap": None,
            }
    return out


# ── Spin-off valuation override ───────────────────────────────────────────────

def _recent_spinoffs(as_of: date) -> dict[str, date]:
    """
    Parents with at least one spin-off ex-date in [as_of − 365d, as_of].

    Returns {ticker: most_recent_ex_date_in_window}.
    Trailing P/E and trailing market cap are not comparable for these names
    until a full year has passed; package.py sets both to null.
    """
    if not _ACTIONS.exists():
        return {}
    with open(_ACTIONS) as fh:
        events = yaml.safe_load(fh).get("events", []) or []

    cutoff = as_of - timedelta(days=SPINOFF_LOOKBACK_DAYS)
    out: dict[str, date] = {}
    for ev in events:
        parent = ev.get("parent")
        for raw in ev.get("ex_dates") or []:
            ex = date.fromisoformat(raw) if isinstance(raw, str) else raw
            if cutoff <= ex <= as_of:
                if parent not in out or ex > out[parent]:
                    out[parent] = ex
    return out


def _spinoff_note(ticker: str, ex_date: date) -> str:
    return (
        f"{ticker}: P/E and market cap omitted – trailing figures not "
        f"comparable after the {ex_date.isoformat()} spin-off."
    )


# ── Sanity checks ─────────────────────────────────────────────────────────────

def _check_dividend_yield(stocks: list[dict]) -> None:
    for s in stocks:
        dy = s.get("dividend_yield")
        if dy is None:
            continue
        if not (0.0 <= dy <= DIVIDEND_YIELD_MAX_PCT):
            raise RuntimeError(
                f"{s['ticker']}: dividend_yield {dy} outside [0, {DIVIDEND_YIELD_MAX_PCT}] % – "
                f"unit error?"
            )


def _check_pe(stocks: list[dict]) -> None:
    for s in stocks:
        pe = s.get("pe_trailing")
        if pe is None:
            continue
        if pe <= 0:
            raise RuntimeError(f"{s['ticker']}: pe_trailing {pe} ≤ 0")


def _check_implied_shares_drift(stocks: list[dict], prev_pkg: Path | None) -> None:
    """
    For each ticker, implied_shares = market_cap / last_close.
    Compare with the previous package's implied shares; raise if any drift > 20 %.
    Skip gracefully when no previous package exists.
    """
    if prev_pkg is None or not prev_pkg.exists():
        logger.info("No previous package → skipping implied-shares drift check")
        return
    prev = json.loads(prev_pkg.read_text(encoding="utf-8"))
    prev_map = {s["ticker"]: s for s in prev.get("stocks", [])}

    violations = []
    for s in stocks:
        mc, px = s.get("market_cap"), s.get("last_close")
        if mc is None or not px:
            continue
        new_sh = mc / px
        p = prev_map.get(s["ticker"])
        if not p or p.get("market_cap") is None or not p.get("last_close"):
            continue
        old_sh = p["market_cap"] / p["last_close"]
        if old_sh == 0:
            continue
        drift = abs(new_sh / old_sh - 1)
        if drift > SHARES_DRIFT_TOLERANCE:
            violations.append(
                f"{s['ticker']}: implied shares drift {drift:.1%} "
                f"(old {old_sh:,.0f} → new {new_sh:,.0f})"
            )
    if violations:
        raise RuntimeError(
            "Implied-shares drift exceeds tolerance:\n  " + "\n  ".join(violations)
        )


# ── Serialization ─────────────────────────────────────────────────────────────

def _canonical_json(obj: object) -> str:
    raw   = json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=2)
    lines = [line.rstrip() for line in raw.splitlines()]
    return "\n".join(lines) + "\n"


# ── Main build ────────────────────────────────────────────────────────────────

def build_package(
    run_date: date,
    out_dir: Path | None = None,
    prev_package: Path | None = None,
) -> Path:
    """
    Build the weekly data package.

    package.json contains only data known at `as_of` (the last US trading
    session before run_date). No retrieval timestamps. No dates > as_of.
    Retrieval timestamps and raw-source fields go to package_meta.json.
    """
    from data.adjust_prices import build_clean_index
    from data.earnings      import fetch_earnings
    from data.macro         import fetch_macro

    if out_dir is None:
        out_dir = _ROOT / "data" / "weekly"

    tickers, universe_meta, info_map = _load_universe()
    all_tickers = tickers + ["DIA"]

    as_of = _alpaca_last_session(run_date)
    logger.info("run_date=%s  as_of=%s (%s)", run_date, as_of, as_of.strftime("%A"))

    start = run_date - timedelta(days=140)
    logger.info("Downloading prices %s … %s (exclusive)", start, run_date)
    prices = yf.download(
        all_tickers,
        start=str(start),
        end=str(run_date),
        auto_adjust=False,
        progress=False,
    )["Close"]
    prices.index = pd.to_datetime(prices.index)
    prices = prices.dropna(how="all")

    as_of_ts = pd.Timestamp(as_of)
    if as_of_ts not in prices.index:
        raise RuntimeError(f"close for {as_of} not available in price data")
    as_of_pos = prices.index.get_loc(as_of_ts)

    clean_idx = build_clean_index(prices.copy())

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
    stocks: list[dict] = []
    meta_stocks: list[dict] = []
    row = prices.iloc[as_of_pos]
    for tkr in sorted(tickers):
        raw_close  = row.get(tkr)
        last_close = round(float(raw_close), 2) if (raw_close is not None and pd.notna(raw_close)) else None
        f  = fundamentals.get(tkr, {})

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
        meta_stocks.append({
            "ticker":             tkr,
            "shares_outstanding": f.get("shares_outstanding"),
            "yahoo_market_cap":   f.get("market_cap"),
            "market_cap_source":  "Yahoo marketCap field (covers all share classes)",
        })

    # ── Spin-off valuation override ───────────────────────────────────────
    # Trailing P/E and trailing market cap are not comparable for a stock
    # whose parent had a spin-off in the past 12 months; null both out and
    # record a line in data_notes.
    recent = _recent_spinoffs(as_of)
    data_notes: list[str] = []
    verification_date = universe_meta.get("verification_date", "unknown")
    data_notes.append(
        f"Universe: Dow Jones Industrial Average constituents as of "
        f"{verification_date}. Alphabet (GOOGL) replaced Verizon (VZ) on 2026-06-29."
    )
    if recent:
        tickers_affected = {s["ticker"] for s in stocks} & recent.keys()
        for s in stocks:
            if s["ticker"] in tickers_affected:
                s["pe_trailing"] = None
                s["market_cap"]  = None
        for t in sorted(tickers_affected):
            data_notes.append(_spinoff_note(t, recent[t]))

    # ── Sanity checks (raise) ─────────────────────────────────────────────
    _check_dividend_yield(stocks)
    _check_pe(stocks)
    _check_implied_shares_drift(stocks, prev_package)

    # ── Earnings: only in-window entries, plus a note ─────────────────────
    all_entries = fetch_earnings(run_date=run_date)
    in_window = [e for e in all_entries if e.get("next_earnings_date") is not None]
    earnings = {
        "entries": in_window,
        "note": (
            "Other universe companies have no earnings date in this window "
            "according to Yahoo (dates unverified)."
        ),
    }

    # ── Macro: strict as_of filter (no look-ahead) ────────────────────────
    macro_entries = fetch_macro(as_of=as_of)
    for m in macro_entries:
        if m.get("obs_date") and m["obs_date"] > str(as_of):
            raise RuntimeError(
                f"Macro series {m['id']} obs_date {m['obs_date']} > as_of {as_of}"
            )

    # ── Agent-visible package ─────────────────────────────────────────────
    package = {
        "meta": {
            "run_date":         str(run_date),
            "as_of":            str(as_of),
            "universe_version": universe_meta.get("status", "unknown"),
            "universe_tickers": len(tickers),
        },
        "market":     market,
        "stocks":     stocks,
        "earnings":   earnings,
        "macro":      macro_entries,
        "data_notes": data_notes,
    }

    # ── Meta (not hashed, not seen by the agent) ──────────────────────────
    meta_file = {
        "run_date":     str(run_date),
        "as_of":        str(as_of),
        "retrieved_at": retrieved_at,
        "sources":      [
            "Yahoo Finance (prices, fundamentals, earnings)",
            "FRED (macro)",
        ],
        "stocks":       meta_stocks,
    }

    # ── Write ─────────────────────────────────────────────────────────────
    pkg_dir  = out_dir / str(run_date)
    pkg_dir.mkdir(parents=True, exist_ok=True)
    pkg_path  = pkg_dir / "package.json"
    meta_path = pkg_dir / "package_meta.json"

    pkg_path.write_text(_canonical_json(package), encoding="utf-8")
    meta_path.write_text(_canonical_json(meta_file), encoding="utf-8")

    sha = hashlib.sha256(pkg_path.read_bytes()).hexdigest()
    (pkg_dir / "package.sha256").write_text(sha + "\n", encoding="utf-8")

    logger.info("package.json: %d bytes", len(pkg_path.read_bytes()))
    logger.info("SHA-256: %s", sha)

    return pkg_path


# ── Verification ──────────────────────────────────────────────────────────────

def verify_package(path: Path | str) -> bool:
    path     = Path(path)
    sha_path = path.parent / "package.sha256"
    if not sha_path.exists():
        raise FileNotFoundError(f"Hash file not found: {sha_path}")
    expected = sha_path.read_text(encoding="utf-8").strip()
    actual   = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"Hash mismatch:\n  expected: {expected}\n  actual:   {actual}")
    return True
