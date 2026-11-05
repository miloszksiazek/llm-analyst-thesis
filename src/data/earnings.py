"""
Fetch next earnings dates for a list of tickers from Yahoo Finance.

Used to build the earnings-calendar section of the agent's weekly data package.

Rule: only dates in [run_date, run_date + 14 days] are kept; dates outside that
window (past or too far ahead) are stored as null. Status is always
"unverified (Yahoo)" because the calendar API does not expose a confirmed flag.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import yaml
import yfinance as yf

logger = logging.getLogger(__name__)

_UNIVERSE = Path(__file__).parents[2] / "config" / "universe.yaml"
WINDOW_DAYS = 14


def _load_universe() -> list[dict]:
    with open(_UNIVERSE) as fh:
        return yaml.safe_load(fh)["universe"]


def _parse_next_date(ticker: str) -> date | None:
    """
    Return the next earnings date from Yahoo's calendar, or None.

    Yahoo's Earnings Date field is a list; a one-element list and a
    two-element list are both returned without a confirmed/estimated flag —
    the API gives no such distinction. We take the earliest date in the list.
    """
    try:
        cal = yf.Ticker(ticker).calendar
        if not isinstance(cal, dict):
            return None
        raw = cal.get("Earnings Date")
        if not raw:
            return None
        dates = raw if isinstance(raw, list) else [raw]
        parsed = []
        for d in dates:
            if hasattr(d, "date"):
                parsed.append(d.date())
            elif isinstance(d, date):
                parsed.append(d)
        return min(parsed) if parsed else None
    except Exception as exc:
        logger.warning("%s: calendar fetch failed – %s", ticker, exc)
        return None


def fetch_earnings(
    tickers: list[str] | None = None,
    run_date: date | None = None,
) -> list[dict]:
    """
    Fetch next earnings dates for a list of tickers.

    Parameters
    ----------
    tickers : list[str] | None
        Defaults to the 30-ticker universe in config/universe.yaml.
    run_date : date | None
        The reference date (the Sunday the pipeline runs). Defaults to
        date.today(). Only dates in [run_date, run_date + 14 days] are kept;
        all others are stored as null.

    Returns
    -------
    list of dict:
        ticker, company, next_earnings_date, status
    """
    if tickers is None:
        universe  = _load_universe()
        tickers   = [s["ticker"]  for s in universe]
        companies = {s["ticker"]: s["company"] for s in universe}
    else:
        companies = {t: t for t in tickers}

    today    = run_date or date.today()
    deadline = today + timedelta(days=WINDOW_DAYS)
    results  = []

    for tkr in tickers:
        raw_date = _parse_next_date(tkr)

        if raw_date is not None and today <= raw_date <= deadline:
            ed     = str(raw_date)
            status = "unverified (Yahoo)"
        else:
            ed     = None
            status = "none"

        logger.info("%s: raw=%s → stored=%s", tkr, raw_date, ed)
        results.append({
            "ticker":              tkr,
            "company":             companies.get(tkr, tkr),
            "next_earnings_date":  ed,
            "status":              status,
        })

    return results


def earnings_preview(entries: list[dict]) -> str:
    """Format fetch_earnings() output as a plain-text table."""
    reporting = [e for e in entries if e["next_earnings_date"] is not None]
    lines = ["  Earnings within 14-day window:"]
    if reporting:
        for e in sorted(reporting, key=lambda x: x["next_earnings_date"]):
            lines.append(
                f"    {e['ticker']:<6} {e['company']:<40} "
                f"{e['next_earnings_date']}  [{e['status']}]"
            )
    else:
        lines.append("    (none)")
    nulls = [e["ticker"] for e in entries if e["next_earnings_date"] is None]
    lines.append(f"\n  Null (outside window or no data): {', '.join(nulls)}")
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    entries = fetch_earnings()          # uses date.today()
    print("\nEarnings calendar\n")
    print(earnings_preview(entries))
