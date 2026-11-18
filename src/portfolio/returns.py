"""
Weekly TOTAL returns for every stock in the universe.

For each Monday M in the schedule:
    entry_date = first NYSE session with date >= M        (typically M)
    exit_date  = first NYSE session with date >= M + 7d   (typically next Monday)
    total_return[t]
        = Open[exit_date, t] / Open[entry_date, t] - 1
          + sum(cash_dividend[t, d] for d in (entry_date, exit_date]) / Open[entry_date, t]

Two deviations from the formula:

  * Spin-off week: if a ticker has a spin-off ex-date in (entry_date, exit_date]
    (`config/corporate_actions.yaml`), its weekly return is REPLACED by DIA's
    total return for the same week. Mirrors the price-cleaning rule used by
    XGBoost features: do not let a one-off distribution bleed into the series.
    The substitution is logged so the ledger can mark it.

  * Delisting: if Open[exit_date, t] is missing (NaN) but Open[entry_date, t]
    is present, we treat the stock as sold at its last observed price before
    `exit_date` and compute the return against that. If Open[entry_date, t]
    itself is missing, the return is NaN and the ledger excludes the stock
    from that week's ranking (frozen decision, 2026-11-04).

Inputs are the same raw frames XGBoost uses — raw opens/closes from
`data/raw/history.parquet` plus the cash-dividend table from
`data/raw/dividends.parquet`. No auto-adjusted prices.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yaml

from data.calendar import first_session_on_or_after

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parents[2]
_ACTIONS = _ROOT / "config" / "corporate_actions.yaml"


def _spinoff_lookup() -> dict[str, set[pd.Timestamp]]:
    if not _ACTIONS.exists():
        return {}
    with open(_ACTIONS) as fh:
        events = yaml.safe_load(fh).get("events", []) or []
    out: dict[str, set[pd.Timestamp]] = {}
    for ev in events:
        bucket = out.setdefault(ev["parent"], set())
        for raw in ev.get("ex_dates") or []:
            bucket.add(pd.Timestamp(raw))
    return out


def weekly_entries(mondays: list[date], sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Resolve each scheduled Monday to its (entry, exit) trading sessions.

    If a Monday is a US holiday, entry is the next trading session; the
    agent is not re-run (the decision was made Sunday). Exit is the entry
    of the following week.
    """
    rows: list[dict] = []
    for m in mondays:
        entry = first_session_on_or_after(m)
        rows.append({"monday": pd.Timestamp(m), "entry_date": pd.Timestamp(entry)})
    df = pd.DataFrame(rows)
    df["exit_date"] = df["entry_date"].shift(-1)
    # For the last week in the list, exit is the first session on/after M+7d
    last = df.iloc[-1]
    df.loc[df.index[-1], "exit_date"] = pd.Timestamp(
        first_session_on_or_after(last["monday"].date() + timedelta(days=7))
    )
    return df


@dataclass
class WeeklyReturns:
    """Container the ledger consumes."""
    table: pd.DataFrame            # rows: entry_date; columns: ticker; values: total return
    schedule: pd.DataFrame         # monday, entry_date, exit_date
    spinoff_weeks: dict[pd.Timestamp, list[str]]  # entry_date -> [tickers substituted with DIA]
    missing: dict[pd.Timestamp, list[str]]        # entry_date -> [tickers with NaN return]


def _ticker_total_return(
    t: str,
    entry: pd.Timestamp,
    exit_: pd.Timestamp,
    opens: pd.DataFrame,
    dividends: pd.DataFrame,
) -> float | None:
    if t not in opens.columns:
        return None
    o_entry = opens.at[entry, t] if entry in opens.index else None
    if pd.isna(o_entry) or o_entry in (None, 0):
        return None

    if exit_ in opens.index:
        o_exit = opens.at[exit_, t]
    else:
        o_exit = float("nan")
    if pd.isna(o_exit):
        # Delisted: use last observed open strictly before exit_
        prior = opens.loc[entry:exit_, t].dropna()
        if len(prior) <= 1:
            return None
        o_exit = float(prior.iloc[-1])

    div_sum = 0.0
    if not dividends.empty and t in dividends.columns:
        mask = (dividends.index > entry) & (dividends.index <= exit_)
        if mask.any():
            div_sum = float(dividends.loc[mask, t].sum())

    return float(o_exit) / float(o_entry) - 1.0 + div_sum / float(o_entry)


def build_weekly_returns(
    mondays: list[date],
    opens: pd.DataFrame,
    dividends: pd.DataFrame,
    tickers: list[str],
) -> WeeklyReturns:
    """
    Build the weekly total-return table for `tickers` + DIA.

    `opens`     DataFrame of raw opening prices (DatetimeIndex, ticker cols, must include DIA).
    `dividends` DataFrame of cash dividends (DatetimeIndex ex-dates, ticker cols, 0 where no payout).
    """
    if "DIA" not in opens.columns:
        raise ValueError("opens must include a 'DIA' column (needed for spin-off substitution)")

    spin = _spinoff_lookup()
    schedule = weekly_entries(mondays, opens.index)

    # DIA weekly total return (we substitute into spin-off weeks)
    dia_tr: dict[pd.Timestamp, float] = {}
    for _, r in schedule.iterrows():
        tr = _ticker_total_return("DIA", r["entry_date"], r["exit_date"], opens, dividends)
        dia_tr[r["entry_date"]] = tr if tr is not None else float("nan")

    cols = list(dict.fromkeys(tickers + ["DIA"]))
    table = pd.DataFrame(index=pd.DatetimeIndex([r["entry_date"] for _, r in schedule.iterrows()]),
                         columns=cols, dtype=float)
    spinoff_weeks: dict[pd.Timestamp, list[str]] = {}
    missing: dict[pd.Timestamp, list[str]] = {}

    for _, r in schedule.iterrows():
        entry, exit_ = r["entry_date"], r["exit_date"]
        subs_here: list[str] = []
        miss_here: list[str] = []
        for t in cols:
            # Spin-off substitution
            if t in spin and any(entry < d <= exit_ for d in spin[t]):
                table.at[entry, t] = dia_tr[entry]
                subs_here.append(t)
                logger.info("Spin-off week %s: %s return -> DIA (%.4f)",
                            entry.date(), t, dia_tr[entry])
                continue
            tr = _ticker_total_return(t, entry, exit_, opens, dividends)
            if tr is None:
                miss_here.append(t)
                table.at[entry, t] = float("nan")
            else:
                table.at[entry, t] = tr
        if subs_here:
            spinoff_weeks[entry] = subs_here
        if miss_here:
            missing[entry] = miss_here

    return WeeklyReturns(table=table, schedule=schedule.set_index("entry_date"),
                         spinoff_weeks=spinoff_weeks, missing=missing)
