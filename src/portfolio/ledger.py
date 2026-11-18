"""
Portfolio ledgers for every strategy in the study.

A strategy is a sequence of weekly decisions = {ticker: target weight}. The
ledger compounds those decisions forward one week at a time using the total
returns from `portfolio.returns.WeeklyReturns.table`.

Strategies implemented here:

    claude_web / claude_noweb   aggregate ranking (average rank over N runs)
                                -> top 4 at 25% each
    xgb                         prediction ranking -> top 4 at 25% each
    ew_1n                       1/30 (every universe stock), rebalanced weekly
    dia                         buy-and-hold DIA, dividends reinvested
                                (= compounding DIA weekly total return)
    random                      caller supplies weekly picks (see benchmarks.py)

Conventions
    - Start capital: 100,000 USD.
    - Fractional shares. Weights target exactly 1/K per holding at each entry.
    - No costs in the headline run. `cost_bps` parameter charges per-side
      transaction costs on rebalance turnover; total cost = turnover × bps/1e4
      with turnover = Σ|w_new - w_old|  (captures both buys and sells).
    - Fallback: if a week's ranking is missing/invalid the strategy KEEPS the
      prior week's holdings (`keep_prev_portfolio`, settings.yaml). The week
      is recorded with `ranking_missing=True` so IC can exclude it.

Picks input format: `dict[pd.Timestamp, list[str]]` keyed by entry_date, list
of 4 tickers (or `None` / missing key to trigger the fallback). The ledger
does not check that length is exactly 4 — ew_1n uses 30, random uses 4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class LedgerWeek:
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    holdings: dict[str, float]   # ticker -> weight (sums to 1 on entry)
    weekly_return: float          # net of costs
    turnover: float               # Σ|w_new - w_old|
    cost: float                   # fraction of portfolio paid as costs
    value_end: float              # portfolio value at exit (start_value * Π(1+r))
    ranking_missing: bool = False


@dataclass
class LedgerResult:
    strategy: str
    start_value: float
    weeks: list[LedgerWeek] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        rows = [
            {
                "entry_date":     w.entry_date,
                "exit_date":      w.exit_date,
                "weekly_return":  w.weekly_return,
                "turnover":       w.turnover,
                "cost":           w.cost,
                "value_end":      w.value_end,
                "ranking_missing": w.ranking_missing,
                "holdings":       w.holdings,
            }
            for w in self.weeks
        ]
        return pd.DataFrame(rows)

    def cumulative(self) -> pd.Series:
        s = pd.Series(
            [w.value_end for w in self.weeks],
            index=[w.entry_date for w in self.weeks],
            name=self.strategy,
        )
        return s


def _step(
    entry: pd.Timestamp,
    exit_: pd.Timestamp,
    target: dict[str, float] | None,
    prev_weights: dict[str, float],
    returns_row: pd.Series,
    value_start: float,
    cost_bps: float,
) -> LedgerWeek:
    """Execute one weekly step: rebalance to `target`, apply weekly returns, charge costs."""
    ranking_missing = False
    if target is None or len(target) == 0:
        target = dict(prev_weights) if prev_weights else {}
        ranking_missing = True

    # Drop target tickers with NaN returns this week (delisted / untraded):
    # redistribute their weight pro-rata to the survivors (frozen rule, 2026-11-04).
    valid = {t: w for t, w in target.items() if t in returns_row.index and pd.notna(returns_row[t])}
    if not valid:
        # Degenerate: no valid tickers -> hold cash (0 return, 0 turnover, flag missing)
        return LedgerWeek(
            entry_date=entry, exit_date=exit_, holdings={}, weekly_return=0.0,
            turnover=0.0, cost=0.0, value_end=value_start, ranking_missing=True,
        )
    s = sum(valid.values())
    valid = {t: w / s for t, w in valid.items()}

    all_tickers = set(prev_weights) | set(valid)
    turnover = sum(
        abs(valid.get(t, 0.0) - prev_weights.get(t, 0.0)) for t in all_tickers
    )
    cost = turnover * cost_bps / 10_000.0

    gross_r = sum(valid[t] * float(returns_row[t]) for t in valid)
    net_r = gross_r - cost
    value_end = value_start * (1.0 + net_r)

    return LedgerWeek(
        entry_date=entry, exit_date=exit_, holdings=valid,
        weekly_return=net_r, turnover=turnover, cost=cost,
        value_end=value_end, ranking_missing=ranking_missing,
    )


def run_strategy(
    name: str,
    picks: dict[pd.Timestamp, list[str] | None],
    weekly_returns,            # portfolio.returns.WeeklyReturns
    start_value: float = 100_000.0,
    cost_bps: float = 0.0,
    weight_per_name: float | None = None,
) -> LedgerResult:
    """
    Compound weekly decisions through the schedule.

    `picks` : entry_date -> list of tickers to hold at target weight.
              Missing / None picks trigger the fallback (keep prev holdings).
    `weight_per_name` : override for equal weighting. None -> 1/len(picks).
    """
    sched = weekly_returns.schedule
    table = weekly_returns.table
    result = LedgerResult(strategy=name, start_value=start_value)

    prev_weights: dict[str, float] = {}
    value = start_value
    for entry, row in sched.iterrows():
        exit_ = row["exit_date"]
        pk = picks.get(entry)
        if pk is None:
            target = None
        else:
            w = weight_per_name if weight_per_name is not None else 1.0 / len(pk)
            target = {t: w for t in pk}
        wk = _step(entry, exit_, target, prev_weights,
                   table.loc[entry], value, cost_bps)
        prev_weights = wk.holdings
        value = wk.value_end
        result.weeks.append(wk)

    return result


# --- Strategy constructors ---------------------------------------------------

def ew_1n_picks(weekly_returns, tickers: Iterable[str]) -> dict[pd.Timestamp, list[str]]:
    """Picks dict for 1/N over the full universe (excluding DIA)."""
    tickers = [t for t in tickers if t != "DIA"]
    return {entry: list(tickers) for entry in weekly_returns.schedule.index}


def dia_picks(weekly_returns) -> dict[pd.Timestamp, list[str]]:
    """DIA buy-and-hold. Weekly rebalance is a no-op (always 100% DIA)."""
    return {entry: ["DIA"] for entry in weekly_returns.schedule.index}


def aggregate_ranking(run_rankings: list[dict[str, int]]) -> list[str]:
    """
    Average-rank aggregation across N single-run rankings.

    Each run ranking is {ticker: rank_1_to_30}. Returns tickers sorted by
    mean rank ascending (best first). Ties broken by ticker name (stable).
    """
    if not run_rankings:
        return []
    tickers = sorted({t for r in run_rankings for t in r})
    means = []
    for t in tickers:
        ranks = [r[t] for r in run_rankings if t in r]
        if not ranks:
            continue
        means.append((float(np.mean(ranks)), t))
    means.sort(key=lambda x: (x[0], x[1]))
    return [t for _, t in means]


def picks_from_ranking(ranking: list[str], k: int = 4) -> list[str]:
    """Top-k of an ordered ranking."""
    return list(ranking[:k])
