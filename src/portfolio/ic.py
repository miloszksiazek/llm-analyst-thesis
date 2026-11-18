"""
Weekly rank Information Coefficient (IC).

For each week, IC = Spearman correlation between a ranking and the realized
total returns of the 30 stocks that week. Weeks where the ranking is missing
(fallback triggered, delisting left < 3 valid tickers) are excluded.

A ranking is passed as `dict[ticker -> rank]` where lower rank = "more
preferred". Agent aggregate ranking uses the average rank across the N
single runs; this module also accepts a list of single-run rankings and
reports IC for each.

Summary stats per ranking source:
    mean_ic   mean across included weeks
    std_ic    sample std (ddof=1)
    t_stat    mean_ic / (std_ic / sqrt(n))      -- treats weekly ICs as iid
    hit_rate  share of weeks with IC > 0
    n_weeks   number of weeks with a valid IC
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


@dataclass
class ICSeries:
    label: str
    weekly: pd.Series   # index = entry_date, values = Spearman rho


def weekly_rank_ic(
    rankings: dict[pd.Timestamp, dict[str, float] | None],
    returns_table: pd.DataFrame,
) -> pd.Series:
    """
    Compute Spearman IC for each week present in both inputs.

    `rankings[entry_date]` must be a dict ticker->rank. Missing/None values
    yield a NaN for that week (excluded from the summary).
    """
    ics = {}
    for entry, rnk in rankings.items():
        if rnk is None or entry not in returns_table.index:
            ics[entry] = float("nan")
            continue
        r = returns_table.loc[entry]
        # Restrict to tickers in both the ranking and with non-NaN returns.
        common = [t for t in rnk if t in r.index and pd.notna(r[t])]
        if len(common) < 3:
            ics[entry] = float("nan")
            continue
        x = np.array([rnk[t] for t in common], dtype=float)
        y = np.array([float(r[t]) for t in common], dtype=float)
        rho, _ = spearmanr(x, y)
        # Note: Spearman on ranks x and returns y yields -IC because lower rank =
        # better. Negate so positive IC means good ranking.
        ics[entry] = float(-rho) if np.isfinite(rho) else float("nan")
    return pd.Series(ics, name="ic").sort_index()


def summarize(series: pd.Series) -> dict:
    s = series.dropna()
    n = int(len(s))
    if n == 0:
        return {"n_weeks": 0, "mean_ic": float("nan"), "std_ic": float("nan"),
                "t_stat": float("nan"), "hit_rate": float("nan")}
    mean = float(s.mean())
    std = float(s.std(ddof=1)) if n > 1 else float("nan")
    t = mean / (std / np.sqrt(n)) if (std and std > 0 and n > 1) else float("nan")
    hit = float((s > 0).mean())
    return {"n_weeks": n, "mean_ic": mean, "std_ic": std, "t_stat": t, "hit_rate": hit}


def ranking_from_order(order: list[str]) -> dict[str, int]:
    """Convert an ordered list (best first) to a rank dict. Ranks start at 1."""
    return {t: i + 1 for i, t in enumerate(order)}


def average_rank_from_runs(run_rankings: list[dict[str, int]]) -> dict[str, float]:
    """Average rank across runs (used as the agent's aggregate ranking)."""
    tickers = sorted({t for r in run_rankings for t in r})
    out = {}
    for t in tickers:
        ranks = [r[t] for r in run_rankings if t in r]
        if ranks:
            out[t] = float(np.mean(ranks))
    return out
