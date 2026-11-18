"""
Random-portfolio benchmark.

Generates N random 4-stock portfolios (uniform over all `C(30, 4) = 27,405`
combinations in a week, independent draws across weeks) and compounds each
one through the weekly return table. Output is the empirical distribution of
cumulative returns at every week, plus a helper that scores a real strategy
against it.

Fixed seed so the distribution is reproducible across reruns of the analysis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from portfolio.ledger import run_strategy

SEED = 20261110
N_DEFAULT = 10_000


@dataclass
class RandomBenchmark:
    cumulative: pd.DataFrame        # rows: entry_date, cols: 0..N-1, values: value_end
    tickers: list[str]
    seed: int
    n: int

    def final_returns(self) -> pd.Series:
        """Terminal cumulative return for each random portfolio (as fraction, not %)."""
        start = 100_000.0
        return self.cumulative.iloc[-1] / start - 1.0

    def percentile_of(self, cumulative_series: pd.Series) -> dict:
        """Report the percentile of each week's value of `cumulative_series`."""
        aligned = cumulative_series.reindex(self.cumulative.index)
        pcts = {}
        for dt in self.cumulative.index:
            draws = self.cumulative.loc[dt].values
            v = aligned.loc[dt]
            if pd.isna(v):
                pcts[dt] = float("nan")
                continue
            pcts[dt] = float((draws < v).mean() * 100.0)
        final_v = aligned.iloc[-1]
        terminal_pct = pcts[self.cumulative.index[-1]] if len(pcts) else float("nan")
        return {"weekly_percentiles": pcts, "terminal_percentile": terminal_pct,
                "terminal_value": float(final_v) if pd.notna(final_v) else None}


def run_random(
    weekly_returns,
    tickers: list[str],
    n: int = N_DEFAULT,
    k: int = 4,
    seed: int = SEED,
    start_value: float = 100_000.0,
    cost_bps: float = 0.0,
) -> RandomBenchmark:
    """Build the random-portfolio cumulative-value matrix."""
    tickers = [t for t in tickers if t != "DIA"]
    rng = np.random.default_rng(seed)
    n_tickers = len(tickers)
    entries = list(weekly_returns.schedule.index)

    # Pre-sample all picks as indices into `tickers`, shape (n, n_weeks, k).
    # Each (sim, week) is k distinct indices drawn without replacement.
    all_picks = np.empty((n, len(entries), k), dtype=np.int32)
    for s in range(n):
        for w in range(len(entries)):
            all_picks[s, w] = rng.choice(n_tickers, size=k, replace=False)

    cum = pd.DataFrame(index=pd.DatetimeIndex(entries), columns=range(n), dtype=float)
    for s in range(n):
        picks = {
            entries[w]: [tickers[i] for i in all_picks[s, w]]
            for w in range(len(entries))
        }
        res = run_strategy(f"random_{s}", picks, weekly_returns,
                           start_value=start_value, cost_bps=cost_bps)
        cum.iloc[:, s] = [wk.value_end for wk in res.weeks]
    return RandomBenchmark(cumulative=cum, tickers=tickers, seed=seed, n=n)
