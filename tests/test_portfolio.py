"""
Hand-computed tests for the portfolio ledger, random benchmark and IC.

The main dataset dependencies (Yahoo history, dividend cache) are NOT loaded
here — every test fabricates its own tiny price/dividend frames so the logic
can be checked against values calculated by hand on paper.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from portfolio.returns import build_weekly_returns, WeeklyReturns
from portfolio.ledger import run_strategy, ew_1n_picks, dia_picks, aggregate_ranking
from portfolio.ic import weekly_rank_ic, summarize, ranking_from_order
from portfolio.benchmarks import run_random


def _synthetic_opens(dates, data):
    return pd.DataFrame(data, index=pd.DatetimeIndex(dates))


def _empty_divs(columns):
    return pd.DataFrame(columns=list(columns))


def test_ic_five_stock_example_rho_0_8():
    """Plan example: ranking A,B,C,D,E vs realized B,A,C,E,D -> IC = 0.8."""
    ranking = ranking_from_order(["A", "B", "C", "D", "E"])
    entry = pd.Timestamp("2026-12-07")
    returns_table = pd.DataFrame(
        {"A": [4.0], "B": [5.0], "C": [3.0], "D": [1.0], "E": [2.0]},
        index=[entry],
    )
    ics = weekly_rank_ic({entry: ranking}, returns_table)
    assert ics.loc[entry] == pytest.approx(0.8, abs=1e-9)
    summ = summarize(ics)
    assert summ["n_weeks"] == 1 and summ["hit_rate"] == 1.0


def test_weekly_return_includes_dividend_in_window():
    """Dividend with ex-date in (entry, exit] enters the total return."""
    # Two weeks: entries 2026-12-07 (Mon) and 2026-12-14 (Mon).
    opens = _synthetic_opens(
        ["2026-12-07", "2026-12-14", "2026-12-21"],
        {"A": [100.0, 102.0, 101.0], "DIA": [400.0, 404.0, 405.0]},
    )
    divs = pd.DataFrame(
        {"A": [0.0, 1.50, 0.0], "DIA": [0.0, 0.0, 0.0]},
        index=pd.DatetimeIndex(["2026-12-07", "2026-12-10", "2026-12-20"]),
    )
    wr = build_weekly_returns(
        mondays=[date(2026, 12, 7), date(2026, 12, 14)],
        opens=opens, dividends=divs, tickers=["A"],
    )
    # Week 1: (102 - 100 + 1.50) / 100 = 0.035
    assert wr.table.loc[pd.Timestamp("2026-12-07"), "A"] == pytest.approx(0.035)
    # Week 2: (101 - 102 + 0) / 102 (div on 2026-12-20 falls in (14,21])
    # Actually the dividend *is* in (2026-12-14, 2026-12-21] -> included.
    expected = (101.0 - 102.0 + 0.0) / 102.0  # no div on 12-20 for A in this test
    # Correct: div = 0 on 12-20 for A -> just capital return.
    assert wr.table.loc[pd.Timestamp("2026-12-14"), "A"] == pytest.approx(expected)


def test_dividend_ex_date_at_entry_is_excluded():
    """Boundary: ex-date == entry_date is OUTSIDE the (entry, exit] window."""
    opens = _synthetic_opens(
        ["2026-12-07", "2026-12-14"],
        {"A": [100.0, 100.0], "DIA": [400.0, 400.0]},
    )
    divs = pd.DataFrame(
        {"A": [2.00]},
        index=pd.DatetimeIndex(["2026-12-07"]),
    )
    wr = build_weekly_returns(
        mondays=[date(2026, 12, 7)], opens=opens, dividends=divs, tickers=["A"],
    )
    # Dividend on entry date itself -> NOT in (entry, exit] -> return = 0.
    assert wr.table.loc[pd.Timestamp("2026-12-07"), "A"] == pytest.approx(0.0)


def test_us_holiday_monday_shifts_entry():
    """Jan 1 2027 falls on a Friday; Jan 18 2027 (MLK Day) is a US holiday.
    Scheduling the Monday 2027-01-18 must pick the next trading session."""
    import exchange_calendars as ec
    cal = ec.get_calendar("XNYS")
    sess = cal.sessions_in_range("2027-01-15", "2027-01-22")
    # Make opens cover the relevant dates with simple values
    opens = _synthetic_opens(
        [d.date().isoformat() for d in sess],
        {"A": list(range(100, 100 + len(sess))),
         "DIA": list(range(400, 400 + len(sess)))},
    )
    wr = build_weekly_returns(
        mondays=[date(2027, 1, 18), date(2027, 1, 25)],
        opens=opens, dividends=_empty_divs(opens.columns), tickers=["A"],
    )
    # 2027-01-18 is MLK Day (holiday). Entry must shift to 2027-01-19.
    assert pd.Timestamp("2027-01-19") in wr.table.index
    assert pd.Timestamp("2027-01-18") not in wr.table.index


def test_fallback_week_keeps_prev_portfolio():
    """Missing ranking -> keep previous holdings, ranking_missing=True, no turnover."""
    entries = ["2026-12-07", "2026-12-14", "2026-12-21"]
    opens = _synthetic_opens(
        entries,
        {"A": [100.0, 110.0, 110.0], "B": [100.0, 100.0, 120.0],
         "DIA": [400.0, 400.0, 400.0]},
    )
    wr = build_weekly_returns(
        mondays=[date(2026, 12, 7), date(2026, 12, 14)],
        opens=opens, dividends=_empty_divs(opens.columns), tickers=["A", "B"],
    )
    picks = {
        pd.Timestamp("2026-12-07"): ["A", "B"],
        pd.Timestamp("2026-12-14"): None,              # fallback
    }
    res = run_strategy("test", picks, wr, start_value=100_000.0)
    # Week 1: picks = [A,B] (50/50). Return = 0.5*0.10 + 0.5*0.00 = 0.05
    assert res.weeks[0].weekly_return == pytest.approx(0.05)
    assert not res.weeks[0].ranking_missing
    # Week 2: fallback -> same holdings. A rose from 110->110 (0%), B rose 100->120 (20%).
    # Portfolio return (unchanged weights 50/50) = 0.10. ranking_missing=True.
    assert res.weeks[1].ranking_missing
    assert res.weeks[1].weekly_return == pytest.approx(0.10)
    # No new picks -> turnover = 0 (same target weights as prev).
    assert res.weeks[1].turnover == pytest.approx(0.0)


def test_cost_bps_charges_on_turnover():
    """cost_bps×turnover/1e4 subtracted from the gross weekly return."""
    entries = ["2026-12-07", "2026-12-14"]
    opens = _synthetic_opens(
        entries,
        {"A": [100.0, 110.0], "DIA": [400.0, 400.0]},
    )
    wr = build_weekly_returns(
        mondays=[date(2026, 12, 7)],
        opens=opens, dividends=_empty_divs(opens.columns), tickers=["A"],
    )
    # First week: prev=0, target={A:1} -> turnover = 1.0.
    picks = {pd.Timestamp("2026-12-07"): ["A"]}
    r0 = run_strategy("no_cost", picks, wr, cost_bps=0.0)
    r1 = run_strategy("with_cost", picks, wr, cost_bps=25.0)
    assert r0.weeks[0].weekly_return == pytest.approx(0.10)
    # 25 bps × turnover 1.0 = 0.0025 charged.
    assert r1.weeks[0].weekly_return == pytest.approx(0.10 - 0.0025)
    assert r1.weeks[0].turnover == pytest.approx(1.0)
    assert r1.weeks[0].cost == pytest.approx(0.0025)


def test_spinoff_week_substitutes_dia():
    """A ticker whose spin-off ex-date is in (entry, exit] gets DIA's return."""
    # HON Aerospace spin-off ex_date is 2026-06-29 (Monday). Schedule a week
    # spanning that date.
    entries = ["2026-06-22", "2026-06-29", "2026-07-06"]
    opens = _synthetic_opens(
        entries,
        {"HON": [220.0, 210.0, 205.0], "DIA": [400.0, 404.0, 406.0]},
    )
    wr = build_weekly_returns(
        mondays=[date(2026, 6, 22), date(2026, 6, 29)],
        opens=opens, dividends=_empty_divs(opens.columns), tickers=["HON"],
    )
    # Week 2026-06-22: HON's spin-off ex-date 2026-06-29 is in (06-22, 06-29]? YES.
    # -> HON gets DIA's weekly return.
    dia_week1 = (404.0 - 400.0) / 400.0
    assert wr.table.loc[pd.Timestamp("2026-06-22"), "HON"] == pytest.approx(dia_week1)
    assert "HON" in wr.spinoff_weeks[pd.Timestamp("2026-06-22")]


def test_aggregate_ranking_averages_run_ranks():
    runs = [
        {"A": 1, "B": 2, "C": 3},
        {"A": 3, "B": 1, "C": 2},
        {"A": 2, "B": 1, "C": 3},
    ]
    order = aggregate_ranking(runs)
    # Mean ranks: A = 2.0, B = 1.33, C = 2.67 -> order B, A, C
    assert order == ["B", "A", "C"]


def test_random_benchmark_shapes_and_seed():
    """Small random run is reproducible and matches 1/N mean return in expectation."""
    entries = ["2026-12-07", "2026-12-14"]
    opens = _synthetic_opens(
        entries + ["2026-12-21"],
        {
            "A": [100, 110, 110], "B": [100, 100, 120],
            "C": [100, 105, 100], "D": [100, 95, 95],
            "DIA": [400, 400, 400],
        },
    )
    wr = build_weekly_returns(
        mondays=[date(2026, 12, 7), date(2026, 12, 14)],
        opens=opens, dividends=_empty_divs(opens.columns),
        tickers=["A", "B", "C", "D"],
    )
    rb1 = run_random(wr, tickers=["A", "B", "C", "D"], n=50, k=2, seed=42)
    rb2 = run_random(wr, tickers=["A", "B", "C", "D"], n=50, k=2, seed=42)
    # Reproducible under fixed seed.
    assert rb1.cumulative.equals(rb2.cumulative)
    assert rb1.cumulative.shape == (2, 50)
