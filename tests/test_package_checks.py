"""
Unit tests for sanity checks in src/data/package.py.
They cover the pure validators, not the Yahoo/FRED-dependent build_package().
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from datetime import date

from data.package import (
    _check_dividend_yield,
    _check_pe,
    _check_implied_shares_drift,
    _recent_spinoffs,
    _spinoff_note,
)


def test_dividend_yield_in_range_ok():
    _check_dividend_yield([{"ticker": "AAPL", "dividend_yield": 0.5}])


def test_dividend_yield_null_ok():
    _check_dividend_yield([{"ticker": "BRK", "dividend_yield": None}])


def test_dividend_yield_too_high_raises():
    with pytest.raises(RuntimeError, match="dividend_yield"):
        _check_dividend_yield([{"ticker": "CVX", "dividend_yield": 334.0}])


def test_pe_null_ok():
    _check_pe([{"ticker": "X", "pe_trailing": None}])


def test_pe_zero_raises():
    with pytest.raises(RuntimeError, match="pe_trailing"):
        _check_pe([{"ticker": "X", "pe_trailing": 0.0}])


def test_pe_negative_raises():
    with pytest.raises(RuntimeError, match="pe_trailing"):
        _check_pe([{"ticker": "X", "pe_trailing": -3.1}])


def test_implied_shares_drift_skipped_without_prev(tmp_path):
    _check_implied_shares_drift(
        [{"ticker": "X", "last_close": 100.0, "market_cap": 1_000_000}],
        prev_pkg=None,
    )


def test_implied_shares_drift_within_tolerance(tmp_path):
    prev = tmp_path / "prev.json"
    prev.write_text(json.dumps({
        "stocks": [{"ticker": "X", "last_close": 100.0, "market_cap": 1_000_000}],
    }))
    _check_implied_shares_drift(
        [{"ticker": "X", "last_close": 110.0, "market_cap": 1_100_000}],  # same 10k shares
        prev_pkg=prev,
    )


def test_recent_spinoffs_hon_2026_window():
    """HON Aerospace ex-date 2026-06-29: within 12 months of 2026-10-30."""
    recent = _recent_spinoffs(date(2026, 10, 30))
    assert "HON" in recent, f"HON should be flagged; got {recent}"
    assert recent["HON"] == date(2026, 6, 29)


def test_recent_spinoffs_hon_outside_window():
    """A year later – HON falls out of the 12-month lookback."""
    recent = _recent_spinoffs(date(2027, 7, 1))
    assert "HON" not in recent, f"HON should NOT be flagged; got {recent}"


def test_spinoff_override_nulls_valuation_fields():
    """The override step nulls pe_trailing and market_cap for affected tickers,
    and the note mentions the ticker and ex-date."""
    stocks = [
        {"ticker": "HON", "pe_trailing": 10.91, "market_cap": 69_000_000_000, "dividend_yield": 1.27, "last_close": 218.33},
        {"ticker": "AAPL", "pe_trailing": 38.13, "market_cap": 5_000_000_000_000, "dividend_yield": 0.32, "last_close": 344.63},
    ]
    recent = _recent_spinoffs(date(2026, 10, 30))
    notes = []
    for s in stocks:
        if s["ticker"] in recent:
            s["pe_trailing"] = None
            s["market_cap"]  = None
            notes.append(_spinoff_note(s["ticker"], recent[s["ticker"]]))

    hon  = next(s for s in stocks if s["ticker"] == "HON")
    aapl = next(s for s in stocks if s["ticker"] == "AAPL")
    assert hon["pe_trailing"] is None and hon["market_cap"] is None
    assert aapl["pe_trailing"] == 38.13 and aapl["market_cap"] == 5_000_000_000_000
    assert len(notes) == 1 and "HON" in notes[0] and "2026-06-29" in notes[0]


def test_implied_shares_drift_raises(tmp_path):
    prev = tmp_path / "prev.json"
    # prev: 10 000 shares.
    prev.write_text(json.dumps({
        "stocks": [{"ticker": "X", "last_close": 100.0, "market_cap": 1_000_000}],
    }))
    # new: 5 000 shares → 50 % drift.
    with pytest.raises(RuntimeError, match="drift"):
        _check_implied_shares_drift(
            [{"ticker": "X", "last_close": 100.0, "market_cap": 500_000}],
            prev_pkg=prev,
        )
