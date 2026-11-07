"""
Unit tests for sanity checks in src/data/package.py.
They cover the pure validators, not the Yahoo/FRED-dependent build_package().
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from data.package import (
    _check_dividend_yield,
    _check_pe,
    _check_implied_shares_drift,
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
