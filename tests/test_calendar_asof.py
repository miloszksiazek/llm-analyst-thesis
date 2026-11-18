"""Strict-as_of tests using the offline NYSE calendar."""

from __future__ import annotations

from datetime import date
from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.calendar import last_session_before
from data.package  import MissingAsOfError, _expected_as_of


def test_last_session_before_sunday_2026_11_01() -> None:
    """run_date 2026-11-01 (Sun) → as_of 2026-10-30 (Fri)."""
    assert last_session_before(date(2026, 11, 1)) == date(2026, 10, 30)


def test_last_session_before_sunday_2026_11_08() -> None:
    """run_date 2026-11-08 (Sun) → as_of 2026-11-06 (Fri)."""
    assert last_session_before(date(2026, 11, 8)) == date(2026, 11, 6)


def _fake_prices(dates: list[str]) -> pd.DataFrame:
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    return pd.DataFrame({"AAPL": range(len(idx))}, index=idx)


def test_expected_as_of_strict_raises_when_friday_missing() -> None:
    """If Yahoo/Alpaca returned a frame that lacks 2026-11-06, we must raise
    rather than silently fall back to 2026-11-05."""
    prices = _fake_prices([
        "2026-10-30", "2026-11-02", "2026-11-03",
        "2026-11-04", "2026-11-05",
        # 2026-11-06 deliberately missing
    ])
    with pytest.raises(MissingAsOfError):
        _expected_as_of(date(2026, 11, 8), prices)


def test_expected_as_of_returns_friday_when_present() -> None:
    prices = _fake_prices([
        "2026-10-26", "2026-10-27", "2026-10-28",
        "2026-10-29", "2026-10-30",
    ])
    assert _expected_as_of(date(2026, 11, 1), prices) == date(2026, 10, 30)
