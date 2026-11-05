"""
Spin-off return cleaning for XGBoost price features.

For each confirmed corporate action in config/corporate_actions.yaml, the
affected stock's daily return on the event date is replaced with DIA's return
on that same date. A cleaned price index (starting at 1.0) is then rebuilt
from the adjusted return series.

ALL XGBoost features that involve price (momentum, moving-average distance,
volatility, return vs DIA) must be computed from the cleaned index returned
by build_clean_index(), both for training and in the weekly live pipeline.
See CLAUDE.md rule 10.

The function is idempotent: if the event-date return in the input data is
already close to DIA's return (because Yahoo has already adjusted), replacing
it again changes nothing materially.
"""

from __future__ import annotations

import logging
import yaml
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

_CONFIG = Path(__file__).parents[2] / "config" / "corporate_actions.yaml"


def _load_events() -> list[dict]:
    with open(_CONFIG) as fh:
        return yaml.safe_load(fh)["events"]


def clean_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Replace spin-off event-date returns with DIA's return on that date.

    Parameters
    ----------
    prices : pd.DataFrame
        DatetimeIndex rows, ticker columns (must include 'DIA').
        Pass Yahoo Finance or Alpaca daily close prices.

    Returns
    -------
    pd.DataFrame
        Daily returns (pct_change) with event-date returns replaced.
        First row is NaN (same as pct_change convention).
    """
    if "DIA" not in prices.columns:
        raise ValueError("prices must include a 'DIA' column")

    if not isinstance(prices.index, pd.DatetimeIndex):
        prices = prices.copy()
        prices.index = pd.to_datetime(prices.index)

    rets = prices.pct_change()

    for ev in _load_events():
        ticker = ev["parent"]
        if ticker not in rets.columns:
            continue
        for dt_str in ev["ex_dates"]:
            dt = pd.Timestamp(dt_str)
            if dt not in rets.index:
                logger.warning("Event %s: date %s not in index – skipped", ev["id"], dt_str)
                continue
            before = float(rets.loc[dt, ticker])
            dia    = float(rets.loc[dt, "DIA"])
            rets.loc[dt, ticker] = dia
            logger.info(
                "Event %s: %s return on %s replaced %.4f → %.4f (DIA)",
                ev["id"], ticker, dt_str, before, dia,
            )

    return rets


def build_clean_index(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Build a cleaned price index starting at 1.0 for every ticker.

    Parameters
    ----------
    prices : pd.DataFrame
        Raw close prices (DatetimeIndex, ticker columns, must include 'DIA').

    Returns
    -------
    pd.DataFrame
        Price index with same shape; each column starts at 1.0 on the first
        available date and is compounded from the cleaned daily returns.
    """
    rets = clean_returns(prices)
    # Fill the first NaN with 0 so cumprod starts at 1.0
    filled = rets.fillna(0)
    index = (1 + filled).cumprod()
    return index


def assert_clean(prices: pd.DataFrame) -> None:
    """
    Assert that every event-date |stock_return - DIA_return| ≤ 1e-9.
    Raises AssertionError with the event id if any event date is not clean.
    Used in tests.
    """
    if not isinstance(prices.index, pd.DatetimeIndex):
        prices = prices.copy()
        prices.index = pd.to_datetime(prices.index)

    rets = prices.pct_change()

    for ev in _load_events():
        ticker = ev["parent"]
        if ticker not in rets.columns:
            continue
        for dt_str in ev["ex_dates"]:
            dt = pd.Timestamp(dt_str)
            if dt not in rets.index:
                continue
            dev = abs(float(rets.loc[dt, ticker]) - float(rets.loc[dt, "DIA"]))
            assert dev < 1e-9, (
                f"Event {ev['id']} date {dt_str}: "
                f"stock={rets.loc[dt, ticker]:.6f} DIA={rets.loc[dt, 'DIA']:.6f} "
                f"deviation={dev:.2e} (not clean)"
            )
