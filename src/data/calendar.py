"""
NYSE trading calendar helpers.

Uses the offline `exchange_calendars` library (XNYS) pinned in
requirements.txt. No network, no Alpaca; the calendar is baked into the
package. Replaces the previous "last row in prices" heuristic — the as-of
is now computed from the official exchange schedule and the pipeline
raises if that session is missing from the price data.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from functools import lru_cache

import exchange_calendars as ec
import pandas as pd

NYSE = "XNYS"


@lru_cache(maxsize=1)
def _cal():
    return ec.get_calendar(NYSE)


def last_session_before(run_date: date) -> date:
    """Last NYSE trading session strictly before `run_date` (local date)."""
    cal = _cal()
    start = (pd.Timestamp(run_date) - pd.Timedelta(days=14)).date()
    end   = (pd.Timestamp(run_date) - pd.Timedelta(days=1)).date()
    sessions = cal.sessions_in_range(str(start), str(end))
    if len(sessions) == 0:
        raise RuntimeError(f"No NYSE sessions in [{start}, {end}]")
    return sessions[-1].date()


def first_session_on_or_after(run_date: date) -> date:
    """First NYSE trading session on or after `run_date`."""
    cal = _cal()
    start = pd.Timestamp(run_date).date()
    end   = (pd.Timestamp(run_date) + pd.Timedelta(days=14)).date()
    sessions = cal.sessions_in_range(str(start), str(end))
    if len(sessions) == 0:
        raise RuntimeError(f"No NYSE sessions in [{start}, {end}]")
    return sessions[0].date()


def session_open_utc(session: date) -> datetime:
    """UTC timestamp of the regular open for `session`."""
    cal = _cal()
    ts = cal.session_open(pd.Timestamp(session))
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.to_pydatetime().astimezone(timezone.utc)


def is_past_next_open(run_date: date, now_utc: datetime | None = None) -> bool:
    """True if `now_utc` is after the first NYSE open on-or-after `run_date`.

    Used by the pipeline as a look-ahead guard: once the Monday session has
    opened, backfilling a run_date = that-Sunday would mean picking stocks
    after seeing the opening tape.
    """
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    nxt = first_session_on_or_after(run_date)
    open_utc = session_open_utc(nxt)
    return now_utc >= open_utc
