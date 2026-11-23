"""Portfolio ledgers, returns, IC, and benchmarks."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml

_ROOT = Path(__file__).parents[2]
_SETTINGS = _ROOT / "config" / "settings.yaml"


def load_test_start() -> date:
    """Return the first run_date (Sunday) that counts as the forward test.

    Earlier run_dates are the pilot: weeks whose entry_date (first NYSE session
    on-or-after run_date + 1 day) is on-or-before this Sunday are excluded by
    default from the ledger and IC. See docs/zmiany_do_planu.md.
    """
    cfg = yaml.safe_load(_SETTINGS.read_text(encoding="utf-8")) or {}
    raw = cfg.get("test_start")
    if raw is None:
        raise KeyError("config/settings.yaml is missing 'test_start'")
    return date.fromisoformat(str(raw))
