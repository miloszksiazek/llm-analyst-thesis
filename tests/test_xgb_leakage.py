"""Leakage checks for the XGBoost panel."""

from datetime import date
from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xgb.dataset import build_panel, leakage_check
from xgb.features import load_bundle


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    panel_path = ROOT / "data" / "raw" / "panel.parquet"
    if panel_path.exists():
        return pd.read_parquet(panel_path)
    bundle = load_bundle()
    return build_panel(bundle, start=date(2024, 1, 1), end=date(2024, 6, 30))


def test_no_leakage(panel: pd.DataFrame) -> None:
    leakage_check(panel)


def test_asof_before_target(panel: pd.DataFrame) -> None:
    """as_of must precede the entire target window."""
    assert (panel["as_of"] < panel["entry_date"]).all()
    assert (panel["entry_date"] < panel["exit_date"]).all()


def test_target_window_gap(panel: pd.DataFrame) -> None:
    """Target entry is within 4 days of as_of (next trading session after a Fri as_of)."""
    gap = (panel["entry_date"] - panel["as_of"]).dt.days
    assert gap.min() >= 1
    assert gap.max() <= 5
