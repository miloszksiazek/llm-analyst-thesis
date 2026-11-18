"""Tests for scripts/check_summary.py.

The script is the alert gate for both weekly_run.yml and watchdog.yml: it
MUST exit non-zero when summary.json does not exist (e.g. the run failed
and its folder was moved to data/failed/, or the look-ahead guard aborted
before the folder could be created).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT   = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_summary.py"


def _run(path: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(path)],
        capture_output=True, text=True,
    )


def test_missing_file_exits_nonzero(tmp_path: Path) -> None:
    missing = tmp_path / "nope" / "summary.json"
    result = _run(missing)
    assert result.returncode != 0
    assert "missing" in (result.stderr + result.stdout).lower()


def test_all_ok_exits_zero(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({
        "strategies": {
            "claude_web":   {"status": "ok", "picks": ["AAPL", "MSFT", "V", "JPM"]},
            "claude_noweb": {"status": "ok", "picks": ["AAPL", "HD",   "V", "JPM"]},
            "xgb":          {"status": "ok", "picks": ["GS",   "HD",   "BA", "CAT"]},
        }
    }))
    assert _run(path).returncode == 0


def test_non_ok_strategy_exits_nonzero(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({
        "strategies": {
            "claude_web":   {"status": "fallback", "picks": None},
            "claude_noweb": {"status": "ok",       "picks": ["A","B","C","D"]},
            "xgb":          {"status": "ok",       "picks": ["E","F","G","H"]},
        }
    }))
    result = _run(path)
    assert result.returncode != 0
    assert "claude_web" in (result.stderr + result.stdout)


def test_empty_strategies_exits_nonzero(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"strategies": {}}))
    assert _run(path).returncode != 0
