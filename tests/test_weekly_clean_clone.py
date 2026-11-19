"""
Regression: `run_week` on a clean clone where data/scratch does not yet exist.

The dress rehearsal on 2026-11-18 failed on GitHub because the runner's
`data/scratch/` directory is gitignored and absent, and `tempfile.mkdtemp`
does NOT create parent directories — it raised FileNotFoundError before any
logging. This test reproduces that exact precondition and asserts the fix
(src/pipeline/weekly.py:_ensure_runtime_dirs) holds.

The test does not hit the network. The three pipeline steps are monkeypatched
so run_week only exercises its directory-setup and bookkeeping path.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeline import weekly  # noqa: E402


@pytest.fixture
def clean_root(tmp_path: Path) -> Path:
    """A root with NO data/ subdirs (simulates `git clone` on a runner)."""
    assert not (tmp_path / "data").exists()
    return tmp_path


def _stub_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Short-circuit all network-touching steps."""

    def _fail_package(run_date, pkg_root):
        raise RuntimeError("stubbed: package step not exercised in this test")

    # Package raising stops the pipeline cleanly: agent + xgb steps are gated
    # on `package_path is not None` and are skipped.
    monkeypatch.setattr(weekly, "_step_package", _fail_package)
    # Pretend we're not past the next NYSE open regardless of wall clock.
    monkeypatch.setattr(weekly, "is_past_next_open", lambda rd, now_utc=None: False)


def test_run_week_creates_scratch_on_clean_clone(
    clean_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_steps(monkeypatch)

    # The guard: this used to raise FileNotFoundError before logging anything.
    rc = weekly.run_week(
        run_date=date(2026, 11, 22), mode="test", allow_late=False, root=clean_root,
    )

    # Pipeline reported a failure (package stub raised), but exited cleanly.
    assert rc == 1
    assert (clean_root / "data" / "scratch").is_dir()
    assert (clean_root / "data" / "raw").is_dir()

    # The run's artifacts were moved to data/test_runs/, never data/weekly/.
    test_runs = clean_root / "data" / "test_runs"
    assert test_runs.is_dir()
    runs = list(test_runs.iterdir())
    assert len(runs) == 1, f"expected exactly one test run, got {runs}"

    summary = json.loads((runs[0] / "summary.json").read_text(encoding="utf-8"))
    assert summary["mode"] == "test"
    assert any(e["step"] == "package" for e in summary["errors"]), summary["errors"]
    # Every strategy failed because the package failed.
    assert all(s["status"] == "failed" for s in summary["strategies"].values())

    assert not (clean_root / "data" / "weekly").exists()


def test_ensure_runtime_dirs_is_idempotent(clean_root: Path) -> None:
    weekly._ensure_runtime_dirs(clean_root)
    weekly._ensure_runtime_dirs(clean_root)  # must not raise
    assert (clean_root / "data" / "scratch").is_dir()
    assert (clean_root / "data" / "raw").is_dir()
