"""
Weekly pipeline entry point.

Usage:
    PYTHONPATH=src python -m pipeline.weekly \
        --run-date 2026-11-22 [--mode production|test] [--allow-late]

For a given `run_date`:
    1. Build the data package → package.json (+ meta, sha256).
    2. Run claude_web and claude_noweb in parallel (n_runs from
       config/settings.yaml, default 3; effort=high).
    3. Produce the XGBoost ranking from the saved provisional/frozen model.
    4. Write summary.json: picks per strategy, costs, validation status,
       package SHA-256, data warnings.

Atomicity
---------
All artifacts are written to a temporary directory. On full success the
directory is moved to data/weekly/<run_date>/. If any step failed, it is
moved to data/failed/<run_date>_<UTC timestamp>/ instead, so re-runs of
the same run_date remain possible.

Guards
------
* Refuses to run if data/weekly/<run_date>/ already exists and every
  strategy there is 'ok' – exits 0 with a short log line.
* Refuses to run after the first NYSE open on-or-after `run_date` unless
  --allow-late is passed. When overridden, summary.json records
  "late_run": true.

Modes
-----
* production (default) – outputs land in data/weekly/ or data/failed/.
* test                – outputs always land in
                        data/test_runs/<run_date>_<UTC timestamp>/
                        and NEVER in data/weekly/.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import logging
import shutil
import subprocess
import sys
import tempfile
import traceback
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from agent.run         import run_variant, VARIANTS
from data.calendar     import is_past_next_open
from data.package      import build_package, verify_package
from xgb.features      import load_bundle
from xgb.fetch_history import fetch as fetch_history
from xgb.predict       import load_model, rank as xgb_rank

logger = logging.getLogger(__name__)

_PROVISIONAL_MODEL = _ROOT / "models" / "xgb_provisional.json"
_FROZEN_MODEL      = _ROOT / "models" / "xgb_frozen.json"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_ROOT
        ).decode().strip()
    except Exception:
        return None


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _pick_model() -> Path:
    if _FROZEN_MODEL.exists():
        return _FROZEN_MODEL
    if _PROVISIONAL_MODEL.exists():
        return _PROVISIONAL_MODEL
    raise FileNotFoundError(
        f"No XGBoost model found. Expected {_FROZEN_MODEL} or {_PROVISIONAL_MODEL}"
    )


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )


def _move_atomic(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


# ── Already-done shortcut ────────────────────────────────────────────────────

def _already_complete(run_date: date, root: Path) -> bool:
    final = root / "data" / "weekly" / str(run_date) / "summary.json"
    if not final.exists():
        return False
    try:
        s = json.loads(final.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    strategies = s.get("strategies", {})
    if not strategies:
        return False
    return all(st.get("status") == "ok" for st in strategies.values())


# ── Steps ────────────────────────────────────────────────────────────────────

def _step_package(run_date: date, pkg_root: Path) -> tuple[Path, list[str], str | None, str | None]:
    """Build package into pkg_root/<run_date>/. Returns path, warnings, sha, source."""
    prev_dirs = sorted((_ROOT / "data" / "weekly").glob("*/package.json"))
    prev = prev_dirs[-1] if prev_dirs else None
    pkg = build_package(run_date=run_date, out_dir=pkg_root, prev_package=prev)
    verify_package(pkg)
    sha = (pkg.parent / "package.sha256").read_text(encoding="utf-8").strip()
    meta_file = pkg.parent / "package_meta.json"
    warnings: list[str] = []
    source: str | None = None
    if meta_file.exists():
        mf = json.loads(meta_file.read_text(encoding="utf-8"))
        warnings = list(mf.get("sanity_warnings") or [])
        source = mf.get("price_source")
    return pkg, warnings, sha, source


def _step_agent(variant: str, package_path: Path, run_date: date, out_dir: Path) -> dict:
    return run_variant(variant, package_path, run_date, out_dir)


def _step_xgb(package_path: Path, out_dir: Path) -> dict:
    model_path = _pick_model()
    model, meta = load_model(model_path)
    pkg = json.loads(package_path.read_text(encoding="utf-8"))
    as_of = pd.Timestamp(pkg["meta"]["as_of"])
    fetch_history(force=True)
    bundle = load_bundle()
    ranking = xgb_rank(as_of, model, meta, bundle=bundle)

    xgb_dir = out_dir / "xgb"
    xgb_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {"rank": int(r), "ticker": row.ticker, "score": float(row.score)}
        for r, row in zip(ranking.index, ranking.itertuples(index=False))
    ]
    model_status = meta.get(
        "status", "frozen" if model_path == _FROZEN_MODEL else "provisional"
    )
    out = {
        "model_path":             str(model_path.relative_to(_ROOT)),
        "model_status":           model_status,
        "trained_through":        meta.get("trained_through"),
        "git_commit_at_training": meta.get("git_commit"),
        "as_of":                  str(as_of.date()),
        "picks":                  [r["ticker"] for r in rows[:4]],
        "ranking":                rows,
    }
    _write_json(xgb_dir / "prediction.json", out)
    return out


# ── Orchestrator ─────────────────────────────────────────────────────────────

def run_week(
    run_date: date,
    mode: str = "production",
    allow_late: bool = False,
    root: Path = _ROOT,
    now_utc: datetime | None = None,
) -> int:
    """Run the pipeline. Returns an exit code (0 ok, 1 step failed, 2 refused)."""
    if mode not in ("production", "test"):
        raise ValueError(f"mode must be 'production' or 'test', got {mode!r}")

    # Already-done shortcut (production only)
    if mode == "production" and _already_complete(run_date, root):
        logger.info("data/weekly/%s already complete – nothing to do", run_date)
        return 0

    # Look-ahead guard
    late = is_past_next_open(run_date, now_utc=now_utc)
    if late and not allow_late:
        logger.error(
            "Refusing to run: current time is past the first NYSE open on-or-after "
            "%s. Pass --allow-late to override.", run_date,
        )
        return 2

    # Build into a temp directory first.
    tmp_root  = Path(tempfile.mkdtemp(prefix=f"weekly_{run_date}_", dir=str(root / "data" / "scratch")))
    out_dir   = tmp_root / str(run_date)
    out_dir.mkdir(parents=True, exist_ok=True)
    errors:   list[dict] = []
    warnings: list[str]  = []
    stamp = _utc_stamp()

    # 1) Package
    package_path: Path | None = None
    package_sha256: str | None = None
    package_source: str | None = None
    try:
        package_path, warnings, package_sha256, package_source = _step_package(run_date, tmp_root)
        logger.info("Package built: sha=%s source=%s warnings=%d",
                    package_sha256[:12] if package_sha256 else "?",
                    package_source, len(warnings))
    except Exception as exc:
        logger.exception("Package step failed")
        errors.append({"step": "package", "error": str(exc),
                       "traceback": traceback.format_exc()})

    # 2) Agents (parallel) – only if package succeeded
    agent_results: dict[str, dict | None] = {v: None for v in VARIANTS}
    if package_path is not None:
        with cf.ThreadPoolExecutor(max_workers=len(VARIANTS)) as ex:
            futures = {
                ex.submit(_step_agent, v, package_path, run_date, out_dir): v
                for v in VARIANTS
            }
            for fut in cf.as_completed(futures):
                v = futures[fut]
                try:
                    agent_results[v] = fut.result()
                except Exception as exc:
                    logger.exception("Agent %s failed", v)
                    errors.append({"step": f"agent/{v}", "error": str(exc),
                                   "traceback": traceback.format_exc()})

    # 3) XGBoost – only if package succeeded
    xgb_result: dict | None = None
    if package_path is not None:
        try:
            xgb_result = _step_xgb(package_path, out_dir)
        except Exception as exc:
            logger.exception("XGB step failed")
            errors.append({"step": "xgb", "error": str(exc),
                           "traceback": traceback.format_exc()})

    # 4) Summary
    summary: dict[str, Any] = {
        "run_date":      str(run_date),
        "mode":          mode,
        "finished_at":   datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit":    _git_commit(),
        "late_run":      bool(late and allow_late),
        "warnings":      warnings,
        "package": {
            "path":         f"data/weekly/{run_date}/package.json",
            "sha256":       package_sha256,
            "price_source": package_source,
        },
        "strategies":    {},
        "xgb":           xgb_result,
        "errors":        errors,
    }
    for v in VARIANTS:
        ar = agent_results[v]
        if ar is None:
            summary["strategies"][v] = {"status": "failed", "picks": None}
            continue
        if ar.get("fallback_applied", False):
            summary["strategies"][v] = {
                "status":           "fallback",
                "picks":            None,
                "n_valid_runs":     ar.get("n_valid_runs"),
                "n_runs_requested": ar.get("n_runs"),
                "total_cost_usd":   ar.get("total_cost_usd"),
            }
        else:
            summary["strategies"][v] = {
                "status":           "ok",
                "picks":            ar.get("picks"),
                "n_valid_runs":     ar.get("n_valid_runs"),
                "n_runs_requested": ar.get("n_runs"),
                "total_cost_usd":   ar.get("total_cost_usd"),
            }
    if xgb_result is not None:
        summary["strategies"]["xgb"] = {
            "status": "ok", "picks": xgb_result["picks"],
            "model":  xgb_result["model_path"],
        }
    else:
        summary["strategies"]["xgb"] = {"status": "failed", "picks": None}

    _write_json(out_dir / "summary.json", summary)
    if errors:
        _write_json(out_dir / "errors.json", errors)

    # 5) Decide final destination
    all_ok = (not errors) and all(
        st["status"] == "ok" for st in summary["strategies"].values()
    )
    if mode == "test":
        dest = root / "data" / "test_runs" / f"{run_date}_{stamp}"
        _move_atomic(out_dir, dest)
        shutil.rmtree(tmp_root, ignore_errors=True)
        logger.info("[test] outputs saved to %s", dest.relative_to(root))
        return 0 if all_ok else 1

    if all_ok:
        dest = root / "data" / "weekly" / str(run_date)
        if dest.exists():
            # Preempted by a concurrent writer; keep the fresh run on the side.
            dest = root / "data" / "failed" / f"{run_date}_{stamp}_duplicate"
    else:
        dest = root / "data" / "failed" / f"{run_date}_{stamp}"
    _move_atomic(out_dir, dest)
    shutil.rmtree(tmp_root, ignore_errors=True)
    logger.info("Outputs saved to %s", dest.relative_to(root))
    return 0 if all_ok else 1


# ── CLI ──────────────────────────────────────────────────────────────────────

def _parse_run_date(raw: str | None) -> date:
    if raw:
        return date.fromisoformat(raw)
    return date.today()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    p = argparse.ArgumentParser()
    p.add_argument("--run-date",  help="YYYY-MM-DD (defaults to today UTC)")
    p.add_argument("--mode",      default="production", choices=("production", "test"))
    p.add_argument("--allow-late", action="store_true",
                   help="Run even if the first NYSE open after run_date has passed")
    args = p.parse_args()
    run_date = _parse_run_date(args.run_date)
    logger.info("Starting weekly pipeline for %s (mode=%s)", run_date, args.mode)
    rc = run_week(run_date, mode=args.mode, allow_late=args.allow_late)
    logger.info("Pipeline finished with exit code %d", rc)
    return rc


if __name__ == "__main__":
    sys.exit(main())
