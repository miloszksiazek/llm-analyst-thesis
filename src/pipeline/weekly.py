"""
Weekly pipeline entry point.

Usage:
    PYTHONPATH=src python -m pipeline.weekly --run-date 2026-11-22

For a given `run_date` (a Sunday, or any date for backfills):
    1. Build the data package → data/weekly/<run_date>/package.json (+ meta, sha256).
    2. Run claude_web and claude_noweb in parallel (n_runs from config/settings.yaml,
       default 3; effort=high; see src/agent/run.py).
    3. Produce the XGBoost ranking from the saved provisional/frozen model.
    4. Write summary.json with picks per strategy, costs, validation status and
       the package SHA-256.

Guarantees
----------
* Never overwrites an existing data/weekly/<run_date>/ folder – aborts first.
* On any step failure the partial artifacts are kept, an error.json is written
  with the traceback, and the process exits with a non-zero status so the
  GitHub Actions workflow fails and sends an email.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import logging
import os
import subprocess
import sys
import traceback
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from agent.run         import run_variant, VARIANTS
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


def _pick_model() -> Path:
    """Prefer the frozen model if present; otherwise fall back to provisional."""
    if _FROZEN_MODEL.exists():
        return _FROZEN_MODEL
    if _PROVISIONAL_MODEL.exists():
        return _PROVISIONAL_MODEL
    raise FileNotFoundError(
        f"No XGBoost model found. Expected {_FROZEN_MODEL} or {_PROVISIONAL_MODEL}"
    )


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str, ensure_ascii=False),
                    encoding="utf-8")


# ── Steps ────────────────────────────────────────────────────────────────────

def _step_package(run_date: date, out_dir: Path) -> Path:
    prev_dirs = sorted((out_dir.parent).glob("*/package.json"))
    prev = prev_dirs[-1] if prev_dirs else None
    pkg = build_package(run_date=run_date, out_dir=out_dir.parent, prev_package=prev)
    verify_package(pkg)
    return pkg


def _step_agent(variant: str, package_path: Path, run_date: date, out_dir: Path) -> dict:
    return run_variant(variant, package_path, run_date, out_dir)


def _step_xgb(package_path: Path, out_dir: Path) -> dict:
    model_path = _pick_model()
    model, meta = load_model(model_path)
    pkg = json.loads(package_path.read_text(encoding="utf-8"))
    as_of = pd.Timestamp(pkg["meta"]["as_of"])
    # Ensure the cached Yahoo history is present and current (data/raw/
    # is gitignored, so it may be missing on a clean GitHub runner).
    fetch_history(force=True)
    bundle = load_bundle()
    ranking = xgb_rank(as_of, model, meta, bundle=bundle)

    xgb_dir = out_dir / "xgb"
    xgb_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {"rank": int(r), "ticker": row.ticker, "score": float(row.score)}
        for r, row in zip(ranking.index, ranking.itertuples(index=False))
    ]
    out = {
        "model_path":      str(model_path.relative_to(_ROOT)),
        "model_status":    meta.get("status", "frozen" if model_path == _FROZEN_MODEL else "provisional"),
        "trained_through": meta.get("trained_through"),
        "git_commit_at_training": meta.get("git_commit"),
        "as_of":           str(as_of.date()),
        "picks":           [r["ticker"] for r in rows[:4]],
        "ranking":         rows,
    }
    _write_json(xgb_dir / "prediction.json", out)
    return out


# ── Orchestrator ─────────────────────────────────────────────────────────────

def run_week(run_date: date, root: Path = _ROOT) -> int:
    """Run the full pipeline for `run_date`. Returns an exit code (0 ok, 1 fail)."""
    weekly_dir = root / "data" / "weekly"
    out_dir    = weekly_dir / str(run_date)

    if out_dir.exists():
        logger.error("Refusing to overwrite existing folder %s", out_dir)
        return 2

    out_dir.mkdir(parents=True, exist_ok=False)
    errors: list[dict] = []

    # 1) Package
    package_path: Path | None = None
    package_sha256: str | None = None
    package_source: str | None = None
    try:
        package_path = _step_package(run_date, out_dir)
        package_sha256 = (out_dir / "package.sha256").read_text(encoding="utf-8").strip()
        meta_file = out_dir / "package_meta.json"
        if meta_file.exists():
            package_source = json.loads(meta_file.read_text()).get("price_source")
        logger.info("Package built: %s (sha=%s..., source=%s)",
                    package_path, package_sha256[:12], package_source)
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
    summary = {
        "run_date":       str(run_date),
        "finished_at":    datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit":     _git_commit(),
        "package": {
            "path":          str(package_path.relative_to(root)) if package_path else None,
            "sha256":        package_sha256,
            "price_source":  package_source,
        },
        "strategies":     {},
        "xgb":            xgb_result,
        "errors":         errors,
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
        return 1
    return 0


# ── CLI ──────────────────────────────────────────────────────────────────────

def _parse_run_date(raw: str | None) -> date:
    if raw:
        return date.fromisoformat(raw)
    # Default: today (the workflow schedules on Sunday evening UTC).
    return date.today()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    p = argparse.ArgumentParser()
    p.add_argument("--run-date", help="YYYY-MM-DD (defaults to today UTC)")
    args = p.parse_args()
    run_date = _parse_run_date(args.run_date)
    logger.info("Starting weekly pipeline for %s", run_date)
    rc = run_week(run_date)
    logger.info("Pipeline finished with exit code %d", rc)
    return rc


if __name__ == "__main__":
    sys.exit(main())
