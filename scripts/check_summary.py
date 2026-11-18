#!/usr/bin/env python3
"""Exit non-zero if a weekly summary.json is missing or has non-ok strategies.

Used by .github/workflows/weekly_run.yml (alert step) and watchdog.yml.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: check_summary.py path/to/summary.json", file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"FAIL: {path} is missing", file=sys.stderr)
        return 1

    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"FAIL: {path} is not valid JSON: {exc}", file=sys.stderr)
        return 1

    strategies = summary.get("strategies") or {}
    if not strategies:
        print(f"FAIL: {path} has no strategies recorded", file=sys.stderr)
        return 1

    bad = {k: v.get("status") for k, v in strategies.items() if v.get("status") != "ok"}
    if bad:
        print(f"FAIL: non-ok strategies in {path}: {bad}", file=sys.stderr)
        return 1

    print(f"OK: all strategies ok in {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
