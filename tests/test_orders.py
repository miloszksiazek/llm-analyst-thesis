"""Unit tests for `broker.orders`.

The rebalance math is pure — no Alpaca client involved — so these tests
cover the only logic that needs to be correct before we ever go live.
Also covers the dry-run routing and the live-only idempotency gate.
"""

from __future__ import annotations

import json
from pathlib import Path

from broker.orders import (
    compute_rebalance, PlannedOrder, TARGET_WEIGHT,
    _dryrun_ledger_path, _has_live_ledger, _needs_work, _live_ledger_path,
)


PICKS = ["AAPL", "MSFT", "JPM", "GOOGL"]


def _by_symbol(plan: list[PlannedOrder]) -> dict[str, PlannedOrder]:
    return {o.symbol: o for o in plan}


def test_flat_account_buys_target_in_each_pick():
    plan = compute_rebalance({}, PICKS, equity=100_000.0)
    bysym = _by_symbol(plan)
    assert set(bysym) == set(PICKS)
    target = 100_000.0 * TARGET_WEIGHT
    for sym in PICKS:
        assert bysym[sym].side == "buy"
        assert bysym[sym].reason == "new"
        assert bysym[sym].notional == target


def test_existing_pick_at_target_generates_no_order():
    target = 100_000.0 * TARGET_WEIGHT
    positions = {sym: target for sym in PICKS}
    plan = compute_rebalance(positions, PICKS, equity=100_000.0)
    assert plan == []


def test_overweight_pick_is_trimmed():
    target = 25_000.0
    # AAPL overweight, others at target
    positions = {"AAPL": 30_000.0, "MSFT": target, "JPM": target, "GOOGL": target}
    plan = compute_rebalance(positions, PICKS, equity=100_000.0)
    assert len(plan) == 1
    (o,) = plan
    assert o.symbol == "AAPL"
    assert o.side == "sell"
    assert o.reason == "trim"
    assert o.notional == 5_000.0


def test_underweight_pick_gets_top_up():
    target = 25_000.0
    positions = {"AAPL": 20_000.0, "MSFT": target, "JPM": target, "GOOGL": target}
    plan = compute_rebalance(positions, PICKS, equity=100_000.0)
    assert len(plan) == 1
    (o,) = plan
    assert o.symbol == "AAPL"
    assert o.side == "buy"
    assert o.reason == "top_up"
    assert o.notional == 5_000.0


def test_non_pick_positions_are_liquidated():
    positions = {"NVDA": 30_000.0, "AAPL": 25_000.0, "MSFT": 25_000.0,
                 "JPM": 25_000.0, "GOOGL": 25_000.0}
    plan = compute_rebalance(positions, PICKS, equity=130_000.0)
    bysym = _by_symbol(plan)
    # NVDA is liquidated; picks all underweight relative to new 32.5k target.
    assert bysym["NVDA"].side == "sell"
    assert bysym["NVDA"].reason == "liquidate"
    assert bysym["NVDA"].notional == 30_000.0
    for sym in PICKS:
        assert bysym[sym].side == "buy"
        assert bysym[sym].reason == "top_up"
        assert bysym[sym].notional == 130_000.0 * TARGET_WEIGHT - 25_000.0


def test_tiny_drift_is_ignored():
    # 1-cent drift per pick must not generate an order (below MIN_NOTIONAL).
    target = 25_000.0
    positions = {sym: target - 0.01 for sym in PICKS}
    plan = compute_rebalance(positions, PICKS, equity=100_000.0)
    assert plan == []


def test_full_rotation_liquidates_four_and_buys_four():
    # Prior-week picks {A,B,C,D} fully replaced by {E,F,G,H}.
    prev = ["A", "B", "C", "D"]
    new = ["E", "F", "G", "H"]
    positions = {sym: 25_000.0 for sym in prev}
    plan = compute_rebalance(positions, new, equity=100_000.0)
    bysym = _by_symbol(plan)
    assert set(bysym) == set(prev) | set(new)
    for sym in prev:
        assert bysym[sym].reason == "liquidate"
        assert bysym[sym].side == "sell"
        assert bysym[sym].notional == 25_000.0
    for sym in new:
        assert bysym[sym].reason == "new"
        assert bysym[sym].side == "buy"
        assert bysym[sym].notional == 25_000.0


def test_small_non_pick_leftover_is_ignored():
    # A 2-dollar stale position in a non-pick is below MIN_NOTIONAL; skip.
    positions = {"STALE": 2.0, "AAPL": 25_000.0, "MSFT": 25_000.0,
                 "JPM": 25_000.0, "GOOGL": 25_000.0}
    plan = compute_rebalance(positions, PICKS, equity=100_002.0)
    # Only tiny underweight top-ups allowed (0.50 per pick = below min),
    # STALE below min => no order expected.
    assert plan == []


# ── Dry-run routing + live-only idempotency ─────────────────────────────────

def _write(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def test_dryrun_path_lives_under_scratch(tmp_path, monkeypatch):
    # The path helper is deterministic — just assert the layout.
    p = _dryrun_ledger_path("2026-11-22", "claude_web")
    assert p.parent.parent.name == "orders_dryrun"
    assert p.parent.name == "2026-11-22"
    assert p.name == "claude_web.json"
    assert "data/scratch" in str(p)


def test_live_ledger_blocks_rerun(tmp_path):
    run_dir = tmp_path / "2026-11-22"
    live_path = _live_ledger_path(run_dir, "claude_web")
    _write(live_path, {"strategy": "claude_web", "live": True, "picks": []})
    assert _has_live_ledger(run_dir, "claude_web") is True


def test_dryrun_file_in_weekly_does_not_block_rerun(tmp_path):
    # A leftover dry-run ledger (live=false) mistakenly stored in
    # data/weekly/<run>/orders/ must not be treated as a submitted ledger.
    run_dir = tmp_path / "2026-11-22"
    _write(_live_ledger_path(run_dir, "claude_web"),
           {"strategy": "claude_web", "live": False, "picks": []})
    assert _has_live_ledger(run_dir, "claude_web") is False


def test_needs_work_true_when_no_live_ledgers(tmp_path):
    run_dir = tmp_path / "2026-11-22"
    run_dir.mkdir()
    summary = {"strategies": {"claude_web": {"status": "ok"},
                               "claude_noweb": {"status": "ok"},
                               "xgb": {"status": "ok"}}}
    assert _needs_work(run_dir, summary) is True


def test_needs_work_false_when_all_live_ledgers_present(tmp_path):
    run_dir = tmp_path / "2026-11-22"
    for s in ("claude_web", "claude_noweb", "xgb"):
        _write(_live_ledger_path(run_dir, s),
               {"strategy": s, "live": True, "picks": []})
    summary = {"strategies": {"claude_web": {}, "claude_noweb": {}, "xgb": {}}}
    assert _needs_work(run_dir, summary) is False


def test_needs_work_ignores_dryrun_file_at_live_path(tmp_path):
    # A stray live=false file at the live path still counts as missing.
    run_dir = tmp_path / "2026-11-22"
    for s in ("claude_web", "claude_noweb"):
        _write(_live_ledger_path(run_dir, s),
               {"strategy": s, "live": True, "picks": []})
    _write(_live_ledger_path(run_dir, "xgb"),
           {"strategy": "xgb", "live": False, "picks": []})
    summary = {"strategies": {"claude_web": {}, "claude_noweb": {}, "xgb": {}}}
    assert _needs_work(run_dir, summary) is True
