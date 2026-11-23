"""Unit tests for `broker.orders.compute_rebalance`.

The rebalance math is pure — no Alpaca client involved — so these tests
cover the only logic that needs to be correct before we ever go live.
"""

from __future__ import annotations

from broker.orders import compute_rebalance, PlannedOrder, TARGET_WEIGHT


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
