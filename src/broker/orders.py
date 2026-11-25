"""
Monday broker job: rebalance the three paper accounts to the picks published
by the most recent Sunday run.

Usage
-----
    PYTHONPATH=src python -m broker.orders [--run-date YYYY-MM-DD] [--dry-run]

Behaviour
---------
* Locates `data/weekly/<Sunday>/summary.json` (latest if `--run-date` omitted).
* Refuses to run if the Alpaca market clock reports `is_open=False`.
* For each strategy in {claude_web, claude_noweb, xgb}:
    - skip if a LIVE ledger already exists at
      `data/weekly/<Sunday>/orders/<strategy>.json` with `live=true`
      (idempotent — dry-run ledgers never block a later live submission);
    - skip if the strategy status is `fallback` (keep last week's positions)
      or `failed` (no decision) — write a short note either way;
    - otherwise rebalance the account so each of the 4 picks sits at exactly
      25% of the account's current equity.
* Rebalance arithmetic: notional diff relative to each pick's current market
  value; non-pick positions are fully liquidated via `close_position`. See
  `compute_rebalance()`. Orders are fractional, market, time_in_force=day.
* After submitting, polls each order briefly for the fill, then writes
  `data/weekly/<Sunday>/orders/<strategy>.json` with the plan, the submitted
  orders, and whatever fill status came back.

Dry-run
-------
Dry-run writes to `data/scratch/orders_dryrun/<run_date>/<strategy>.json`
(gitignored under `data/scratch/`). It never touches `data/weekly/` so an
experimental dry-run can't be mistaken for a submitted ledger and can't
block a later live rebalance.

Safety flag
-----------
`config/settings.yaml:broker_live` must be true before any real orders are
submitted. While it is false, the script runs through every step in
`--dry-run` mode (orders computed, nothing sent) regardless of the CLI flag.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(_ROOT / "src"))

_SETTINGS = _ROOT / "config" / "settings.yaml"
_WEEKLY = _ROOT / "data" / "weekly"
PAPER_URL = "https://paper-api.alpaca.markets"

STRATEGIES = ("claude_web", "claude_noweb", "xgb")

# Per-pick weight and the dollar-threshold below which we don't bother placing
# a rebalance order (Alpaca's notional minimum is $1; we use $5 to avoid the
# round-trip on cents-sized drifts after a fresh fill).
TARGET_WEIGHT = 0.25
MIN_NOTIONAL = 5.0


# ── Pure helpers (unit-testable) ────────────────────────────────────────────

@dataclass(frozen=True)
class PlannedOrder:
    symbol: str
    side: str            # "buy" | "sell"
    notional: float      # USD
    reason: str          # "liquidate" | "trim" | "top_up" | "new"


def compute_rebalance(
    current_positions: dict[str, float],
    picks: list[str],
    equity: float,
    target_weight: float = TARGET_WEIGHT,
    min_notional: float = MIN_NOTIONAL,
) -> list[PlannedOrder]:
    """Compute the notional rebalance orders.

    `current_positions` : {symbol: current_market_value_usd}. Only symbols the
        account currently holds; absent symbols are assumed flat.
    `picks` : the 4 target tickers (any length is accepted, caller validates).
    `equity` : account equity in USD before any orders.

    Non-picks that are held → one full-liquidation sell order per symbol.
    Picks → one order per symbol closing the gap to `equity * target_weight`.
    Orders below `min_notional` are dropped.
    """
    target = equity * target_weight
    orders: list[PlannedOrder] = []

    for sym, mv in current_positions.items():
        if sym in picks:
            continue
        if mv > min_notional:
            orders.append(PlannedOrder(sym, "sell", round(mv, 2), "liquidate"))

    for sym in picks:
        cur = float(current_positions.get(sym, 0.0))
        diff = target - cur
        if diff > min_notional:
            reason = "new" if cur == 0.0 else "top_up"
            orders.append(PlannedOrder(sym, "buy", round(diff, 2), reason))
        elif -diff > min_notional:
            orders.append(PlannedOrder(sym, "sell", round(-diff, 2), "trim"))

    return orders


def _load_settings() -> dict:
    return yaml.safe_load(_SETTINGS.read_text(encoding="utf-8")) or {}


def _latest_run_dir(after: date | None = None) -> Path | None:
    if not _WEEKLY.exists():
        return None
    candidates = sorted(p for p in _WEEKLY.iterdir() if p.is_dir())
    for p in reversed(candidates):
        try:
            d = date.fromisoformat(p.name)
        except ValueError:
            continue
        if after is not None and d < after:
            return None
        return p
    return None


def _env_keys(strategy: str) -> tuple[str, str]:
    key = os.getenv(f"ALPACA_KEY_{strategy.upper()}")
    secret = os.getenv(f"ALPACA_SECRET_{strategy.upper()}")
    if not key or not secret:
        raise RuntimeError(
            f"Missing ALPACA_KEY_{strategy.upper()} / ALPACA_SECRET_{strategy.upper()}"
        )
    return key, secret


# ── Alpaca I/O (thin wrappers so the core logic stays test-friendly) ────────

def _make_client(strategy: str):
    from alpaca.trading.client import TradingClient
    key, secret = _env_keys(strategy)
    return TradingClient(api_key=key, secret_key=secret, paper=True,
                         url_override=PAPER_URL)


def _market_is_open(client) -> bool:
    return bool(client.get_clock().is_open)


def _snapshot_account(client) -> tuple[float, dict[str, float]]:
    acct = client.get_account()
    equity = float(acct.equity)
    positions = {p.symbol: float(p.market_value) for p in client.get_all_positions()}
    return equity, positions


def _submit(client, order: PlannedOrder) -> dict:
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    side = OrderSide.BUY if order.side == "buy" else OrderSide.SELL
    req = MarketOrderRequest(
        symbol=order.symbol,
        notional=order.notional,
        side=side,
        time_in_force=TimeInForce.DAY,
    )
    submitted = client.submit_order(req)
    return {"id": str(submitted.id), "status": str(submitted.status)}


def _poll_fill(client, order_id: str, timeout_s: int = 30) -> dict:
    from alpaca.trading.enums import OrderStatus
    terminal = {OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED,
                OrderStatus.EXPIRED, OrderStatus.DONE_FOR_DAY}
    for _ in range(timeout_s):
        o = client.get_order_by_id(order_id)
        if o.status in terminal:
            return {
                "id": str(o.id),
                "status": str(o.status),
                "filled_qty": float(o.filled_qty) if o.filled_qty else 0.0,
                "filled_avg_price": float(o.filled_avg_price) if o.filled_avg_price else None,
                "filled_at": o.filled_at.isoformat() if o.filled_at else None,
            }
        time.sleep(1)
    o = client.get_order_by_id(order_id)
    return {
        "id": str(o.id),
        "status": str(o.status),
        "filled_qty": float(o.filled_qty) if o.filled_qty else 0.0,
        "filled_avg_price": float(o.filled_avg_price) if o.filled_avg_price else None,
        "filled_at": o.filled_at.isoformat() if o.filled_at else None,
        "note": "timeout before terminal status",
    }


# ── Per-strategy orchestration ──────────────────────────────────────────────

def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def _live_ledger_path(run_dir: Path, strategy: str) -> Path:
    return run_dir / "orders" / f"{strategy}.json"


def _dryrun_ledger_path(run_date: str, strategy: str) -> Path:
    return _ROOT / "data" / "scratch" / "orders_dryrun" / run_date / f"{strategy}.json"


def _has_live_ledger(run_dir: Path, strategy: str) -> bool:
    """True iff a LIVE ledger already exists for this strategy.

    Only ledgers with `live=true` block a later submission. Dry-run ledgers
    (if they ever end up here) and corrupt files are treated as absent.
    """
    p = _live_ledger_path(run_dir, strategy)
    if not p.exists():
        return False
    try:
        return bool(json.loads(p.read_text(encoding="utf-8")).get("live", False))
    except (json.JSONDecodeError, OSError):
        return False


def process_strategy(
    strategy: str,
    picks: list[str] | None,
    status: str,
    run_dir: Path,
    live: bool,
) -> dict:
    """Rebalance one paper account. Returns the ledger record written to disk."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    if live:
        out_path = _live_ledger_path(run_dir, strategy)
        if _has_live_ledger(run_dir, strategy):
            return {"strategy": strategy, "skipped": "live orders file already exists",
                    "path": str(out_path.relative_to(_ROOT))}
    else:
        out_path = _dryrun_ledger_path(run_dir.name, strategy)

    record: dict = {
        "strategy":     strategy,
        "run_dir":      run_dir.name,
        "submitted_at": now,
        "picks":        picks,
        "status":       status,
        "live":         live,
    }

    if status == "fallback":
        record["note"] = "strategy used fallback – no orders placed, positions held"
        _write_json(out_path, record)
        return record
    if status != "ok" or not picks:
        record["note"] = f"strategy status={status!r}, no orders placed"
        _write_json(out_path, record)
        return record

    client = _make_client(strategy)
    equity, positions = _snapshot_account(client)
    plan = compute_rebalance(positions, picks, equity)

    record.update({
        "equity_before":    equity,
        "positions_before": positions,
        "orders_planned":   [p.__dict__ for p in plan],
    })

    if not live:
        record["note"] = "dry-run: orders computed, nothing submitted"
        _write_json(out_path, record)
        return record

    # Liquidate non-picks first so the resulting cash is available for the
    # subsequent pick buys. (Alpaca paper fills these in milliseconds.)
    submitted: list[dict] = []
    for p in plan:
        if p.reason == "liquidate":
            submitted.append({**p.__dict__, **_submit(client, p)})
    # Then adjust picks.
    for p in plan:
        if p.reason != "liquidate":
            submitted.append({**p.__dict__, **_submit(client, p)})

    fills = [_poll_fill(client, s["id"]) for s in submitted]
    record["orders_submitted"] = submitted
    record["fills"] = fills
    _write_json(out_path, record)
    return record


# ── Entry point ────────────────────────────────────────────────────────────

def _latest_incomplete_run_dir() -> Path | None:
    """Return the latest `data/weekly/<Sunday>/` whose summary.json still has
    at least one strategy without a live orders file. Returns None if every
    dated run dir is either fully submitted, missing summary.json, or
    unreadable.
    """
    if not _WEEKLY.exists():
        return None
    candidates = sorted(p for p in _WEEKLY.iterdir() if p.is_dir())
    for p in reversed(candidates):
        try:
            date.fromisoformat(p.name)
        except ValueError:
            continue
        summary_path = p / "summary.json"
        if not summary_path.exists():
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if _needs_work(p, summary):
            return p
    return None


def _find_run_dir(run_date: str | None) -> Path:
    if run_date:
        p = _WEEKLY / run_date
        if not p.exists():
            raise FileNotFoundError(f"No weekly dir for {run_date}: {p}")
        return p
    # External scheduler fires repeatedly; pick the latest Sunday still
    # missing a live orders file so retries after a partial submission land
    # on the same dir. Fall back to the latest dir overall only when every
    # recent run is already fully submitted – the caller then exits cleanly
    # via the _needs_work shortcut.
    p = _latest_incomplete_run_dir()
    if p is not None:
        return p
    p = _latest_run_dir()
    if p is None:
        raise FileNotFoundError("No completed weekly runs found in data/weekly/")
    return p


def _needs_work(run_dir: Path, summary: dict) -> bool:
    """True iff at least one strategy still has no LIVE ledger.

    Dry-run ledgers live in `data/scratch/` and never count here. A live
    ledger is one whose JSON contains `"live": true`; anything else
    (including a leftover dry-run file accidentally in `data/weekly/orders/`)
    is treated as absent so a later submission can run.
    """
    for name in STRATEGIES:
        if name not in summary.get("strategies", {}):
            continue
        if not _has_live_ledger(run_dir, name):
            return True
    return False


def run(run_date: str | None, dry_run: bool) -> int:
    run_dir = _find_run_dir(run_date)
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        logger.error("No summary.json in %s", run_dir)
        return 1
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    if not _needs_work(run_dir, summary):
        logger.info("All orders files already present under %s – nothing to do",
                    run_dir.relative_to(_ROOT))
        return 0

    settings = _load_settings()
    live = bool(settings.get("broker_live", False)) and not dry_run
    if not live:
        logger.info("Running in DRY-RUN mode (broker_live=%s, --dry-run=%s)",
                    settings.get("broker_live"), dry_run)

    # Market-open gate. Use any one account's client — paper clocks are shared.
    try:
        clock_client = _make_client("xgb")
    except RuntimeError as exc:
        logger.error("Cannot reach Alpaca: %s", exc)
        return 1
    if not _market_is_open(clock_client):
        logger.info("Market is closed – exiting without sending orders.")
        return 0

    results = []
    for name in STRATEGIES:
        strat = summary.get("strategies", {}).get(name)
        if strat is None:
            logger.info("Strategy %s not in summary – skipping", name)
            continue
        try:
            rec = process_strategy(
                strategy=name,
                picks=strat.get("picks"),
                status=strat.get("status", "unknown"),
                run_dir=run_dir,
                live=live,
            )
            results.append(rec)
            logger.info("%s: %s", name, {k: v for k, v in rec.items()
                                          if k in ("skipped", "note", "status")})
        except Exception:
            logger.exception("Failed to process %s", name)
            results.append({"strategy": name, "error": True})
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--run-date", help="Sunday run_date (YYYY-MM-DD). Defaults to the latest.")
    p.add_argument("--dry-run", action="store_true",
                   help="Compute orders but do not submit (forces dry-run regardless of settings).")
    args = p.parse_args()
    return run(args.run_date, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
