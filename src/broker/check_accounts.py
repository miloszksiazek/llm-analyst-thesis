"""Read-only Alpaca account diagnostics. No orders placed."""

import os
from datetime import date, timedelta

from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetCalendarRequest

load_dotenv()

ACCOUNTS = {
    "claude_web": {
        "key": os.getenv("ALPACA_KEY_CLAUDE_WEB"),
        "secret": os.getenv("ALPACA_SECRET_CLAUDE_WEB"),
    },
    "claude_noweb": {
        "key": os.getenv("ALPACA_KEY_CLAUDE_NOWEB"),
        "secret": os.getenv("ALPACA_SECRET_CLAUDE_NOWEB"),
    },
    "xgb": {
        "key": os.getenv("ALPACA_KEY_XGB"),
        "secret": os.getenv("ALPACA_SECRET_XGB"),
    },
}

PAPER_URL = "https://paper-api.alpaca.markets"


def print_account_info(name: str, client: TradingClient) -> str:
    acct = client.get_account()
    # fractional_trading_enabled is not in the Account model for paper accounts;
    # Alpaca paper accounts support fractional trading by default.
    fractional = getattr(acct, "fractional_trading_enabled", "not in API response (paper: enabled by default)")
    print(f"\n--- {name} ---")
    print(f"  account_number          : {acct.account_number}")
    print(f"  status                  : {acct.status}")
    print(f"  cash                    : {acct.cash}")
    print(f"  equity                  : {acct.equity}")
    print(f"  fractional_trading      : {fractional}")
    return acct.account_number


def main() -> None:
    clients = {
        name: TradingClient(
            api_key=creds["key"],
            secret_key=creds["secret"],
            paper=True,
            url_override=PAPER_URL,
        )
        for name, creds in ACCOUNTS.items()
    }

    # ── Account info ──────────────────────────────────────────────────────────
    account_numbers: dict[str, str] = {}
    for name, client in clients.items():
        account_numbers[name] = print_account_info(name, client)

    nums = list(account_numbers.values())
    if len(set(nums)) == len(nums):
        print(f"\n[OK] All three account numbers are distinct.")
        for n, v in account_numbers.items():
            print(f"  {n}: {v}")
    else:
        print(f"\n[WARNING] Duplicate account numbers detected: {account_numbers}")

    # ── Market clock ─────────────────────────────────────────────────────────
    first_client = next(iter(clients.values()))
    clock = first_client.get_clock()
    print("\n--- Market Clock ---")
    print(f"  is_open    : {clock.is_open}")
    print(f"  next_open  : {clock.next_open}")
    print(f"  next_close : {clock.next_close}")

    # ── Trading calendar & missing Mondays ───────────────────────────────────
    cal_start = date(2026, 12, 7)
    cal_end = date(2027, 6, 21)
    calendar = first_client.get_calendar(
        GetCalendarRequest(start=cal_start, end=cal_end)
    )
    # Calendar dates may come back as date or datetime objects
    trading_days = {
        c.date.date() if hasattr(c.date, "date") else c.date for c in calendar
    }

    # Collect every Monday in [cal_start, cal_end]
    d = cal_start
    while d.weekday() != 0:  # advance to first Monday
        d += timedelta(days=1)

    missing: list[date] = []
    while d <= cal_end:
        if d not in trading_days:
            missing.append(d)
        d += timedelta(weeks=1)

    print(f"\n--- Mondays NOT a trading day ({cal_start} to {cal_end}) ---")
    if missing:
        for m in missing:
            print(f"  {m}")
    else:
        print("  (none — all Mondays are trading days)")


if __name__ == "__main__":
    main()
