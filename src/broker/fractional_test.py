"""
Fractional-trading smoke test.
  1. For every ticker in the draft universe: print tradable/fractionable/exchange.
  2. On the xgb paper account: buy $10 notional AAPL, confirm fractional fill,
     then close the position and confirm zero open positions.
Run only when the market is open.
"""

import os
import time
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrderByIdRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus

load_dotenv()

DRAFT_UNIVERSE = [
    "AAPL", "AMGN", "AMZN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS",
    "GOOGL", "GS", "HD", "HON", "IBM", "JNJ", "JPM", "KO", "MCD", "MMM",
    "MRK", "MSFT", "NKE", "NVDA", "PG", "SHW", "TRV", "UNH", "V", "WMT",
]

PAPER_URL = "https://paper-api.alpaca.markets"


def make_client(env_key: str, env_secret: str) -> TradingClient:
    return TradingClient(
        api_key=os.getenv(env_key),
        secret_key=os.getenv(env_secret),
        paper=True,
        url_override=PAPER_URL,
    )


def check_assets(client: TradingClient) -> None:
    print("=" * 60)
    print("Asset check for draft universe")
    print("=" * 60)
    not_found: list[str] = []
    not_fractionable: list[str] = []

    for ticker in DRAFT_UNIVERSE:
        try:
            asset = client.get_asset(ticker)
            fractionable = getattr(asset, "fractionable", False)
            tradable = getattr(asset, "tradable", False)
            exchange = getattr(asset, "exchange", "?")
            flag = ""
            if not fractionable:
                flag += " [NOT FRACTIONABLE]"
                not_fractionable.append(ticker)
            if not tradable:
                flag += " [NOT TRADABLE]"
            print(f"  {ticker:<6}  tradable={tradable}  fractionable={fractionable}  exchange={exchange}{flag}")
        except Exception as e:
            print(f"  {ticker:<6}  ERROR: {e}")
            not_found.append(ticker)

    print()
    if not_found:
        print(f"[WARN] Not found: {not_found}")
    if not_fractionable:
        print(f"[WARN] Not fractionable: {not_fractionable}")
    if not not_found and not not_fractionable:
        print("[OK] All 30 tickers found and fractionable.")


def fractional_buy_test(client: TradingClient) -> None:
    print("\n" + "=" * 60)
    print("Fractional buy test: $10 notional AAPL on xgb account")
    print("=" * 60)

    # Confirm market is open
    clock = client.get_clock()
    if not clock.is_open:
        print(f"[SKIP] Market is closed. Next open: {clock.next_open}")
        return

    # Place $10 notional market buy
    order_req = MarketOrderRequest(
        symbol="AAPL",
        notional=10,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
    )
    order = client.submit_order(order_req)
    print(f"Order submitted: id={order.id}  status={order.status}")

    # Wait for fill (poll up to 30 s)
    for _ in range(30):
        order = client.get_order_by_id(order.id)
        if order.status == OrderStatus.FILLED:
            break
        time.sleep(1)
    else:
        print(f"[WARN] Order not filled after 30 s — status: {order.status}")
        return

    filled_qty = float(order.filled_qty)
    fill_price = float(order.filled_avg_price)
    filled_at = order.filled_at
    print(f"  Filled qty   : {filled_qty:.6f} shares  (fractional = {filled_qty < 1})")
    print(f"  Fill price   : ${fill_price:.4f}")
    print(f"  Fill time    : {filled_at}")
    print(f"  Fill value   : ${filled_qty * fill_price:.4f}")

    # Close position
    print("\nClosing AAPL position …")
    client.close_position("AAPL")

    # Wait for close to settle (poll up to 15 s)
    for _ in range(15):
        time.sleep(1)
        try:
            pos = client.get_open_position("AAPL")
            # still exists, keep waiting
        except Exception:
            break  # position gone

    # Confirm zero open positions
    positions = client.get_all_positions()
    if len(positions) == 0:
        print("[OK] No open positions — account is flat.")
    else:
        print(f"[WARN] {len(positions)} open position(s) remain: {[p.symbol for p in positions]}")


def main() -> None:
    # Any account works for asset lookup
    any_client = make_client("ALPACA_KEY_XGB", "ALPACA_SECRET_XGB")
    check_assets(any_client)

    # Fractional buy/close on xgb only
    xgb_client = make_client("ALPACA_KEY_XGB", "ALPACA_SECRET_XGB")
    fractional_buy_test(xgb_client)


if __name__ == "__main__":
    main()
