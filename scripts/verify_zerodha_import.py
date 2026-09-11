"""
Verifies discover_zerodha_nifty_option_legs()/PositionStore.import_from_zerodha()
against a REAL Zerodha (kiteconnect) account.

Every field name and sign-convention this module relies on was cross-checked
against Zerodha's own published API docs, SDK source, and forum (see
nifty_live/position_store.py's module docstring for the specifics) -- but
none of that was verified end-to-end against a real account's actual
response from this environment. Run this FIRST against your real account,
the same way scripts/verify_find_atm.py and scripts/debug_breeze.py are the
first real-account checks for the Breeze side, before trusting
import_from_zerodha() with real position state.

Auth flow (kiteconnect, NOT the same flow as Breeze):
  1. Log in at https://kite.zerodha.com/connect/login?api_key=YOUR_API_KEY
     -- after login you're redirected to your registered redirect URL with
     a `request_token` query parameter.
  2. Exchange that for an access_token (valid for the trading day only):
       from kiteconnect import KiteConnect
       kite = KiteConnect(api_key="...")
       data = kite.generate_session("request_token_from_step_1", api_secret="...")
       access_token = data["access_token"]
  3. Set KITE_API_KEY / KITE_ACCESS_TOKEN as environment variables and run
     this script.

Run (from the repo root): python3 scripts/verify_zerodha_import.py

Not named test_*.py deliberately (same convention as verify_find_atm.py) --
requires a live session and a real held position to be useful, so it must
never be auto-collected by a bare `pytest` run.
"""

import os
import json

from kiteconnect import KiteConnect

from nifty_live.position_store import discover_zerodha_nifty_option_legs, PositionStore

API_KEY = os.environ.get("KITE_API_KEY", "")
ACCESS_TOKEN = os.environ.get("KITE_ACCESS_TOKEN", "")

if not (API_KEY and ACCESS_TOKEN):
    print("Missing credentials -- set KITE_API_KEY and KITE_ACCESS_TOKEN (see this script's docstring for the login flow).")
    raise SystemExit(1)

kite = KiteConnect(api_key=API_KEY)
kite.set_access_token(ACCESS_TOKEN)

print("=== STEP 1: raw kite.positions() ===")
positions_response = kite.positions()
print(json.dumps(positions_response, indent=2, default=str))
net_positions = positions_response.get("net", [])
print(f"\n{len(net_positions)} net position(s), {sum(1 for p in net_positions if p.get('quantity', 0) != 0)} currently open (nonzero quantity).")

print("\n=== STEP 2: kite.instruments('NFO') -- fetching full dump (can take a few seconds) ===")
instruments = kite.instruments("NFO")
print(f"{len(instruments)} NFO instruments returned.")
nifty_options = [i for i in instruments if i.get("segment") == "NFO-OPT" and str(i.get("tradingsymbol", "")).startswith("NIFTY")]
print(f"{len(nifty_options)} of those are NIFTY-prefixed options (before the digit-after-prefix filter this module applies).")
if nifty_options:
    print("Sample row (first match) -- check this matches the CONFIRMED schema in position_store.py's docstring:")
    print(json.dumps(nifty_options[0], indent=2, default=str))

print("\n=== STEP 3: discover_zerodha_nifty_option_legs() ===")
discovered = discover_zerodha_nifty_option_legs(kite)
if not discovered:
    print("No open NIFTY options legs discovered -- either you're flat, or something in the "
          "filtering (segment/prefix/instrument_type) didn't match what STEP 1/2 printed above. "
          "Compare against the raw responses before assuming this is wrong.")
else:
    for leg in discovered:
        print(f"  {leg.tradingsymbol}: expiry={leg.expiry} strike={leg.strike} right={leg.right.value} "
              f"direction={leg.direction.value} net_qty={leg.net_quantity} lots={leg.lots} "
              f"avg_price={leg.average_price} product={leg.product}")

print("\n=== STEP 4: sanity checks ===")
print("For each leg above, manually confirm against the Kite web/app position screen:")
print("  - direction (long/short) matches what you actually hold")
print("  - strike/expiry/right match the actual contract")
print("  - lots matches your actual lot count (net_qty / lot_size)")
print("If any of these disagree, do NOT trust import_from_zerodha() yet -- open an issue/note")
print("here with the exact raw STEP 1/2 output so the join logic can be corrected against real data.")
