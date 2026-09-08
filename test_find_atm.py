"""
Tests find_atm_strike() against your real Breeze account.

Set credentials as environment variables first:
  export BREEZE_API_KEY="..."
  export BREEZE_API_SECRET="..."
  export BREEZE_SESSION_TOKEN="..."

Run: python3 test_find_atm.py
"""

import os
import datetime as dt

from data_layer_breeze import NiftyOptionsDataBreeze
API_KEY = os.environ.get("BREEZE_API_KEY", "y65381Uf4G298til8!l712891x5361e9")
API_SECRET = os.environ.get("BREEZE_API_SECRET", "y65381Uf4G298til8!l712891x5361e9")
SESSION_TOKEN = os.environ.get("BREEZE_SESSION_TOKEN", "56948362")


if not (API_KEY and API_SECRET and SESSION_TOKEN):
    print("Missing credentials — set BREEZE_API_KEY, BREEZE_API_SECRET, BREEZE_SESSION_TOKEN.")
    raise SystemExit(1)

data = NiftyOptionsDataBreeze(API_KEY, API_SECRET, SESSION_TOKEN)

# Same confirmed-working contract/date window as before — feel free to swap
# for a different expiry/date once this checks out.
EXPIRY = dt.date(2026, 9, 8)
APPROX_SPOT = 23813  # from the real spot value confirmed earlier in this project
AS_OF = dt.datetime(2026, 9, 2, 11, 50)

print(f"Finding ATM strike for expiry={EXPIRY}, as_of={AS_OF}, approx_spot={APPROX_SPOT}...")
result = data.find_atm_strike(
    expiry=EXPIRY,
    approx_spot=APPROX_SPOT,
    as_of=AS_OF,
    strike_step=100,
    strike_range=5,
)

print(f"\nBest (ATM) strike: {result['strike']}")
print(f"  Call price: {result['call_price']}")
print(f"  Put price:  {result['put_price']}")
print(f"  |C - P|:    {result['diff']}")

print("\nAll candidates checked, sorted by strike:")
for c in sorted(result["candidates"], key=lambda x: x["strike"]):
    marker = "  <-- ATM" if c["strike"] == result["strike"] else ""
    print(f"  {c['strike']}: call={c['call_price']:.2f} put={c['put_price']:.2f} diff={c['diff']:.2f}{marker}")

print("\nSanity check: the ATM strike should be reasonably close to approx_spot")
print(f"(within a strike or two of {APPROX_SPOT}, given the current week's cost-of-carry).")
print("If the winning strike is far off, or the diff at the winner isn't clearly")
print("smaller than its neighbors, worth widening strike_range or double-checking")
print("the expiry/date window has real trading data for all candidate strikes.")
