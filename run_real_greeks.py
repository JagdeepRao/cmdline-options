"""
End-to-end example: pulls a real option leg AND its matching NIFTY spot from
Breeze, then computes real IV/Greeks — no placeholder values, everything
comes from your actual account data.

Set credentials as environment variables first:
  export BREEZE_API_KEY="..."
  export BREEZE_API_SECRET="..."
  export BREEZE_SESSION_TOKEN="..."

Run: python3 run_real_greeks.py
"""

import os
import datetime as dt

from data_layer_breeze import NiftyOptionsDataBreeze
from pricing import add_greeks_to_option_df

API_KEY = os.environ.get("BREEZE_API_KEY", "y65381Uf4G298til8!l712891x5361e9")
API_SECRET = os.environ.get("BREEZE_API_SECRET", "y65381Uf4G298til8!l712891x5361e9")
SESSION_TOKEN = os.environ.get("BREEZE_SESSION_TOKEN", "56948362")

if not (API_KEY and API_SECRET and SESSION_TOKEN):
    print("Missing credentials — set BREEZE_API_KEY, BREEZE_API_SECRET, BREEZE_SESSION_TOKEN.")
    raise SystemExit(1)

data = NiftyOptionsDataBreeze(API_KEY, API_SECRET, SESSION_TOKEN)

# Same contract confirmed working earlier in this project — swap these for
# whatever expiry/strike/date range you actually want to check.
EXPIRY = dt.date(2026, 9, 8)
STRIKE = 24600
FROM_DATE = dt.date(2026, 9, 1)
TO_DATE = dt.date(2026, 9, 5)

print("Fetching option leg...")
option_df = data.get_option_historical(
    expiry=EXPIRY, strike=STRIKE, right="call",
    from_date=FROM_DATE, to_date=TO_DATE, interval="1minute",
)
print(f"  {len(option_df)} option rows fetched")

print("Fetching matching NIFTY spot...")
spot_df = data.get_index_historical(
    from_date=FROM_DATE, to_date=TO_DATE, interval="1minute",
)
print(f"  {len(spot_df)} spot rows fetched")

if option_df.empty or spot_df.empty:
    print("One of the pulls came back empty — check expiry/strike/date range before proceeding.")
    raise SystemExit(1)

print("\nComputing real IV/Greeks (using current r=5.25%, q=1.2%)...")
result = add_greeks_to_option_df(option_df, spot_df)

print("\nFirst 10 rows with real computed Greeks:")
print(result[["datetime", "close", "spot_close", "strike_price", "T", "iv", "delta", "gamma", "theta", "vega"]].head(10))

print(f"\nIV range across this data: {result['iv'].min():.4f} to {result['iv'].max():.4f}")
print("(Sanity check: NIFTY weekly ATM/near-ATM IV is typically somewhere in the")
print(" 10-20%+ range depending on current market conditions — if the numbers here")
print(" look wildly outside that, worth double-checking the strike/spot alignment.)")
