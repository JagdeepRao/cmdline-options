"""
Diagnostic script — run this BEFORE trying the full data layer again.
Prints the raw response from Breeze at each step so we can see exactly
what's coming back, rather than guessing from an empty DataFrame.

Set your credentials as environment variables first:
  export BREEZE_API_KEY="..."
  export BREEZE_API_SECRET="..."
  export BREEZE_SESSION_TOKEN="..."

Run: python3 debug_breeze.py
"""

import os
import datetime as dt
from breeze_connect import BreezeConnect

API_KEY = os.environ.get("BREEZE_API_KEY", "")
API_SECRET = os.environ.get("BREEZE_API_SECRET", "")
SESSION_TOKEN = os.environ.get("BREEZE_SESSION_TOKEN", "")

if not (API_KEY and API_SECRET and SESSION_TOKEN):
    print("Missing credentials — set BREEZE_API_KEY, BREEZE_API_SECRET, BREEZE_SESSION_TOKEN.")
    exit(1)

breeze = BreezeConnect(api_key=API_KEY)
session_result = breeze.generate_session(api_secret=API_SECRET, session_token=SESSION_TOKEN)
print("=== generate_session() result ===")
print(session_result)
print()

# ─────────────────────────────────────────────
# STEP 1: simplest possible call — NIFTY index, 1-day candles, a safely
# historical week (well before today, no strike/expiry involved at all).
# If THIS comes back empty or errors, the problem is auth/connectivity,
# not strike/expiry selection.
# ─────────────────────────────────────────────
print("=== STEP 1: NIFTY index, 1day interval, safely historical range ===")
resp1 = breeze.get_historical_data_v2(
    interval="1day",
    from_date="2026-08-01T07:00:00.000Z",
    to_date="2026-08-10T07:00:00.000Z",
    stock_code="NIFTY",
    exchange_code="NSE",
    product_type="cash",
)
print("Raw response:", resp1)
print()

# ─────────────────────────────────────────────
# STEP 2: if step 1 worked, use its data to find a real recent spot price,
# then try a REAL, currently-listed option contract. Replace the expiry
# below with an actual current/recent NIFTY weekly expiry date (check
# your broker's option chain for the exact date) and a strike near the
# spot printed in step 1.
# ─────────────────────────────────────────────
print("=== STEP 2: single option leg, replace expiry/strike with real values ===")
REAL_EXPIRY = "2026-09-11T07:00:00.000Z"  # <-- confirm this is an actual listed weekly expiry
REAL_STRIKE = "24600"                      # <-- confirm this strike actually exists for that expiry

resp2 = breeze.get_historical_data_v2(
    interval="1day",  # start with 1day, not 1minute — fewer ways to get an empty/odd result
    from_date="2026-09-01T07:00:00.000Z",
    to_date="2026-09-05T07:00:00.000Z",
    stock_code="NIFTY",
    exchange_code="NFO",
    product_type="options",
    expiry_date=REAL_EXPIRY,
    right="call",
    strike_price=REAL_STRIKE,
)
print("Raw response:", resp2)