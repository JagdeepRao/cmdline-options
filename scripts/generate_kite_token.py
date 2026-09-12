"""
Runs the Kite Connect login flow's second half: exchanges a request_token
(obtained by logging in through a browser) for an access_token, and writes
the result straight into your local .env file -- so the daily "regenerate
KITE_ACCESS_TOKEN" chore is one script run, not a manual copy-paste.

CONFIRMED from the installed kiteconnect SDK source (KiteConnect.login_url()
and KiteConnect.generate_session()) -- not assumed:

  STEP 1 (browser): visit
      https://kite.zerodha.com/connect/login?api_key=YOUR_API_KEY&v=3
  and log in. Zerodha redirects to your Kite Connect app's registered
  redirect URL with a `request_token` query parameter attached -- copy
  that value out of the resulting URL.

  STEP 2 (this script): exchanges that request_token for an access_token
  by POSTing to https://api.kite.trade/session/token with api_key,
  request_token, and a checksum = SHA256(api_key + request_token +
  api_secret) -- api_secret itself is NEVER sent, only used to compute
  this checksum. kiteconnect's generate_session() does this POST for you;
  this script just wraps that call and persists the result.

Needs KITE_API_KEY and KITE_API_SECRET already set in .env (both
long-lived, from your Kite Connect app registration -- see .env.example).
KITE_ACCESS_TOKEN does NOT need to be set yet; this script writes it.

Run (from the repo root): python3 scripts/generate_kite_token.py
"""

import os

from kiteconnect import KiteConnect
from dotenv import find_dotenv, set_key

from nifty_backtester.env_loader import load_env

load_env()
API_KEY = os.environ.get("KITE_API_KEY", "")
API_SECRET = os.environ.get("KITE_API_SECRET", "")

if not (API_KEY and API_SECRET):
    print("Missing credentials -- set KITE_API_KEY and KITE_API_SECRET in .env first (see .env.example).")
    raise SystemExit(1)

kite = KiteConnect(api_key=API_KEY)

print("STEP 1: open this URL in a browser and log in:\n")
print(f"  {kite.login_url()}\n")
print("After login, you'll be redirected to your app's registered redirect URL")
print("with a request_token=... query parameter attached to it. Copy just that")
print("token value (not the whole URL).\n")

request_token = input("Paste the request_token here: ").strip()
if not request_token:
    print("No request_token entered -- aborting.")
    raise SystemExit(1)

print("\nSTEP 2: exchanging request_token for an access_token...")
try:
    session_data = kite.generate_session(request_token, api_secret=API_SECRET)
except Exception as e:
    print(f"\nToken exchange failed: {e}")
    print("Common causes: request_token already used/expired (they're single-use "
          "and short-lived -- restart from STEP 1 if this happens), or a wrong "
          "KITE_API_SECRET.")
    raise SystemExit(1)

access_token = session_data["access_token"]
print(f"\nSuccess. Logged in as: {session_data.get('user_name', '(name not returned)')} "
      f"({session_data.get('user_id', '?')})")

dotenv_path = find_dotenv(usecwd=True) or ".env"
set_key(dotenv_path, "KITE_ACCESS_TOKEN", access_token)
print(f"\nKITE_ACCESS_TOKEN written to {dotenv_path}.")
print("This token is valid for today's trading session only -- re-run this script tomorrow.")
