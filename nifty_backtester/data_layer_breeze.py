"""
Data layer for the NIFTY options backtester — ICICI Breeze Connect edition.

Breeze differs from Kite in a useful way: there's no separate instrument-master
lookup step. You request historical data directly by stock_code + expiry_date +
strike_price + right — Breeze resolves the actual contract internally.

CONFIRMED by inspecting the installed breeze-connect SDK source directly
(v1.x, checked in this session — re-verify if you're on a different version):
  - Dates are ISO strings in the form "YYYY-MM-DDTHH:MM:SS.000Z"
  - get_historical_data_v2(interval=, from_date=, to_date=, stock_code=,
    exchange_code=, product_type=, expiry_date=, right=, strike_price=)
    is the method used here — it's more permissive than v1 (explicitly
    supports product_type="cash" for the index, and more exchange codes)
  - VALID INTERVALS ARE ONLY: "1second" (v2 only), "1minute", "5minute",
    "30minute", "1day". THERE IS NO NATIVE "15minute" (or other arbitrary)
    INTERVAL. This layer therefore always fetches at native "1minute" and
    lets market_data.BreezeMarketDataProvider combine bars into whatever
    timeframe a strategy needs (15-min sold-leg signal, 1hr overlay
    direction, etc.) via the shared _resample_ohlc() helper — see
    market_data.py, not this file, for that logic.

CONFIRMED FIELD NAMES — verified against a real successful pull (2026-09),
full column list: close, datetime, exchange_code, expiry_date, high, low,
open, open_interest, product_type, right, stock_code, strike_price, volume.
open_interest is available but not currently used anywhere — worth pulling
into the position/liquidity logic later (e.g. skip strikes with thin OI)
if useful.

NOT independently confirmed (no live network access to verify from here):
  - Exact max date-range-per-call Breeze enforces (CHUNK_DAYS below is a
    conservative placeholder, not a documented limit — tune based on what
    your own calls actually accept/reject)
  - Exact current rate limit (the sleep below is a cautious guess)

Requires: pip install breeze-connect pandas pyarrow --break-system-packages

Session tokens are obtained via the login URL flow
(https://api.icicidirect.com/apiuser/login?api_key=YOUR_API_KEY) and, like
most broker APIs, need periodic regeneration — this script assumes you
already have a valid session_token when you run it.
"""

import time
import math
import datetime as dt
from pathlib import Path

import pandas as pd
from breeze_connect import BreezeConnect

from .data_cache import DataCache

CACHE_DIR = Path("./data_cache")
CACHE_DIR.mkdir(exist_ok=True)

# UNCONFIRMED placeholders — see module docstring. Start conservative and
# widen only after confirming Breeze accepts a larger range without error.
CHUNK_DAYS = {
    "1minute": 7,
    "5minute": 30,
    "30minute": 90,
    "1day": 365,
}
REQUEST_SLEEP_SECONDS = 0.5  # conservative placeholder, not a documented Breeze rate limit


def _breeze_date(d, time_of_day: dt.time = None) -> str:
    """Formats a date/datetime into Breeze's expected ISO string.

    BUG FIX: previously defaulted every bare `date` to a fixed 07:00:00,
    which meant a same-day from_date/to_date pair (e.g. a short window
    within one day) collapsed to an identical timestamp — a zero-width
    query that silently returns no data. Callers now pass explicit
    start/end times so a single-day range still has real width.
    """
    if isinstance(d, dt.date) and not isinstance(d, dt.datetime):
        d = dt.datetime.combine(d, time_of_day or dt.time(7, 0, 0))
    return d.strftime("%Y-%m-%dT%H:%M:%S.000Z")


class NiftyOptionsDataBreeze:
    def __init__(self, api_key: str, api_secret: str, session_token: str):
        self.breeze = BreezeConnect(api_key=api_key)
        self.breeze.generate_session(api_secret=api_secret, session_token=session_token)
        self._cache = DataCache(CACHE_DIR)

    # ─────────────────────────────────────────────
    # STRIKE RESOLUTION — broker-agnostic, same logic as the Pine Script panel
    # ─────────────────────────────────────────────
    @staticmethod
    def nearest_otm_strikes(spot: float, strike_step: int = 100) -> tuple[int, int]:
        """Nearest OTM call strike (smallest multiple of strike_step strictly
        above spot) and nearest OTM put strike (largest multiple strictly below)."""
        call_strike = math.floor(spot / strike_step) * strike_step + strike_step
        put_strike = math.ceil(spot / strike_step) * strike_step - strike_step
        return int(call_strike), int(put_strike)

    # ─────────────────────────────────────────────
    # HISTORICAL DATA — OPTIONS
    # ─────────────────────────────────────────────
    def get_option_historical(
        self,
        expiry: dt.date,
        strike: int,
        right: str,  # "call" or "put"
        from_date: dt.date,
        to_date: dt.date,
        interval: str = "1minute",
    ) -> pd.DataFrame:
        cache_key = f"NIFTY_{expiry}_{strike}_{right}_{interval}"

        def fetch_fn(fd: dt.date, td: dt.date) -> pd.DataFrame:
            chunk_days = CHUNK_DAYS.get(interval, 7)
            all_rows = []
            cursor = fd
            while cursor <= td:
                chunk_end = min(cursor + dt.timedelta(days=chunk_days), td)
                try:
                    resp = self.breeze.get_historical_data_v2(
                        interval=interval,
                        from_date=_breeze_date(cursor, dt.time(0, 0, 0)),
                        to_date=_breeze_date(chunk_end, dt.time(23, 59, 59)),
                        stock_code="NIFTY",
                        exchange_code="NFO",
                        product_type="options",
                        expiry_date=_breeze_date(expiry),
                        right=right,
                        strike_price=str(strike),
                    )
                    if resp.get("Success"):
                        all_rows.extend(resp["Success"])
                    elif resp.get("Error"):
                        print(f"Breeze error for {cursor} to {chunk_end}: {resp['Error']}")
                except Exception as e:
                    print(f"Fetch failed for {cursor} to {chunk_end}: {e}")
                cursor = chunk_end + dt.timedelta(days=1)
                time.sleep(REQUEST_SLEEP_SECONDS)
            return pd.DataFrame(all_rows)

        return self._cache.get(cache_key, from_date, to_date, fetch_fn)

    # ─────────────────────────────────────────────
    # HISTORICAL DATA — UNDERLYING INDEX (for strike resolution / spot reference)
    # ─────────────────────────────────────────────
    def get_index_historical(
        self,
        from_date: dt.date,
        to_date: dt.date,
        interval: str = "1minute",
    ) -> pd.DataFrame:
        cache_key = f"NIFTY_INDEX_{interval}"

        def fetch_fn(fd: dt.date, td: dt.date) -> pd.DataFrame:
            chunk_days = CHUNK_DAYS.get(interval, 7)
            all_rows = []
            cursor = fd
            while cursor <= td:
                chunk_end = min(cursor + dt.timedelta(days=chunk_days), td)
                try:
                    resp = self.breeze.get_historical_data_v2(
                        interval=interval,
                        from_date=_breeze_date(cursor, dt.time(0, 0, 0)),
                        to_date=_breeze_date(chunk_end, dt.time(23, 59, 59)),
                        stock_code="NIFTY",
                        exchange_code="NSE",
                        product_type="cash",
                    )
                    if resp.get("Success"):
                        all_rows.extend(resp["Success"])
                    elif resp.get("Error"):
                        print(f"Breeze error for {cursor} to {chunk_end}: {resp['Error']}")
                except Exception as e:
                    print(f"Fetch failed for {cursor} to {chunk_end}: {e}")
                cursor = chunk_end + dt.timedelta(days=1)
                time.sleep(REQUEST_SLEEP_SECONDS)
            return pd.DataFrame(all_rows)

        return self._cache.get(cache_key, from_date, to_date, fetch_fn)

    def find_atm_strike(
        self,
        expiry: dt.date,
        approx_spot: float,
        as_of: dt.datetime,
        strike_step: int = 100,
        strike_range: int = 5,
    ) -> dict:
        """Finds the ATM strike for a given expiry via put-call parity: the
        strike minimizing |Call price - Put price| is the market-implied
        forward price, which is the correct ATM reference when there's no
        matching futures contract for that expiry (e.g. weekly options).

        FOR BACKTESTING (historical dates). Breeze's get_option_chain_quotes
        endpoint would do this in 2 calls instead of looping per strike, but
        it has no from_date/to_date parameter anywhere in the installed SDK
        — confirmed by reading the source, not assumed — so it's live-quotes
        only. See find_atm_strike_live() below for that version, once you're
        past backtesting.

        Scans strike_range strikes on either side of approx_spot (rounded to
        the nearest strike_step), fetching a short window of 1-minute candles
        around as_of for both the call and put at each candidate strike, and
        compares their prices at the candle nearest as_of.

        Returns: {"strike": int, "call_price": float, "put_price": float,
                  "diff": float, "candidates": [...]} — candidates lists every
        strike checked with its diff, so you can inspect how close the next-
        best strikes were rather than just trusting a single winner blindly.
        """
        center = round(approx_spot / strike_step) * strike_step
        candidates = []

        # small window around as_of — just need the nearest quote, not a
        # full history, to compare C vs P at this specific moment
        window_start = as_of - dt.timedelta(minutes=5)
        window_end = as_of + dt.timedelta(minutes=5)

        for i in range(-strike_range, strike_range + 1):
            strike = center + i * strike_step
            call_df = self.get_option_historical(
                expiry, strike, "call", window_start.date(), window_end.date(), interval="1minute"
            )
            put_df = self.get_option_historical(
                expiry, strike, "put", window_start.date(), window_end.date(), interval="1minute"
            )
            if call_df.empty or put_df.empty:
                continue

            call_df["datetime"] = pd.to_datetime(call_df["datetime"])
            put_df["datetime"] = pd.to_datetime(put_df["datetime"])
            call_idx = (call_df["datetime"] - as_of).abs().idxmin()
            put_idx = (put_df["datetime"] - as_of).abs().idxmin()

            call_price = float(call_df.loc[call_idx, "close"])
            put_price = float(put_df.loc[put_idx, "close"])
            candidates.append({
                "strike": strike,
                "call_price": call_price,
                "put_price": put_price,
                "diff": abs(call_price - put_price),
            })

        if not candidates:
            raise ValueError(
                f"No usable call/put data found for expiry={expiry} within "
                f"{strike_range} strikes of {center} around {as_of} — "
                f"widen strike_range or check the expiry/date are valid."
            )

        best = min(candidates, key=lambda c: c["diff"])
        return {**best, "candidates": candidates}

    def find_atm_strike_live(self, expiry: dt.date) -> dict:
        """FOR LIVE/PAPER USE ONLY — not usable for backtesting (see
        find_atm_strike() above for why). Uses get_option_chain_quotes to
        pull the entire call chain and entire put chain for an expiry in
        two calls, then finds the strike minimizing |C - P|.

        UNCONFIRMED: the response field names below (assumed 'strike_price'
        and 'ltp') are a reasonable guess based on this SDK's conventions
        elsewhere, but this endpoint's exact response schema was not
        independently verified against a live account from this environment
        — there's no way to test it without hitting Breeze's live quote
        system. Run this once against your real account and check the raw
        response structure before trusting it; adjust the field names below
        if they don't match.
        """
        expiry_str = _breeze_date(expiry)

        call_chain = self.breeze.get_option_chain_quotes(
            stock_code="NIFTY", exchange_code="NFO",
            expiry_date=expiry_str, product_type="options", right="call", strike_price="",
        )
        put_chain = self.breeze.get_option_chain_quotes(
            stock_code="NIFTY", exchange_code="NFO",
            expiry_date=expiry_str, product_type="options", right="put", strike_price="",
        )

        calls = {row["strike_price"]: float(row["ltp"]) for row in call_chain.get("Success", [])}
        puts = {row["strike_price"]: float(row["ltp"]) for row in put_chain.get("Success", [])}

        common_strikes = set(calls) & set(puts)
        if not common_strikes:
            raise ValueError("No overlapping strikes between call and put chains — check the raw response structure, field names may not match what's assumed here.")

        candidates = [
            {"strike": s, "call_price": calls[s], "put_price": puts[s], "diff": abs(calls[s] - puts[s])}
            for s in common_strikes
        ]
        best = min(candidates, key=lambda c: c["diff"])
        return {**best, "candidates": candidates}

    # ─────────────────────────────────────────────
    # CONVENIENCE: straddle + hedge legs for a given day
    # ─────────────────────────────────────────────
    def get_straddle_and_hedge_data(
        self,
        expiry: dt.date,
        spot_at_entry: float,
        from_date: dt.date,
        to_date: dt.date,
        interval: str = "1minute",
    ) -> dict:
        """Given the spot price at the time you'd resolve strikes (e.g. that
        morning's open), fetches the ATM straddle legs and nearest-OTM hedge
        legs for the specified date range. ATM strike here uses the same
        round-to-nearest-100 logic as the OTM strikes — adjust if your ATM
        selection differs from nearest-100 rounding."""
        atm_strike = round(spot_at_entry / 100) * 100
        otm_call_strike, otm_put_strike = self.nearest_otm_strikes(spot_at_entry)

        return {
            "straddle_call": self.get_option_historical(expiry, atm_strike, "call", from_date, to_date, interval),
            "straddle_put": self.get_option_historical(expiry, atm_strike, "put", from_date, to_date, interval),
            "hedge_call": self.get_option_historical(expiry, otm_call_strike, "call", from_date, to_date, interval),
            "hedge_put": self.get_option_historical(expiry, otm_put_strike, "put", from_date, to_date, interval),
        }


if __name__ == "__main__":
    import os
    from nifty_backtester.env_loader import load_env  # absolute: relative imports break if this file is run directly

    load_env()  # picks up BREEZE_API_KEY/BREEZE_API_SECRET/BREEZE_SESSION_TOKEN from a local .env if present
    API_KEY = os.environ.get("BREEZE_API_KEY", "")
    API_SECRET = os.environ.get("BREEZE_API_SECRET", "")
    SESSION_TOKEN = os.environ.get("BREEZE_SESSION_TOKEN", "")

    if not (API_KEY and API_SECRET and SESSION_TOKEN):
        print("Set BREEZE_API_KEY, BREEZE_API_SECRET, and BREEZE_SESSION_TOKEN first.")
    else:
        data = NiftyOptionsDataBreeze(API_KEY, API_SECRET, SESSION_TOKEN)

        call_strike, put_strike = data.nearest_otm_strikes(spot=24537)
        print(f"Nearest OTM call strike: {call_strike}, put strike: {put_strike}")

        # Example uses the expiry/strike combination confirmed working —
        # swap in whatever real expiry/strike you actually need for your run.
        expiry = dt.date(2026, 9, 8)
        df = data.get_option_historical(
            expiry=expiry,
            strike=24600,
            right="call",
            from_date=dt.date(2026, 9, 1),
            to_date=dt.date(2026, 9, 5),
            interval="1minute",
        )
        print(df.head())
