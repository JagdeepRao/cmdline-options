"""
Sample-data-backed stand-in for NiftyOptionsDataBreeze.

Same public interface (get_option_historical, get_index_historical,
find_atm_strike, nearest_otm_strikes, get_straddle_and_hedge_data) but
sources bars from sample_data's synthetic generators instead of a live
Breeze session, routed through the SAME DataCache class used by the real
data layer. This is what requirement #3 (sample data + tests) is built on:
strategy/pricing/metrics code can be exercised end-to-end without Breeze
credentials, and the exact same DataCache code path (including retry
logic) gets test coverage.

Intentionally duplicates nearest_otm_strikes/find_atm_strike/
get_straddle_and_hedge_data logic from data_layer_breeze.py rather than
importing it, since importing would pull in the breeze_connect dependency
(and its network calls) just to reuse a few pure-Python helper methods —
keeping this module import-safe with zero external/network dependencies
is the whole point.
"""

import math
import datetime as dt
from pathlib import Path
from functools import partial

import pandas as pd

from .data_cache import DataCache
from .sample_data import generate_index_bars, generate_option_bars

SAMPLE_CACHE_DIR = Path("./sample_data_cache")


class NiftyOptionsDataSample:
    def __init__(self, cache_dir: Path = SAMPLE_CACHE_DIR, index_start_price: float = 24800.0):
        self.cache = DataCache(cache_dir)
        self.index_start_price = index_start_price

    @staticmethod
    def nearest_otm_strikes(spot: float, strike_step: int = 100) -> tuple[int, int]:
        call_strike = math.floor(spot / strike_step) * strike_step + strike_step
        put_strike = math.ceil(spot / strike_step) * strike_step - strike_step
        return int(call_strike), int(put_strike)

    def get_option_historical(
        self,
        expiry: dt.date,
        strike: int,
        right: str,
        from_date: dt.date,
        to_date: dt.date,
        interval: str = "1minute",
    ) -> pd.DataFrame:
        cache_key = f"SAMPLE_NIFTY_{expiry}_{strike}_{right}_{interval}"
        fetch_fn = partial(
            generate_option_bars,
            expiry=expiry, strike=strike, right=right, interval=interval,
            index_spot_at_start=self.index_start_price,
        )
        return self.cache.get(cache_key, from_date, to_date, fetch_fn)

    def get_index_historical(
        self,
        from_date: dt.date,
        to_date: dt.date,
        interval: str = "1minute",
    ) -> pd.DataFrame:
        cache_key = f"SAMPLE_NIFTY_INDEX_{interval}"
        fetch_fn = partial(generate_index_bars, interval=interval, start_price=self.index_start_price)
        return self.cache.get(cache_key, from_date, to_date, fetch_fn)

    def find_atm_strike(
        self,
        expiry: dt.date,
        approx_spot: float,
        as_of: dt.datetime,
        strike_step: int = 100,
        strike_range: int = 5,
    ) -> dict:
        """Same put-call-parity ATM resolution as the real data layer, but
        against synthetic bars — useful for exercising the algorithm and
        for strategy dry-runs without live data."""
        center = round(approx_spot / strike_step) * strike_step
        candidates = []
        window_start = as_of - dt.timedelta(minutes=5)
        window_end = as_of + dt.timedelta(minutes=5)

        for i in range(-strike_range, strike_range + 1):
            strike = center + i * strike_step
            call_df = self.get_option_historical(expiry, strike, "call", window_start.date(), window_end.date())
            put_df = self.get_option_historical(expiry, strike, "put", window_start.date(), window_end.date())
            if call_df.empty or put_df.empty:
                continue
            call_df["datetime"] = pd.to_datetime(call_df["datetime"])
            put_df["datetime"] = pd.to_datetime(put_df["datetime"])
            call_idx = (call_df["datetime"] - as_of).abs().idxmin()
            put_idx = (put_df["datetime"] - as_of).abs().idxmin()
            call_price = float(call_df.loc[call_idx, "close"])
            put_price = float(put_df.loc[put_idx, "close"])
            candidates.append({"strike": strike, "call_price": call_price, "put_price": put_price,
                                "diff": abs(call_price - put_price)})

        if not candidates:
            raise ValueError(f"No sample data generated for expiry={expiry} near {center}.")

        best = min(candidates, key=lambda c: c["diff"])
        return {**best, "candidates": candidates}

    def get_straddle_and_hedge_data(
        self,
        expiry: dt.date,
        spot_at_entry: float,
        from_date: dt.date,
        to_date: dt.date,
        interval: str = "1minute",
    ) -> dict:
        atm_strike = round(spot_at_entry / 100) * 100
        otm_call_strike, otm_put_strike = self.nearest_otm_strikes(spot_at_entry)
        return {
            "straddle_call": self.get_option_historical(expiry, atm_strike, "call", from_date, to_date, interval),
            "straddle_put": self.get_option_historical(expiry, atm_strike, "put", from_date, to_date, interval),
            "hedge_call": self.get_option_historical(expiry, otm_call_strike, "call", from_date, to_date, interval),
            "hedge_put": self.get_option_historical(expiry, otm_put_strike, "put", from_date, to_date, interval),
        }
