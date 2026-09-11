"""
Shared base for every "cached-style" options data layer -- i.e. every data
layer that reads option/index bars keyed by (expiry, strike, right,
interval) out of some local store and answers find_atm_strike /
nearest_otm_strikes / get_straddle_and_hedge_data against them the same
way regardless of where the bars actually came from.

Before this module existed, NiftyOptionsDataCached and NiftyOptionsDataSample
each carried their own copy of find_atm_strike/nearest_otm_strikes/
get_straddle_and_hedge_data -- byte-for-byte identical logic, duplicated on
purpose at the time so data_layer_sample.py could stay free of the
breeze_connect import pulled in by data_layer_breeze.py. That constraint
doesn't actually apply to data_layer_cached.py (it never imported breeze
either), and duplicating the ATM-resolution algorithm across two files
risked exactly the kind of drift a shared base class exists to prevent --
so it's centralized here instead, with NO dependency on breeze_connect,
preserving the original import-safety property for every subclass.

Subclasses need only implement the two abstract data-access methods
(get_option_historical, get_index_historical); find_atm_strike,
nearest_otm_strikes, and get_straddle_and_hedge_data are inherited as-is.

NiftyOptionsDataBreeze (data_layer_breeze.py) is deliberately NOT folded
into this hierarchy -- it already implements the identical algorithm
against live Breeze data, and merging it in here would force this
breeze_connect-free module to either import breeze_connect or grow an
awkward optional-dependency seam. Three call sites for the same ~15-line
algorithm (here, live, and the historical find_atm_strike_live variant) is
an acceptable amount of duplication against that cost -- revisit only if a
fourth copy shows up.
"""

from __future__ import annotations
import math
import datetime as dt
from pathlib import Path


class BaseCachedOptionsDataLayer:
    """Common ATM-resolution / OTM-strike / straddle-assembly logic for any
    data layer backed by a local (non-live) store of option/index bars.

    Subclasses must implement:
      get_option_historical(expiry, strike, right, from_date, to_date, interval="1minute") -> DataFrame
      get_index_historical(from_date, to_date, interval="1minute") -> DataFrame

    and are free to raise whatever "data not available" exception fits
    their own semantics (NiftyOptionsDataCached raises CachedDataUnavailable
    and never fetches; NiftyOptionsDataSample never raises -- it always
    generates deterministically instead). find_atm_strike below treats any
    exception raised by get_option_historical as "this candidate strike
    isn't usable" and moves on to the next one, so either behavior works
    without this base class needing to know which.
    """

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)

    # ---- subclasses must implement these ----
    def get_option_historical(
        self,
        expiry: dt.date,
        strike: int,
        right: str,
        from_date: dt.date,
        to_date: dt.date,
        interval: str = "1minute",
    ):
        raise NotImplementedError

    def get_index_historical(
        self,
        from_date: dt.date,
        to_date: dt.date,
        interval: str = "1minute",
    ):
        raise NotImplementedError

    # ---- shared logic, identical across every subclass ----
    @staticmethod
    def nearest_otm_strikes(spot: float, strike_step: int = 100) -> tuple[int, int]:
        """Nearest OTM call strike (smallest multiple of strike_step strictly
        above spot) and nearest OTM put strike (largest multiple strictly
        below)."""
        call_strike = math.floor(spot / strike_step) * strike_step + strike_step
        put_strike = math.ceil(spot / strike_step) * strike_step - strike_step
        return int(call_strike), int(put_strike)

    def find_atm_strike(
        self,
        expiry: dt.date,
        approx_spot: float,
        as_of: dt.datetime,
        strike_step: int = 100,
        strike_range: int = 5,
    ) -> dict:
        """Put-call-parity ATM resolution: the strike minimizing
        |call_price - put_price| is the market-implied forward, which is
        the correct ATM reference when there's no matching futures contract
        for the expiry (e.g. weekly options). Scans strike_range strikes
        either side of approx_spot (rounded to strike_step), using each
        candidate's nearest-to-as_of bar.

        Any candidate strike this layer can't produce data for (whatever
        exception get_option_historical raises, or an empty result) is
        silently skipped rather than treated as a hard failure -- a
        curated real snapshot may only cover a handful of strikes near the
        money, and synthetic data always covers every strike, so "skip
        what's missing" is the right behavior for both.
        """
        center = round(approx_spot / strike_step) * strike_step
        candidates = []
        window_start = as_of - dt.timedelta(minutes=5)
        window_end = as_of + dt.timedelta(minutes=5)

        for i in range(-strike_range, strike_range + 1):
            strike = center + i * strike_step
            try:
                call_df = self.get_option_historical(expiry, strike, "call", window_start.date(), window_end.date())
                put_df = self.get_option_historical(expiry, strike, "put", window_start.date(), window_end.date())
            except Exception:
                continue
            if call_df.empty or put_df.empty:
                continue

            call_df["datetime"] = _to_datetime_col(call_df["datetime"])
            put_df["datetime"] = _to_datetime_col(put_df["datetime"])
            call_idx = (call_df["datetime"] - as_of).abs().idxmin()
            put_idx = (put_df["datetime"] - as_of).abs().idxmin()
            call_price = float(call_df.loc[call_idx, "close"])
            put_price = float(put_df.loc[put_idx, "close"])
            candidates.append({
                "strike": strike, "call_price": call_price, "put_price": put_price,
                "diff": abs(call_price - put_price),
            })

        if not candidates:
            raise ValueError(
                f"No usable call/put data for expiry={expiry} within {strike_range} "
                f"strikes of {center} around {as_of} -- widen strike_range, or (for a "
                f"cached layer) commit more strikes for this expiry/date."
            )

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


def _to_datetime_col(series):
    import pandas as pd
    return pd.to_datetime(series)
