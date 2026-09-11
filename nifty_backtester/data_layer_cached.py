"""
Credential-free data layer that reads ONLY from a pre-populated cache
directory of the exact parquet files data_cache.DataCache writes -- no
network calls, no BreezeConnect session, ever.

WHY THIS EXISTS: once you've pulled real data from Breeze locally (via
NiftyOptionsDataBreeze with live credentials, which fills up
./data_cache/*.parquet), commit the specific parquet files you want to
keep into real_data_cache/ at the repo root and push. Anyone without live
credentials -- including a fresh session here -- can then load that exact
real data, run it through the strategy/backtest layers, and look at real
scenarios instead of only synthetic ones. See real_data_cache/README.md
for the exact workflow and cache-key naming convention.

This is deliberately NOT a general substitute for NiftyOptionsDataBreeze:
any request for data that isn't already in the committed parquet files
raises CachedDataUnavailable immediately rather than trying to fetch --
there is no live session behind this class to fetch with. It intentionally
does NOT go through data_cache.DataCache.get() for reads, since that
class's job is "fetch what's missing" (with retries/backoff) -- there is
nothing to fetch here, so failing fast beats retrying a fetch_fn that can
only ever raise.

Same public interface as NiftyOptionsDataBreeze / NiftyOptionsDataSample
(get_option_historical, get_index_historical, find_atm_strike,
nearest_otm_strikes, get_straddle_and_hedge_data) -- so it's a drop-in for
BreezeMarketDataProvider, or any script's data-source selection (see
data_sources.resolve_data_layer).
"""

import math
import datetime as dt
from pathlib import Path

import pandas as pd

DEFAULT_CACHE_DIR = Path(__file__).parent.parent / "real_data_cache"


class CachedDataUnavailable(Exception):
    """Raised when requested data isn't already present in the committed
    cache -- there is no live session here to fetch it with. The message
    always says what IS available for that key, if anything, so you know
    whether to widen the request or go pull more data with live credentials."""


class NiftyOptionsDataCached:
    def __init__(self, cache_dir: Path = DEFAULT_CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        if not self.cache_dir.exists():
            raise FileNotFoundError(
                f"No cached data directory at {self.cache_dir} -- see "
                f"real_data_cache/README.md for how to populate and commit one."
            )

    @staticmethod
    def nearest_otm_strikes(spot: float, strike_step: int = 100) -> tuple[int, int]:
        call_strike = math.floor(spot / strike_step) * strike_step + strike_step
        put_strike = math.ceil(spot / strike_step) * strike_step - strike_step
        return int(call_strike), int(put_strike)

    def _read_slice(self, cache_key: str, from_date: dt.date, to_date: dt.date) -> pd.DataFrame:
        cache_path = self.cache_dir / f"{cache_key}.parquet"
        if not cache_path.exists():
            raise CachedDataUnavailable(
                f"No committed data for '{cache_key}' ({cache_path.name} not found in "
                f"{self.cache_dir}) -- nothing has been cached and committed for this "
                f"contract/index/interval yet."
            )
        df = pd.read_parquet(cache_path)
        df["datetime"] = pd.to_datetime(df["datetime"])
        mask = (
            (df["datetime"] >= pd.Timestamp(from_date))
            & (df["datetime"] < pd.Timestamp(to_date) + pd.Timedelta(days=1))
        )
        result = df[mask].reset_index(drop=True)
        if result.empty:
            cached_min, cached_max = df["datetime"].min(), df["datetime"].max()
            raise CachedDataUnavailable(
                f"'{cache_key}' is committed but only covers "
                f"{cached_min.date()}..{cached_max.date()}; requested "
                f"{from_date}..{to_date} falls outside that."
            )
        return result

    def get_option_historical(
        self,
        expiry: dt.date,
        strike: int,
        right: str,
        from_date: dt.date,
        to_date: dt.date,
        interval: str = "1minute",
    ) -> pd.DataFrame:
        cache_key = f"NIFTY_{expiry}_{strike}_{right}_{interval}"
        return self._read_slice(cache_key, from_date, to_date)

    def get_index_historical(
        self,
        from_date: dt.date,
        to_date: dt.date,
        interval: str = "1minute",
    ) -> pd.DataFrame:
        cache_key = f"NIFTY_INDEX_{interval}"
        return self._read_slice(cache_key, from_date, to_date)

    def find_atm_strike(
        self,
        expiry: dt.date,
        approx_spot: float,
        as_of: dt.datetime,
        strike_step: int = 100,
        strike_range: int = 5,
    ) -> dict:
        """Same put-call-parity ATM resolution as the live/sample layers,
        but every candidate strike that isn't in the committed cache is
        silently skipped (not treated as a hard failure) -- you may well
        have committed data for only a handful of strikes near the money,
        which is exactly the realistic "curated real snapshot" use case."""
        center = round(approx_spot / strike_step) * strike_step
        candidates = []
        window_start = as_of - dt.timedelta(minutes=5)
        window_end = as_of + dt.timedelta(minutes=5)

        for i in range(-strike_range, strike_range + 1):
            strike = center + i * strike_step
            try:
                call_df = self.get_option_historical(expiry, strike, "call", window_start.date(), window_end.date())
                put_df = self.get_option_historical(expiry, strike, "put", window_start.date(), window_end.date())
            except CachedDataUnavailable:
                continue
            if call_df.empty or put_df.empty:
                continue

            call_df["datetime"] = pd.to_datetime(call_df["datetime"])
            put_df["datetime"] = pd.to_datetime(put_df["datetime"])
            call_idx = (call_df["datetime"] - as_of).abs().idxmin()
            put_idx = (put_df["datetime"] - as_of).abs().idxmin()
            call_price = float(call_df.loc[call_idx, "close"])
            put_price = float(put_df.loc[put_idx, "close"])
            candidates.append({
                "strike": strike, "call_price": call_price, "put_price": put_price,
                "diff": abs(call_price - put_price),
            })

        if not candidates:
            raise CachedDataUnavailable(
                f"No committed call/put data for expiry={expiry} within {strike_range} "
                f"strikes of {center} around {as_of} -- commit more strikes for this "
                f"expiry/date if you need ATM resolution here."
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
