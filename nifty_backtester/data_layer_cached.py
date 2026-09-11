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

import datetime as dt
from pathlib import Path

import pandas as pd

from .data_layer_base import BaseCachedOptionsDataLayer

DEFAULT_CACHE_DIR = Path(__file__).parent.parent / "real_data_cache"


class CachedDataUnavailable(Exception):
    """Raised when requested data isn't already present in the committed
    cache -- there is no live session here to fetch it with. The message
    always says what IS available for that key, if anything, so you know
    whether to widen the request or go pull more data with live credentials."""


class NiftyOptionsDataCached(BaseCachedOptionsDataLayer):
    """find_atm_strike / nearest_otm_strikes / get_straddle_and_hedge_data
    are inherited unchanged from BaseCachedOptionsDataLayer -- this class
    only implements the two raw data-access methods, using a store that
    NEVER fetches: any key not already committed under cache_dir raises
    CachedDataUnavailable immediately (see module docstring)."""

    def __init__(self, cache_dir: Path = DEFAULT_CACHE_DIR):
        super().__init__(cache_dir)
        if not self.cache_dir.exists():
            raise FileNotFoundError(
                f"No cached data directory at {self.cache_dir} -- see "
                f"real_data_cache/README.md for how to populate and commit one."
            )

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

    # find_atm_strike, nearest_otm_strikes, get_straddle_and_hedge_data are
    # inherited from BaseCachedOptionsDataLayer unchanged. The base's
    # find_atm_strike catches a broad `Exception` around each candidate
    # strike lookup (see its docstring), so CachedDataUnavailable raised by
    # get_option_historical above is already treated as "skip this
    # candidate" with zero cached-layer-specific code needed here.
