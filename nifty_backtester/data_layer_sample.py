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

find_atm_strike/nearest_otm_strikes/get_straddle_and_hedge_data now come
from data_layer_base.BaseCachedOptionsDataLayer, shared with
NiftyOptionsDataCached (previously duplicated byte-for-byte between the
two). That base module has zero breeze_connect dependency itself, so
importing it here preserves this module's original import-safety
guarantee: NiftyOptionsDataSample still never pulls in breeze_connect or
makes a network call.
"""

import datetime as dt
from pathlib import Path
from functools import partial

import pandas as pd

from .data_cache import DataCache
from .data_layer_base import BaseCachedOptionsDataLayer
from .sample_data import generate_index_bars, generate_option_bars

SAMPLE_CACHE_DIR = Path("./sample_data_cache")


class NiftyOptionsDataSample(BaseCachedOptionsDataLayer):
    """find_atm_strike / nearest_otm_strikes / get_straddle_and_hedge_data
    are inherited unchanged from BaseCachedOptionsDataLayer -- this class
    only implements the two raw data-access methods, generating
    deterministic bars on demand (via sample_data.py) through the same
    DataCache class the real Breeze layer uses, so cache-hit/miss and
    retry behavior get identical test coverage against either backend.

    TEST-ENVIRONMENT PROPERTY (unchanged by this refactor, worth stating
    explicitly since it's exactly what SampleLiveDataFeed in nifty_live/
    depends on later): this class NEVER makes a network call and NEVER
    imports breeze_connect -- every bar is generated locally and
    deterministically. That's what makes it safe to use as the backing
    data for tests, demos, and (via ScenarioBoundedDataLayer/
    SampleLiveDataFeed) a stand-in "live" feed with zero live session."""

    def __init__(self, cache_dir: Path = SAMPLE_CACHE_DIR, index_start_price: float = 24800.0):
        super().__init__(cache_dir)
        self.cache = DataCache(cache_dir)
        self.index_start_price = index_start_price

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

    # find_atm_strike, nearest_otm_strikes, get_straddle_and_hedge_data are
    # inherited from BaseCachedOptionsDataLayer unchanged -- generation
    # always succeeds here (sample_data has no "missing" concept), so the
    # base's per-candidate try/except is simply never triggered on this path.
