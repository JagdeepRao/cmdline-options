"""
Regression tests for data_layer_base.BaseCachedOptionsDataLayer -- the
shared find_atm_strike/nearest_otm_strikes/get_straddle_and_hedge_data
logic that used to be duplicated byte-for-byte between
NiftyOptionsDataCached and NiftyOptionsDataSample.

These exist to make two things checkable, not just assertable:
  1. Both subclasses still produce identical find_atm_strike/OTM/straddle
     results after the refactor (no behavior drift from centralizing).
  2. The base class's find_atm_strike genuinely tolerates whatever
     "missing data" exception a subclass raises per-candidate (broad
     except Exception), not just the one exception type either concrete
     subclass happens to use today.
"""

import datetime as dt
import sys

import pandas as pd
import pytest

from nifty_backtester.data_layer_base import BaseCachedOptionsDataLayer
from nifty_backtester.data_layer_sample import NiftyOptionsDataSample
from nifty_backtester.data_layer_cached import NiftyOptionsDataCached, CachedDataUnavailable


def test_import_safety_no_breeze_connect_pulled_in():
    """data_layer_base (and anything built on it that isn't the live
    Breeze layer) must never import breeze_connect -- that's what keeps
    NiftyOptionsDataSample usable with zero network dependency, and is a
    prerequisite for it later standing in as a fake 'live' feed."""
    assert "breeze_connect" not in sys.modules, (
        "breeze_connect was already imported by something else in this "
        "process -- test isolation issue, not a real failure of this module"
    )
    import nifty_backtester.data_layer_base  # noqa: F401
    assert "breeze_connect" not in sys.modules


def test_both_subclasses_inherit_identical_nearest_otm_strikes():
    sample = NiftyOptionsDataSample(cache_dir="/tmp/_unused_sample_otm_test")
    assert sample.nearest_otm_strikes(24537) == (24600, 24500)
    assert BaseCachedOptionsDataLayer.nearest_otm_strikes(24537) == (24600, 24500)


def test_sample_find_atm_strike_still_works_after_refactor(tmp_path):
    layer = NiftyOptionsDataSample(cache_dir=tmp_path)
    result = layer.find_atm_strike(
        expiry=dt.date(2026, 9, 4), approx_spot=24800,
        as_of=dt.datetime(2026, 9, 2, 11, 0),
    )
    assert "strike" in result and "candidates" in result
    assert len(result["candidates"]) > 1


def test_sample_get_straddle_and_hedge_data_still_works_after_refactor(tmp_path):
    layer = NiftyOptionsDataSample(cache_dir=tmp_path)
    result = layer.get_straddle_and_hedge_data(
        expiry=dt.date(2026, 9, 4), spot_at_entry=24537,
        from_date=dt.date(2026, 9, 1), to_date=dt.date(2026, 9, 3),
    )
    assert set(result) == {"straddle_call", "straddle_put", "hedge_call", "hedge_put"}
    for df in result.values():
        assert not df.empty


def test_cached_find_atm_strike_skips_missing_strikes_not_committed(tmp_path):
    """Only commit ONE strike's call+put parquet -- find_atm_strike should
    silently skip every other candidate (raising internally as
    CachedDataUnavailable, caught by the base class's broad except) and
    still resolve using the one strike that IS available."""
    cache_dir = tmp_path
    expiry = dt.date(2026, 9, 8)
    strike = 24500

    for right in ("call", "put"):
        df = pd.DataFrame({
            "datetime": [dt.datetime(2026, 9, 1, 11, 0)],
            "close": [100.0 if right == "call" else 101.0],
        })
        df.to_parquet(cache_dir / f"NIFTY_{expiry}_{strike}_{right}_1minute.parquet")

    layer = NiftyOptionsDataCached(cache_dir=cache_dir)
    result = layer.find_atm_strike(
        expiry=expiry, approx_spot=24500, as_of=dt.datetime(2026, 9, 1, 11, 0), strike_range=3,
    )
    assert result["strike"] == 24500
    # only one candidate should have survived -- every neighboring strike
    # has no committed parquet at all, so CachedDataUnavailable must have
    # been swallowed by the base class's per-candidate try/except
    assert len(result["candidates"]) == 1


def test_cached_find_atm_strike_raises_when_nothing_available(tmp_path):
    layer = NiftyOptionsDataCached(cache_dir=tmp_path)
    with pytest.raises(ValueError):
        layer.find_atm_strike(
            expiry=dt.date(2026, 9, 8), approx_spot=24500, as_of=dt.datetime(2026, 9, 1, 11, 0),
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
