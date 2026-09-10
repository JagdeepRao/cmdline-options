"""
Tests for data_cache.DataCache: cache-hit/miss behavior, edge-extension,
and the retry/backoff logic added on top of fetch_fn calls. Also a smoke
test for data_layer_sample.NiftyOptionsDataSample, which is what lets the
rest of the test suite (and any credential-free run) exercise the exact
same DataCache code path the real Breeze-backed layer uses.

Retry tests use tiny delays (DataCache(..., initial_delay_seconds=0.01))
rather than the production defaults, so this file runs in well under a
second instead of tens of seconds.
"""

import datetime as dt
import pandas as pd
import pytest

from data_cache import DataCache
from data_layer_sample import NiftyOptionsDataSample


def _df(dates: list[dt.date], value: float = 1.0) -> pd.DataFrame:
    return pd.DataFrame({
        "datetime": [dt.datetime.combine(d, dt.time(9, 15)) for d in dates],
        "close": [value] * len(dates),
    })


def test_cache_miss_then_hit(tmp_path):
    cache = DataCache(tmp_path)
    calls = []

    def fetch_fn(fd, td):
        calls.append((fd, td))
        return _df([fd, td])

    d1 = dt.date(2026, 9, 1)
    d5 = dt.date(2026, 9, 5)

    result1 = cache.get("KEY", d1, d5, fetch_fn)
    assert len(calls) == 1
    assert not result1.empty

    calls.clear()
    result2 = cache.get("KEY", d1, d5, fetch_fn)
    assert len(calls) == 0, "second call for the exact same range should be a pure cache hit, no fetch"
    assert len(result2) == len(result1)


def test_edge_extension_only_fetches_missing_tail(tmp_path):
    cache = DataCache(tmp_path)
    calls = []

    def fetch_fn(fd, td):
        calls.append((fd, td))
        # one row per day in the requested range
        days = []
        d = fd
        while d <= td:
            days.append(d)
            d += dt.timedelta(days=1)
        return _df(days)

    cache.get("KEY", dt.date(2026, 9, 1), dt.date(2026, 9, 5), fetch_fn)
    assert len(calls) == 1

    calls.clear()
    result = cache.get("KEY", dt.date(2026, 9, 1), dt.date(2026, 9, 10), fetch_fn)
    assert len(calls) == 1, "extending forward should only fetch the new tail, not the whole range again"
    fetched_from, fetched_to = calls[0]
    assert fetched_from == dt.date(2026, 9, 6), f"expected the gap to start right after the cached max, got {fetched_from}"
    assert fetched_to == dt.date(2026, 9, 10)
    assert len(result) == 10  # 9/1 .. 9/10 inclusive


def test_retry_succeeds_after_transient_exceptions(tmp_path):
    cache = DataCache(tmp_path, max_retries=4, initial_delay_seconds=0.01, backoff_multiplier=1.0)
    attempts = {"n": 0}

    def flaky_fetch_fn(fd, td):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ConnectionError("simulated throttle")
        return _df([fd])

    result = cache.get("KEY", dt.date(2026, 9, 1), dt.date(2026, 9, 1), flaky_fetch_fn)
    assert attempts["n"] == 3, "should have failed twice then succeeded on the third attempt"
    assert not result.empty


def test_retry_exhausted_reraises(tmp_path):
    cache = DataCache(tmp_path, max_retries=3, initial_delay_seconds=0.01, backoff_multiplier=1.0)
    attempts = {"n": 0}

    def always_fails(fd, td):
        attempts["n"] += 1
        raise ConnectionError("permanently throttled")

    with pytest.raises(ConnectionError):
        cache.get("KEY", dt.date(2026, 9, 1), dt.date(2026, 9, 1), always_fails)
    assert attempts["n"] == 3, "should have attempted exactly max_retries times before giving up"


def test_breeze_style_error_payload_retries_then_returns_empty(tmp_path, capsys):
    cache = DataCache(tmp_path, max_retries=2, initial_delay_seconds=0.01, backoff_multiplier=1.0)
    attempts = {"n": 0}

    def error_payload_fetch_fn(fd, td):
        attempts["n"] += 1
        return {"Success": None, "Error": "Access denied, throttled"}

    result = cache.get("KEY", dt.date(2026, 9, 1), dt.date(2026, 9, 1), error_payload_fetch_fn)
    assert attempts["n"] == 2
    assert result.empty
    captured = capsys.readouterr()
    assert "Access denied, throttled" in captured.out, "the provider's actual error message should be printed, not swallowed silently"


def test_breeze_style_error_payload_succeeds_on_retry(tmp_path):
    cache = DataCache(tmp_path, max_retries=3, initial_delay_seconds=0.01, backoff_multiplier=1.0)
    attempts = {"n": 0}

    def fetch_fn(fd, td):
        attempts["n"] += 1
        if attempts["n"] == 1:
            return {"Success": None, "Error": "throttled"}
        return _df([fd])

    result = cache.get("KEY", dt.date(2026, 9, 1), dt.date(2026, 9, 1), fetch_fn)
    assert attempts["n"] == 2
    assert not result.empty


def test_sample_data_layer_smoke(tmp_path):
    """The sample data layer routes through the SAME DataCache class the
    real Breeze layer uses -- this confirms that full pipeline (cache +
    synthetic generator) works end-to-end without any live credentials."""
    layer = NiftyOptionsDataSample(cache_dir=tmp_path)

    index_df = layer.get_index_historical(dt.date(2026, 9, 1), dt.date(2026, 9, 3))
    assert not index_df.empty
    assert {"datetime", "close"}.issubset(index_df.columns)

    option_df = layer.get_option_historical(
        expiry=dt.date(2026, 9, 4), strike=24500, right="call",
        from_date=dt.date(2026, 9, 1), to_date=dt.date(2026, 9, 3),
    )
    assert not option_df.empty
    assert (option_df["close"] > 0).all(), "synthetic option prices should never be zero or negative"

    atm_result = layer.find_atm_strike(
        expiry=dt.date(2026, 9, 4), approx_spot=24800,
        as_of=dt.datetime(2026, 9, 2, 11, 0),
    )
    assert "strike" in atm_result and "candidates" in atm_result
    assert len(atm_result["candidates"]) > 1


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
