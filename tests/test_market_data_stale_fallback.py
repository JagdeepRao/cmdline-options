"""
Tests for market_data._most_recent_price_at_or_before and its wiring
through BreezeMarketDataProvider.get_spot/get_option_price.

Regression tests for a real failure found in review (not hypothesized):
'ValueError: No option data available for strike=22700 right=call
expiry=2025-09-30 around 2025-09-23 15:30:00' -- caused by two real-world
boundary conditions:
  1. The last 1-minute candle of an NSE session is timestamped 15:29, not
     15:30 -- querying exactly at market close needs to fall back to the
     most recent prior print instead of raising when that exact day's
     fetch comes back empty.
  2. A given strike may genuinely not trade at all on a given day (thin
     OTM/ITM weeklies) -- Breeze then returns zero rows for the WHOLE
     day, not just a gap at one timestamp.
Also covers a related look-ahead-bias bug found while fixing the above:
the old code picked the "nearest by absolute distance" bar, which could
silently select a bar timestamped AFTER as_of.
"""

import datetime as dt

import pandas as pd
import pytest

from nifty_backtester.market_data import BreezeMarketDataProvider, _most_recent_price_at_or_before


def _df(rows: list[tuple[str, float]]) -> pd.DataFrame:
    """rows: [(iso_datetime, close), ...]"""
    return pd.DataFrame({"datetime": [r[0] for r in rows], "close": [r[1] for r in rows]})


class FakeDayByDayDataLayer:
    """Test double: get_option_historical/get_index_historical are called
    ONE DAY AT A TIME by _most_recent_price_at_or_before (from_date ==
    to_date always) -- this fake just looks up a pre-seeded DataFrame per
    calendar date, returning an empty frame for any date not seeded,
    exactly mimicking a real illiquid-contract/no-trades day."""

    def __init__(self, by_date: dict[dt.date, pd.DataFrame], daily_by_date: dict[dt.date, pd.DataFrame] = None):
        self.by_date = by_date
        self.daily_by_date = daily_by_date or {}
        self.calls: list[tuple[dt.date, dt.date]] = []

    def get_option_historical(self, expiry, strike, right, from_date, to_date, interval="1minute"):
        self.calls.append((from_date, to_date))
        assert from_date == to_date, "provider should fetch one day at a time"
        if interval == "1day":
            return self.daily_by_date.get(from_date, pd.DataFrame())
        return self.by_date.get(from_date, pd.DataFrame())

    def get_index_historical(self, from_date, to_date, interval="1minute"):
        self.calls.append((from_date, to_date))
        assert from_date == to_date
        if interval == "1day":
            return self.daily_by_date.get(from_date, pd.DataFrame())
        return self.by_date.get(from_date, pd.DataFrame())


def test_same_day_data_used_directly_no_fallback(capsys):
    layer = FakeDayByDayDataLayer({
        dt.date(2025, 9, 23): _df([("2025-09-23 15:28:00", 100.0), ("2025-09-23 15:29:00", 101.0)]),
    })
    provider = BreezeMarketDataProvider(layer)
    price = provider.get_option_price(22700, "call", dt.date(2025, 9, 30), dt.datetime(2025, 9, 23, 15, 30, 0))
    assert price == 101.0  # the 15:29 bar -- nearest AT/BEFORE 15:30
    assert len(layer.calls) == 1, "should not have needed to fall back to an earlier day"
    assert "stale" not in capsys.readouterr().out


def test_market_close_boundary_finds_1529_candle_when_1530_does_not_exist(capsys):
    """Regression test for boundary condition #1: the real failure was at
    exactly 15:30:00 -- the day's last candle (15:29) must still resolve
    correctly rather than requiring an exact-timestamp match."""
    layer = FakeDayByDayDataLayer({
        dt.date(2025, 9, 23): _df([("2025-09-23 15:29:00", 55.5)]),  # no 15:30 bar at all
    })
    provider = BreezeMarketDataProvider(layer)
    price = provider.get_option_price(22700, "call", dt.date(2025, 9, 30), dt.datetime(2025, 9, 23, 15, 30, 0))
    assert price == 55.5


def test_market_close_post_market_print_found_for_same_day(capsys):
    """At market close (15:30:00), NSE post-market closing prints (e.g. 15:39:00)
    on the SAME day must be resolved without falling back to a prior day."""
    layer = FakeDayByDayDataLayer({
        dt.date(2024, 1, 2): _df([("2024-01-02 15:39:00", 120.0)]),
    })
    provider = BreezeMarketDataProvider(layer)
    price = provider.get_option_price(22700, "call", dt.date(2024, 1, 9), dt.datetime(2024, 1, 2, 15, 30, 0))
    assert price == 120.0
    out = capsys.readouterr().out
    assert "stale" not in out


def test_empty_day_falls_back_to_most_recent_prior_day_with_warning(capsys):
    """Regression test for boundary condition #2: a whole day with zero
    prints for this specific contract (thin/illiquid strike) must fall
    back to the most recent day that DOES have data, not raise."""
    layer = FakeDayByDayDataLayer({
        dt.date(2025, 9, 22): _df([("2025-09-22 15:29:00", 48.0)]),
        # 2025-09-23 deliberately has NO entry -- simulates a real no-trades day
    })
    provider = BreezeMarketDataProvider(layer)
    price = provider.get_option_price(22700, "call", dt.date(2025, 9, 30), dt.datetime(2025, 9, 23, 15, 30, 0))
    assert price == 48.0
    assert len(layer.calls) >= 2  # tried 9/23 (1min & 1day empty), then 9/22 (found it)
    out = capsys.readouterr().out
    assert "stale" in out and "2025-09-22" in out


def test_never_uses_a_bar_after_as_of_even_if_numerically_nearest(capsys):
    """The bug this replaced picked 'nearest by absolute distance', which
    could select a bar AFTER as_of if it happened to be closer -- direct
    look-ahead bias in a backtest. Must always prefer the most recent bar
    AT OR BEFORE as_of, even when a later bar is numerically closer."""
    layer = FakeDayByDayDataLayer({
        dt.date(2025, 9, 23): _df([
            ("2025-09-23 10:00:00", 10.0),   # 5 minutes before as_of
            ("2025-09-23 10:05:30", 999.0),  # 30 seconds AFTER as_of -- must NEVER be picked
        ]),
    })
    provider = BreezeMarketDataProvider(layer)
    price = provider.get_option_price(22700, "call", dt.date(2025, 9, 30), dt.datetime(2025, 9, 23, 10, 5, 0))
    assert price == 10.0, "must use the 10:00:00 bar, not the numerically-closer-but-future 10:05:30 bar"


def test_falls_back_multiple_days_when_several_consecutive_days_are_empty(capsys):
    layer = FakeDayByDayDataLayer({
        dt.date(2025, 9, 18): _df([("2025-09-18 15:29:00", 33.0)]),
        # 9/19, 9/22, 9/23 all empty (9/20-21 is a weekend, not queried since we go by calendar day here)
    })
    provider = BreezeMarketDataProvider(layer)
    price = provider.get_option_price(22700, "call", dt.date(2025, 9, 30), dt.datetime(2025, 9, 23, 15, 30, 0))
    assert price == 33.0


def test_same_day_1day_daily_candle_fallback_when_intraday_missing(capsys):
    layer = FakeDayByDayDataLayer(
        by_date={},  # 1min intraday empty
        daily_by_date={
            dt.date(2025, 9, 23): _df([("2025-09-23 00:00:00", 250.0)]),
        },
    )
    provider = BreezeMarketDataProvider(layer, max_stale_lookback_days=2)
    price = provider.get_option_price(22700, "call", dt.date(2025, 9, 30), dt.datetime(2025, 9, 23, 15, 30, 0))
    assert price == 250.0
    out = capsys.readouterr().out
    assert "official NSE daily closing price" in out


def test_raises_clearly_when_nothing_found_within_lookback_window():
    layer = FakeDayByDayDataLayer({})  # nothing seeded at all
    provider = BreezeMarketDataProvider(layer, max_stale_lookback_days=2)
    with pytest.raises(ValueError, match="no usable data"):
        provider.get_option_price(22700, "call", dt.date(2025, 9, 30), dt.datetime(2025, 9, 23, 15, 30, 0))


def test_get_spot_uses_the_same_fallback():
    layer = FakeDayByDayDataLayer({
        dt.date(2025, 9, 22): _df([("2025-09-22 15:29:00", 24555.5)]),
    })
    provider = BreezeMarketDataProvider(layer)
    spot = provider.get_spot(dt.datetime(2025, 9, 23, 15, 30, 0))
    assert spot == 24555.5


def test_max_stale_lookback_days_is_configurable():
    layer = FakeDayByDayDataLayer({
        dt.date(2025, 9, 20): _df([("2025-09-20 15:29:00", 40.0)]),  # 3 calendar days before 9/23
    })
    too_strict = BreezeMarketDataProvider(layer, max_stale_lookback_days=1)
    with pytest.raises(ValueError):
        too_strict.get_option_price(22700, "call", dt.date(2025, 9, 30), dt.datetime(2025, 9, 23, 15, 30, 0))

    lenient = BreezeMarketDataProvider(layer, max_stale_lookback_days=5)
    price = lenient.get_option_price(22700, "call", dt.date(2025, 9, 30), dt.datetime(2025, 9, 23, 15, 30, 0))
    assert price == 40.0


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
