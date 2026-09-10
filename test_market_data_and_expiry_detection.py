"""
Tests for:
  - market_data._resample_ohlc: proper OHLC resampling vs the naive
    row-skipping it replaced (regression test for the bug found while
    building the real-data validation scripts).
  - expiry_utils' calendar loading/lookup functions, which replaced the
    original day-of-week heuristic ("next Thursday") after NIFTY's weekly
    expiry day itself changed exchange-side.
"""

import datetime as dt
import warnings

import pandas as pd
import pytest

from market_data import _resample_ohlc, SyntheticMarketDataProvider
from expiry_utils import (
    load_expiry_calendar, get_next_expiry, get_prior_trading_day_for_expiry,
    select_monthly_hedge_expiry_from_calendar, select_monthly_hedge_expiry,
)

warnings.filterwarnings("ignore", message="Using the SHIPPED TEMPLATE")


def test_resample_ohlc_takes_last_close_not_first():
    """The bug: naive row-skipping (df.iloc[::N]) samples the close at the
    FIRST minute of each N-minute window. A proper resample takes the LAST
    close in the window -- these differ whenever price actually moves
    within the window, which real (and even synthetic) data always does."""
    timestamps = [dt.datetime(2026, 9, 1, 9, 15) + dt.timedelta(minutes=i) for i in range(30)]
    closes = list(range(100, 130))  # strictly increasing, so first != last within any window
    df = pd.DataFrame({"datetime": timestamps, "close": closes, "high": closes, "low": closes, "open": closes})

    resampled = _resample_ohlc(df, freq_minutes=15)
    naive_skip = df.iloc[::15].reset_index(drop=True)

    # proper resample's close for the first 15-min bin should be the LAST
    # close in that bin (index 14 -> close=114), not the first (index 0 -> close=100)
    assert resampled.iloc[0]["close"] == 114
    assert naive_skip.iloc[0]["close"] == 100
    assert resampled.iloc[0]["close"] != naive_skip.iloc[0]["close"], (
        "this test is only meaningful if proper resampling and naive row-skipping actually disagree"
    )
    assert resampled.iloc[0]["high"] == 114
    assert resampled.iloc[0]["low"] == 100
    assert resampled.iloc[0]["open"] == 100


def test_synthetic_provider_option_price_series_uses_proper_resample():
    """End-to-end check that SyntheticMarketDataProvider.option_price_series
    routes through the proper resample rather than the old naive skip."""
    start = dt.datetime(2026, 9, 1, 9, 15)
    end = dt.datetime(2026, 9, 1, 11, 0)
    provider = SyntheticMarketDataProvider(start, end, initial_spot=24500, annual_vol=0.2, flat_iv=0.13, seed=1)

    series_1min = provider.option_price_series(24500, "call", dt.date(2026, 9, 4), start, end, freq_minutes=1)
    series_15min = provider.option_price_series(24500, "call", dt.date(2026, 9, 4), start, end, freq_minutes=15)

    assert len(series_15min) < len(series_1min)
    first_bin_1min_closes = series_1min[series_1min["datetime"] <= start + dt.timedelta(minutes=14)]["close"]
    last_close_in_first_bin = first_bin_1min_closes.iloc[-1]
    assert abs(series_15min.iloc[0]["close"] - last_close_in_first_bin) < 1e-9


# ─────────────────────────────────────────────
# EXPIRY CALENDAR
# ─────────────────────────────────────────────

def test_load_expiry_calendar_default_shipped_template():
    calendar = load_expiry_calendar()
    assert set(calendar.columns) >= {"expiry_type", "expiry_date", "prior_trading_day"}
    assert set(calendar["expiry_type"]) <= {"weekly", "monthly"}
    assert (calendar["prior_trading_day"] < calendar["expiry_date"]).all()


def test_load_expiry_calendar_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_expiry_calendar(tmp_path / "does_not_exist.csv")


def test_load_expiry_calendar_missing_column_raises(tmp_path):
    bad_path = tmp_path / "bad_calendar.csv"
    bad_path.write_text("expiry_type,expiry_date\nweekly,2026-09-08\n")
    with pytest.raises(ValueError):
        load_expiry_calendar(bad_path)


def test_load_expiry_calendar_bad_ordering_raises(tmp_path):
    """prior_trading_day on/after expiry_date is a data-entry error --
    should fail loudly rather than silently produce a wrong expiry-eve close."""
    bad_path = tmp_path / "bad_calendar.csv"
    bad_path.write_text("expiry_type,expiry_date,prior_trading_day\nweekly,2026-09-08,2026-09-09\n")
    with pytest.raises(ValueError):
        load_expiry_calendar(bad_path)


def test_load_expiry_calendar_unknown_type_raises(tmp_path):
    bad_path = tmp_path / "bad_calendar.csv"
    bad_path.write_text("expiry_type,expiry_date,prior_trading_day\nquarterly,2026-09-08,2026-09-04\n")
    with pytest.raises(ValueError):
        load_expiry_calendar(bad_path)


def test_get_next_expiry_finds_first_on_or_after():
    calendar = load_expiry_calendar()
    expiry, prior = get_next_expiry(calendar, dt.date(2026, 9, 2), "weekly")
    # per the shipped template: weekly expiries are 9/1, 9/8, 9/15... -- the
    # first one on/after 9/2 should be 9/8
    assert expiry == dt.date(2026, 9, 8)
    assert prior < expiry


def test_get_next_expiry_exact_match_on_the_expiry_date_itself():
    calendar = load_expiry_calendar()
    expiry, prior = get_next_expiry(calendar, dt.date(2026, 9, 8), "weekly")
    assert expiry == dt.date(2026, 9, 8)


def test_get_next_expiry_beyond_calendar_coverage_raises():
    calendar = load_expiry_calendar()
    with pytest.raises(ValueError):
        get_next_expiry(calendar, dt.date(2030, 1, 1), "weekly")


def test_get_prior_trading_day_for_expiry_exact_match():
    calendar = load_expiry_calendar()
    prior = get_prior_trading_day_for_expiry(calendar, dt.date(2026, 9, 8), "weekly")
    assert prior < dt.date(2026, 9, 8)


def test_get_prior_trading_day_for_expiry_unknown_date_raises():
    calendar = load_expiry_calendar()
    with pytest.raises(ValueError):
        get_prior_trading_day_for_expiry(calendar, dt.date(2026, 9, 3), "weekly")  # not an expiry date in the template


def test_select_monthly_hedge_expiry_from_calendar_rolls_when_close():
    calendar = load_expiry_calendar()
    # per the template, Sept 2026 monthly expiry is 2026-09-29
    as_of = dt.date(2026, 9, 20)  # 9 days out -- should roll to October's monthly expiry
    expiry, prior = select_monthly_hedge_expiry_from_calendar(calendar, as_of, min_days=15)
    assert expiry == dt.date(2026, 10, 27)


def test_select_monthly_hedge_expiry_from_calendar_uses_current_when_far_enough():
    calendar = load_expiry_calendar()
    as_of = dt.date(2026, 9, 1)  # comfortably >15 days from 2026-09-29
    expiry, prior = select_monthly_hedge_expiry_from_calendar(calendar, as_of, min_days=15)
    assert expiry == dt.date(2026, 9, 29)


def test_select_monthly_hedge_expiry_pure_arithmetic_version_unchanged():
    """The older pure-date-arithmetic helper (for callers that already
    resolved both candidate dates themselves) should still work as before."""
    as_of = dt.date(2026, 9, 20)
    current_month_expiry = dt.date(2026, 9, 30)  # 10 days away -> too close
    next_month_expiry = dt.date(2026, 10, 30)
    assert select_monthly_hedge_expiry(as_of, current_month_expiry, next_month_expiry, min_days=15) == next_month_expiry


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
