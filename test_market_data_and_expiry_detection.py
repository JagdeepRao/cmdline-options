"""
Tests for:
  - market_data._resample_ohlc: proper OHLC resampling vs the naive
    row-skipping it replaced (regression test for the bug found while
    building the real-data validation scripts).
  - run_backtest_from_range's expiry auto-detection helpers.
"""

import datetime as dt
import pandas as pd
import pytest

from market_data import _resample_ohlc, SyntheticMarketDataProvider
from run_backtest_from_range import _next_or_same_thursday, _last_thursday_of_month, _derive_monthly_hedge_expiry


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
    # OHLC columns should reflect the whole window, not just one row
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
    # the 15-min series' close values should be a SUBSET of actual 1-min
    # closes that occurred at the LAST minute of each bin, not the first
    first_bin_1min_closes = series_1min[series_1min["datetime"] <= start + dt.timedelta(minutes=14)]["close"]
    last_close_in_first_bin = first_bin_1min_closes.iloc[-1]
    assert abs(series_15min.iloc[0]["close"] - last_close_in_first_bin) < 1e-9


@pytest.mark.parametrize("input_date,expected", [
    (dt.date(2026, 9, 1), dt.date(2026, 9, 3)),   # Tuesday -> that week's Thursday
    (dt.date(2026, 9, 3), dt.date(2026, 9, 3)),   # Thursday -> itself
    (dt.date(2026, 9, 4), dt.date(2026, 9, 10)),  # Friday -> NEXT week's Thursday (correctly, not the one just passed)
])
def test_next_or_same_thursday(input_date, expected):
    assert _next_or_same_thursday(input_date) == expected


def test_last_thursday_of_month():
    # September 2026: last Thursday is the 24th (verified against a calendar)
    assert _last_thursday_of_month(2026, 9) == dt.date(2026, 9, 24)


def test_derive_monthly_hedge_expiry_rolls_when_close():
    # as_of near the end of the month, close to that month's own last
    # Thursday -> should roll to next month
    as_of = dt.date(2026, 9, 20)  # last Thursday of Sept 2026 is the 24th -- only 4 days away
    result = _derive_monthly_hedge_expiry(as_of, min_days=15)
    assert result == _last_thursday_of_month(2026, 10)


def test_derive_monthly_hedge_expiry_uses_current_month_when_far_enough():
    as_of = dt.date(2026, 9, 1)  # last Thursday of Sept 2026 is the 24th -- comfortably >15 days away
    result = _derive_monthly_hedge_expiry(as_of, min_days=15)
    assert result == _last_thursday_of_month(2026, 9)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
