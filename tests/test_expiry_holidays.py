"""
Tests for the holiday-awareness pieces added to expiry_utils.py:
  - load_holidays() / is_trading_day() / get_holiday_shifted_trading_day()
  - validate_calendar_against_holidays()
  - the shipped expiry_calendar.csv + nse_holidays.csv pair itself, which
    must stay mutually consistent (this is the regression test for the
    real bug found in review: the shipped calendar previously listed
    2026-11-24 as a monthly expiry, but 2026-11-24 is a real NSE holiday).
"""

import datetime as dt
import warnings

import pandas as pd
import pytest

from nifty_backtester.expiry_utils import (
    load_expiry_calendar, load_holidays, is_trading_day,
    get_holiday_shifted_trading_day, validate_calendar_against_holidays,
    DEFAULT_HOLIDAYS_PATH,
)

warnings.filterwarnings("ignore", message="Using the SHIPPED TEMPLATE")


def test_load_holidays_returns_a_set_of_dates():
    holidays = load_holidays()
    assert isinstance(holidays, set)
    assert dt.date(2026, 9, 14) in holidays   # Ganesh Chaturthi -- the date that started this
    assert dt.date(2026, 11, 24) in holidays  # Guru Nanak Jayanti -- the one that was silently wrong in the calendar


def test_load_holidays_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_holidays(tmp_path / "does_not_exist.csv")


def test_is_trading_day_excludes_weekends_and_holidays():
    holidays = {dt.date(2026, 9, 14)}
    assert is_trading_day(dt.date(2026, 9, 15), holidays)          # ordinary Tuesday
    assert not is_trading_day(dt.date(2026, 9, 14), holidays)      # listed holiday (also happens to be a weekday)
    assert not is_trading_day(dt.date(2026, 9, 12), holidays)      # Saturday
    assert not is_trading_day(dt.date(2026, 9, 13), holidays)      # Sunday


def test_get_holiday_shifted_trading_day_steps_back_over_holiday_and_weekend():
    holidays = {dt.date(2026, 11, 24)}
    # 2026-11-24 is a Tuesday holiday -- should step back to Monday 2026-11-23
    assert get_holiday_shifted_trading_day(dt.date(2026, 11, 24), holidays) == dt.date(2026, 11, 23)
    # an already-valid trading day should be returned unchanged
    assert get_holiday_shifted_trading_day(dt.date(2026, 11, 23), holidays) == dt.date(2026, 11, 23)


def test_get_holiday_shifted_trading_day_steps_back_over_weekend_only():
    # 2026-09-13 is a Sunday, no holiday involved -- should land on Friday 2026-09-11
    assert get_holiday_shifted_trading_day(dt.date(2026, 9, 13), set()) == dt.date(2026, 9, 11)


def test_validate_calendar_against_holidays_passes_for_clean_calendar():
    calendar = pd.DataFrame([
        {"expiry_type": "weekly", "expiry_date": dt.date(2026, 9, 15), "prior_trading_day": dt.date(2026, 9, 11)},
    ])
    validate_calendar_against_holidays(calendar, {dt.date(2026, 9, 14)})  # should not raise


def test_validate_calendar_against_holidays_catches_expiry_on_holiday():
    calendar = pd.DataFrame([
        {"expiry_type": "monthly", "expiry_date": dt.date(2026, 11, 24), "prior_trading_day": dt.date(2026, 11, 20)},
    ])
    with pytest.raises(ValueError, match="2026-11-24"):
        validate_calendar_against_holidays(calendar, {dt.date(2026, 11, 24)})


def test_validate_calendar_against_holidays_catches_prior_trading_day_on_holiday():
    calendar = pd.DataFrame([
        {"expiry_type": "weekly", "expiry_date": dt.date(2026, 10, 6), "prior_trading_day": dt.date(2026, 10, 2)},
    ])
    with pytest.raises(ValueError, match="2026-10-02"):
        validate_calendar_against_holidays(calendar, {dt.date(2026, 10, 2)})


def test_validate_calendar_against_holidays_catches_weekend():
    calendar = pd.DataFrame([
        {"expiry_type": "weekly", "expiry_date": dt.date(2026, 9, 13), "prior_trading_day": dt.date(2026, 9, 11)},
    ])  # 2026-09-13 is a Sunday
    with pytest.raises(ValueError, match="weekend"):
        validate_calendar_against_holidays(calendar, set())


def test_validate_calendar_against_holidays_reports_every_violation_not_just_first():
    calendar = pd.DataFrame([
        {"expiry_type": "monthly", "expiry_date": dt.date(2026, 11, 24), "prior_trading_day": dt.date(2026, 11, 20)},
        {"expiry_type": "weekly", "expiry_date": dt.date(2026, 10, 6), "prior_trading_day": dt.date(2026, 10, 2)},
    ])
    holidays = {dt.date(2026, 11, 24), dt.date(2026, 10, 2)}
    with pytest.raises(ValueError) as exc_info:
        validate_calendar_against_holidays(calendar, holidays)
    assert "2 entr" in str(exc_info.value)
    assert "2026-11-24" in str(exc_info.value)
    assert "2026-10-02" in str(exc_info.value)


def test_shipped_calendar_and_holiday_list_are_mutually_consistent():
    """Regression test for the actual bug found in review: the shipped
    expiry_calendar.csv previously had monthly expiry_date=2026-11-24,
    which nse_holidays.csv lists as a real NSE holiday (Guru Nanak
    Jayanti). Both shipped files must stay consistent with each other."""
    calendar = load_expiry_calendar()
    holidays = load_holidays()
    validate_calendar_against_holidays(calendar, holidays)  # should not raise


def test_default_holidays_path_points_at_shipped_file():
    assert DEFAULT_HOLIDAYS_PATH.name == "nse_holidays.csv"
    assert DEFAULT_HOLIDAYS_PATH.exists()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
