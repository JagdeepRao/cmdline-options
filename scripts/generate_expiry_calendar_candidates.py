"""
Generates CANDIDATE expiry_calendar.csv rows for a date range, by combining:
  - expiry_utils.WEEKLY_EXPIRY_WEEKDAY_REGIMES (the documented history of
    which weekday NIFTY's weekly/monthly expiry falls on -- Thursday
    through 2025-08-28, Tuesday from 2025-09-01; see that table's
    docstring for sourcing/caveats), and
  - expiry_utils.load_holidays() (nse_holidays.csv), applying the
    documented NSE convention: if the regime's weekday falls on a holiday,
    expiry shifts to the prior trading day.

THIS DOES NOT WRITE expiry_calendar.csv DIRECTLY, and its output is NOT
treated as authoritative anywhere else in this codebase. That's
deliberate: expiry_calendar.csv is the single source of truth precisely
BECAUSE the engine never derives an expiry date from a weekday rule (see
expiry_utils.py's module docstring) -- rules like "Tuesday" have already
changed exchange-side before and will again. This script exists only to
save a human the manual work of applying the known regime + holiday rules
by hand; REVIEW every candidate row against your broker's contract master
or an official NSE circular before pasting it into expiry_calendar.csv.

Monthly expiry here is generated as the LAST occurrence of the regime's
weekday in each calendar month (holiday-shifted the same way as weekly) --
confirmed as the current rule for the post-2025-09-01 Tuesday regime; if
you're generating candidates that span an earlier regime, double-check
this rule held then too before trusting those rows.

Usage (from the repo root):
  python3 scripts/generate_expiry_calendar_candidates.py --from-date 2026-12-01 --to-date 2027-02-28
"""

import argparse
import calendar as calendar_module
import datetime as dt

from nifty_backtester.expiry_utils import (
    WEEKLY_EXPIRY_WEEKDAY_REGIMES, load_holidays, get_holiday_shifted_trading_day, is_trading_day,
)


def _weekday_for(d: dt.date) -> int:
    """Looks up which weekday is the expiry weekday on date `d`, per
    WEEKLY_EXPIRY_WEEKDAY_REGIMES. Raises if `d` falls in a gap between
    regimes (shouldn't happen with the shipped table, but fail loudly
    rather than silently picking the wrong regime if one is ever added
    with a gap)."""
    for start, end, weekday in WEEKLY_EXPIRY_WEEKDAY_REGIMES:
        if d >= start and (end is None or d <= end):
            return weekday
    raise ValueError(
        f"{d} falls outside every known regime in WEEKLY_EXPIRY_WEEKDAY_REGIMES -- "
        f"extend that table (with a confirmed source) before generating candidates this far out."
    )


def _prior_trading_day(expiry_date: dt.date, holidays: set) -> dt.date:
    cursor = expiry_date - dt.timedelta(days=1)
    return get_holiday_shifted_trading_day(cursor, holidays)


def _weekly_candidates(from_date: dt.date, to_date: dt.date, holidays: set) -> list[dict]:
    rows = []
    cursor = from_date
    while cursor <= to_date:
        weekday = _weekday_for(cursor)
        days_ahead = (weekday - cursor.weekday()) % 7
        raw_expiry = cursor + dt.timedelta(days=days_ahead)
        if raw_expiry > to_date:
            break
        expiry = get_holiday_shifted_trading_day(raw_expiry, holidays)
        rows.append({
            "expiry_type": "weekly", "expiry_date": expiry,
            "prior_trading_day": _prior_trading_day(expiry, holidays),
            "note": "" if expiry == raw_expiry else f"holiday-shifted from {raw_expiry}",
        })
        cursor = raw_expiry + dt.timedelta(days=1)
    return rows


def _monthly_candidates(from_date: dt.date, to_date: dt.date, holidays: set) -> list[dict]:
    rows = []
    year, month = from_date.year, from_date.month
    while dt.date(year, month, 1) <= to_date:
        _, last_day = calendar_module.monthrange(year, month)
        month_end = dt.date(year, month, last_day)
        weekday = _weekday_for(month_end)
        # last occurrence of `weekday` on/before month_end
        offset = (month_end.weekday() - weekday) % 7
        raw_expiry = month_end - dt.timedelta(days=offset)
        if from_date <= raw_expiry <= to_date or from_date <= raw_expiry:
            expiry = get_holiday_shifted_trading_day(raw_expiry, holidays)
            if expiry >= from_date:
                rows.append({
                    "expiry_type": "monthly", "expiry_date": expiry,
                    "prior_trading_day": _prior_trading_day(expiry, holidays),
                    "note": "" if expiry == raw_expiry else f"holiday-shifted from {raw_expiry}",
                })
        month += 1
        if month > 12:
            month, year = 1, year + 1
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--to-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--holidays", default=None, help="path to nse_holidays.csv (default: the shipped one)")
    args = parser.parse_args()

    from_date = dt.datetime.strptime(args.from_date, "%Y-%m-%d").date()
    to_date = dt.datetime.strptime(args.to_date, "%Y-%m-%d").date()
    holidays = load_holidays(args.holidays) if args.holidays else load_holidays()

    weekly = _weekly_candidates(from_date, to_date, holidays)
    monthly = _monthly_candidates(from_date, to_date, holidays)

    print("expiry_type,expiry_date,prior_trading_day")
    for row in sorted(weekly + monthly, key=lambda r: (r["expiry_type"], r["expiry_date"])):
        print(f"{row['expiry_type']},{row['expiry_date']},{row['prior_trading_day']}")

    shifted = [r for r in weekly + monthly if r["note"]]
    if shifted:
        print("\n# Rows shifted off their regular weekday by a holiday -- double-check these against")
        print("# an official NSE circular or your broker's contract master before committing:")
        for r in shifted:
            print(f"#   {r['expiry_type']} {r['expiry_date']}: {r['note']}")

    print(
        "\n# REVIEW EVERY ROW ABOVE before pasting into expiry_calendar.csv -- this is generated "
        "from a documented weekday-regime table + nse_holidays.csv, not fetched from NSE itself. "
        "See this script's module docstring for what's confirmed vs. not."
    )


if __name__ == "__main__":
    main()
