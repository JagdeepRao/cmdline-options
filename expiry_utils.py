"""
Expiry-calendar helpers shared across the engine.

Kept as a standalone module (rather than folded into strategy.py or the
data layer) because these are pure calendar-arithmetic rules that don't
depend on positions, indicators, or Breeze — they should be trivially
unit-testable on their own and reusable wherever an expiry decision needs
to be made.
"""

import datetime as dt
from typing import Optional


def select_monthly_hedge_expiry(
    as_of: dt.date,
    current_month_expiry: dt.date,
    next_month_expiry: dt.date,
    min_days: int = 15,
) -> dt.date:
    """Point 1's monthly-hedge selection rule: use the current calendar
    month's monthly expiry as the hedge, UNLESS it's less than `min_days`
    away from `as_of`, in which case roll to next month's monthly expiry
    instead.

    `current_month_expiry` / `next_month_expiry` are passed in rather than
    computed here (e.g. 'last Thursday of the month', with holiday
    adjustments) because that calendar logic belongs with the data layer /
    broker's own expiry list, not duplicated here from assumptions about
    NSE's holiday calendar. Fetch the two nearest monthly expiries from
    whatever expiry-list source you have (Breeze's option chain response,
    or a maintained NSE expiry calendar) and pass them straight through.
    """
    days_to_current = (current_month_expiry - as_of).days
    if days_to_current >= min_days:
        return current_month_expiry
    return next_month_expiry


def is_expiry_eve_close_bar(
    as_of: dt.datetime,
    expiry_date: dt.date,
    close_time: dt.time = dt.time(15, 30),
) -> bool:
    """True on/after the force-close moment the day before expiry (point
    6f) — the sold/hedge straddles (and, per config, the OTM legs) must be
    flat before expiry day's extra margin requirements kick in.

    Simplification stated plainly: 'the day before' is calendar-date minus
    one day, not the previous NSE TRADING day. For a Thursday expiry that's
    Wednesday, which is correct; it would be wrong for a Monday expiry
    (naively landing on Sunday) — Breeze/NSE weekly expiries are
    consistently non-Monday in practice, but if that ever changes, this
    needs a real trading-calendar 'previous business day' lookup instead.
    """
    expiry_eve = expiry_date - dt.timedelta(days=1)
    return as_of.date() == expiry_eve and as_of.time() >= close_time


def is_on_or_after_expiry(as_of: dt.datetime, expiry_date: dt.date) -> bool:
    """True once the calendar date has reached expiry itself — a hard
    backstop so nothing is ever left open past expiry regardless of
    whether the expiry-eve close bar was hit exactly."""
    return as_of.date() >= expiry_date
