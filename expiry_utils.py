"""
Expiry-calendar helpers shared across the engine.

WHY A DATA FILE INSTEAD OF A RULE: NIFTY's weekly expiry day itself has
changed exchange-side (Thursday, then other days at different points) --
any "next Thursday" / "day before expiry = calendar date minus one" style
rule is a bet on a convention that has already proven not to hold. The
only durable fix is to stop computing expiry dates and instead look them
up from an explicit, maintained source of truth: expiry_calendar.csv.

expiry_calendar.csv columns:
  expiry_type       "weekly" or "monthly"
  expiry_date       the contract's actual expiry date (YYYY-MM-DD)
  prior_trading_day the last real NSE trading day before that expiry,
                     with holidays already accounted for -- NOT
                     necessarily expiry_date minus one calendar day (a
                     Monday holiday would push this back to the prior
                     Friday, for example)

THE SHIPPED expiry_calendar.csv IS AN ILLUSTRATIVE TEMPLATE, not the real
NSE calendar -- it exists so the test suite and demo scripts have
something to load. Replace it with the actual expiry/holiday calendar
(from NSE's published circulars or your broker's contract master) before
running this against real trading decisions. load_expiry_calendar() prints
a one-time warning to make this hard to miss.
"""

import datetime as dt
import warnings
from pathlib import Path
from typing import Optional

import pandas as pd

DEFAULT_CALENDAR_PATH = Path(__file__).parent / "expiry_calendar.csv"


def load_expiry_calendar(path: Path = DEFAULT_CALENDAR_PATH) -> pd.DataFrame:
    """Loads and validates the expiry calendar. Returns a DataFrame sorted
    by expiry_date with proper date dtypes, columns: expiry_type,
    expiry_date, prior_trading_day.

    Raises FileNotFoundError with a clear message if the file is missing,
    and ValueError if required columns are absent or a row's
    prior_trading_day isn't strictly before its expiry_date (a sign the
    file was hand-edited incorrectly) -- fail loudly here rather than
    silently producing a wrong expiry-eve close later.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Expiry calendar not found at {path}. This file is required -- "
            f"there is no day-of-week fallback (that's the whole point: NSE's "
            f"weekly expiry day itself has changed before, so guessing from a "
            f"rule is exactly what this file replaces)."
        )

    df = pd.read_csv(path)
    required_cols = {"expiry_type", "expiry_date", "prior_trading_day"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Expiry calendar at {path} is missing required column(s): {missing}")

    df["expiry_date"] = pd.to_datetime(df["expiry_date"]).dt.date
    df["prior_trading_day"] = pd.to_datetime(df["prior_trading_day"]).dt.date

    bad_rows = df[df["prior_trading_day"] >= df["expiry_date"]]
    if not bad_rows.empty:
        raise ValueError(
            f"Expiry calendar at {path} has {len(bad_rows)} row(s) where "
            f"prior_trading_day is not strictly before expiry_date -- check "
            f"for a data-entry error:\n{bad_rows.to_string(index=False)}"
        )

    unknown_types = set(df["expiry_type"]) - {"weekly", "monthly"}
    if unknown_types:
        raise ValueError(f"Expiry calendar at {path} has unrecognized expiry_type value(s): {unknown_types}")

    if path == DEFAULT_CALENDAR_PATH:
        warnings.warn(
            "Using the SHIPPED TEMPLATE expiry_calendar.csv -- this is illustrative "
            "sample data, not the real NSE calendar. Replace it with the actual "
            "expiry/holiday calendar before running against real trading decisions.",
            stacklevel=2,
        )

    return df.sort_values(["expiry_type", "expiry_date"]).reset_index(drop=True)


def get_next_expiry(
    calendar: pd.DataFrame,
    as_of: dt.date,
    expiry_type: str,
) -> tuple[dt.date, dt.date]:
    """Returns (expiry_date, prior_trading_day) for the first expiry of
    `expiry_type` on or after `as_of`. Raises ValueError if the calendar
    doesn't cover that far -- extend expiry_calendar.csv rather than
    silently falling back to a guessed date."""
    subset = calendar[(calendar["expiry_type"] == expiry_type) & (calendar["expiry_date"] >= as_of)]
    if subset.empty:
        raise ValueError(
            f"No {expiry_type} expiry on/after {as_of} found in the calendar -- "
            f"it doesn't cover this far forward. Extend expiry_calendar.csv."
        )
    row = subset.iloc[0]
    return row["expiry_date"], row["prior_trading_day"]


def get_prior_trading_day_for_expiry(
    calendar: pd.DataFrame,
    expiry_date: dt.date,
    expiry_type: str,
) -> dt.date:
    """Exact-match lookup for when the caller already knows a SPECIFIC
    expiry date (e.g. it was passed explicitly on the command line) and
    just needs that contract's prior trading day -- as opposed to
    get_next_expiry's 'first expiry on/after some date' search."""
    match = calendar[(calendar["expiry_type"] == expiry_type) & (calendar["expiry_date"] == expiry_date)]
    if match.empty:
        raise ValueError(
            f"No {expiry_type} expiry dated {expiry_date} found in the calendar -- "
            f"add it to expiry_calendar.csv, or double check the date."
        )
    return match.iloc[0]["prior_trading_day"]


def select_monthly_hedge_expiry_from_calendar(
    calendar: pd.DataFrame,
    as_of: dt.date,
    min_days: int = 15,
) -> tuple[dt.date, dt.date]:
    """Calendar-backed version of select_monthly_hedge_expiry: looks up the
    current and next monthly expiries from the calendar (instead of a
    'last Thursday of the month' formula, which has the same fragility
    problem as the weekly day-of-week assumption) and applies the same
    >=15-day rule. Returns (expiry_date, prior_trading_day)."""
    current_expiry, current_prior = get_next_expiry(calendar, as_of, "monthly")
    days_to_current = (current_expiry - as_of).days
    if days_to_current >= min_days:
        return current_expiry, current_prior
    next_expiry, next_prior = get_next_expiry(calendar, current_expiry + dt.timedelta(days=1), "monthly")
    return next_expiry, next_prior


def select_monthly_hedge_expiry(
    as_of: dt.date,
    current_month_expiry: dt.date,
    next_month_expiry: dt.date,
    min_days: int = 15,
) -> dt.date:
    """Pure date-arithmetic version, kept for callers that already have
    both candidate dates in hand (e.g. from a calendar lookup done
    elsewhere) and just want the >=15-day rule applied. Prefer
    select_monthly_hedge_expiry_from_calendar for new code -- it sources
    the candidate dates itself rather than asking the caller to have
    already resolved them correctly."""
    days_to_current = (current_month_expiry - as_of).days
    if days_to_current >= min_days:
        return current_month_expiry
    return next_month_expiry


def is_expiry_eve_close_bar(
    as_of: dt.datetime,
    prior_trading_day: dt.date,
    close_time: dt.time = dt.time(15, 30),
) -> bool:
    """True on/after the force-close moment on the actual last trading day
    before expiry (point 6f) — the sold/hedge straddles (and, per config,
    the OTM legs) must be flat before expiry day's extra margin
    requirements kick in.

    Takes prior_trading_day directly (looked up from the expiry calendar
    via get_next_expiry) rather than deriving it as expiry_date minus one
    calendar day -- that arithmetic breaks on any holiday-adjacent expiry,
    and doesn't survive the exchange changing which weekday expiry falls
    on in the first place.
    """
    return as_of.date() == prior_trading_day and as_of.time() >= close_time


def is_on_or_after_expiry(as_of: dt.datetime, expiry_date: dt.date) -> bool:
    """True once the calendar date has reached expiry itself — a hard
    backstop so nothing is ever left open past expiry regardless of
    whether the expiry-eve close bar was hit exactly."""
    return as_of.date() >= expiry_date
