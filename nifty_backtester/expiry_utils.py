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
DEFAULT_HOLIDAYS_PATH = Path(__file__).parent / "nse_holidays.csv"

# Used only as a SANITY CHECK inside get_next_expiry (see its docstring) --
# never to compute an expiry date itself. Real weekly contracts are ~7
# days apart and monthly ~28-35 days apart, so a "nearest match" further
# out than this all but certainly means the calendar has a GAP around
# as_of (e.g. it only covers a different year entirely) rather than as_of
# genuinely landing in a quiet stretch.
_MAX_PLAUSIBLE_GAP_DAYS = {"weekly": 10, "monthly": 40}

# WEEKLY-EXPIRY WEEKDAY HISTORY, for documentation and for
# scripts/generate_expiry_calendar_candidates.py ONLY -- this is
# deliberately NEVER read by load_expiry_calendar/get_next_expiry/etc. at
# runtime. The whole point of expiry_calendar.csv as a data file (see this
# module's top-of-file docstring) is that the engine never computes an
# expiry date from a weekday rule, because that rule keeps changing
# exchange-side. This table exists purely so a human generating/reviewing
# a calendar update doesn't have to independently research the history
# each time; it is a candidate-generation aid, not a source of truth.
#
# CONFIRMED (web search, cross-checked against multiple financial-news
# sources reporting the same NSE/SEBI circulars -- re-verify against an
# official NSE circular before trusting a real trading decision on it):
#   - Thursday was NIFTY's weekly AND monthly expiry day for roughly 25
#     years, through 2025-08-28.
#   - NSE originally announced (2025-03-04 circular) a move to MONDAY
#     effective 2025-04-05 -- this was announced, then explicitly
#     DEFERRED/paused (2025-03-27 circular) before ever taking effect.
#     NIFTY expiry was NEVER actually on Monday in production.
#   - SEBI then directed exchanges to standardize on Tuesday-or-Thursday
#     only (circular, ~May 2025). NSE chose Tuesday; the change took
#     effect for contracts from 2025-09-01 (last Thursday expiry was
#     2025-08-28). NIFTY weekly AND monthly expiry has been Tuesday since.
#   - Monthly expiry is the LAST Tuesday of the calendar month (under the
#     current, post-2025-09-01 regime) -- shifted to the prior trading day
#     if that Tuesday is a holiday, same rule as weekly (see
#     get_holiday_shifted_trading_day below).
WEEKLY_EXPIRY_WEEKDAY_REGIMES = [
    # (regime_start, regime_end_inclusive_or_None, weekday) -- weekday is
    # Python's Monday=0..Sunday=6 convention. Periods are CONTIGUOUS (each
    # regime's end is the day immediately before the next one's start) --
    # this must never have a gap, since _weekday_for() below is looked up
    # for every calendar date a scan touches, not just actual expiry
    # dates; a gap here previously broke the candidate generator right at
    # this exact transition (found in review -- see
    # generate_expiry_calendar_candidates.py's _weekly_candidates).
    (dt.date(2000, 6, 12), dt.date(2025, 8, 31), 3),   # Thursday, from NIFTY F&O launch through the day before the Tuesday regime began (last actual Thursday expiry: 2025-08-28, confirmed via NSE circular FAOP68747 and Zerodha's own bulletin)
    (dt.date(2025, 9, 1), None, 1),                     # Tuesday, current regime (None = still in effect); first actual Tuesday expiry: 2025-09-02 (2025-09-01 itself was a Monday)
]


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


def load_holidays(path: Path = DEFAULT_HOLIDAYS_PATH) -> set[dt.date]:
    """Loads nse_holidays.csv into a set of dt.date. Raises FileNotFoundError
    with a clear message if missing -- same fail-loud posture as
    load_expiry_calendar, since a silently-empty holiday set would make
    validate_calendar_against_holidays below pass vacuously and give false
    confidence."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Holiday list not found at {path}. This is required for "
            f"validate_calendar_against_holidays() / "
            f"scripts/generate_expiry_calendar_candidates.py -- there is no "
            f"fallback (a missing file must not silently mean 'no holidays')."
        )
    df = pd.read_csv(path)
    if "holiday_date" not in df.columns:
        raise ValueError(f"Holiday list at {path} is missing required column 'holiday_date'")
    return set(pd.to_datetime(df["holiday_date"]).dt.date)


def is_trading_day(d: dt.date, holidays: set[dt.date]) -> bool:
    """Not a weekend and not in the holiday set. Used by
    validate_calendar_against_holidays and the candidate-calendar
    generator -- never by the live engine (see WEEKLY_EXPIRY_WEEKDAY_REGIMES
    docstring above for why runtime code never derives expiry/trading days
    from a rule)."""
    return d.weekday() < 5 and d not in holidays


def get_holiday_shifted_trading_day(d: dt.date, holidays: set[dt.date]) -> dt.date:
    """If `d` itself isn't a valid trading day (weekend or holiday), steps
    backward one day at a time until it finds one that is -- this is the
    documented NSE convention ('if expiry day is a market holiday, expiry
    moves to the prior trading day'), used only by the candidate generator,
    never to silently reinterpret an already-committed expiry_calendar.csv
    entry at runtime."""
    while not is_trading_day(d, holidays):
        d -= dt.timedelta(days=1)
    return d


def validate_calendar_against_holidays(
    calendar: pd.DataFrame,
    holidays: set[dt.date],
) -> None:
    """Cross-checks an already-loaded calendar (as returned by
    load_expiry_calendar) against a holiday set. Raises ValueError listing
    EVERY violation found (not just the first) if any expiry_date or
    prior_trading_day:
      - falls on a weekend, or
      - falls on a listed holiday.

    This is deliberately NOT called automatically from load_expiry_calendar
    -- wiring it in as a default would make load_expiry_calendar() raise
    for anyone whose calendar and holiday files were sourced/updated at
    different times (a realistic, recoverable situation, not necessarily a
    data-entry error the way expiry_utils' existing prior_trading_day-must-
    precede-expiry_date check is). Call this explicitly wherever you want
    the extra guarantee -- e.g. before committing an updated
    expiry_calendar.csv, or as a CI/test-suite check.
    """
    violations = []
    for _, row in calendar.iterrows():
        for col in ("expiry_date", "prior_trading_day"):
            d = row[col]
            if d.weekday() >= 5:
                violations.append(
                    f"{row['expiry_type']} {col}={d} ({d.strftime('%A')}) falls on a weekend"
                )
            elif d in holidays:
                violations.append(
                    f"{row['expiry_type']} {col}={d} ({d.strftime('%A')}) is a listed NSE holiday"
                )

    if violations:
        raise ValueError(
            f"Calendar has {len(violations)} entr(y/ies) landing on a weekend or "
            f"listed holiday -- these dates were never real trading days, so any "
            f"expiry/prior_trading_day computed from them is wrong:\n" +
            "\n".join(f"  - {v}" for v in violations)
        )


def get_next_expiry(
    calendar: pd.DataFrame,
    as_of: dt.date,
    expiry_type: str,
) -> tuple[dt.date, dt.date]:
    """Returns (expiry_date, prior_trading_day) for the first expiry of
    `expiry_type` on or after `as_of`. Raises ValueError if the calendar
    doesn't cover that far -- extend expiry_calendar.csv rather than
    silently falling back to a guessed date.

    Also raises ValueError if the nearest match found is implausibly far
    from `as_of` (see _MAX_PLAUSIBLE_GAP_DAYS) -- regression guard for a
    real bug found in review: if the calendar has a GAP (e.g. it only has
    2026 rows and as_of is in 2025), "first expiry_date >= as_of" happily
    matches the first 2026 row and returns an expiry over a YEAR away
    without ever raising, silently corrupting every downstream campaign/
    backtest date. A calendar gap should fail loudly here, at the lookup,
    not surface later as a confusing 'no option data available' error deep
    inside a data layer.
    """
    subset = calendar[(calendar["expiry_type"] == expiry_type) & (calendar["expiry_date"] >= as_of)]
    if subset.empty:
        raise ValueError(
            f"No {expiry_type} expiry on/after {as_of} found in the calendar -- "
            f"it doesn't cover this far forward. Extend expiry_calendar.csv."
        )
    row = subset.iloc[0]
    gap_days = (row["expiry_date"] - as_of).days
    max_gap = _MAX_PLAUSIBLE_GAP_DAYS.get(expiry_type, 40)
    if gap_days > max_gap:
        raise ValueError(
            f"Nearest {expiry_type} expiry on/after {as_of} is {row['expiry_date']} -- "
            f"{gap_days} days away, which is implausible for a {expiry_type} contract "
            f"(expected within ~{max_gap} days). This almost certainly means the calendar "
            f"has a GAP around {as_of} (e.g. it covers a much later year but nothing near "
            f"{as_of} itself) rather than {as_of} genuinely being this far from the next "
            f"expiry -- extend expiry_calendar.csv to actually cover {as_of} instead of "
            f"trusting this result."
        )
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
