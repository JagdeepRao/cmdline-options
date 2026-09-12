"""
Shared scaffolding for "campaign" strategies -- any strategy shaped as a
month-long position that opens on the first trading day of the month and
fully closes on the monthly expiry-eve, with a fresh one starting the
following month. campaign_strategy.py (funded strangle, "Campaign 1") and
campaign_straddle_strategy.py (sold weekly straddle / bought monthly
straddle, "Campaign 2") both build on this -- the whole point of factoring
it out is that different campaign SHAPES can be run and compared against
each other (same expiry-resolution rules, same month-chaining behavior,
same leg open/close/equity-marking primitives) without each one
re-implementing that machinery its own slightly-different way.

What's intentionally NOT here: the bar grid. Campaign 1 only needs to act
on expiry-eves (mark-only bars the rest of the time); Campaign 2 acts
every single trading day (a daily drift check). That difference is core
to each strategy's own shape, not shared plumbing -- each module builds
its own grid function.
"""

from __future__ import annotations
import datetime as dt
from dataclasses import dataclass

import pandas as pd

from .strategy import Leg, MultiLegPosition, Right, Direction
from .expiry_utils import is_trading_day, get_next_expiry
from .market_data import MarketDataProvider

DEFAULT_MIN_DAYS_TO_MONTHLY_EXPIRY_AT_START = 10


@dataclass
class MonthlyCampaignSchedule:
    monthly_expiry: dt.date
    monthly_expiry_prior_trading_day: dt.date
    # (expiry_date, prior_trading_day) pairs, ascending, ALWAYS ending with
    # the cycle whose expiry_date == monthly_expiry (both shipped-template
    # and real post-2025-09-01 NIFTY calendars have monthly expiry coincide
    # with a weekly expiry date -- see expiry_utils.WEEKLY_EXPIRY_WEEKDAY_REGIMES).
    weekly_cycles: list[tuple[dt.date, dt.date]]


def resolve_monthly_campaign_schedule(
    calendar: pd.DataFrame,
    campaign_start: dt.date,
    min_days: int = DEFAULT_MIN_DAYS_TO_MONTHLY_EXPIRY_AT_START,
) -> MonthlyCampaignSchedule:
    """Resolves the monthly expiry (+ prior trading day) and the full
    sequence of weekly cycles a campaign starting on campaign_start will
    roll through, ending with the cycle that coincides with monthly expiry.

    Raises ValueError if campaign_start is within `min_days` of the next
    monthly expiry ("cannot start close to monthly expiry") -- normal
    usage (first trading day of the month) comfortably clears this; this
    guard exists for misuse (e.g. resuming mid-month), not normal use.

    EXCEPTION RULE: if the first weekly cycle on/after campaign_start is
    already at or past its own expiry-eve on campaign_start itself (i.e.
    entering it now would require rolling it the very same day -- this
    includes campaign_start landing exactly ON that cycle's expiry date),
    that cycle is skipped and entry happens against the NEXT weekly cycle.
    """
    monthly_expiry, monthly_prior = get_next_expiry(calendar, campaign_start, "monthly")
    days_to_monthly = (monthly_expiry - campaign_start).days
    if days_to_monthly < min_days:
        raise ValueError(
            f"campaign_start={campaign_start} is only {days_to_monthly} day(s) before the next "
            f"monthly expiry ({monthly_expiry}) -- below the {min_days}-day minimum runway a "
            f"campaign needs (spec: 'cannot start close to monthly expiry'). Start the campaign "
            f"on the first trading day of the following month instead."
        )

    weekly_rows = calendar[
        (calendar["expiry_type"] == "weekly")
        & (calendar["expiry_date"] >= campaign_start)
        & (calendar["expiry_date"] <= monthly_expiry)
    ].sort_values("expiry_date")
    cycles = list(zip(weekly_rows["expiry_date"], weekly_rows["prior_trading_day"]))
    if not cycles:
        raise ValueError(
            f"No weekly expiries found between campaign_start={campaign_start} and "
            f"monthly_expiry={monthly_expiry} -- extend expiry_calendar.csv."
        )

    if cycles[0][1] <= campaign_start:
        cycles = cycles[1:]  # the "already at its own eve" exception rule
    if not cycles:
        raise ValueError(
            f"After skipping the weekly cycle already at its own expiry-eve on "
            f"campaign_start={campaign_start}, no weekly cycle remains before "
            f"monthly_expiry={monthly_expiry} -- extend expiry_calendar.csv or push "
            f"campaign_start earlier."
        )
    if cycles[-1][0] != monthly_expiry:
        raise ValueError(
            f"Calendar inconsistency: the last weekly cycle before monthly_expiry="
            f"{monthly_expiry} is {cycles[-1][0]}, not the monthly expiry itself -- every "
            f"campaign here assumes monthly expiry always coincides with a weekly expiry date "
            f"in the calendar (true for the shipped template; confirm it for whatever calendar "
            f"you're using)."
        )

    return MonthlyCampaignSchedule(monthly_expiry, monthly_prior, cycles)


def first_trading_day_of_month(year: int, month: int, holidays: set[dt.date]) -> dt.date:
    """First actual NSE trading day of the given month -- the 1st itself
    if that's a trading day, else the first weekday/non-holiday after it."""
    d = dt.date(year, month, 1)
    while not is_trading_day(d, holidays):
        d += dt.timedelta(days=1)
    return d


def next_month(year: int, month: int) -> tuple[int, int]:
    return (year + 1, 1) if month == 12 else (year, month + 1)


def weekly_roll_map(schedule: MonthlyCampaignSchedule) -> dict[dt.date, dt.date]:
    """eve_date -> next weekly expiry to roll into, for every weekly cycle
    EXCEPT the final one (whose expiry == monthly_expiry) -- that eve is
    always a full campaign close, never a plain roll, so it's deliberately
    absent from this map; callers check monthly_expiry_prior_trading_day
    separately and first."""
    roll_map: dict[dt.date, dt.date] = {}
    for k in range(len(schedule.weekly_cycles)):
        expiry_k, prior_k = schedule.weekly_cycles[k]
        if expiry_k == schedule.monthly_expiry:
            continue
        roll_map[prior_k] = schedule.weekly_cycles[k + 1][0]
    return roll_map


# ─────────────────────────────────────────────
# LEG OPEN/CLOSE/MARK HELPERS -- identical shape needed by every campaign
# ─────────────────────────────────────────────

def open_campaign_leg(position: MultiLegPosition, provider: MarketDataProvider, action_log: list[str],
                       tag: str, right: str, strike: int, expiry: dt.date, quantity: int,
                       as_of: dt.datetime, reason: str, position_label: str = "campaign") -> Leg:
    price = provider.get_option_price(strike, right, expiry, as_of)
    leg = Leg(
        tag=tag, right=Right.CALL if right == "call" else Right.PUT,
        direction=Direction.LONG if tag.startswith("long_") else Direction.SHORT,
        strike=strike, expiry=expiry, entry_time=as_of, entry_price=price, quantity=quantity,
    )
    position.add_leg(leg)
    action_log.append(
        f"{as_of}: OPEN {position_label}.{tag} strike={strike} expiry={expiry} qty={quantity} "
        f"price={price:.2f} -- {reason}"
    )
    return leg


def close_campaign_leg(position: MultiLegPosition, provider: MarketDataProvider, trade_pnls: list[float],
                        action_log: list[str], tag: str, as_of: dt.datetime, reason: str,
                        position_label: str = "campaign") -> None:
    leg = position.get_leg(tag)
    if leg is None:
        return
    price = provider.get_option_price(leg.strike, leg.right.value, leg.expiry, as_of)
    leg.close(as_of, price)
    trade_pnls.append(leg.pnl())
    action_log.append(f"{as_of}: CLOSE {position_label}.{tag} @ {price:.2f} -- {reason}")


def mark_campaign_equity(position: MultiLegPosition, provider: MarketDataProvider,
                          as_of: dt.datetime, initial_capital: float) -> float:
    marks = {leg.tag: provider.get_option_price(leg.strike, leg.right.value, leg.expiry, as_of) for leg in position.open_legs()}
    return initial_capital + position.total_pnl(marks)
