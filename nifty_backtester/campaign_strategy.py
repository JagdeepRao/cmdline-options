"""
Campaign strategy: a "funded strangle" theta engine.

SHAPE (per spec):
  - Entry (first trading day of the month, or the day after the prior
    campaign's monthly expiry -- a fresh calendar month always has enough
    runway; see resolve_campaign_expiry_schedule's min_days guard for the
    explicit check): buy 3x ATM-ish call + 3x ATM-ish put at MONTHLY
    expiry, strike chosen so each leg's premium is close to
    `long_target_premium` (200 by default). ITM is an acceptable match,
    not just OTM -- the search scans both directions from ATM.
  - Fund both sides independently: for calls, search EVERY pair of call
    strikes (same weekly expiry as the funding legs -- see below) whose
    combined credit exceeds 3x the long call premium, and take the pair
    that maximizes combined theta capture (the higher-premium leg of the
    winning pair is tagged "near", the lower-premium leg "far" -- this is
    a label applied to the result, not a constraint on the search itself).
    Same independently for puts, funded only against the put side's cost.
  - The four funding legs are WEEKLY (not monthly), sold at whatever the
    nearest weekly expiry is when the campaign opens. Every week, on the
    weekly expiry-eve, all four funding legs are closed and immediately
    re-sold at the SAME (locked-at-entry) strikes for the next weekly
    expiry -- deliberately a blind roll with no delta/breach defense: if
    the underlying has moved far enough that a locked strike is now deep
    ITM, that's treated as intentional protection against mean reversion,
    not a bug (confirmed explicitly, not assumed).
  - The long legs are NEVER touched mid-month -- they ride at their
    entry strikes until the whole campaign closes.
  - On the trading day before MONTHLY expiry, every leg (long and short)
    is closed. Nothing reopens -- the campaign is over for that month.
    Because the shipped expiry_calendar.csv (and, per the confirmed
    NSE regime, real NIFTY calendars generally) always has monthly expiry
    coincide with a weekly expiry date, this naturally means the LAST
    weekly-eve in a campaign closes everything instead of rolling -- no
    special-casing needed beyond checking "is this eve the monthly eve."
  - Both the weekly-roll moment and the monthly-close moment happen at the
    SAME configurable time of day (`CampaignConfig.roll_and_close_time`,
    confirmed as one single knob, not two) -- this is what
    scripts/run_campaign_backtest.py's --sweep-times sweeps to compare
    strategy performance across different times of day.

WHY A SEPARATE ENGINE, NOT backtest_engine.FullBacktestConfig: that engine
is explicitly single-weekly-cycle scope (its own module docstring says so).
A campaign spans MULTIPLE consecutive weekly cycles inside one monthly
window while two legs never move at all -- different enough in shape from
DeltaThresholdStrategy/FixedMoveStrategy/SoldLegSignalStrategy (which all
assume "recenter to a freshly-resolved ATM/delta/signal-based strike every
time") that bolting it on would fight the existing abstractions rather than
reuse them. What IS reused, unchanged: Leg/MultiLegPosition (strategy.py),
solve_iv_and_greeks (pricing.py), the MarketDataProvider interface
(market_data.py) via its get_spot/get_option_price/find_atm_strike methods,
is_trading_day/load_holidays (expiry_utils.py), and metrics.full_report for
scoring a result -- only the position-shape/adjustment-timing logic here is
new.

BAR GRID: unlike backtest_engine's 15-min grid (needed there because
delta/RSI/Supertrend signals are evaluated continuously), a campaign only
ever ACTS at a handful of moments (entry, each weekly eve, the monthly
eve) -- so the grid here is one bar per NSE trading day (skipping weekends
and nse_holidays.csv), with that day's single timestamp set to
roll_and_close_time on an action day (entry day, any weekly eve, the
monthly eve) and to ordinary market close (15:30) otherwise. This keeps a
multi-month backtest fast without losing the daily resolution
metrics.full_report wants for a meaningful equity curve.
"""

from __future__ import annotations
import datetime as dt
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from .strategy import Leg, MultiLegPosition, Right, Direction
from .pricing import solve_iv_and_greeks
from .expiry_utils import is_trading_day, load_holidays, get_next_expiry, DEFAULT_HOLIDAYS_PATH
from .market_data import MarketDataProvider
from .campaign_common import (
    MonthlyCampaignSchedule as CampaignExpirySchedule,
    resolve_monthly_campaign_schedule as resolve_campaign_expiry_schedule,
    first_trading_day_of_month, next_month as _next_month,
    open_campaign_leg as _open_leg, close_campaign_leg as _close_leg, mark_campaign_equity as _mark_equity,
    DEFAULT_MIN_DAYS_TO_MONTHLY_EXPIRY_AT_START,
)


# ─────────────────────────────────────────────
# CONFIG / RESULT TYPES
# ─────────────────────────────────────────────

@dataclass
class CampaignConfig:
    strike_step: int = 100
    long_quantity: int = 3
    short_quantity: int = 1
    long_target_premium: float = 200.0
    short_far_target_premium: float = 150.0    # documents the intent; the search itself only uses this to seed
    short_near_target_premium: float = 450.0   # logging -- the actual pick is whichever pair maximizes theta subject to the credit constraint
    strike_search_range: int = 40              # steps each direction from ATM, for both long-premium search and short-combo search
    roll_and_close_time: dt.time = dt.time(15, 30)
    initial_capital: float = 100_000.0
    min_days_to_monthly_expiry_at_start: int = DEFAULT_MIN_DAYS_TO_MONTHLY_EXPIRY_AT_START
    r: float = 0.0525
    q: float = 0.012


@dataclass
class CampaignStrikes:
    long_call_strike: int
    long_call_premium: float
    long_put_strike: int
    long_put_premium: float
    short_call_near_strike: int
    short_call_near_premium: float
    short_call_far_strike: int
    short_call_far_premium: float
    short_put_near_strike: int
    short_put_near_premium: float
    short_put_far_strike: int
    short_put_far_premium: float


@dataclass
class CampaignResult:
    equity_curve: pd.Series
    trade_pnls: list[float] = field(default_factory=list)
    action_log: list[str] = field(default_factory=list)
    strikes: Optional[CampaignStrikes] = None
    schedule: Optional[CampaignExpirySchedule] = None
    position: Optional[MultiLegPosition] = None


# NOTE: expiry-schedule resolution and month-chaining helpers
# (CampaignExpirySchedule/resolve_campaign_expiry_schedule/
# first_trading_day_of_month/_next_month) now live in campaign_common.py,
# shared with campaign_straddle_strategy.py -- imported above under their
# original names here so nothing else in this module (or its tests) needed
# to change.


# ─────────────────────────────────────────────
# STRIKE / PREMIUM SEARCH
# ─────────────────────────────────────────────

def _find_strike_by_target_premium(
    provider: MarketDataProvider,
    expiry: dt.date,
    as_of: dt.datetime,
    right: str,
    target_premium: float,
    strike_step: int,
    search_range: int,
) -> tuple[int, float]:
    """Scans strike_step-spaced strikes on BOTH sides of the put-call-parity
    ATM (ITM and OTM alike -- confirmed acceptable, not just OTM) and
    returns the (strike, premium) whose premium is closest to
    target_premium. Strikes the provider can't price are skipped, matching
    data_layer_base.find_atm_strike's own skip-on-failure precedent."""
    center = provider.find_atm_strike(expiry, as_of, strike_step)
    best = None  # (strike, premium, |premium - target|)
    for i in range(-search_range, search_range + 1):
        strike = center + i * strike_step
        try:
            price = provider.get_option_price(strike, right, expiry, as_of)
        except Exception:
            continue
        diff = abs(price - target_premium)
        if best is None or diff < best[2]:
            best = (strike, price, diff)

    if best is None:
        raise ValueError(
            f"No usable {right} data for expiry={expiry} within {search_range} strikes of ATM "
            f"around {as_of} -- widen strike_search_range, or (for a cached layer) commit more "
            f"strikes for this expiry/date."
        )
    return best[0], best[1]


def _find_max_theta_funding_pair(
    provider: MarketDataProvider,
    expiry: dt.date,
    as_of: dt.datetime,
    right: str,
    required_credit: float,
    strike_step: int,
    search_range: int,
    r: float,
    q: float,
) -> tuple[tuple[int, float, float], tuple[int, float, float]]:
    """Searches every pair of strike_step-spaced strikes (ITM and OTM) for
    `right`, keeps only pairs whose combined premium exceeds
    required_credit, and returns the pair that maximizes combined theta
    capture (sum of |theta| across the two short legs -- theta is negative
    for a long option, so being short captures its magnitude). Raises
    ValueError if no pair clears the credit bar (confirmed: this is a hard
    failure, not a "use the best available anyway" fallback).

    Returns (near, far) where each is (strike, premium, theta) and "near"
    is whichever leg of the winning pair has the HIGHER premium (a label
    on the result, not a search constraint -- see module docstring)."""
    center = provider.find_atm_strike(expiry, as_of, strike_step)
    candidates = []
    for i in range(-search_range, search_range + 1):
        strike = center + i * strike_step
        try:
            price = provider.get_option_price(strike, right, expiry, as_of)
        except Exception:
            continue
        spot = provider.get_spot(as_of)
        greeks = solve_iv_and_greeks(
            price=price, spot=spot, strike=strike, now=as_of, expiry_date=expiry,
            right=right.capitalize(), r=r, q=q,
        )
        theta = greeks["theta"]
        if theta != theta:  # NaN -- degenerate quote, solver couldn't converge
            continue
        candidates.append((strike, price, theta))

    if len(candidates) < 2:
        raise ValueError(
            f"Only {len(candidates)} usable {right} strike(s) found for expiry={expiry} around "
            f"{as_of} -- need at least 2 to form a funding pair. Widen strike_search_range."
        )

    best_pair = None
    best_theta_capture = None
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            leg_a, leg_b = candidates[i], candidates[j]
            combined_credit = leg_a[1] + leg_b[1]
            if combined_credit <= required_credit:
                continue
            theta_capture = abs(leg_a[2]) + abs(leg_b[2])
            if best_theta_capture is None or theta_capture > best_theta_capture:
                best_theta_capture = theta_capture
                best_pair = (leg_a, leg_b)

    if best_pair is None:
        raise ValueError(
            f"No pair of {right} strikes found where combined premium exceeds the required "
            f"credit of {required_credit:.2f} for expiry={expiry} around {as_of} -- this month's "
            f"campaign genuinely can't be funded on the {right} side at this strike_search_range. "
            f"Widen strike_search_range, or accept that this campaign shouldn't open this month."
        )

    leg_a, leg_b = best_pair
    near, far = (leg_a, leg_b) if leg_a[1] >= leg_b[1] else (leg_b, leg_a)
    return near, far


def select_campaign_entry(
    provider: MarketDataProvider,
    config: CampaignConfig,
    monthly_expiry: dt.date,
    entry_weekly_expiry: dt.date,
    entry_time: dt.datetime,
) -> CampaignStrikes:
    """Runs the full entry-selection algorithm: long call/put strikes by
    target premium at monthly_expiry, then independently funds each side
    (call funding legs must clear 3x the long call premium; put funding
    legs must clear 3x the long put premium) at entry_weekly_expiry via
    the max-theta search above."""
    long_call_strike, long_call_premium = _find_strike_by_target_premium(
        provider, monthly_expiry, entry_time, "call", config.long_target_premium,
        config.strike_step, config.strike_search_range,
    )
    long_put_strike, long_put_premium = _find_strike_by_target_premium(
        provider, monthly_expiry, entry_time, "put", config.long_target_premium,
        config.strike_step, config.strike_search_range,
    )

    required_call_credit = config.long_quantity * long_call_premium
    required_put_credit = config.long_quantity * long_put_premium

    call_near, call_far = _find_max_theta_funding_pair(
        provider, entry_weekly_expiry, entry_time, "call", required_call_credit,
        config.strike_step, config.strike_search_range, config.r, config.q,
    )
    put_near, put_far = _find_max_theta_funding_pair(
        provider, entry_weekly_expiry, entry_time, "put", required_put_credit,
        config.strike_step, config.strike_search_range, config.r, config.q,
    )

    return CampaignStrikes(
        long_call_strike=long_call_strike, long_call_premium=long_call_premium,
        long_put_strike=long_put_strike, long_put_premium=long_put_premium,
        short_call_near_strike=call_near[0], short_call_near_premium=call_near[1],
        short_call_far_strike=call_far[0], short_call_far_premium=call_far[1],
        short_put_near_strike=put_near[0], short_put_near_premium=put_near[1],
        short_put_far_strike=put_far[0], short_put_far_premium=put_far[1],
    )


# ─────────────────────────────────────────────
# BAR GRID
# ─────────────────────────────────────────────

def _campaign_time_grid(
    schedule: CampaignExpirySchedule,
    campaign_start: dt.date,
    holidays: set[dt.date],
    roll_and_close_time: dt.time,
) -> list[dt.datetime]:
    """One bar per NSE trading day from campaign_start through
    monthly_expiry. Action days (entry day, every weekly eve, the monthly
    eve) use roll_and_close_time; every other day uses ordinary market
    close (15:30) purely for equity marking."""
    action_dates = {prior for _, prior in schedule.weekly_cycles} | {campaign_start}
    grid = []
    d = campaign_start
    while d <= schedule.monthly_expiry:
        if is_trading_day(d, holidays):
            t = roll_and_close_time if d in action_dates else dt.time(15, 30)
            grid.append(dt.datetime.combine(d, t))
        d += dt.timedelta(days=1)
    return grid


# NOTE: leg open/close/equity-mark helpers (_open_leg/_close_leg/
# _mark_equity) now live in campaign_common.py as open_campaign_leg/
# close_campaign_leg/mark_campaign_equity, imported above under their
# original names here.

SHORT_TAGS = ["short_call_near", "short_call_far", "short_put_near", "short_put_far"]
LONG_TAGS = ["long_call", "long_put"]
_SHORT_RIGHT_BY_TAG = {"short_call_near": "call", "short_call_far": "call", "short_put_near": "put", "short_put_far": "put"}


# ─────────────────────────────────────────────
# MAIN ENGINE
# ─────────────────────────────────────────────

def run_campaign_backtest(
    provider: MarketDataProvider,
    calendar: pd.DataFrame,
    holidays: set[dt.date],
    campaign_start: dt.date,
    config: Optional[CampaignConfig] = None,
) -> CampaignResult:
    config = config or CampaignConfig()
    schedule = resolve_campaign_expiry_schedule(calendar, campaign_start, config.min_days_to_monthly_expiry_at_start)

    grid = _campaign_time_grid(schedule, campaign_start, holidays, config.roll_and_close_time)
    if not grid:
        raise ValueError("Empty campaign time grid -- check campaign_start/monthly_expiry and the holiday list.")

    entry_time = grid[0]
    entry_weekly_expiry = schedule.weekly_cycles[0][0]
    strikes = select_campaign_entry(provider, config, schedule.monthly_expiry, entry_weekly_expiry, entry_time)

    # eve_date -> next weekly expiry to roll into (absent for the FINAL
    # cycle, whose expiry == monthly_expiry -- that eve is handled as a
    # full close, not a roll; see module docstring)
    roll_map: dict[dt.date, dt.date] = {}
    for k in range(len(schedule.weekly_cycles)):
        expiry_k, prior_k = schedule.weekly_cycles[k]
        if expiry_k == schedule.monthly_expiry:
            continue
        roll_map[prior_k] = schedule.weekly_cycles[k + 1][0]

    position = MultiLegPosition(name="campaign")
    trade_pnls: list[float] = []
    action_log: list[str] = []
    equity_points: list[tuple[dt.datetime, float]] = []

    _open_leg(position, provider, action_log, "long_call", "call", strikes.long_call_strike,
              schedule.monthly_expiry, config.long_quantity, entry_time,
              f"campaign entry, target premium {config.long_target_premium}")
    _open_leg(position, provider, action_log, "long_put", "put", strikes.long_put_strike,
              schedule.monthly_expiry, config.long_quantity, entry_time,
              f"campaign entry, target premium {config.long_target_premium}")
    for tag in SHORT_TAGS:
        strike = getattr(strikes, f"{tag}_strike")
        _open_leg(position, provider, action_log, tag, _SHORT_RIGHT_BY_TAG[tag], strike,
                  entry_weekly_expiry, config.short_quantity, entry_time,
                  "campaign entry, funding leg (max-theta pick clearing the credit requirement) -- strike locked for the rest of the campaign")

    equity_points.append((entry_time, _mark_equity(position, provider, entry_time, config.initial_capital)))

    for as_of in grid[1:]:
        d = as_of.date()
        if d == schedule.monthly_expiry_prior_trading_day:
            for tag in SHORT_TAGS + LONG_TAGS:
                _close_leg(position, provider, trade_pnls, action_log, tag, as_of,
                           "monthly expiry eve -- closing entire campaign, not reopening")
        elif d in roll_map:
            next_expiry = roll_map[d]
            for tag in SHORT_TAGS:
                _close_leg(position, provider, trade_pnls, action_log, tag, as_of,
                           f"weekly expiry eve -- rolling to {next_expiry} at the same locked strike")
            for tag in SHORT_TAGS:
                strike = getattr(strikes, f"{tag}_strike")
                _open_leg(position, provider, action_log, tag, _SHORT_RIGHT_BY_TAG[tag], strike,
                          next_expiry, config.short_quantity, as_of,
                          "rolled -- same locked strike as campaign entry, blind roll (intentional: "
                          "protects against mean reversion even if now deep ITM)")

        equity_points.append((as_of, _mark_equity(position, provider, as_of, config.initial_capital)))

    equity_curve = pd.Series(
        [v for _, v in equity_points],
        index=pd.DatetimeIndex([t for t, _ in equity_points]),
    )
    return CampaignResult(
        equity_curve=equity_curve, trade_pnls=trade_pnls, action_log=action_log,
        strikes=strikes, schedule=schedule, position=position,
    )


def run_chained_campaigns(
    provider_factory,
    calendar: pd.DataFrame,
    holidays: set[dt.date],
    first_campaign_start: dt.date,
    num_months: int,
    config: Optional[CampaignConfig] = None,
) -> list[CampaignResult]:
    """Runs num_months consecutive campaigns, each starting the first
    trading day of the month after the previous one's monthly expiry (spec
    point 8: "start again with next month"). provider_factory is a
    one-arg callable, provider_factory(campaign_start) -> provider, called
    fresh before each campaign -- for a live/cached Breeze-backed provider
    this is typically `lambda _: provider` (the same object reused); for
    SyntheticMarketDataProvider each campaign needs its own instance built
    over THAT campaign's own date window, so the caller controls that via
    the factory rather than this function guessing at window sizing.
    """
    config = config or CampaignConfig()
    results = []
    campaign_start = first_campaign_start
    for _ in range(num_months):
        provider = provider_factory(campaign_start)
        result = run_campaign_backtest(provider, calendar, holidays, campaign_start, config)
        results.append(result)
        next_year, next_month = _next_month(result.schedule.monthly_expiry.year, result.schedule.monthly_expiry.month)
        campaign_start = first_trading_day_of_month(next_year, next_month, holidays)
    return results
