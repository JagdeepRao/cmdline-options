"""
Campaign 2: sold-weekly-straddle / bought-monthly-straddle engine -- the
simpler, more-frequently-adjusted counterpart to campaign_strategy.py's
funded strangle ("Campaign 1"). Built on the shared scaffolding in
campaign_common.py specifically so the two can be run over the same
campaign_start/months and compared head-to-head (see
scripts/compare_campaigns.py) -- that comparison is the actual point:
whether the extra complexity of Campaign 1's funded strangle beats this
simpler daily-recentered straddle, or not.

SHAPE:
  - Entry (first trading day of the month; same runway/exception rules as
    Campaign 1 -- see campaign_common.resolve_monthly_campaign_schedule):
    buy 1x ATM call + 1x ATM put at MONTHLY expiry (the hedge), sell 1x
    ATM call + 1x ATM put at the nearest WEEKLY expiry (the short
    straddle). ATM is resolved via the provider's put-call-parity
    find_atm_strike, same as everywhere else in this codebase.
  - EVERY trading day (not just expiry-eves), at one configurable time of
    day (`adjustment_time`), check the short straddle: if the CURRENT
    ATM strike for its expiry is more than `recenter_threshold_points`
    away from the strike it currently holds, recenter it -- close both
    short legs and reopen fresh ATM call+put at the current ATM strike,
    same weekly expiry.
  - On the weekly expiry-eve, the short straddle unconditionally rolls to
    the next weekly expiry at a freshly-resolved ATM strike, regardless of
    the drift threshold -- the contract is about to stop existing, so
    there's nothing to "not recenter" into. An eve day therefore always
    rolls; a non-eve day only recenters if the drift threshold is
    breached. These are mutually exclusive for a given day by
    construction (the eve dates are excluded from the drift check).
  - The long monthly straddle is NEVER touched mid-month, same discipline
    as Campaign 1's long legs.
  - On the monthly expiry-eve, everything (short straddle + long straddle)
    closes. Nothing reopens. A fresh campaign starts the following month.

BAR GRID: unlike Campaign 1 (which only needs to act on expiry-eves),
this strategy genuinely checks every single trading day -- so the grid
here is simply one bar per NSE trading day at adjustment_time, no
separate "just marking" timestamp is needed.
"""

from __future__ import annotations
import datetime as dt
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from .strategy import MultiLegPosition
from .expiry_utils import is_trading_day
from .market_data import MarketDataProvider
from .campaign_common import (
    MonthlyCampaignSchedule, resolve_monthly_campaign_schedule,
    first_trading_day_of_month, next_month, weekly_roll_map,
    open_campaign_leg, close_campaign_leg, mark_campaign_equity,
    DEFAULT_MIN_DAYS_TO_MONTHLY_EXPIRY_AT_START,
)

POSITION_LABEL = "campaign_straddle"


@dataclass
class StraddleCampaignConfig:
    strike_step: int = 100
    long_quantity: int = 1
    short_quantity: int = 1
    recenter_threshold_points: float = 100.0
    adjustment_time: dt.time = dt.time(15, 30)
    initial_capital: float = 100_000.0
    min_days_to_monthly_expiry_at_start: int = DEFAULT_MIN_DAYS_TO_MONTHLY_EXPIRY_AT_START


@dataclass
class StraddleCampaignResult:
    equity_curve: pd.Series
    trade_pnls: list[float] = field(default_factory=list)
    action_log: list[str] = field(default_factory=list)
    schedule: Optional[MonthlyCampaignSchedule] = None
    position: Optional[MultiLegPosition] = None
    num_recenters: int = 0  # drift-triggered recenters only, NOT weekly rolls -- see run_straddle_campaign_backtest


def _daily_grid(schedule: MonthlyCampaignSchedule, campaign_start: dt.date,
                 holidays: set[dt.date], adjustment_time: dt.time) -> list[dt.datetime]:
    grid = []
    d = campaign_start
    while d <= schedule.monthly_expiry:
        if is_trading_day(d, holidays):
            grid.append(dt.datetime.combine(d, adjustment_time))
        d += dt.timedelta(days=1)
    return grid


def run_straddle_campaign_backtest(
    provider: MarketDataProvider,
    calendar: pd.DataFrame,
    holidays: set[dt.date],
    campaign_start: dt.date,
    config: Optional[StraddleCampaignConfig] = None,
) -> StraddleCampaignResult:
    config = config or StraddleCampaignConfig()
    schedule = resolve_monthly_campaign_schedule(calendar, campaign_start, config.min_days_to_monthly_expiry_at_start)

    grid = _daily_grid(schedule, campaign_start, holidays, config.adjustment_time)
    if not grid:
        raise ValueError("Empty campaign time grid -- check campaign_start/monthly_expiry and the holiday list.")

    entry_time = grid[0]
    entry_weekly_expiry = schedule.weekly_cycles[0][0]
    roll_map = weekly_roll_map(schedule)  # eve_date -> next weekly expiry, excludes the monthly-coincident eve

    position = MultiLegPosition(name=POSITION_LABEL)
    trade_pnls: list[float] = []
    action_log: list[str] = []
    equity_points: list[tuple[dt.datetime, float]] = []
    num_recenters = 0

    monthly_atm = provider.find_atm_strike(schedule.monthly_expiry, entry_time, config.strike_step)
    weekly_atm = provider.find_atm_strike(entry_weekly_expiry, entry_time, config.strike_step)

    open_campaign_leg(position, provider, action_log, "long_call", "call", monthly_atm,
                       schedule.monthly_expiry, config.long_quantity, entry_time,
                       "campaign entry -- monthly ATM hedge", POSITION_LABEL)
    open_campaign_leg(position, provider, action_log, "long_put", "put", monthly_atm,
                       schedule.monthly_expiry, config.long_quantity, entry_time,
                       "campaign entry -- monthly ATM hedge", POSITION_LABEL)
    open_campaign_leg(position, provider, action_log, "short_call", "call", weekly_atm,
                       entry_weekly_expiry, config.short_quantity, entry_time,
                       "campaign entry -- weekly ATM short straddle", POSITION_LABEL)
    open_campaign_leg(position, provider, action_log, "short_put", "put", weekly_atm,
                       entry_weekly_expiry, config.short_quantity, entry_time,
                       "campaign entry -- weekly ATM short straddle", POSITION_LABEL)

    equity_points.append((entry_time, mark_campaign_equity(position, provider, entry_time, config.initial_capital)))

    current_weekly_expiry = entry_weekly_expiry

    for as_of in grid[1:]:
        d = as_of.date()

        if d == schedule.monthly_expiry_prior_trading_day:
            for tag in ("short_call", "short_put", "long_call", "long_put"):
                close_campaign_leg(position, provider, trade_pnls, action_log, tag, as_of,
                                    "monthly expiry eve -- closing entire campaign, not reopening", POSITION_LABEL)

        elif d in roll_map:
            next_expiry = roll_map[d]
            for tag in ("short_call", "short_put"):
                close_campaign_leg(position, provider, trade_pnls, action_log, tag, as_of,
                                    f"weekly expiry eve -- unconditional roll to {next_expiry}", POSITION_LABEL)
            new_atm = provider.find_atm_strike(next_expiry, as_of, config.strike_step)
            open_campaign_leg(position, provider, action_log, "short_call", "call", new_atm, next_expiry,
                               config.short_quantity, as_of, "rolled to fresh ATM for the new weekly expiry", POSITION_LABEL)
            open_campaign_leg(position, provider, action_log, "short_put", "put", new_atm, next_expiry,
                               config.short_quantity, as_of, "rolled to fresh ATM for the new weekly expiry", POSITION_LABEL)
            current_weekly_expiry = next_expiry

        else:
            short_call_leg = position.get_leg("short_call")
            if short_call_leg is not None:
                current_atm = provider.find_atm_strike(current_weekly_expiry, as_of, config.strike_step)
                drift = abs(current_atm - short_call_leg.strike)
                if drift > config.recenter_threshold_points:
                    for tag in ("short_call", "short_put"):
                        close_campaign_leg(
                            position, provider, trade_pnls, action_log, tag, as_of,
                            f"ATM strike {current_atm} is {drift:.0f}pts from held strike "
                            f"{short_call_leg.strike} (> {config.recenter_threshold_points}) -- recentering",
                            POSITION_LABEL,
                        )
                    open_campaign_leg(position, provider, action_log, "short_call", "call", current_atm,
                                       current_weekly_expiry, config.short_quantity, as_of,
                                       "recentered to current ATM", POSITION_LABEL)
                    open_campaign_leg(position, provider, action_log, "short_put", "put", current_atm,
                                       current_weekly_expiry, config.short_quantity, as_of,
                                       "recentered to current ATM", POSITION_LABEL)
                    num_recenters += 1

        equity_points.append((as_of, mark_campaign_equity(position, provider, as_of, config.initial_capital)))

    equity_curve = pd.Series(
        [v for _, v in equity_points],
        index=pd.DatetimeIndex([t for t, _ in equity_points]),
    )
    return StraddleCampaignResult(
        equity_curve=equity_curve, trade_pnls=trade_pnls, action_log=action_log,
        schedule=schedule, position=position, num_recenters=num_recenters,
    )


def run_chained_straddle_campaigns(
    provider_factory,
    calendar: pd.DataFrame,
    holidays: set[dt.date],
    first_campaign_start: dt.date,
    num_months: int,
    config: Optional[StraddleCampaignConfig] = None,
) -> list[StraddleCampaignResult]:
    """Runs num_months consecutive campaigns, each starting the first
    trading day of the month after the previous one's monthly expiry.
    provider_factory is a one-arg callable, provider_factory(campaign_start)
    -> provider -- see campaign_strategy.run_chained_campaigns for the same
    convention and why SyntheticMarketDataProvider needs a fresh instance
    per campaign while a Breeze-backed provider typically doesn't."""
    config = config or StraddleCampaignConfig()
    results = []
    campaign_start = first_campaign_start
    for _ in range(num_months):
        provider = provider_factory(campaign_start)
        result = run_straddle_campaign_backtest(provider, calendar, holidays, campaign_start, config)
        results.append(result)
        next_year, next_month_num = next_month(result.schedule.monthly_expiry.year, result.schedule.monthly_expiry.month)
        campaign_start = first_trading_day_of_month(next_year, next_month_num, holidays)
    return results
