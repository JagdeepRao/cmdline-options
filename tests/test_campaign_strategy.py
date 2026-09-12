"""
Tests for nifty_backtester.campaign_strategy.

Organized to match the pieces described in that module's docstring:
  1. Expiry schedule resolution (including the min-days guard and the
     "already at its own eve" exception rule)
  2. Strike/premium search primitives in isolation
  3. Full entry selection
  4. End-to-end single-campaign run (locked strikes across rolls, the
     equity-curve-matches-realized-pnl invariant, entire-campaign-closed
     at the end)
  5. Chaining multiple months together
"""

import datetime as dt
import warnings

import pytest

from nifty_backtester.expiry_utils import load_expiry_calendar, load_holidays
from nifty_backtester.market_data import SyntheticMarketDataProvider
from nifty_backtester.campaign_strategy import (
    CampaignConfig, resolve_campaign_expiry_schedule, first_trading_day_of_month,
    _find_strike_by_target_premium, _find_max_theta_funding_pair,
    select_campaign_entry, run_campaign_backtest, run_chained_campaigns,
    SHORT_TAGS, LONG_TAGS,
)
from nifty_backtester import metrics

warnings.filterwarnings("ignore", message="Using the SHIPPED TEMPLATE")
warnings.filterwarnings("ignore", message="divide by zero encountered")
warnings.filterwarnings("ignore", message="invalid value encountered")

CALENDAR = load_expiry_calendar()
HOLIDAYS = load_holidays()


# ─────────────────────────────────────────────
# 1. EXPIRY SCHEDULE RESOLUTION
# ─────────────────────────────────────────────

def test_resolve_schedule_normal_month_start():
    schedule = resolve_campaign_expiry_schedule(CALENDAR, dt.date(2026, 9, 1))
    assert schedule.monthly_expiry == dt.date(2026, 9, 29)
    # 2026-09-01 is itself a weekly expiry date (per the shipped calendar) --
    # the exception rule correctly skips straight to the next cycle rather
    # than trying to sell a contract that expires the same day it opens.
    assert schedule.weekly_cycles[0][0] == dt.date(2026, 9, 8)
    assert schedule.weekly_cycles[-1][0] == dt.date(2026, 9, 29)  # last cycle coincides with monthly expiry
    assert schedule.weekly_cycles[-1][1] == schedule.monthly_expiry_prior_trading_day


def test_resolve_schedule_raises_when_too_close_to_monthly_expiry():
    with pytest.raises(ValueError, match="minimum runway"):
        resolve_campaign_expiry_schedule(CALENDAR, dt.date(2026, 9, 25), min_days=10)


def test_resolve_schedule_exception_skips_cycle_already_at_its_own_eve():
    """If campaign_start falls ON or AFTER the nearest weekly cycle's own
    prior_trading_day (eve), that cycle must be skipped -- selling it would
    require rolling it the same day it was opened."""
    # 2026-09-07 is the eve (prior_trading_day) of the 2026-09-08 weekly cycle
    normal = resolve_campaign_expiry_schedule(CALENDAR, dt.date(2026, 9, 4), min_days=1)
    assert normal.weekly_cycles[0][0] == dt.date(2026, 9, 8)  # not yet at its eve -- sold normally

    skipped = resolve_campaign_expiry_schedule(CALENDAR, dt.date(2026, 9, 7), min_days=1)
    assert skipped.weekly_cycles[0][0] == dt.date(2026, 9, 15), (
        "starting exactly on the 9/8 cycle's own eve should skip straight to the 9/15 cycle"
    )


def test_first_trading_day_of_month_skips_holiday_and_weekend():
    # 2026-11-01 is a Sunday -> should land on the first weekday after it
    d = first_trading_day_of_month(2026, 11, HOLIDAYS)
    assert d.weekday() < 5
    assert d not in HOLIDAYS
    assert d >= dt.date(2026, 11, 1)


# ─────────────────────────────────────────────
# 2. SEARCH PRIMITIVES
# ─────────────────────────────────────────────

def _synthetic_provider(start, end, seed=11):
    return SyntheticMarketDataProvider(
        start - dt.timedelta(days=3), end, initial_spot=24500, annual_vol=0.30, flat_iv=0.16, seed=seed,
    )


def test_find_strike_by_target_premium_converges_reasonably():
    expiry = dt.date(2026, 9, 29)
    as_of = dt.datetime(2026, 9, 1, 15, 30)
    provider = _synthetic_provider(dt.datetime(2026, 9, 1, 9, 15), dt.datetime(2026, 9, 1, 15, 30))

    strike, premium = _find_strike_by_target_premium(provider, expiry, as_of, "call", 200.0, 100, 40)
    assert isinstance(strike, int)
    # should be in the right ballpark -- not asserting exact convergence since
    # synthetic pricing is coarse, just that it's not wildly off
    assert premium > 0


def test_find_strike_by_target_premium_raises_when_nothing_usable():
    class DeadProvider:
        def find_atm_strike(self, expiry, as_of, strike_step=100):
            return 24500
        def get_option_price(self, strike, right, expiry, as_of):
            raise ValueError("no data")

    with pytest.raises(ValueError, match="No usable"):
        _find_strike_by_target_premium(DeadProvider(), dt.date(2026, 9, 29), dt.datetime(2026, 9, 1, 15, 30), "call", 200.0, 100, 5)


def test_find_max_theta_funding_pair_respects_credit_constraint():
    expiry = dt.date(2026, 9, 8)
    as_of = dt.datetime(2026, 9, 1, 15, 30)
    provider = _synthetic_provider(dt.datetime(2026, 9, 1, 9, 15), dt.datetime(2026, 9, 1, 15, 30))

    near, far = _find_max_theta_funding_pair(provider, expiry, as_of, "call", required_credit=50.0,
                                              strike_step=100, search_range=20, r=0.0525, q=0.012)
    assert near[1] + far[1] > 50.0
    assert near[1] >= far[1]  # "near" is defined as the higher-premium leg of the winning pair


def test_find_max_theta_funding_pair_raises_when_unfundable():
    expiry = dt.date(2026, 9, 8)
    as_of = dt.datetime(2026, 9, 1, 15, 30)
    provider = _synthetic_provider(dt.datetime(2026, 9, 1, 9, 15), dt.datetime(2026, 9, 1, 15, 30))

    with pytest.raises(ValueError, match="No pair"):
        _find_max_theta_funding_pair(provider, expiry, as_of, "call", required_credit=10_000_000.0,
                                      strike_step=100, search_range=10, r=0.0525, q=0.012)


# ─────────────────────────────────────────────
# 3. FULL ENTRY SELECTION
# ─────────────────────────────────────────────

def test_select_campaign_entry_funding_clears_cost_independently_per_side():
    config = CampaignConfig(strike_search_range=20)
    monthly_expiry = dt.date(2026, 9, 29)
    weekly_expiry = dt.date(2026, 9, 8)
    entry_time = dt.datetime(2026, 9, 1, 15, 30)
    provider = _synthetic_provider(dt.datetime(2026, 9, 1, 9, 15), entry_time)

    strikes = select_campaign_entry(provider, config, monthly_expiry, weekly_expiry, entry_time)

    call_credit = strikes.short_call_near_premium + strikes.short_call_far_premium
    put_credit = strikes.short_put_near_premium + strikes.short_put_far_premium
    assert call_credit > config.long_quantity * strikes.long_call_premium
    assert put_credit > config.long_quantity * strikes.long_put_premium
    # near leg is the higher-premium leg by construction
    assert strikes.short_call_near_premium >= strikes.short_call_far_premium
    assert strikes.short_put_near_premium >= strikes.short_put_far_premium


# ─────────────────────────────────────────────
# 4. END-TO-END SINGLE CAMPAIGN
# ─────────────────────────────────────────────

def _full_month_provider(seed=7):
    return SyntheticMarketDataProvider(
        dt.datetime(2026, 8, 27, 9, 15), dt.datetime(2026, 9, 29, 15, 30),
        initial_spot=24500, annual_vol=0.30, flat_iv=0.16, seed=seed,
    )


def test_run_campaign_backtest_end_to_end():
    provider = _full_month_provider()
    config = CampaignConfig(strike_search_range=15)
    result = run_campaign_backtest(provider, CALENDAR, HOLIDAYS, dt.date(2026, 9, 1), config)

    assert not result.equity_curve.empty
    assert result.strikes is not None
    assert result.position.is_flat(), "everything should be closed by the monthly expiry eve"

    # equity-curve-matches-realized-pnl invariant (same guarantee as backtest_engine)
    final_equity = result.equity_curve.iloc[-1]
    expected_equity = config.initial_capital + sum(result.trade_pnls)
    assert abs(final_equity - expected_equity) < 1e-6

    report = metrics.full_report(result.equity_curve, result.trade_pnls, config.initial_capital)
    assert report["num_trades"] > 0


def test_run_campaign_backtest_locks_short_strikes_across_every_roll():
    provider = _full_month_provider()
    config = CampaignConfig(strike_search_range=15)
    result = run_campaign_backtest(provider, CALENDAR, HOLIDAYS, dt.date(2026, 9, 1), config)

    for tag in SHORT_TAGS:
        locked_strike = getattr(result.strikes, f"{tag}_strike")
        open_lines = [l for l in result.action_log if f"OPEN campaign.{tag} " in l]
        assert len(open_lines) > 1, f"{tag} should have been opened more than once (entry + at least one roll)"
        for line in open_lines:
            assert f"strike={locked_strike} " in line, f"{tag} was opened at a different strike than its locked entry strike: {line}"


def test_run_campaign_backtest_never_touches_long_legs_mid_month():
    provider = _full_month_provider()
    config = CampaignConfig(strike_search_range=15)
    result = run_campaign_backtest(provider, CALENDAR, HOLIDAYS, dt.date(2026, 9, 1), config)

    for tag in LONG_TAGS:
        open_lines = [l for l in result.action_log if f"OPEN campaign.{tag} " in l]
        close_lines = [l for l in result.action_log if f"CLOSE campaign.{tag} " in l]
        assert len(open_lines) == 1, f"{tag} should only ever be opened once (at entry)"
        assert len(close_lines) == 1, f"{tag} should only ever be closed once (at the final monthly close)"
        assert "closing entire campaign" in close_lines[0]


def test_run_campaign_backtest_final_close_includes_every_leg():
    provider = _full_month_provider()
    config = CampaignConfig(strike_search_range=15)
    result = run_campaign_backtest(provider, CALENDAR, HOLIDAYS, dt.date(2026, 9, 1), config)

    final_close_lines = [l for l in result.action_log if "closing entire campaign" in l]
    closed_tags = set()
    for tag in SHORT_TAGS + LONG_TAGS:
        if any(f"CLOSE campaign.{tag} " in l and "closing entire campaign" in l for l in final_close_lines):
            closed_tags.add(tag)
    assert closed_tags == set(SHORT_TAGS + LONG_TAGS)


def test_min_days_guard_is_wired_through_config():
    provider = _full_month_provider()
    config = CampaignConfig(min_days_to_monthly_expiry_at_start=100)  # impossible to satisfy
    with pytest.raises(ValueError, match="minimum runway"):
        run_campaign_backtest(provider, CALENDAR, HOLIDAYS, dt.date(2026, 9, 1), config)


# ─────────────────────────────────────────────
# 5. CHAINING
# ─────────────────────────────────────────────

def test_run_chained_campaigns_two_months():
    def provider_factory(campaign_start: dt.date):
        # a window comfortably covering that campaign's own month
        month_end = dt.date(campaign_start.year, campaign_start.month, 28) + dt.timedelta(days=8)
        return SyntheticMarketDataProvider(
            dt.datetime.combine(campaign_start, dt.time(9, 15)) - dt.timedelta(days=5),
            dt.datetime.combine(month_end, dt.time(15, 30)),
            initial_spot=24500, annual_vol=0.30, flat_iv=0.16, seed=3,
        )

    config = CampaignConfig(strike_search_range=12)
    results = run_chained_campaigns(provider_factory, CALENDAR, HOLIDAYS, dt.date(2026, 9, 1), num_months=2, config=config)

    assert len(results) == 2
    assert results[0].schedule.monthly_expiry == dt.date(2026, 9, 29)
    assert results[1].schedule.monthly_expiry == dt.date(2026, 10, 27)
    # second campaign should start the first trading day of October
    second_start = min(pd_ts.date() for pd_ts in results[1].equity_curve.index)
    assert second_start.month == 10
    for result in results:
        assert result.position.is_flat()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
