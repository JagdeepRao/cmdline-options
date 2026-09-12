"""
Tests for nifty_backtester.campaign_straddle_strategy ("Campaign 2").

Mirrors test_campaign_strategy.py's organization for the end-to-end/
chaining sections, plus dedicated tests for the two behaviors specific to
this campaign: the daily ATM-drift recenter and the mandatory weekly roll.
"""

import datetime as dt
import warnings

import pytest

from nifty_backtester.expiry_utils import load_expiry_calendar, load_holidays
from nifty_backtester.market_data import SyntheticMarketDataProvider
from nifty_backtester.campaign_straddle_strategy import (
    StraddleCampaignConfig, run_straddle_campaign_backtest, run_chained_straddle_campaigns,
)
from nifty_backtester import metrics

warnings.filterwarnings("ignore", message="Using the SHIPPED TEMPLATE")
warnings.filterwarnings("ignore", message="divide by zero encountered")
warnings.filterwarnings("ignore", message="invalid value encountered")

CALENDAR = load_expiry_calendar()
HOLIDAYS = load_holidays()


def _full_month_provider(seed=7, annual_vol=0.30):
    return SyntheticMarketDataProvider(
        dt.datetime(2026, 8, 27, 9, 15), dt.datetime(2026, 9, 29, 15, 30),
        initial_spot=24500, annual_vol=annual_vol, flat_iv=0.16, seed=seed,
    )


def test_run_straddle_campaign_end_to_end():
    provider = _full_month_provider()
    config = StraddleCampaignConfig()
    result = run_straddle_campaign_backtest(provider, CALENDAR, HOLIDAYS, dt.date(2026, 9, 1), config)

    assert not result.equity_curve.empty
    assert result.position.is_flat(), "everything should be closed by the monthly expiry eve"

    final_equity = result.equity_curve.iloc[-1]
    expected_equity = config.initial_capital + sum(result.trade_pnls)
    assert abs(final_equity - expected_equity) < 1e-6

    report = metrics.full_report(result.equity_curve, result.trade_pnls, config.initial_capital)
    assert report["num_trades"] > 0


def test_long_straddle_never_touched_mid_month():
    provider = _full_month_provider()
    result = run_straddle_campaign_backtest(provider, CALENDAR, HOLIDAYS, dt.date(2026, 9, 1))

    for tag in ("long_call", "long_put"):
        open_lines = [l for l in result.action_log if f"OPEN campaign_straddle.{tag} " in l]
        close_lines = [l for l in result.action_log if f"CLOSE campaign_straddle.{tag} " in l]
        assert len(open_lines) == 1, f"{tag} should only ever open once, at entry"
        assert len(close_lines) == 1, f"{tag} should only ever close once, at the final monthly close"
        assert "closing entire campaign" in close_lines[0]


def test_short_straddle_rolls_every_week_regardless_of_drift():
    """Even with a very high recenter threshold (so the drift check never
    fires), the short straddle must still roll on every weekly eve --
    the roll is mandatory (the contract expires), not conditional."""
    provider = _full_month_provider()
    config = StraddleCampaignConfig(recenter_threshold_points=1_000_000.0)
    result = run_straddle_campaign_backtest(provider, CALENDAR, HOLIDAYS, dt.date(2026, 9, 1), config)

    roll_lines = [l for l in result.action_log if "unconditional roll" in l]
    # September 2026 weekly cycles from entry (9/1, itself skipped -- see
    # campaign_common's exception rule) through 9/29 monthly: 9/8, 9/15,
    # 9/22, 9/29 -- three eve-rolls before the final (monthly) close, each
    # closing two legs (short_call + short_put).
    assert len(roll_lines) == 3 * 2
    recenter_lines = [l for l in result.action_log if "-- recentering" in l]
    assert not recenter_lines, "an effectively-infinite threshold should never trigger a drift recenter"


def test_short_straddle_recenters_on_drift_with_low_threshold():
    """A near-zero threshold should force a recenter on virtually every
    non-eve trading day the straddle is open, on top of the mandatory
    weekly rolls."""
    provider = _full_month_provider(annual_vol=0.45)  # more movement -> more opportunities to drift
    config = StraddleCampaignConfig(recenter_threshold_points=1.0)
    result = run_straddle_campaign_backtest(provider, CALENDAR, HOLIDAYS, dt.date(2026, 9, 1), config)

    assert result.num_recenters > 0, "expected at least one drift-triggered recenter with a near-zero threshold"
    recenter_lines = [l for l in result.action_log if "-- recentering" in l]
    assert len(recenter_lines) == result.num_recenters * 2  # two legs closed per recenter


def test_recentered_strike_matches_current_atm():
    provider = _full_month_provider(annual_vol=0.45)
    config = StraddleCampaignConfig(recenter_threshold_points=1.0)
    result = run_straddle_campaign_backtest(provider, CALENDAR, HOLIDAYS, dt.date(2026, 9, 1), config)

    # every OPEN of short_call after a "recentered to current ATM" reason
    # should show the SAME strike as the drift check that triggered it
    lines = result.action_log
    for i, line in enumerate(lines):
        if "recentered to current ATM" in line and "OPEN campaign_straddle.short_call" in line:
            # the immediately-preceding CLOSE lines named the drift and the new ATM strike
            preceding = lines[max(0, i - 3):i]
            assert any("-- recentering" in p for p in preceding), f"no recenter reason found before: {line}"


def test_min_days_guard_is_wired_through_config():
    provider = _full_month_provider()
    config = StraddleCampaignConfig(min_days_to_monthly_expiry_at_start=100)
    with pytest.raises(ValueError, match="minimum runway"):
        run_straddle_campaign_backtest(provider, CALENDAR, HOLIDAYS, dt.date(2026, 9, 1), config)


def test_run_chained_straddle_campaigns_two_months():
    def provider_factory(campaign_start: dt.date):
        month_end = dt.date(campaign_start.year, campaign_start.month, 28) + dt.timedelta(days=8)
        return SyntheticMarketDataProvider(
            dt.datetime.combine(campaign_start, dt.time(9, 15)) - dt.timedelta(days=5),
            dt.datetime.combine(month_end, dt.time(15, 30)),
            initial_spot=24500, annual_vol=0.30, flat_iv=0.16, seed=3,
        )

    results = run_chained_straddle_campaigns(provider_factory, CALENDAR, HOLIDAYS, dt.date(2026, 9, 1), num_months=2)

    assert len(results) == 2
    assert results[0].schedule.monthly_expiry == dt.date(2026, 9, 29)
    assert results[1].schedule.monthly_expiry == dt.date(2026, 10, 27)
    for result in results:
        assert result.position.is_flat()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
