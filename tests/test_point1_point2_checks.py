"""
Checks for:
  - DeltaThresholdStrategy's extra_thresholds generalization
  - FixedMoveStrategy's position_name generalization
  - DirectionalOverlayStrategy's rewritten spread logic (long leg opens
    immediately on regime flip; short leg waits for its own 15-min signal;
    whole spread closes together on the next flip)
  - expiry_utils' monthly-hedge selection and expiry-eve close timing
"""

import datetime as dt
import pandas as pd
import numpy as np

from nifty_backtester.strategy import (
    Leg, MultiLegPosition, Right, Direction, ActionType, Action,
    DeltaThresholdStrategy, DeltaIndicator, FixedMoveStrategy,
    DirectionalOverlayStrategy, SoldLegSignalStrategy, RSICrossAdapter,
    RSIIndicator,
)
from nifty_backtester.expiry_utils import select_monthly_hedge_expiry, is_expiry_eve_close_bar, is_on_or_after_expiry


class ConstDelta(DeltaIndicator):
    """Test double: DeltaIndicator subclass that just returns a fixed value,
    so threshold tests don't need real pricing/vol machinery."""
    def __init__(self, fixed_value):
        self.fixed_value = fixed_value

    def value(self, as_of):
        return self.fixed_value


def test_delta_threshold_extra_position():
    pos = MultiLegPosition(name="directional_overlay")
    pos.add_leg(Leg(tag="overlay_short_call", right=Right.CALL, direction=Direction.SHORT, strike=100,
                     expiry=dt.date(2026, 9, 8), entry_time=dt.datetime(2026, 9, 1, 9, 15), entry_price=5.0))
    positions = {"directional_overlay": pos}

    strat = DeltaThresholdStrategy(
        delta_indicators={"overlay_short_call": ConstDelta(0.70)},
        extra_thresholds={"directional_overlay": 0.65},
    )
    actions = strat.evaluate(positions, dt.datetime(2026, 9, 1, 10, 0))
    assert len(actions) == 1 and actions[0].position_name == "directional_overlay"
    print("PASS: DeltaThresholdStrategy recenters an extra_thresholds position")


def test_fixed_move_custom_position():
    pos = MultiLegPosition(name="directional_overlay")
    pos.add_leg(Leg(tag="overlay_short_call", right=Right.CALL, direction=Direction.SHORT, strike=100,
                     expiry=dt.date(2026, 9, 8), entry_time=dt.datetime(2026, 9, 1, 9, 15), entry_price=5.0))
    positions = {"directional_overlay": pos}

    strat = FixedMoveStrategy(spot_lookup=lambda ts: 24650.0, position_name="directional_overlay", move_points=100)
    strat.set_reference(24500.0)
    actions = strat.evaluate(positions, dt.datetime(2026, 9, 1, 10, 0))
    assert len(actions) == 1 and actions[0].position_name == "directional_overlay"
    print("PASS: FixedMoveStrategy recenters a custom position_name")


class FixedRegimeAdapter:
    """Test double standing in for RSICrossAdapter/SupertrendEMAAdapter with
    a scripted regime sequence, so overlay tests don't need real 1hr data."""
    def __init__(self, regime_by_time: dict):
        self.regime_by_time = regime_by_time

    def crossed_bullish(self, as_of):
        return False

    def regime(self, as_of):
        # last scripted regime at/before as_of
        applicable = [t for t in self.regime_by_time if t <= as_of]
        if not applicable:
            return "neutral"
        return self.regime_by_time[max(applicable)]


def test_directional_overlay_long_first_then_gated_short():
    t0 = dt.datetime(2026, 9, 1, 9, 15)
    t1 = dt.datetime(2026, 9, 1, 10, 15)  # regime turns bullish here
    t2 = dt.datetime(2026, 9, 1, 11, 15)  # 15-min signal permits selling CALLS here
    t3 = dt.datetime(2026, 9, 1, 13, 15)  # regime flips bearish here (selling PUTs not yet safe)
    t4 = dt.datetime(2026, 9, 1, 14, 15)  # 15-min signal permits selling PUTS here

    hourly_regime = FixedRegimeAdapter({t0: "neutral", t1: "bullish", t3: "bearish"})

    # Distinct per-side 15-min signals so the call-side and put-side gating
    # can be demonstrated independently rather than sharing one timeline.
    call_signal = FixedRegimeAdapter({t0: "bullish", t2: "bearish"})
    put_signal = FixedRegimeAdapter({t0: "bullish", t4: "bearish"})
    short_leg_strategy = SoldLegSignalStrategy(
        signal_adapters={"overlay_short_call": call_signal, "overlay_short_put": put_signal},
        delta_indicators={"overlay_short_call": ConstDelta(0.3), "overlay_short_put": ConstDelta(0.3)},
        position_name="directional_overlay",
    )

    class NoOpCore:
        def evaluate(self, positions, as_of):
            return []

    overlay = DirectionalOverlayStrategy(
        core_strategy=NoOpCore(),
        hourly_regime_signal=hourly_regime,
        short_leg_strategy=short_leg_strategy,
    )

    pos = MultiLegPosition(name="directional_overlay")
    positions = {"directional_overlay": pos}

    def apply(actions, ts):
        for a in actions:
            if a.type == ActionType.OPEN_LEG:
                right = Right.CALL if "call" in a.leg_tag else Right.PUT
                pos.add_leg(Leg(tag=a.leg_tag, right=right, direction=Direction.LONG if "long" in a.leg_tag else Direction.SHORT,
                                 strike=100, expiry=dt.date(2026, 9, 8), entry_time=ts, entry_price=1.0))
            elif a.type == ActionType.CLOSE_LEG:
                leg = pos.get_leg(a.leg_tag)
                if leg:
                    leg.close(ts, 1.0)

    apply(overlay.evaluate(positions, t0), t0)
    assert pos.is_flat(), "should still be flat before the regime turns bullish"

    apply(overlay.evaluate(positions, t1), t1)
    tags_open = {l.tag for l in pos.open_legs()}
    assert tags_open == {"overlay_long_call"}, f"expected only the long leg open right after the flip, got {tags_open}"

    apply(overlay.evaluate(positions, t2), t2)
    tags_open = {l.tag for l in pos.open_legs()}
    assert tags_open == {"overlay_long_call", "overlay_short_call"}, (
        f"expected short leg to join once its own 15-min signal permitted selling, got {tags_open}")

    apply(overlay.evaluate(positions, t3), t3)
    tags_open = {l.tag for l in pos.open_legs()}
    assert tags_open == {"overlay_long_put"}, (
        f"expected old spread closed AND the new opposite long leg opened in the same bar "
        f"('close the spread and open the opposite spread'), short leg not yet safe to sell -- got {tags_open}")

    apply(overlay.evaluate(positions, t4), t4)
    tags_open = {l.tag for l in pos.open_legs()}
    assert tags_open == {"overlay_long_put", "overlay_short_put"}, (
        f"expected the put-side short leg to join once ITS OWN 15-min signal permitted selling, got {tags_open}")

    print("PASS: overlay opens the long leg immediately, gates each side's short leg on its own "
          "independent 15-min signal, and flips (closes old spread, opens new opposite long leg) on regime flip")


def test_monthly_hedge_expiry_selection():
    as_of = dt.date(2026, 9, 20)
    current_month = dt.date(2026, 9, 30)   # 10 days away -> too close
    next_month = dt.date(2026, 10, 30)
    chosen = select_monthly_hedge_expiry(as_of, current_month, next_month, min_days=15)
    assert chosen == next_month, "should roll to next month when current month's expiry is <15 days out"

    as_of2 = dt.date(2026, 9, 1)
    current_month2 = dt.date(2026, 9, 30)  # 29 days away -> fine
    chosen2 = select_monthly_hedge_expiry(as_of2, current_month2, next_month, min_days=15)
    assert chosen2 == current_month2, "should use current month's expiry when it's far enough out"
    print("PASS: monthly hedge expiry selection follows the >=15-day rule")


def test_expiry_eve_close_bar():
    expiry = dt.date(2026, 9, 11)  # Friday, say
    eve = dt.date(2026, 9, 10)     # the actual prior trading day, as looked up from a calendar
    assert not is_expiry_eve_close_bar(dt.datetime.combine(eve, dt.time(15, 0)), eve)
    assert is_expiry_eve_close_bar(dt.datetime.combine(eve, dt.time(15, 30)), eve)
    assert is_expiry_eve_close_bar(dt.datetime.combine(eve, dt.time(15, 45)), eve)
    assert not is_expiry_eve_close_bar(dt.datetime.combine(expiry - dt.timedelta(days=2), dt.time(15, 30)), eve)
    assert is_on_or_after_expiry(dt.datetime.combine(expiry, dt.time(9, 15)), expiry)
    assert not is_on_or_after_expiry(dt.datetime.combine(eve, dt.time(15, 45)), expiry)
    print("PASS: expiry-eve close bar (now keyed to an explicit prior_trading_day, not expiry_date - 1) "
          "and on-or-after-expiry checks behave as expected")


if __name__ == "__main__":
    test_delta_threshold_extra_position()
    test_fixed_move_custom_position()
    test_directional_overlay_long_first_then_gated_short()
    test_monthly_hedge_expiry_selection()
    test_expiry_eve_close_bar()
    print("\nAll checks passed.")
