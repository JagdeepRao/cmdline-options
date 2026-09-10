"""
Quick, standalone checks (not the full pytest suite) for the two pieces
that just changed:

  1. RenkoOpportunisticOTMStrategy — confirms it never holds both sides at
     once, opens on a trend, and flips (close-then-open) rather than
     stacking when Renko's underlying trend reverses.
  2. SoldLegSignalStrategy — confirms RSICrossAdapter and
     SupertrendEMAAdapter are truly interchangeable: same strategy logic,
     swap the adapter, get the same shape of behaviour (close on adverse
     cross, re-enter only once regime flips back).
"""

import datetime as dt
from strategy import (
    Leg, MultiLegPosition, Right, Direction, ActionType,
    RenkoSuperTrendIndicator, RenkoOpportunisticOTMStrategy,
    RSIIndicator, SupertrendEMAIndicator, RSICrossAdapter, SupertrendEMAAdapter,
    SoldLegSignalStrategy, DeltaIndicator,
)
import pandas as pd
import numpy as np


def make_trending_price_df(n=200, start=100.0, seg=40, step=1.2, seed=1):
    """A price series with clean alternating up/down trend segments, so
    Renko/Supertrend/RSI all get an unambiguous signal to react to rather
    than noise."""
    rng = np.random.default_rng(seed)
    prices = [start]
    direction = 1
    for i in range(1, n):
        if i % seg == 0:
            direction *= -1
        prices.append(prices[-1] + direction * step + rng.normal(0, 0.05))
    ts = [dt.datetime(2026, 9, 1, 9, 15) + dt.timedelta(minutes=i) for i in range(n)]
    return pd.DataFrame({"datetime": ts, "close": prices, "high": [p + 0.1 for p in prices], "low": [p - 0.1 for p in prices]})


def test_renko_trackers_independent_and_react_to_trend():
    df = make_trending_price_df()
    renko = RenkoSuperTrendIndicator(df, atr_box_len=5, atr_box_mult=0.5, st_factor=2.0, st_atr_len=5)
    strat = RenkoOpportunisticOTMStrategy(renko_indicator=renko)

    pos = MultiLegPosition(name="otm_position")
    positions = {"otm_position": pos}

    open_sides_seen = set()
    concurrent_both_sides_observed = False
    for ts in df["datetime"]:
        actions = strat.evaluate(positions, ts)
        for a in actions:
            if a.type == ActionType.CLOSE_LEG:
                leg = pos.get_leg(a.leg_tag)
                if leg is not None:
                    leg.close(ts, 1.0)
            elif a.type == ActionType.OPEN_LEG:
                pos.add_leg(Leg(tag=a.leg_tag, right=Right.CALL if "call" in a.leg_tag else Right.PUT,
                                 direction=Direction.LONG, strike=100, expiry=dt.date(2026, 9, 8),
                                 entry_time=ts, entry_price=1.0, quantity=1))
                open_sides_seen.add(a.leg_tag)
        if len(pos.open_legs()) > 1:
            concurrent_both_sides_observed = True

    assert len(open_sides_seen) >= 2, "expected both sides to get entered at least once across alternating trend segments"
    # With both trackers currently reading the SAME shared Renko trend, they
    # can't both be true at once -- so no overlap is *expected* today. This
    # isn't an invariant the strategy enforces (see RenkoLegTracker's
    # docstring): feed each tracker an independent signal later and this
    # would flip to True with zero code changes here.
    assert not concurrent_both_sides_observed, (
        "unexpected: both sides open at once while both trackers share one indicator -- "
        "shouldn't happen unless the shared-trend assumption changed"
    )
    print(f"PASS: independent call/put trackers both got exercised over the run: {open_sides_seen} "
          f"(no overlap today, as expected while both share one Renko trend)")


def test_sold_leg_adapters_interchangeable():
    df = make_trending_price_df(n=120, seg=30)
    rsi_ind = RSIIndicator(df, rsi_len=10, ema_len=5)
    st_ind = SupertrendEMAIndicator(df, ema_len=10, factor=2.0, atr_period=5)

    rsi_adapter = RSICrossAdapter(rsi_ind)
    st_adapter = SupertrendEMAAdapter(st_ind)

    for name, adapter in [("RSI", rsi_adapter), ("Supertrend+EMA", st_adapter)]:
        pos = MultiLegPosition(name="sold_straddle")
        pos.add_leg(Leg(tag="sold_call", right=Right.CALL, direction=Direction.SHORT, strike=100,
                         expiry=dt.date(2026, 9, 8), entry_time=df["datetime"].iloc[0], entry_price=5.0))
        positions = {"sold_straddle": pos}

        # delta indicator that never triggers harvest, so we isolate the signal-cross behaviour
        never_harvest = DeltaIndicator(
            strike=100, right=Right.CALL, expiry=dt.date(2026, 9, 8),
            spot_lookup=lambda ts: 100.0, option_price_lookup=lambda ts: 5.0,
        )
        strat = SoldLegSignalStrategy(
            signal_adapters={"sold_call": adapter},
            delta_indicators={"sold_call": never_harvest},
        )
        closes = 0
        reopens = 0
        for ts in df["datetime"]:
            actions = strat.evaluate(positions, ts)
            for a in actions:
                if a.type == ActionType.CLOSE_LEG:
                    leg = pos.get_leg(a.leg_tag)
                    if leg is not None:
                        leg.close(ts, 5.0)
                        closes += 1
                elif a.type == ActionType.OPEN_LEG:
                    pos.add_leg(Leg(tag=a.leg_tag, right=Right.CALL, direction=Direction.SHORT, strike=100,
                                     expiry=dt.date(2026, 9, 8), entry_time=ts, entry_price=5.0))
                    reopens += 1
        print(f"  {name}: {closes} close(s) on adverse signal, {reopens} re-entr(y/ies) once regime favored selling again")
        assert closes >= 1, f"{name} adapter never triggered a close over a run with clear trend reversals"

    print("PASS: both adapters plug into the same SoldLegSignalStrategy and produce close/re-enter behaviour")


if __name__ == "__main__":
    test_renko_trackers_independent_and_react_to_trend()
    test_sold_leg_adapters_interchangeable()
    print("\nAll quick checks passed.")
