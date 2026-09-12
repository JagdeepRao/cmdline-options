"""
Sequenced tests for nifty_live.live_engine.run_live_monitor.

  1. Merges actions from multiple strategies, notifies every one when
     dedupe=False (the default)
  2. dedupe=True suppresses a repeated identical (position, leg, type) key
     across consecutive steps
  3. dedupe=True still re-notifies once a suppressed condition clears and
     later recurs (rising-edge, not one-shot-forever)
  4. `positions` is never mutated -- this engine is notify-only
  5. End-to-end smoke test: a REAL DeltaThresholdStrategy against a REAL
     ReplayLiveFeed (NiftyOptionsDataSample-backed) and REAL DeltaIndicator
     objects, proving the whole new pipeline holds together -- the live
     engine's counterpart to test_full_backtest_engine.py's role for the
     batch engine.
"""

import datetime as dt

from nifty_backtester.strategy import (
    Leg, MultiLegPosition, Right, Direction, Action, ActionType,
    AdjustmentStrategy, DeltaThresholdStrategy, DeltaIndicator,
)
from nifty_backtester.data_layer_sample import NiftyOptionsDataSample
from nifty_live.replay_feed import ReplayLiveFeed
from nifty_live.notifier import CollectingNotifier
from nifty_live.live_engine import run_live_monitor


class ScriptedStrategy(AdjustmentStrategy):
    """Test double: returns a pre-scripted list of Actions per as_of,
    keyed by call order (index into `steps`) -- so dedup/merge behavior
    can be tested without needing real indicators or price data."""

    def __init__(self, steps: list[list[Action]]):
        self.steps = steps
        self.call_count = 0

    def evaluate(self, positions, as_of):
        actions = self.steps[self.call_count] if self.call_count < len(self.steps) else []
        self.call_count += 1
        return actions


def _action(position_name="sold_straddle", leg_tag="sold_call", reason="test"):
    return Action(type=ActionType.RECENTER, position_name=position_name, leg_tag=leg_tag, reason=reason)


# ─────────────────────────────────────────────
# 1. MERGE + DEDUPE=FALSE (default)
# ─────────────────────────────────────────────

def test_merges_actions_from_multiple_strategies():
    strat_a = ScriptedStrategy([[_action(leg_tag="sold_call")]])
    strat_b = ScriptedStrategy([[_action(leg_tag="sold_put")]])
    notifier = CollectingNotifier()

    sent = run_live_monitor({}, [strat_a, strat_b], [dt.datetime(2026, 9, 1, 9, 15)], notifier)

    assert len(sent) == 2
    assert len(notifier.received) == 2
    tags = {a.leg_tag for a in sent}
    assert tags == {"sold_call", "sold_put"}


def test_dedupe_false_notifies_every_step_even_if_identical():
    steps = [[_action()], [_action()], [_action()]]
    strategy = ScriptedStrategy(steps)
    notifier = CollectingNotifier()

    sent = run_live_monitor({}, [strategy], list(range(3)), notifier, dedupe=False)

    assert len(sent) == 3
    assert len(notifier.received) == 3


# ─────────────────────────────────────────────
# 2 & 3. DEDUPE=TRUE SEMANTICS
# ─────────────────────────────────────────────

def test_dedupe_true_suppresses_repeated_identical_key_across_consecutive_steps():
    steps = [[_action()], [_action()], [_action()]]
    strategy = ScriptedStrategy(steps)
    notifier = CollectingNotifier()

    sent = run_live_monitor({}, [strategy], list(range(3)), notifier, dedupe=True)

    assert len(sent) == 1, "same (position, leg, type) key repeated 3x should notify only once while dedupe=True"


def test_dedupe_true_renotifies_after_condition_clears_and_recurs():
    # fires, then clears (empty), then recurs -- should notify on step 0 and step 2, not step 1
    steps = [[_action()], [], [_action()]]
    strategy = ScriptedStrategy(steps)
    notifier = CollectingNotifier()

    sent = run_live_monitor({}, [strategy], list(range(3)), notifier, dedupe=True)

    assert len(sent) == 2, "a condition that clears and later recurs should be re-notified, not suppressed forever"


def test_dedupe_true_distinguishes_different_keys():
    """Different leg_tag/position_name/type combos are independent keys --
    dedup must not conflate them."""
    steps = [
        [_action(leg_tag="sold_call"), _action(leg_tag="sold_put")],
        [_action(leg_tag="sold_call")],  # sold_put's condition cleared; sold_call's persists
    ]
    strategy = ScriptedStrategy(steps)
    notifier = CollectingNotifier()

    sent = run_live_monitor({}, [strategy], list(range(2)), notifier, dedupe=True)

    assert len(sent) == 2  # sold_call (step0) + sold_put (step0); sold_call's step1 repeat suppressed


# ─────────────────────────────────────────────
# 4. NEVER MUTATES `positions`
# ─────────────────────────────────────────────

def test_positions_dict_never_mutated():
    leg = Leg(tag="sold_call", right=Right.CALL, direction=Direction.SHORT, strike=24500.0,
              expiry=dt.date(2026, 9, 8), entry_time=dt.datetime(2026, 9, 1, 9, 15), entry_price=145.3)
    position = MultiLegPosition(name="sold_straddle")
    position.add_leg(leg)
    positions = {"sold_straddle": position}

    # a strategy that would (if executed) close AND open legs
    steps = [[
        Action(type=ActionType.CLOSE_LEG, position_name="sold_straddle", leg_tag="sold_call", reason="test"),
        Action(type=ActionType.OPEN_LEG, position_name="sold_straddle", leg_tag="sold_put",
               new_leg=Leg(tag="sold_put", right=Right.PUT, direction=Direction.SHORT, strike=24500.0,
                            expiry=dt.date(2026, 9, 8), entry_time=dt.datetime(2026, 9, 1, 9, 15), entry_price=130.0),
               reason="test"),
    ]]
    strategy = ScriptedStrategy(steps)
    notifier = CollectingNotifier()

    run_live_monitor(positions, [strategy], [dt.datetime(2026, 9, 1, 9, 15)], notifier)

    # nothing about the position's actual leg state should have changed
    assert len(positions["sold_straddle"].legs) == 1
    assert positions["sold_straddle"].legs[0] is leg
    assert leg.is_open, "run_live_monitor must never close a leg itself"
    assert positions["sold_straddle"].get_leg("sold_put") is None, "run_live_monitor must never open a leg itself"


# ─────────────────────────────────────────────
# 5. END-TO-END SMOKE TEST (real strategy, real replay feed)
# ─────────────────────────────────────────────

def test_end_to_end_real_strategy_and_replay_feed(tmp_path):
    """Proves the whole new pipeline holds together: PositionStore-shaped
    positions -> real DeltaThresholdStrategy + real DeltaIndicator ->
    ReplayLiveFeed(NiftyOptionsDataSample) -> run_live_monitor ->
    ConsoleNotifier-equivalent (CollectingNotifier here). No crash, and
    positions remain exactly as seeded throughout."""
    from_date, to_date = dt.date(2026, 9, 1), dt.date(2026, 9, 2)
    sample = NiftyOptionsDataSample(cache_dir=tmp_path, index_start_price=24500.0)
    feed = ReplayLiveFeed(sample, from_date, to_date, bar_freq_minutes=60)

    weekly_expiry = dt.date(2026, 9, 4)
    strike = 24500
    entry_time = dt.datetime.combine(from_date, dt.time(9, 15))

    sold_call = Leg(tag="sold_call", right=Right.CALL, direction=Direction.SHORT, strike=strike,
                     expiry=weekly_expiry, entry_time=entry_time,
                     entry_price=feed.provider.get_option_price(strike, "call", weekly_expiry, entry_time))
    sold_put = Leg(tag="sold_put", right=Right.PUT, direction=Direction.SHORT, strike=strike,
                    expiry=weekly_expiry, entry_time=entry_time,
                    entry_price=feed.provider.get_option_price(strike, "put", weekly_expiry, entry_time))

    position = MultiLegPosition(name="sold_straddle")
    position.add_leg(sold_call)
    position.add_leg(sold_put)
    positions = {"sold_straddle": position}

    delta_indicators = {
        "sold_call": DeltaIndicator(
            strike=strike, right=Right.CALL, expiry=weekly_expiry,
            spot_lookup=feed.provider.get_spot,
            option_price_lookup=lambda ts: feed.provider.get_option_price(strike, "call", weekly_expiry, ts),
        ),
        "sold_put": DeltaIndicator(
            strike=strike, right=Right.PUT, expiry=weekly_expiry,
            spot_lookup=feed.provider.get_spot,
            option_price_lookup=lambda ts: feed.provider.get_option_price(strike, "put", weekly_expiry, ts),
        ),
    }
    # low threshold so this short, low-vol synthetic window has a real chance of firing
    strategy = DeltaThresholdStrategy(delta_indicators=delta_indicators, sold_threshold=0.3, hedge_threshold=float("inf"))

    notifier = CollectingNotifier()
    sent = run_live_monitor(positions, [strategy], feed, notifier)

    # the real assertion: it ran to completion across every replay step
    # without crashing, and never touched position state
    assert len(positions["sold_straddle"].legs) == 2
    assert all(leg.is_open for leg in positions["sold_straddle"].legs)
    assert len(sent) == len(notifier.received), "every sent action should have reached the notifier exactly once"
    for action in sent:
        assert action.position_name == "sold_straddle"
        assert action.type.value == "recenter"


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
