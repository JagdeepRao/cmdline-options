"""
Notifier -- where live_engine.run_live_monitor() sends every action a
strategy recommends. Console today (ConsoleNotifier); swapping in the
existing webapp later is a one-class change (implement notify(), nothing
in live_engine.py needs to know or care which Notifier it's talking to).
"""

from __future__ import annotations
import datetime as dt

from nifty_backtester.strategy import Action


class Notifier:
    def notify(self, action: Action, as_of: dt.datetime) -> None:
        raise NotImplementedError


class ConsoleNotifier(Notifier):
    """Prints one line per action, in the same shape as
    backtest_engine's action_log lines, so console output during live
    validation reads exactly like a backtest run's log -- easy to eyeball
    side by side."""

    def notify(self, action: Action, as_of: dt.datetime) -> None:
        leg = action.leg_tag or "(whole position)"
        print(f"{as_of}: [{action.type.value.upper()}] {action.position_name}.{leg} -- {action.reason}")


class CollectingNotifier(Notifier):
    """Test/inspection double: appends every (action, as_of) pair it
    receives to a list, for asserting against in tests without capturing
    stdout."""

    def __init__(self):
        self.received: list[tuple[Action, dt.datetime]] = []

    def notify(self, action: Action, as_of: dt.datetime) -> None:
        self.received.append((action, as_of))
