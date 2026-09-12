"""
live_engine.run_live_monitor() -- the live/replay counterpart to
backtest_engine.run_full_backtest(), and the reason strategy.py's
AdjustmentStrategy/Leg/MultiLegPosition classes were built data-source-
agnostic from the start: this calls the EXACT SAME
AdjustmentStrategy.evaluate(positions, as_of) the backtester uses, driven
by either a ReplayLiveFeed (testing/validation, zero live session) or a
real live MarketDataProvider -- zero changes needed to strategy.py either
way.

CRITICAL DESIGN DECISION, stated plainly rather than left implicit: this
function NEVER executes a trade and NEVER mutates `positions`. It only
evaluates strategies and reports whatever Actions come back to a
Notifier -- decision support, not auto-trading. This matches the stated
workflow: track positions in real time, surface opportunities/adjustments
on the console, validate there, THEN (manually today, via the webapp
later) actually act and update position state (PositionStore.save() after
editing, reloading a fresh seed file, or re-running
PositionStore.import_from_zerodha()) before the next monitoring run.

Because this never mutates positions or rebuilds indicators itself, the
CALLER is responsible for reloading positions and rebuilding any
per-leg indicators (DeltaIndicator, RSIIndicator, etc.) between monitoring
runs if legs actually changed shape (recentered, closed, opened) since the
last run -- see backtest_engine.py's _rebuild_leg_signal for the analogous
logic IF this engine is ever extended to auto-execute; deliberately not
built here.

DEDUPLICATION -- an open UX tradeoff, stated explicitly rather than hidden
behind a "smart" default: `dedupe=True` suppresses re-notifying the exact
same (position_name, leg_tag, action_type) on every step while the
underlying condition stays true (e.g. delta still above threshold) --
without this, a persistent condition spams an identical console line every
single bar until a human acts. But this is a blunt "rising edge only"
rule: for an OPEN_LEG-style recommendation ("you're missing this leg"),
which stays persistently true for as long as the leg genuinely isn't open
(since this engine never opens it for you), dedupe=True means you see it
ONCE and then silence -- even if you never actually acted on it. There is
no periodic re-reminder/cooldown here to distinguish "still true, already
told you" from "still true, you forgot" -- if that distinction matters in
practice, dedupe=False (the default) shows every recommendation on every
step it's true, noisier but never silently hiding an unresolved condition.
Default is dedupe=False deliberately: this is meant for VALIDATING
strategy behavior on the console first (per the stated workflow) -- seeing
a signal repeat bar-by-bar is useful evidence it's behaving as expected,
not noise to hide.
"""

from __future__ import annotations
import datetime as dt
from typing import Iterable

from nifty_backtester.strategy import MultiLegPosition, AdjustmentStrategy, Action
from .notifier import Notifier


def run_live_monitor(
    positions: dict[str, MultiLegPosition],
    strategies: list[AdjustmentStrategy],
    time_steps: Iterable[dt.datetime],
    notifier: Notifier,
    dedupe: bool = False,
) -> list[Action]:
    """Evaluates every strategy in `strategies` at every `as_of` in
    `time_steps` (a ReplayLiveFeed instance, live_polling_clock(), or any
    other iterable of datetimes) and sends every resulting Action to
    `notifier`. Returns every Action actually sent (post-dedupe), for
    callers/tests that want to inspect what fired without a custom
    Notifier.

    Never touches `positions` itself -- see module docstring."""
    all_sent: list[Action] = []
    active_keys: set[tuple] = set()

    for as_of in time_steps:
        actions: list[Action] = []
        for strategy in strategies:
            actions.extend(strategy.evaluate(positions, as_of))

        seen_this_step: set[tuple] = set()
        for action in actions:
            key = (action.position_name, action.leg_tag, action.type)
            seen_this_step.add(key)
            if dedupe and key in active_keys:
                continue
            notifier.notify(action, as_of)
            all_sent.append(action)

        if dedupe:
            active_keys = seen_this_step

    return all_sent
