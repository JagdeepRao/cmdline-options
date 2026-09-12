"""
ReplayLiveFeed -- walks historical (sample or committed-real) bars
bar-by-bar as though they were arriving live, so nifty_live.live_engine
can be exercised end-to-end with ZERO live Breeze/Zerodha session --
building directly on data_layer_scenario.ScenarioBoundedDataLayer's
as_of_cursor (Phase 2), which was designed with exactly this use in mind.

At each replay step `as_of`, the underlying data layer's cursor is
advanced to just PAST as_of -- so the bar AT as_of itself becomes visible,
but nothing after it. This is what makes strategies evaluated against
feed.provider see exactly what a real live feed would have shown at that
moment: no look-ahead, ever. Verified directly in test_replay_feed.py
(comparing against the unwrapped layer's full data), not just asserted.

live_polling_clock() is the much simpler live-mode counterpart: instead of
walking a fixed historical grid, it yields dt.datetime.now() every
interval_seconds, forever. Paired with a REAL
BreezeMarketDataProvider(NiftyOptionsDataBreeze(...)) instead of a replay
feed, the EXACT SAME live_engine.run_live_monitor() loop already works
against actual live data via Breeze's REST historical endpoint (polling
"the last few minutes" each cycle) -- a lower-frequency stand-in for a
true push-based websocket feed (a later phase), not a different code path.
This is why AdjustmentStrategy/MarketDataProvider were built data-source-
agnostic from the start: nothing about live_engine.py or the strategies
needs to know or care which of these two feeds is driving it.
"""

from __future__ import annotations
import datetime as dt
import time
from typing import Iterator, Optional

from nifty_backtester.data_layer_scenario import ScenarioBoundedDataLayer
from nifty_backtester.market_data import BreezeMarketDataProvider
from nifty_backtester.time_grid import trading_time_grid

# How far past `as_of` to advance the cursor at each replay step, so the
# bar AT as_of itself survives truncation (ScenarioBoundedDataLayer's
# cursor excludes bars AT/AFTER the cursor -- see its own docstring) while
# nothing genuinely in the future does. One second is safely smaller than
# the finest native bar resolution used anywhere in this codebase
# (1-minute), so it can never accidentally reveal the NEXT bar early.
_LOOKAHEAD_GUARD = dt.timedelta(seconds=1)


class ReplayLiveFeed:
    """wrapped: a NiftyOptionsDataCached/NiftyOptionsDataSample (or
    anything else BaseCachedOptionsDataLayer-shaped) instance to replay.
    from_date/to_date: the declared bounds AND the replay window.
    bar_freq_minutes: the cadence a caller steps forward at (same knob as
    backtest_engine.FullBacktestConfig.bar_freq_minutes).
    allowed_expiries: optional, forwarded to ScenarioBoundedDataLayer if
    you want requests for other expiries to raise rather than silently
    succeed against whatever the wrapped layer happens to have.

    .provider is a BreezeMarketDataProvider wired to the cursor-bounded
    layer -- pass THIS to whatever strategies/indicators you're testing,
    exactly like you'd pass a real BreezeMarketDataProvider live.

    Iterate a ReplayLiveFeed directly to drive a replay loop -- each
    yielded `as_of` has ALREADY had the cursor advanced correctly for
    that step by the time your loop body runs:

        feed = ReplayLiveFeed(sample_layer, from_date, to_date)
        for as_of in feed:
            actions = strategy.evaluate(positions, as_of)  # feed.provider is cursor-safe here
    """

    def __init__(
        self,
        wrapped,
        from_date: dt.date,
        to_date: dt.date,
        bar_freq_minutes: int = 15,
        allowed_expiries: Optional[set] = None,
    ):
        self.scenario_layer = ScenarioBoundedDataLayer(
            wrapped, from_date=from_date, to_date=to_date,
            allowed_expiries=allowed_expiries, as_of_cursor=None,
        )
        self.provider = BreezeMarketDataProvider(self.scenario_layer)
        self.bar_freq_minutes = bar_freq_minutes
        self._grid = trading_time_grid(
            dt.datetime.combine(from_date, dt.time(9, 15)),
            dt.datetime.combine(to_date, dt.time(15, 30)),
            bar_freq_minutes,
        )

    def __iter__(self) -> Iterator[dt.datetime]:
        for as_of in self._grid:
            self.scenario_layer.advance_cursor(as_of + _LOOKAHEAD_GUARD)
            yield as_of

    def __len__(self) -> int:
        return len(self._grid)


def live_polling_clock(interval_seconds: float = 60.0) -> Iterator[dt.datetime]:
    """Live-mode counterpart to ReplayLiveFeed's iteration: yields
    dt.datetime.now() every interval_seconds, forever -- an INFINITE
    generator; the caller's own loop (e.g. a `break` condition, or simply
    running this as a long-lived process) decides when to stop, not this
    function. See module docstring for how this pairs with a real
    BreezeMarketDataProvider to run live_engine.run_live_monitor() against
    actual live data."""
    while True:
        yield dt.datetime.now()
        time.sleep(interval_seconds)
