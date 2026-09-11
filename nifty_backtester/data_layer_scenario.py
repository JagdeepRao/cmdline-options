"""
ScenarioBoundedDataLayer -- wraps an existing cached-style data layer
(NiftyOptionsDataCached or NiftyOptionsDataSample) with explicit bounds
enforcement, so a caller can't silently pull data outside a declared
window just because the underlying store happens to have it.

WHY THIS EXISTS, beyond what BaseCachedOptionsDataLayer already gives you:
that class shares ATM/OTM/straddle logic across cached/sample, but neither
subclass stops you from asking for, say, 2026-11-01 data out of a
scenario that's only supposed to cover 2026-09-01..2026-09-08 -- if the
underlying cache/generator happens to have (or can generate) data for that
date, it'll just hand it over. For scenario-based comparisons
(run_scenarios.py) that's a real risk: a bug that accidentally widens a
date range wouldn't fail loudly, it would silently blend in data from
outside the declared market-condition window.

TWO KINDS OF ENFORCEMENT, deliberately different in character:

1. DECLARED BOUNDS (from_date/to_date, optional allowed_expiries) --
   a hard *configuration* constraint. Asking outside these bounds is
   always a caller/config error, so it raises immediately
   (ScenarioBoundsViolation) rather than truncating or skipping.

2. as_of_cursor -- a *temporal* constraint, not a config error. This is
   what makes this class double as the foundation for a replay-mode
   "live" feed (see nifty_live.live_feed.SampleLiveDataFeed, built on top
   of this): a live feed asking "what's happened so far" isn't making a
   mistake by asking for a day's worth of data before that day is over --
   it should just get back whatever's actually "arrived" (bars strictly
   before the cursor) and nothing more, exactly like a real websocket feed
   would never hand you tomorrow's candles. So cursor enforcement
   TRUNCATES the result rather than raising. The cursor is monotonic
   (advance_cursor refuses to move it backwards) since "time" for a
   replay should never run backwards either.

Built as a BaseCachedOptionsDataLayer subclass (not a bare wrapper) so
find_atm_strike/nearest_otm_strikes/get_straddle_and_hedge_data are
inherited for free -- they call self.get_option_historical internally,
which is the overridden, bounds-checked version, so every derived method
gets the same enforcement automatically with zero extra code.

KNOWN LIMITATION, stated plainly: find_atm_strike's per-candidate strike
scan (in the base class) catches a broad `Exception` per candidate and
just skips it. If allowed_expiries rejects the requested expiry, EVERY
candidate strike will raise the same ScenarioBoundsViolation and get
silently skipped, and the caller will see the base class's generic
"no usable call/put data" ValueError rather than a clear "expiry not
allowed" message. This matches the base class's existing skip-on-failure
design (real committed data is often sparse across strikes, so "some
candidates fail" has to be tolerated there) -- if you need a crisp error
for a disallowed expiry, call get_option_historical directly rather than
going through find_atm_strike.
"""

from __future__ import annotations
import datetime as dt
from typing import Optional

import pandas as pd

from .data_layer_base import BaseCachedOptionsDataLayer
from .expiry_utils import load_expiry_calendar, get_next_expiry
from .scenarios import Scenario


class ScenarioBoundsViolation(ValueError):
    """Raised when a request falls outside this layer's DECLARED bounds
    (date range or, if set, the allowed-expiries list) -- a configuration
    error, never a "no data yet" situation (that's what as_of_cursor
    truncation is for, which never raises)."""


class ScenarioBoundedDataLayer(BaseCachedOptionsDataLayer):
    def __init__(
        self,
        wrapped,
        from_date: dt.date,
        to_date: dt.date,
        allowed_expiries: Optional[set] = None,
        as_of_cursor: Optional[dt.datetime] = None,
    ):
        """wrapped: a NiftyOptionsDataCached / NiftyOptionsDataSample (or
        anything else exposing get_option_historical/get_index_historical)
        instance to delegate actual data access to.
        from_date/to_date: inclusive declared date bounds -- any request
        (even partially) outside this range raises ScenarioBoundsViolation.
        allowed_expiries: optional set of dt.date; if given, requesting any
        other expiry via get_option_historical raises. None means "no
        expiry restriction" (only the date bounds apply).
        as_of_cursor: optional dt.datetime "replay clock". If set, every
        returned DataFrame is truncated to bars strictly before the
        cursor -- see module docstring for why this truncates rather than
        raises.
        """
        super().__init__(cache_dir=getattr(wrapped, "cache_dir", "."))
        self.wrapped = wrapped
        self.from_date = from_date
        self.to_date = to_date
        self.allowed_expiries = set(allowed_expiries) if allowed_expiries is not None else None
        self.as_of_cursor = as_of_cursor

    @classmethod
    def from_scenario(
        cls,
        wrapped,
        scenario: Scenario,
        calendar_path=None,
        as_of_cursor: Optional[dt.datetime] = None,
    ) -> "ScenarioBoundedDataLayer":
        """Builds bounds directly from a Scenario (scenarios.py): date
        range comes straight from scenario.from_date/to_date; the weekly
        expiry allowlist is scenario.weekly_expiry if the scenario set one
        explicitly, else it's resolved from expiry_calendar.csv the same
        way run_scenarios.py does it (first weekly expiry on/after
        scenario.to_date).

        Only the WEEKLY expiry is allowlisted here -- Scenario doesn't
        carry a monthly-hedge expiry, so a scenario-driven hedge backtest
        should construct ScenarioBoundedDataLayer directly with an explicit
        allowed_expiries={weekly, monthly} instead of using this shortcut.
        """
        if scenario.weekly_expiry is not None:
            weekly_expiry = scenario.weekly_expiry
        else:
            calendar = load_expiry_calendar(calendar_path) if calendar_path else load_expiry_calendar()
            weekly_expiry, _ = get_next_expiry(calendar, scenario.to_date, "weekly")

        return cls(
            wrapped=wrapped,
            from_date=scenario.from_date,
            to_date=scenario.to_date,
            allowed_expiries={weekly_expiry},
            as_of_cursor=as_of_cursor,
        )

    # ---- bounds / cursor enforcement ----
    def _check_bounds(self, from_date: dt.date, to_date: dt.date, expiry: Optional[dt.date] = None) -> None:
        if from_date < self.from_date or to_date > self.to_date:
            raise ScenarioBoundsViolation(
                f"Requested range {from_date}..{to_date} falls outside this scenario's "
                f"declared bounds {self.from_date}..{self.to_date}."
            )
        if expiry is not None and self.allowed_expiries is not None and expiry not in self.allowed_expiries:
            raise ScenarioBoundsViolation(
                f"Expiry {expiry} is not in this scenario's allowed expiries "
                f"{sorted(self.allowed_expiries)}."
            )

    def _apply_cursor(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.as_of_cursor is None or df.empty:
            return df
        df = df.copy()
        df["datetime"] = pd.to_datetime(df["datetime"])
        return df[df["datetime"] < self.as_of_cursor].reset_index(drop=True)

    def advance_cursor(self, new_cursor: dt.datetime) -> None:
        """Moves the replay clock forward. Refuses to move it backwards --
        a replay's notion of "now" should never regress, same as a real
        live feed's clock never would."""
        if self.as_of_cursor is not None and new_cursor < self.as_of_cursor:
            raise ValueError(
                f"Cannot move as_of_cursor backwards: current={self.as_of_cursor}, requested={new_cursor}."
            )
        self.as_of_cursor = new_cursor

    # ---- BaseCachedOptionsDataLayer's two required data-access methods ----
    def get_option_historical(
        self,
        expiry: dt.date,
        strike: int,
        right: str,
        from_date: dt.date,
        to_date: dt.date,
        interval: str = "1minute",
    ) -> pd.DataFrame:
        self._check_bounds(from_date, to_date, expiry=expiry)
        df = self.wrapped.get_option_historical(expiry, strike, right, from_date, to_date, interval)
        return self._apply_cursor(df)

    def get_index_historical(
        self,
        from_date: dt.date,
        to_date: dt.date,
        interval: str = "1minute",
    ) -> pd.DataFrame:
        self._check_bounds(from_date, to_date)
        df = self.wrapped.get_index_historical(from_date, to_date, interval)
        return self._apply_cursor(df)
