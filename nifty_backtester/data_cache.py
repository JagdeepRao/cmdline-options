"""
Transparent incremental data cache.

The caller asks for a date range; the cache checks what it already has,
fetches ONLY the missing portion via a supplied fetch function, merges and
persists the result, and returns the full requested slice — the caller
never needs to think about what's cached vs. what needs fetching.

LIMITATION, stated plainly rather than hidden: this uses an "edge extension"
model — it only detects data missing BEFORE the earliest cached point or
AFTER the latest cached point. It does NOT detect gaps INSIDE an already-
spanned range (e.g. if you separately cached Jan 1-10 and Jan 20-31, a later
request for Jan 1-31 would incorrectly think Jan 11-19 is covered, since it
only compares the overall min/max). This matches the realistic backtest
usage pattern (extending a date range forward/backward, one contiguous
period at a time) but would NOT correctly detect an intentionally
disjoint cache. If you ever fetch disjoint ranges for the same key, clear
that cache file before relying on it, or extend this with real trading-day
gap detection (would need an NSE trading-calendar).

RETRY / THROTTLE HANDLING (added):
Broker APIs (Breeze included) throttle under load and can raise, return an
explicit error payload, or — worse — silently return an empty/partial
result on a rate-limit hit. `fetch_fn` is now called through
`_fetch_with_retry`, which:
  - Catches exceptions from fetch_fn and retries with exponential backoff.
  - Also retries when fetch_fn returns an object that quacks like a Breeze
    error response (a dict-like with an 'Error' key set), since Breeze
    doesn't always raise on a throttle — it can just hand back
    {"Success": None, "Error": "..."}. fetch_fn implementations are free to
    return either a DataFrame or such a dict; this layer normalizes it.
  - ALWAYS prints a message before sleeping, including whatever the
    provider returned (exception text or the error payload), so a throttled
    run is visible in the console rather than just silently pausing.
  - After exhausting retries, re-raises the last exception (for real
    exceptions) or returns an empty DataFrame with a printed warning (for
    the "error payload" case) rather than silently pretending nothing was
    missing — callers should treat a persistently-empty return as "still
    missing", not "confirmed no data".
"""

import time
import datetime as dt
from pathlib import Path
from typing import Callable

import pandas as pd

DEFAULT_MAX_RETRIES = 4
DEFAULT_INITIAL_DELAY_SECONDS = 5.0
DEFAULT_BACKOFF_MULTIPLIER = 2.0


class DataCache:
    def __init__(
        self,
        cache_dir: Path,
        max_retries: int = DEFAULT_MAX_RETRIES,
        initial_delay_seconds: float = DEFAULT_INITIAL_DELAY_SECONDS,
        backoff_multiplier: float = DEFAULT_BACKOFF_MULTIPLIER,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.max_retries = max_retries
        self.initial_delay_seconds = initial_delay_seconds
        self.backoff_multiplier = backoff_multiplier

    def get(
        self,
        cache_key: str,
        from_date: dt.date,
        to_date: dt.date,
        fetch_fn: Callable[[dt.date, dt.date], pd.DataFrame],
    ) -> pd.DataFrame:
        """fetch_fn(from_date, to_date) -> DataFrame with a 'datetime' column.
        Returns the merged (cached + freshly-fetched) data for the full
        [from_date, to_date] range requested.

        fetch_fn may raise, or may return a dict with an 'Error' key set
        (matching Breeze's own error-payload convention) instead of a
        DataFrame — both are treated as transient/retryable failures by
        `_fetch_with_retry` below.
        """
        cache_path = self.cache_dir / f"{cache_key}.parquet"

        cached = pd.DataFrame()
        if cache_path.exists():
            cached = pd.read_parquet(cache_path)
            cached["datetime"] = pd.to_datetime(cached["datetime"])

        missing_ranges = self._find_missing_ranges(cached, from_date, to_date)

        frames = [cached] if not cached.empty else []
        for gap_start, gap_end in missing_ranges:
            fetched = self._fetch_with_retry(fetch_fn, gap_start, gap_end, cache_key)
            if fetched is not None and not fetched.empty:
                fetched = fetched.copy()
                fetched["datetime"] = pd.to_datetime(fetched["datetime"])
                frames.append(fetched)

        if not frames:
            return cached  # nothing cached, nothing fetched (e.g. fetch returned empty)

        combined = pd.concat(frames, ignore_index=True)
        combined = combined.drop_duplicates(subset="datetime").sort_values("datetime").reset_index(drop=True)

        if missing_ranges:  # only re-write the cache file if we actually fetched something new
            combined.to_parquet(cache_path)

        mask = (
            (combined["datetime"] >= pd.Timestamp(from_date))
            & (combined["datetime"] < pd.Timestamp(to_date) + pd.Timedelta(days=1))
        )
        return combined[mask].reset_index(drop=True)

    def _fetch_with_retry(
        self,
        fetch_fn: Callable[[dt.date, dt.date], pd.DataFrame],
        gap_start: dt.date,
        gap_end: dt.date,
        cache_key: str,
    ) -> pd.DataFrame:
        """Calls fetch_fn(gap_start, gap_end), retrying with exponential
        backoff on exceptions or Breeze-style {'Error': ...} payloads.
        Prints a message (including whatever the provider actually returned)
        before every sleep so throttling is visible, not silent.
        """
        delay = self.initial_delay_seconds
        last_error_detail = None

        for attempt in range(1, self.max_retries + 1):
            try:
                result = fetch_fn(gap_start, gap_end)
            except Exception as e:
                last_error_detail = f"exception: {e}"
                if attempt == self.max_retries:
                    print(
                        f"[data_cache:{cache_key}] fetch failed for {gap_start}..{gap_end} "
                        f"after {attempt} attempts — giving up. Last error: {last_error_detail}"
                    )
                    raise
                print(
                    f"[data_cache:{cache_key}] fetch attempt {attempt}/{self.max_retries} for "
                    f"{gap_start}..{gap_end} raised {last_error_detail!r} — retrying in {delay:.1f}s."
                )
                time.sleep(delay)
                delay *= self.backoff_multiplier
                continue

            # Breeze-style error payload instead of a DataFrame/exception
            if isinstance(result, dict) and result.get("Error"):
                last_error_detail = f"provider error payload: {result.get('Error')}"
                if attempt == self.max_retries:
                    print(
                        f"[data_cache:{cache_key}] fetch returned an error payload for "
                        f"{gap_start}..{gap_end} after {attempt} attempts — giving up. "
                        f"Last response: {result}"
                    )
                    return pd.DataFrame()
                print(
                    f"[data_cache:{cache_key}] fetch attempt {attempt}/{self.max_retries} for "
                    f"{gap_start}..{gap_end} returned {last_error_detail} — full response: {result} "
                    f"— retrying in {delay:.1f}s."
                )
                time.sleep(delay)
                delay *= self.backoff_multiplier
                continue

            return result

        return pd.DataFrame()

    @staticmethod
    def _find_missing_ranges(cached: pd.DataFrame, from_date: dt.date, to_date: dt.date) -> list[tuple[dt.date, dt.date]]:
        if cached.empty:
            return [(from_date, to_date)]

        cached_min = cached["datetime"].min().date()
        cached_max = cached["datetime"].max().date()

        try:
            from .expiry_utils import load_holidays, is_trading_day
            holidays = load_holidays()
        except Exception:
            holidays = set()
            def is_trading_day(d, h):
                return d.weekday() < 5

        def _has_trading_day(start: dt.date, end: dt.date) -> bool:
            cur = start
            while cur <= end:
                if is_trading_day(cur, holidays):
                    return True
                cur += dt.timedelta(days=1)
            return False

        missing = []
        if from_date < cached_min:
            gap_start = from_date
            gap_end = cached_min - dt.timedelta(days=1)
            if _has_trading_day(gap_start, gap_end):
                missing.append((gap_start, gap_end))
        if to_date > cached_max:
            gap_start = cached_max + dt.timedelta(days=1)
            gap_end = to_date
            if _has_trading_day(gap_start, gap_end):
                missing.append((gap_start, gap_end))
        return missing

    def clear(self, cache_key: str) -> None:
        """Explicit cache-clear for a single key — use if you know you've
        fetched disjoint ranges for the same key (see module docstring)."""
        cache_path = self.cache_dir / f"{cache_key}.parquet"
        if cache_path.exists():
            cache_path.unlink()
