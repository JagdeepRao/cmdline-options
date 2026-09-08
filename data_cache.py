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
"""

import datetime as dt
from pathlib import Path
from typing import Callable

import pandas as pd


class DataCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)

    def get(
        self,
        cache_key: str,
        from_date: dt.date,
        to_date: dt.date,
        fetch_fn: Callable[[dt.date, dt.date], pd.DataFrame],
    ) -> pd.DataFrame:
        """fetch_fn(from_date, to_date) -> DataFrame with a 'datetime' column.
        Returns the merged (cached + freshly-fetched) data for the full
        [from_date, to_date] range requested."""
        cache_path = self.cache_dir / f"{cache_key}.parquet"

        cached = pd.DataFrame()
        if cache_path.exists():
            cached = pd.read_parquet(cache_path)
            cached["datetime"] = pd.to_datetime(cached["datetime"])

        missing_ranges = self._find_missing_ranges(cached, from_date, to_date)

        frames = [cached] if not cached.empty else []
        for gap_start, gap_end in missing_ranges:
            fetched = fetch_fn(gap_start, gap_end)
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

    @staticmethod
    def _find_missing_ranges(cached: pd.DataFrame, from_date: dt.date, to_date: dt.date) -> list[tuple[dt.date, dt.date]]:
        if cached.empty:
            return [(from_date, to_date)]

        cached_min = cached["datetime"].min().date()
        cached_max = cached["datetime"].max().date()

        missing = []
        if from_date < cached_min:
            missing.append((from_date, cached_min - dt.timedelta(days=1)))
        if to_date > cached_max:
            missing.append((cached_max + dt.timedelta(days=1), to_date))
        return missing

    def clear(self, cache_key: str) -> None:
        """Explicit cache-clear for a single key — use if you know you've
        fetched disjoint ranges for the same key (see module docstring)."""
        cache_path = self.cache_dir / f"{cache_key}.parquet"
        if cache_path.exists():
            cache_path.unlink()