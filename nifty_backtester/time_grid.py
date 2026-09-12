"""
Shared trading-hours time-grid generator, used by both the batch
backtester (backtest_engine.py) and the live/replay engine
(nifty_live.replay_feed.ReplayLiveFeed) -- extracted here so both walk
IDENTICAL bar timestamps (9:15-15:30, Monday-Friday, at a fixed minute
frequency) rather than risking two independently-maintained copies
drifting apart.
"""

import datetime as dt


def trading_time_grid(start: dt.datetime, end: dt.datetime, bar_freq_minutes: int) -> list[dt.datetime]:
    """Every bar_freq_minutes-spaced timestamp within [start, end]
    (inclusive), restricted to 9:15-15:30 on weekdays -- no NSE holiday
    calendar applied here (same "good enough for grid generation, not a
    trading-day source of truth" scope as sample_data.py's
    _trading_timestamps; expiry-eve/force-close logic elsewhere is what
    actually knows about holidays, via expiry_calendar.csv/
    nse_holidays.csv)."""
    grid = []
    cursor = start.date()
    while cursor <= end.date():
        if cursor.weekday() < 5:
            t = dt.datetime.combine(cursor, dt.time(9, 15))
            day_end = dt.datetime.combine(cursor, dt.time(15, 30))
            while t <= day_end:
                if start <= t <= end:
                    grid.append(t)
                t += dt.timedelta(minutes=bar_freq_minutes)
        cursor += dt.timedelta(days=1)
    return grid
