"""
Standalone regression test for nifty_backtester.time_grid.trading_time_grid
-- extracted from backtest_engine.py's own _time_grid (Phase 5) so both
the batch backtester and nifty_live.replay_feed.ReplayLiveFeed walk
identical bar timestamps. This is a thin, direct test of the utility
itself; test_full_backtest_engine.py already exercises it indirectly via
the full engine.
"""

import datetime as dt

from nifty_backtester.time_grid import trading_time_grid


def test_grid_excludes_weekends():
    # 2026-09-05 is a Saturday, 2026-09-06 a Sunday
    start = dt.datetime(2026, 9, 4, 9, 15)   # Friday
    end = dt.datetime(2026, 9, 7, 15, 30)    # Monday
    grid = trading_time_grid(start, end, bar_freq_minutes=60)
    weekdays = {t.weekday() for t in grid}
    assert weekdays == {4, 0}  # Friday (4) and Monday (0) only -- no Sat(5)/Sun(6)


def test_grid_respects_market_open_and_close_times():
    start = dt.datetime(2026, 9, 1, 9, 15)
    end = dt.datetime(2026, 9, 1, 15, 30)
    grid = trading_time_grid(start, end, bar_freq_minutes=15)
    assert grid[0] == dt.datetime(2026, 9, 1, 9, 15)
    assert grid[-1] <= dt.datetime(2026, 9, 1, 15, 30)
    assert all(dt.time(9, 15) <= t.time() <= dt.time(15, 30) for t in grid)


def test_grid_spacing_matches_bar_freq_minutes():
    start = dt.datetime(2026, 9, 1, 9, 15)
    end = dt.datetime(2026, 9, 1, 11, 15)
    grid = trading_time_grid(start, end, bar_freq_minutes=30)
    diffs = {(b - a) for a, b in zip(grid, grid[1:])}
    assert diffs == {dt.timedelta(minutes=30)}


def test_grid_clips_to_the_requested_start_and_end_within_a_day():
    """start/end mid-day should clip the first/last day's bars, not just
    the overall date range."""
    start = dt.datetime(2026, 9, 1, 11, 0)
    end = dt.datetime(2026, 9, 1, 13, 0)
    grid = trading_time_grid(start, end, bar_freq_minutes=30)
    assert grid[0] >= start
    assert grid[-1] <= end


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
