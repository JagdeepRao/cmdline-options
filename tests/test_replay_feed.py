"""
Sequenced tests for nifty_live.replay_feed.

  1. Grid iteration shape (matches trading_time_grid directly)
  2. THE critical guarantee: no look-ahead -- at replay step `as_of`, the
     feed never reveals bars strictly after as_of, verified by comparing
     against the unwrapped (full) data at the same moment.
  3. Correctness: the bar the feed DOES reveal at as_of matches the
     unwrapped layer's actual bar for that exact minute (not just "some
     earlier bar", the RIGHT one).
  4. live_polling_clock()'s yield/sleep behavior.
"""

import datetime as dt
import itertools

import pandas as pd
import pytest

from nifty_backtester.data_layer_sample import NiftyOptionsDataSample
from nifty_backtester.time_grid import trading_time_grid
from nifty_live.replay_feed import ReplayLiveFeed, live_polling_clock


# ─────────────────────────────────────────────
# 1. GRID ITERATION
# ─────────────────────────────────────────────

def test_iteration_yields_same_grid_as_trading_time_grid(tmp_path):
    from_date, to_date = dt.date(2026, 9, 1), dt.date(2026, 9, 2)
    sample = NiftyOptionsDataSample(cache_dir=tmp_path)
    feed = ReplayLiveFeed(sample, from_date, to_date, bar_freq_minutes=60)

    expected = trading_time_grid(
        dt.datetime.combine(from_date, dt.time(9, 15)),
        dt.datetime.combine(to_date, dt.time(15, 30)),
        60,
    )
    assert list(feed) == expected
    assert len(feed) == len(expected)


# ─────────────────────────────────────────────
# 2. NO LOOK-AHEAD (the critical guarantee)
# ─────────────────────────────────────────────

def test_no_lookahead_mid_replay(tmp_path):
    """Partway through a day's replay, the feed must not yet reveal bars
    from later in that same day -- confirmed by comparing against what the
    UNWRAPPED layer actually has for the full day."""
    from_date = to_date = dt.date(2026, 9, 1)
    sample = NiftyOptionsDataSample(cache_dir=tmp_path)
    feed = ReplayLiveFeed(sample, from_date, to_date, bar_freq_minutes=60)

    full_day = sample.get_index_historical(from_date, to_date)
    full_day["datetime"] = pd.to_datetime(full_day["datetime"])
    last_bar_of_day = full_day["datetime"].max()

    steps = list(feed)
    mid_step = steps[len(steps) // 2]  # somewhere in the middle of the trading day

    # re-drive the feed up to (and including) mid_step, then peek at what's visible
    feed2 = ReplayLiveFeed(sample, from_date, to_date, bar_freq_minutes=60)
    for as_of in feed2:
        if as_of == mid_step:
            break

    visible = feed2.scenario_layer.get_index_historical(from_date, to_date)
    visible["datetime"] = pd.to_datetime(visible["datetime"])

    assert not visible.empty
    assert visible["datetime"].max() <= mid_step, "no bar visible mid-replay should be after the current replay step"
    assert visible["datetime"].max() < last_bar_of_day, (
        "sanity check: mid-replay should NOT already see the full day's last bar -- "
        "otherwise this test isn't actually exercising look-ahead prevention"
    )


def test_no_lookahead_via_provider_get_spot(tmp_path):
    """Same guarantee, exercised through the public interface a strategy
    would actually use (feed.provider.get_spot), not the underlying layer
    directly."""
    from_date = to_date = dt.date(2026, 9, 1)
    sample = NiftyOptionsDataSample(cache_dir=tmp_path)
    feed = ReplayLiveFeed(sample, from_date, to_date, bar_freq_minutes=60)

    full_day = sample.get_index_historical(from_date, to_date)
    full_day["datetime"] = pd.to_datetime(full_day["datetime"])

    for as_of in feed:
        spot = feed.provider.get_spot(as_of)
        # the nearest-bar-to-as_of the provider found must itself be <= as_of
        # (cannot be a bar that hasn't "happened" yet in the replay)
        eligible = full_day[full_day["close"] == spot]
        assert not eligible.empty
        assert (eligible["datetime"] <= as_of).any(), (
            f"get_spot({as_of}) returned a value only matching bar(s) after as_of -- look-ahead leak"
        )


# ─────────────────────────────────────────────
# 3. CORRECTNESS (not just "no future data", but "the right past data")
# ─────────────────────────────────────────────

def test_provider_returns_the_correct_bar_at_each_step(tmp_path):
    from_date = to_date = dt.date(2026, 9, 1)
    sample = NiftyOptionsDataSample(cache_dir=tmp_path)
    feed = ReplayLiveFeed(sample, from_date, to_date, bar_freq_minutes=60)

    full_day = sample.get_index_historical(from_date, to_date)
    full_day["datetime"] = pd.to_datetime(full_day["datetime"])
    full_day = full_day.set_index("datetime")["close"]

    for as_of in feed:
        expected_close = full_day.loc[full_day.index <= as_of].iloc[-1]
        actual = feed.provider.get_spot(as_of)
        assert actual == expected_close


# ─────────────────────────────────────────────
# 4. live_polling_clock()
# ─────────────────────────────────────────────

def test_live_polling_clock_yields_now_and_sleeps_between(monkeypatch):
    fixed_now = dt.datetime(2026, 9, 1, 10, 0, 0)
    sleep_calls = []

    monkeypatch.setattr("nifty_live.replay_feed.dt.datetime", type("FakeDT", (), {"now": staticmethod(lambda: fixed_now)}))
    monkeypatch.setattr("nifty_live.replay_feed.time.sleep", lambda s: sleep_calls.append(s))

    clock = live_polling_clock(interval_seconds=5)
    first_three = list(itertools.islice(clock, 3))

    assert first_three == [fixed_now, fixed_now, fixed_now]
    # sleep() only runs when the generator RESUMES after a yield (i.e.
    # between yields, not after the last one taken) -- 3 yields means the
    # generator paused for the 3rd one before ever calling sleep() again,
    # so exactly 2 sleep calls happened, not 3.
    assert sleep_calls == [5, 5]


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
