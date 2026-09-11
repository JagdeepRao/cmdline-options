"""
Sequenced regression tests for data_layer_scenario.ScenarioBoundedDataLayer.

Organized to match the design doc's validation order, so a failure here
points straight at which guarantee broke:
  1. Declared-bounds enforcement (date range + optional expiry allowlist)
  2. as_of_cursor enforcement (truncates, never raises; monotonic advance)
  3. Delegation correctness (wrapping adds constraints, doesn't change data)
  4. from_scenario() construction (explicit vs calendar-resolved expiry)
  5. End-to-end sanity: find_atm_strike still works through the wrapper
"""

import datetime as dt

import pandas as pd
import pytest

from nifty_backtester.data_layer_sample import NiftyOptionsDataSample
from nifty_backtester.data_layer_cached import NiftyOptionsDataCached
from nifty_backtester.data_layer_scenario import ScenarioBoundedDataLayer, ScenarioBoundsViolation
from nifty_backtester.scenarios import Scenario
from nifty_backtester.expiry_utils import load_expiry_calendar, get_next_expiry


# ─────────────────────────────────────────────
# 1. DECLARED BOUNDS
# ─────────────────────────────────────────────

def test_bounds_reject_range_outside_declared_window(tmp_path):
    sample = NiftyOptionsDataSample(cache_dir=tmp_path)
    layer = ScenarioBoundedDataLayer(sample, from_date=dt.date(2026, 9, 1), to_date=dt.date(2026, 9, 8))

    with pytest.raises(ScenarioBoundsViolation):
        layer.get_index_historical(dt.date(2026, 9, 1), dt.date(2026, 9, 10))  # to_date beyond bounds

    with pytest.raises(ScenarioBoundsViolation):
        layer.get_index_historical(dt.date(2026, 8, 25), dt.date(2026, 9, 5))  # from_date before bounds


def test_bounds_boundary_is_inclusive(tmp_path):
    sample = NiftyOptionsDataSample(cache_dir=tmp_path)
    layer = ScenarioBoundedDataLayer(sample, from_date=dt.date(2026, 9, 1), to_date=dt.date(2026, 9, 8))

    # exactly matching the declared bounds must succeed, not raise
    df = layer.get_index_historical(dt.date(2026, 9, 1), dt.date(2026, 9, 8))
    assert not df.empty


def test_expiry_allowlist_rejects_disallowed_expiry(tmp_path):
    sample = NiftyOptionsDataSample(cache_dir=tmp_path)
    layer = ScenarioBoundedDataLayer(
        sample, from_date=dt.date(2026, 9, 1), to_date=dt.date(2026, 9, 8),
        allowed_expiries={dt.date(2026, 9, 8)},
    )

    with pytest.raises(ScenarioBoundsViolation):
        layer.get_option_historical(
            expiry=dt.date(2026, 9, 15), strike=24500, right="call",
            from_date=dt.date(2026, 9, 1), to_date=dt.date(2026, 9, 3),
        )

    # the allowed expiry itself must go through fine
    df = layer.get_option_historical(
        expiry=dt.date(2026, 9, 8), strike=24500, right="call",
        from_date=dt.date(2026, 9, 1), to_date=dt.date(2026, 9, 3),
    )
    assert not df.empty


def test_no_allowed_expiries_means_no_expiry_restriction(tmp_path):
    """allowed_expiries=None (the default) means only the date bounds
    apply -- any expiry should go through as long as dates are in range."""
    sample = NiftyOptionsDataSample(cache_dir=tmp_path)
    layer = ScenarioBoundedDataLayer(sample, from_date=dt.date(2026, 9, 1), to_date=dt.date(2026, 9, 8))

    df = layer.get_option_historical(
        expiry=dt.date(2026, 9, 29), strike=24500, right="put",
        from_date=dt.date(2026, 9, 1), to_date=dt.date(2026, 9, 3),
    )
    assert not df.empty


# ─────────────────────────────────────────────
# 2. AS_OF_CURSOR
# ─────────────────────────────────────────────

def test_cursor_truncates_bars_at_or_after_cursor(tmp_path):
    sample = NiftyOptionsDataSample(cache_dir=tmp_path)
    cursor = dt.datetime(2026, 9, 2, 11, 0)
    layer = ScenarioBoundedDataLayer(
        sample, from_date=dt.date(2026, 9, 1), to_date=dt.date(2026, 9, 8), as_of_cursor=cursor,
    )

    df = layer.get_index_historical(dt.date(2026, 9, 1), dt.date(2026, 9, 3))
    assert not df.empty
    assert (pd.to_datetime(df["datetime"]) < cursor).all(), "no bar at/after the cursor should be returned"

    # sanity: the SAME request against the unwrapped layer does include
    # bars at/after that moment -- confirms truncation is actually doing
    # something, not just a no-op on data that never reached that time anyway
    unwrapped_df = sample.get_index_historical(dt.date(2026, 9, 1), dt.date(2026, 9, 3))
    assert (pd.to_datetime(unwrapped_df["datetime"]) >= cursor).any()


def test_cursor_does_not_raise_when_request_extends_past_it():
    """Unlike declared-bounds violations, asking for a range that extends
    past the cursor is NOT an error -- it's the normal "ask for the whole
    day, get back only what's happened so far" replay pattern."""
    sample = NiftyOptionsDataSample(cache_dir="/tmp/_cursor_no_raise_test")
    cursor = dt.datetime(2026, 9, 2, 11, 0)
    layer = ScenarioBoundedDataLayer(
        sample, from_date=dt.date(2026, 9, 1), to_date=dt.date(2026, 9, 8), as_of_cursor=cursor,
    )
    df = layer.get_index_historical(dt.date(2026, 9, 1), dt.date(2026, 9, 8))  # well past the cursor
    assert not df.empty  # got SOMETHING (bars before the cursor), didn't raise


def test_cursor_none_means_full_data_returned(tmp_path):
    sample = NiftyOptionsDataSample(cache_dir=tmp_path)
    layer = ScenarioBoundedDataLayer(sample, from_date=dt.date(2026, 9, 1), to_date=dt.date(2026, 9, 8))
    with_cursor_off = layer.get_index_historical(dt.date(2026, 9, 1), dt.date(2026, 9, 3))
    direct = sample.get_index_historical(dt.date(2026, 9, 1), dt.date(2026, 9, 3))
    assert len(with_cursor_off) == len(direct)


def test_advance_cursor_forward_succeeds_and_is_monotonic(tmp_path):
    sample = NiftyOptionsDataSample(cache_dir=tmp_path)
    layer = ScenarioBoundedDataLayer(
        sample, from_date=dt.date(2026, 9, 1), to_date=dt.date(2026, 9, 8),
        as_of_cursor=dt.datetime(2026, 9, 1, 10, 0),
    )
    layer.advance_cursor(dt.datetime(2026, 9, 1, 11, 0))
    assert layer.as_of_cursor == dt.datetime(2026, 9, 1, 11, 0)

    with pytest.raises(ValueError):
        layer.advance_cursor(dt.datetime(2026, 9, 1, 10, 30))  # backwards -- must reject


def test_advancing_cursor_reveals_more_bars_on_next_call(tmp_path):
    sample = NiftyOptionsDataSample(cache_dir=tmp_path)
    layer = ScenarioBoundedDataLayer(
        sample, from_date=dt.date(2026, 9, 1), to_date=dt.date(2026, 9, 8),
        as_of_cursor=dt.datetime(2026, 9, 1, 9, 30),
    )
    early = layer.get_index_historical(dt.date(2026, 9, 1), dt.date(2026, 9, 1))
    layer.advance_cursor(dt.datetime(2026, 9, 1, 12, 0))
    later = layer.get_index_historical(dt.date(2026, 9, 1), dt.date(2026, 9, 1))
    assert len(later) > len(early), "advancing the cursor should reveal strictly more bars for the same request"


# ─────────────────────────────────────────────
# 3. DELEGATION CORRECTNESS
# ─────────────────────────────────────────────

def test_delegation_matches_underlying_sample_layer_exactly(tmp_path):
    sample = NiftyOptionsDataSample(cache_dir=tmp_path)
    layer = ScenarioBoundedDataLayer(sample, from_date=dt.date(2026, 9, 1), to_date=dt.date(2026, 9, 8))

    direct = sample.get_option_historical(
        expiry=dt.date(2026, 9, 8), strike=24500, right="call",
        from_date=dt.date(2026, 9, 1), to_date=dt.date(2026, 9, 3),
    )
    via_wrapper = layer.get_option_historical(
        expiry=dt.date(2026, 9, 8), strike=24500, right="call",
        from_date=dt.date(2026, 9, 1), to_date=dt.date(2026, 9, 3),
    )
    pd.testing.assert_frame_equal(direct.reset_index(drop=True), via_wrapper.reset_index(drop=True))


def test_delegation_matches_underlying_cached_layer_exactly(tmp_path):
    expiry, strike = dt.date(2026, 9, 8), 24500
    call_df = pd.DataFrame({
        "datetime": [dt.datetime(2026, 9, 1, 9, 15), dt.datetime(2026, 9, 1, 9, 16)],
        "close": [120.5, 121.0],
    })
    call_df.to_parquet(tmp_path / f"NIFTY_{expiry}_{strike}_call_1minute.parquet")

    cached = NiftyOptionsDataCached(cache_dir=tmp_path)
    layer = ScenarioBoundedDataLayer(cached, from_date=dt.date(2026, 9, 1), to_date=dt.date(2026, 9, 1))

    direct = cached.get_option_historical(expiry, strike, "call", dt.date(2026, 9, 1), dt.date(2026, 9, 1))
    via_wrapper = layer.get_option_historical(expiry, strike, "call", dt.date(2026, 9, 1), dt.date(2026, 9, 1))
    pd.testing.assert_frame_equal(direct.reset_index(drop=True), via_wrapper.reset_index(drop=True))


# ─────────────────────────────────────────────
# 4. from_scenario()
# ─────────────────────────────────────────────

def test_from_scenario_uses_explicit_weekly_expiry_when_given(tmp_path):
    sample = NiftyOptionsDataSample(cache_dir=tmp_path)
    scenario = Scenario(
        name="choppy", market_condition="choppy",
        from_date=dt.date(2026, 9, 8), to_date=dt.date(2026, 9, 15),
        weekly_expiry=dt.date(2026, 9, 15), weekly_expiry_prior_trading_day=dt.date(2026, 9, 11),
    )
    layer = ScenarioBoundedDataLayer.from_scenario(sample, scenario)

    assert layer.from_date == dt.date(2026, 9, 8)
    assert layer.to_date == dt.date(2026, 9, 15)
    assert layer.allowed_expiries == {dt.date(2026, 9, 15)}


def test_from_scenario_resolves_weekly_expiry_from_calendar_when_unset(tmp_path):
    sample = NiftyOptionsDataSample(cache_dir=tmp_path)
    scenario = Scenario(
        name="trend_up", market_condition="trending_up",
        from_date=dt.date(2026, 9, 1), to_date=dt.date(2026, 9, 8),
        # weekly_expiry left unset -- must resolve from expiry_calendar.csv
    )
    layer = ScenarioBoundedDataLayer.from_scenario(sample, scenario)

    calendar = load_expiry_calendar()
    expected_expiry, _ = get_next_expiry(calendar, scenario.to_date, "weekly")
    assert layer.allowed_expiries == {expected_expiry}
    assert expected_expiry == dt.date(2026, 9, 8)  # per the shipped template calendar


# ─────────────────────────────────────────────
# 5. END-TO-END: find_atm_strike/get_straddle_and_hedge_data through the wrapper
# ─────────────────────────────────────────────

def test_find_atm_strike_works_through_wrapper_and_stays_bounds_checked(tmp_path):
    sample = NiftyOptionsDataSample(cache_dir=tmp_path)
    layer = ScenarioBoundedDataLayer(
        sample, from_date=dt.date(2026, 9, 1), to_date=dt.date(2026, 9, 8),
        allowed_expiries={dt.date(2026, 9, 4)},
    )
    result = layer.find_atm_strike(
        expiry=dt.date(2026, 9, 4), approx_spot=24800, as_of=dt.datetime(2026, 9, 2, 11, 0),
    )
    assert "strike" in result and "candidates" in result


def test_find_atm_strike_raises_generic_error_when_expiry_disallowed(tmp_path):
    """Documents the known limitation from the module docstring: a
    disallowed expiry surfaces as the base class's generic 'no usable
    data' error (every per-strike candidate raises and gets skipped),
    not a crisp ScenarioBoundsViolation."""
    sample = NiftyOptionsDataSample(cache_dir=tmp_path)
    layer = ScenarioBoundedDataLayer(
        sample, from_date=dt.date(2026, 9, 1), to_date=dt.date(2026, 9, 8),
        allowed_expiries={dt.date(2026, 9, 8)},  # deliberately NOT 9/4
    )
    with pytest.raises(ValueError):
        layer.find_atm_strike(
            expiry=dt.date(2026, 9, 4), approx_spot=24800, as_of=dt.datetime(2026, 9, 2, 11, 0),
        )


def test_get_straddle_and_hedge_data_works_through_wrapper(tmp_path):
    sample = NiftyOptionsDataSample(cache_dir=tmp_path)
    layer = ScenarioBoundedDataLayer(sample, from_date=dt.date(2026, 9, 1), to_date=dt.date(2026, 9, 8))
    result = layer.get_straddle_and_hedge_data(
        expiry=dt.date(2026, 9, 8), spot_at_entry=24537,
        from_date=dt.date(2026, 9, 1), to_date=dt.date(2026, 9, 3),
    )
    assert set(result) == {"straddle_call", "straddle_put", "hedge_call", "hedge_put"}
    for df in result.values():
        assert not df.empty


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
