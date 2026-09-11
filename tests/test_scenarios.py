"""
Tests for nifty_backtester.scenarios: loading, validation, and the
"no file yet" opt-in behavior other scripts rely on.
"""

import json
import datetime as dt

import pytest

from nifty_backtester.scenarios import load_scenarios, Scenario


def test_missing_file_returns_empty_list(tmp_path):
    assert load_scenarios(tmp_path / "does_not_exist.json") == []


def test_empty_list_file_returns_empty_list(tmp_path):
    path = tmp_path / "scenarios.json"
    path.write_text("[]")
    assert load_scenarios(path) == []


def test_loads_valid_scenarios(tmp_path):
    path = tmp_path / "scenarios.json"
    path.write_text(json.dumps([
        {"name": "trend_up", "market_condition": "trending_up",
         "from_date": "2026-09-01", "to_date": "2026-09-08"},
        {"name": "choppy", "market_condition": "choppy",
         "from_date": "2026-09-08", "to_date": "2026-09-15",
         "weekly_expiry": "2026-09-15", "weekly_expiry_prior_trading_day": "2026-09-11",
         "notes": "range-bound"},
    ]))
    scenarios = load_scenarios(path)
    assert len(scenarios) == 2
    assert scenarios[0] == Scenario(
        name="trend_up", market_condition="trending_up",
        from_date=dt.date(2026, 9, 1), to_date=dt.date(2026, 9, 8),
    )
    assert scenarios[1].weekly_expiry == dt.date(2026, 9, 15)
    assert scenarios[1].weekly_expiry_prior_trading_day == dt.date(2026, 9, 11)
    assert scenarios[1].notes == "range-bound"
    # start/end combine the date with NSE session open/close times
    assert scenarios[0].start == dt.datetime(2026, 9, 1, 9, 15)
    assert scenarios[0].end == dt.datetime(2026, 9, 8, 15, 30)


def test_missing_required_field_raises(tmp_path):
    path = tmp_path / "scenarios.json"
    path.write_text(json.dumps([{"name": "trend_up", "from_date": "2026-09-01", "to_date": "2026-09-08"}]))
    with pytest.raises(ValueError, match="market_condition"):
        load_scenarios(path)


def test_duplicate_names_raise(tmp_path):
    path = tmp_path / "scenarios.json"
    entry = {"name": "dup", "market_condition": "choppy", "from_date": "2026-09-01", "to_date": "2026-09-08"}
    path.write_text(json.dumps([entry, dict(entry)]))
    with pytest.raises(ValueError, match="Duplicate scenario name"):
        load_scenarios(path)


def test_shipped_scenarios_json_is_valid():
    """The real_data_cache/scenarios.json actually shipped in the repo
    must always be valid JSON that load_scenarios can parse -- an empty
    list is fine (that's the shipped default), a malformed file is not."""
    scenarios = load_scenarios()  # default path
    assert isinstance(scenarios, list)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
