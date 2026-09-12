"""
Confirms nifty_backtester.data_sources.resolve_data_layer() actually calls
load_env() before checking for Breeze credentials -- the whole point of
wiring .env support in is that a caller shouldn't need to do anything
beyond keeping .env up to date, so this needs to happen automatically
rather than being left to each script to remember.

Forces prefer="synthetic" throughout so this never risks a real network
call / live Breeze session regardless of what's in the actual environment
this test happens to run in.
"""

from nifty_backtester.data_sources import resolve_data_layer


def test_resolve_data_layer_invokes_load_env(monkeypatch):
    calls = []
    monkeypatch.setattr("nifty_backtester.data_sources.load_env", lambda *a, **k: calls.append((a, k)))

    resolve_data_layer(prefer="synthetic")

    assert calls, "resolve_data_layer() should call load_env() so .env-provided credentials are picked up automatically"


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
