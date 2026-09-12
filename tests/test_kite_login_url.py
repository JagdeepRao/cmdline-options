"""
Regression guard for the one live-SDK behavior scripts/generate_kite_token.py
depends on: KiteConnect.login_url()'s exact format. If a future kiteconnect
version changes this, we want a loud test failure here rather than a
silently stale login URL printed to whoever runs the script.

KiteConnect(api_key=...) makes NO network call at construction (confirmed
by inspecting the installed SDK's __init__ -- it just stores arguments),
so this is safe to run with a fake api_key and no live session.

kiteconnect is only in pyproject.toml's optional "live" extra (not "dev"),
so a `pip install -e ".[dev]"`-only environment won't have it -- this test
skips cleanly rather than failing in that case, same as any other
optional-dependency-gated test would.
"""

import pytest

kiteconnect = pytest.importorskip("kiteconnect")


def test_kite_login_url_format_matches_what_generate_kite_token_documents():
    kite = kiteconnect.KiteConnect(api_key="fake_key_for_test")
    assert kite.login_url() == "https://kite.zerodha.com/connect/login?api_key=fake_key_for_test&v=3"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
