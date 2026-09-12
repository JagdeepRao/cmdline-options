"""
Tests for nifty_backtester.env_loader.load_env.

Uses only explicit dotenv_path (never find_dotenv's upward search), so
these tests are hermetic regardless of whether a real .env happens to
exist somewhere above the test-runner's cwd. Every test cleans up any
environment variable it sets, since load_dotenv writes directly into
os.environ (not into a fixture-scoped copy).
"""

import os

import pytest

from nifty_backtester.env_loader import load_env


@pytest.fixture(autouse=True)
def _clean_test_env_vars():
    """Belt-and-suspenders cleanup around every test in this file, in case
    a test fails partway through its own try/finally."""
    keys = ["ENV_LOADER_TEST_KEY", "ENV_LOADER_TEST_KEY_2"]
    for k in keys:
        os.environ.pop(k, None)
    yield
    for k in keys:
        os.environ.pop(k, None)


def test_load_env_returns_false_when_explicit_path_does_not_exist(tmp_path):
    missing = tmp_path / "does_not_exist.env"
    assert load_env(dotenv_path=missing) is False
    assert "ENV_LOADER_TEST_KEY" not in os.environ


def test_load_env_loads_values_from_explicit_path(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("ENV_LOADER_TEST_KEY=hello_world\nENV_LOADER_TEST_KEY_2=second_value\n")

    loaded = load_env(dotenv_path=env_file)

    assert loaded is True
    assert os.environ["ENV_LOADER_TEST_KEY"] == "hello_world"
    assert os.environ["ENV_LOADER_TEST_KEY_2"] == "second_value"


def test_load_env_does_not_override_existing_value_by_default(tmp_path):
    os.environ["ENV_LOADER_TEST_KEY"] = "already_set_in_shell"
    env_file = tmp_path / ".env"
    env_file.write_text("ENV_LOADER_TEST_KEY=from_dotenv_file\n")

    load_env(dotenv_path=env_file)  # override=False is the default

    assert os.environ["ENV_LOADER_TEST_KEY"] == "already_set_in_shell", (
        "an already-set real environment variable should win over .env by default"
    )


def test_load_env_override_true_replaces_existing_value(tmp_path):
    os.environ["ENV_LOADER_TEST_KEY"] = "already_set_in_shell"
    env_file = tmp_path / ".env"
    env_file.write_text("ENV_LOADER_TEST_KEY=from_dotenv_file\n")

    load_env(dotenv_path=env_file, override=True)

    assert os.environ["ENV_LOADER_TEST_KEY"] == "from_dotenv_file"


def test_load_env_no_dotenv_path_and_none_found_returns_false(tmp_path, monkeypatch):
    """When dotenv_path is omitted, load_env searches upward from cwd via
    find_dotenv -- run from an isolated empty tmp_path with no .env
    anywhere above it (monkeypatch chdir there) so this doesn't
    accidentally pick up a real project .env."""
    monkeypatch.chdir(tmp_path)
    assert load_env() is False


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
