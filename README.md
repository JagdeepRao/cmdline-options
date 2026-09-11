# NIFTY options strategy backtester

A Python project using ICICI Breeze to fetch options prices and evaluate the
efficacy of various strategies and indicators. We treat an options strategy
as a **starting position plus subsequent adjustments** (rather than a fixed
spread/strangle payoff, the way most sites present options strategies) and
backtest that whole lifecycle.

See `STRATEGY.md` for what the code actually does today — position shapes,
management strategies, indicators, expiry-close rules.

## Layout

```
nifty_backtester/        the library: strategy/pricing/metrics/backtest engine,
                          the three data layers, and expiry-calendar handling
scripts/                  CLI entry points -- run these, don't import them
tests/                     pytest suite (credential-free, runs against synthetic data)
real_data_cache/           committed REAL market data + named scenarios (see below)
expiry_calendar.csv → nifty_backtester/expiry_calendar.csv
                          (shipped as an illustrative template -- replace with the
                          real NSE calendar before trusting real trading decisions)
```

## Install

```
pip install -e . --break-system-packages
```
This makes `nifty_backtester` importable everywhere and pulls in
`breeze_connect`, `pandas`, `pyarrow`, `fastparquet`, `vollib`. Add `pytest`
too (`pip install -e ".[dev]" --break-system-packages`) to run the test suite.

## Tests

```
pytest
```
Runs everything under `tests/` against synthetic/sample data — no live
credentials needed, no network calls. (`scripts/verify_find_atm.py` is a
manual verification script requiring a real Breeze session; it's
deliberately not named `test_*.py` so a bare `pytest` never picks it up.)

## Three ways to get data — LIVE, CACHED, or SYNTHETIC

Every data-consuming script in `scripts/` picks a data source the same way,
via `nifty_backtester.data_sources.resolve_data_layer` (see `--data-source`
on any script):

| Source | What it is | Needs |
|---|---|---|
| **LIVE** | `NiftyOptionsDataBreeze` — real account, real network calls | `BREEZE_API_KEY` / `BREEZE_API_SECRET` / `BREEZE_SESSION_TOKEN` env vars |
| **CACHED** | `NiftyOptionsDataCached` — reads only committed parquet files under `real_data_cache/`, real data, zero network calls | nothing — this is what a credential-free session (including this one) runs against |
| **SYNTHETIC** | `NiftyOptionsDataSample` — deterministic fake data | nothing — always available, final fallback |

`--data-source auto` (the default on every script) picks LIVE if
credentials are set, else CACHED if `real_data_cache/` has anything
committed, else SYNTHETIC — and every script's console output and output
filename say which one actually ran, so results are never ambiguous later.

**Workflow for getting real data into a credential-free session:** run any
script locally with real Breeze credentials, then commit the resulting
parquet files under `real_data_cache/`, push, and any future session (this
one included) can run `--data-source cached` against real market history
with no login step. Full instructions, the exact cache-key filename
convention, and size-discipline notes are in `real_data_cache/README.md`.

### Multiple date ranges / market conditions

A single committed date range only shows one market regime. Declare several
named ranges — trending, choppy, a high-IV event week, etc. — in
`real_data_cache/scenarios.json` (schema and an example in
`real_data_cache/scenarios.example.json`), each independently backed by
LIVE/CACHED/SYNTHETIC data, and run the full strategy comparison across all
of them:
```
python3 scripts/run_scenarios.py
```
This is what turns "does strategy X beat strategy Y" into "...and does the
answer change across real market regimes," rather than being stuck with
whichever one date range happens to be on hand.

## Scripts

All run from the repo root, e.g. `python3 scripts/run_backtest_from_range.py ...`.
Every script's own `--help` / docstring has the full option list and examples.

| Script | Purpose |
|---|---|
| `download_option_data.py` | Pull one option leg or the index and annotate it with indicator BUY/SELL signals — indicator-level sanity check |
| `download_and_run_strategy.py` | Run one full strategy config and print every open/close/recenter decision — strategy-level sanity check |
| `run_backtest_from_range.py` | Give it a date range; runs the full strategy/position-combination sweep and returns a metrics table |
| `run_scenarios.py` | Same sweep, but across every named scenario in `real_data_cache/scenarios.json` |
| `compare_strategies.py`, `run_all_strategies_demo.py` | Demo/comparison scripts against synthetic data |
| `debug_breeze.py` | Minimal raw-response diagnostic for a live Breeze session — run this FIRST against a real account |
| `run_real_greeks.py` | Pulls one real option leg + matching spot and computes real IV/Greeks |
| `verify_find_atm.py` | Manual verification of `find_atm_strike()` against a real account (not a pytest test — see Tests above) |
