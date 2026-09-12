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
`breeze_connect`, `pandas`, `pyarrow`, `fastparquet`, `python-dotenv`,
`vollib`. Add `pytest` too (`pip install -e ".[dev]" --break-system-packages`)
to run the test suite, or `pip install -e ".[live]" --break-system-packages`
to add `kiteconnect` for the `nifty_live/` layer.

## Credentials (.env)

Copy `.env.example` to `.env` (already gitignored — never commit the real
file) and fill in your Breeze and/or Kite credentials. Every script that
needs them loads `.env` automatically via `nifty_backtester.env_loader`,
so there's nothing to `export` by hand each session:

```
cp .env.example .env
# edit .env with your real values
```

Both brokers' short-lived tokens need periodic refreshing — `.env.example`
documents which lines to overwrite and how often:
- `BREEZE_SESSION_TOKEN` — regenerated every Breeze login
- `KITE_ACCESS_TOKEN` — regenerated every trading day (Kite Connect's
  access tokens are valid for one day only)

An explicit `export FOO=...` in your shell still takes priority over
whatever's in `.env`, so one-off overrides work without editing the file.

## Tests

```
pytest
```
Runs everything under `tests/` against synthetic/sample data — no live
credentials needed, no network calls. (`scripts/verify_find_atm.py` and
`scripts/verify_zerodha_import.py` are manual verification scripts
requiring a real Breeze/Zerodha session; they're deliberately not named
`test_*.py` so a bare `pytest` never picks them up.)

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

## Live monitoring (`nifty_live/`, in progress)

A separate package, built on the exact same `Leg`/`MultiLegPosition`/
`AdjustmentStrategy` classes as the backtester (zero changes to
`strategy.py` needed) — the backtester stays batch/offline; `nifty_live`
tracks currently-held positions and (eventually) reacts to live Breeze
ticks the same way `backtest_engine.py` reacts to historical bars.

**Phase 3 (current): `PositionStore`.** Seeds position state one of two
ways, both producing the identical JSON schema:
- **Manually** — hand-author a JSON file (schema documented in
  `nifty_live/position_store.py`'s `PositionStore` docstring) for whatever
  you're currently holding.
- **From Zerodha** — `PositionStore.import_from_zerodha()` reads currently
  open positions via `kiteconnect` (read-only; Zerodha's free tier gives
  no live price feed, so it's never used for anything but "what do I
  hold" — Breeze remains the sole price source everywhere, live and
  historical alike). Every field name and sign convention this relies on
  was cross-checked against Zerodha's own docs/SDK/forum — see the module
  docstring for specifics — but **run
  `python3 scripts/verify_zerodha_import.py` against your real account
  before trusting it with real capital**; it prints the raw
  `positions()`/`instruments()` responses alongside what gets derived
  from them, the same pattern as `verify_find_atm.py`/`debug_breeze.py`
  for the Breeze side.

Install the optional `kiteconnect` dependency with
`pip install -e ".[live]" --break-system-packages` (kept separate from the
core install so backtesting-only usage never needs it).

Remaining phases: `chain_downloader` (periodic full-chain snapshots),
`ReplayLiveFeed`/`live_engine` (runs the same `AdjustmentStrategy.evaluate()`
against replayed or live bars), `BreezeWebSocketFeed`, then webapp
integration.
