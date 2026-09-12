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
| `run_campaign_backtest.py` | Runs the funded-strangle campaign strategy for one or more consecutive months, optionally sweeping the weekly-roll/monthly-close time of day to compare metrics (see STRATEGY.md §7) |

## Live monitoring (`nifty_live/`, in progress)

A separate package, built on the exact same `Leg`/`MultiLegPosition`/
`AdjustmentStrategy` classes as the backtester (zero changes to
`strategy.py` needed) — the backtester stays batch/offline; `nifty_live`
tracks currently-held positions and (eventually) reacts to live Breeze
ticks the same way `backtest_engine.py` reacts to historical bars.

**Phase 3: `PositionStore`.** Seeds position state one of two
ways, both producing the identical JSON schema:
- **Manually** — hand-author a JSON file (schema documented in
  `nifty_live/position_store.py`'s `PositionStore` docstring) for whatever
  you're currently holding.
- **From Zerodha** — `PositionStore.import_from_zerodha()` reads currently
  open positions via `kiteconnect` (read-only; Zerodha's free tier gives
  no live price feed, so it's never used for anything but "what do I
  hold" — Breeze remains the sole price source everywhere, live and
  historical alike). Kite's `access_token` is valid for one trading day
  only — run `python3 scripts/generate_kite_token.py` each morning; it
  walks the login/`request_token` exchange and writes the fresh
  `access_token` into `.env` for you (needs `KITE_API_KEY`/
  `KITE_API_SECRET` already in `.env` — see `.env.example`). Every field
  name and sign convention this relies on was cross-checked against
  Zerodha's own docs/SDK/forum — see the module docstring for specifics —
  but **run `python3 scripts/verify_zerodha_import.py` against your real
  account before trusting it with real capital**; it prints the raw
  `positions()`/`instruments()` responses alongside what gets derived
  from them, the same pattern as `verify_find_atm.py`/`debug_breeze.py`
  for the Breeze side.

Install the optional `kiteconnect` dependency with
`pip install -e ".[live]" --break-system-packages` (kept separate from the
core install so backtesting-only usage never needs it).

**Heads up on `kiteconnect`'s own dependency footprint:** it pulls in
`autobahn`, `Twisted`, and `zope.interface` — none of which this codebase
actually uses. Those exist for `KiteTicker`, kiteconnect's WEBSOCKET
client (which needs a paid Zerodha tier you don't have); `PositionStore`
only ever calls the plain-REST `positions()`/`instruments()` methods. But
`kiteconnect/__init__.py` unconditionally imports `KiteTicker`, so a bare
`import kiteconnect` — confirmed directly, not assumed — pulls in ~660
modules and installs a **global Twisted reactor as a side effect**,
before any `KiteConnect` instance is even created. `zope.interface` is
Twisted's own internal dependency (see its PyPI metadata:
"Required-by: Twisted"), not something kiteconnect chose for its own API
and not something this codebase should adopt to match it — our existing
duck-typed contracts (`MarketDataProvider`, `AdjustmentStrategy`, the
`FakeKite` test double) already do what `zope.interface` would offer,
appropriately for our small set of concrete classes rather than a
Twisted/Zope-style plugin registry. Worth remembering before Phase 7
(webapp integration): Twisted allows only one reactor per process, so
anything else in that stack wanting to install its own reactor after
`kiteconnect` has already been imported will hit
`ReactorAlreadyInstalledError`. See `nifty_live/position_store.py`'s
module docstring for the full trace and the mitigation options if this
ever becomes a real constraint.

**Phase 5 (current): `ReplayLiveFeed` / `live_engine`.** Runs the exact
same `AdjustmentStrategy.evaluate()` the batch backtester uses, against
either replayed history (zero live session, for testing/validation) or a
real live feed — with **zero changes to `strategy.py`**, since it was
already built data-source-agnostic:
- `nifty_live/replay_feed.py`'s `ReplayLiveFeed` wraps
  `ScenarioBoundedDataLayer`'s `as_of_cursor` (Phase 2) to walk historical
  (sample or committed-real) bars bar-by-bar as though arriving live —
  verified to have zero look-ahead. `live_polling_clock()` is the real-live
  counterpart (yields `datetime.now()` on an interval); the exact same
  engine loop runs against either.
- `nifty_live/live_engine.py`'s `run_live_monitor()` evaluates every
  configured strategy at each time step and sends whatever it recommends
  to a `Notifier` (`ConsoleNotifier` for now) — **it never executes a
  trade or mutates position state**. This is deliberate decision support,
  matching the intended workflow: watch the console, validate the
  strategy is behaving as expected, act manually (or via the webapp
  later), then update position state yourself before the next run.
- Try it: `python3 scripts/run_live_monitor_demo.py` — runs the whole
  pipeline against sample data with a seeded demo position, zero live
  session needed.

Remaining phases: `BreezeWebSocketFeed` (a lower-latency, push-based
alternative to polling Breeze's REST endpoint — `run_live_monitor()`
already works against a real live feed today via polling, so this is an
efficiency upgrade, not a new code path), then webapp integration. A
periodic full-option-chain downloader was considered and deliberately
**not** built — every strike this codebase trades is already resolved
analytically (ATM via put-call parity, delta-target via a directional
walk, just-OTM via arithmetic), so a bulk chain scan filtered by
volume/OI would have no consumer; a single-strike liquidity check before
executing a live trade is a two-line addition if/when it's actually
needed, not infrastructure worth building speculatively.
