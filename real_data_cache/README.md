# real_data_cache/

Committed, real (non-synthetic) market data pulled from Breeze, so that
anyone WITHOUT live credentials — including a fresh sandbox session — can
still run the strategy/backtest layer against real historical data via
`nifty_backtester.data_layer_cached.NiftyOptionsDataCached`, instead of only
synthetic data.

This directory is checked into git deliberately. It is **not** the same as
the local `data_cache/` directory (gitignored, scratch space that
`NiftyOptionsDataBreeze` fills up automatically during any real run) — that
one accumulates everything you happen to touch; this one holds only what
you've deliberately curated and committed.

## Workflow

1. Run any of the `scripts/*.py` tools locally with real
   `BREEZE_API_KEY` / `BREEZE_API_SECRET` / `BREEZE_SESSION_TOKEN` set, e.g.:
   ```
   python3 scripts/download_and_run_strategy.py --from-date 2026-09-04 --to-date 2026-09-08 \
       --weekly-expiry 2026-09-08 --sold-leg-strategy delta_threshold
   ```
   This fills `./data_cache/*.parquet` (via `DataCache`) as a side effect of
   fetching whatever index/option data that run needed.
2. Copy the specific parquet files for the date range/strikes you want to
   share into `real_data_cache/`:
   ```
   cp data_cache/NIFTY_INDEX_1minute.parquet real_data_cache/
   cp data_cache/NIFTY_2026-09-08_24500_call_1minute.parquet real_data_cache/
   cp data_cache/NIFTY_2026-09-08_24500_put_1minute.parquet real_data_cache/
   # ...every strike/right your chosen strategy config actually touches
   ```
   Copy every file a given run will need — `NiftyOptionsDataCached` never
   fetches, it only reads what's here, and will raise a clear
   `CachedDataUnavailable` naming the missing key if something's absent.
3. `git add real_data_cache/*.parquet && git commit && git push`.
4. In a session without live credentials, point any script at this data
   with `--data-source cached` (or leave the default `auto`, which uses
   CACHED automatically whenever no live credentials are set and this
   directory has anything committed):
   ```
   python3 scripts/run_backtest_from_range.py --from-date 2026-09-04 --to-date 2026-09-08 \
       --data-source cached
   ```

## Cache-key naming (must match exactly — this is what `DataCache` itself writes)

| Data | Filename |
|---|---|
| NIFTY index, interval `I` | `NIFTY_INDEX_{I}.parquet` |
| Option leg | `NIFTY_{expiry:%Y-%m-%d}_{strike}_{call\|put}_{I}.parquet` |

`I` is whatever native Breeze interval you fetched at — `1minute` unless
you deliberately asked for something else. Since scripts always fetch at
native `1minute` and resample locally (see `market_data._resample_ohlc`),
that's normally the only interval you need to commit; any 15-min/1hr
timeframe a strategy needs gets built from it automatically.

## Multiple date ranges / market conditions (`scenarios.json`)

A single committed date range only tells you how a strategy did in one
market regime. `scenarios.json` in this directory lets you name several
distinct ranges — a trending week, a choppy week, a high-IV event week,
etc. — each backed by its own committed parquet data (or live/synthetic,
independently per scenario), and run the full strategy comparison across
all of them at once:

```
python3 scripts/run_scenarios.py
```

See `scenarios.json` in this directory for the schema (a `name`,
`market_condition` label, `from_date`/`to_date`, and optionally an explicit
`weekly_expiry`/`weekly_expiry_prior_trading_day` if you don't want it
resolved from `expiry_calendar.csv` at run time). Add an entry there for
each date range you commit data for, alongside the parquet files.

## Size discipline

Parquet is reasonably compact, but a full week of native 1-minute option
data across many strikes adds up. Only commit the strikes a given
scenario/config actually needs (the ATM straddle plus whatever hedge/OTM
strikes that run's config touches) rather than an entire option chain.
