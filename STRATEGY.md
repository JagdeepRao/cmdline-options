# Strategy & Position Documentation

This documents what the code actually does today: the position shapes, the
management strategies that adjust them, the indicators those strategies
read, the timeframes each indicator runs on, and the expiry-close rules
that apply across all of it. Where something is an assumption rather than
a confirmed decision, it's called out explicitly under **Open questions**
at the end — everything above that line reflects code that exists and has
passing tests, not aspiration.

---

## 1. Position shapes

There are up to four distinct positions in play at once, each independently
open/flat and independently managed. A given backtest run can include any
subset of them.

| Position name         | Legs                                              | Direction        | Expiry            |
|------------------------|----------------------------------------------------|------------------|--------------------|
| `sold_straddle`         | `sold_call`, `sold_put` — both ATM                 | SHORT            | weekly             |
| `hedge_straddle`        | `hedge_call`, `hedge_put` — both ATM               | LONG             | monthly (see §1.2) |
| `directional_overlay`   | `overlay_long_{call/put}`, `overlay_short_{call/put}` | long 75-delta leg LONG, ATM leg SHORT | weekly |
| `otm_position`          | `otm_call`, `otm_put` — just-OTM, entered independently | LONG        | weekly             |

### 1.1 ATM definition

ATM strike is **not** derived from a futures price (weekly options have no
matching weekly futures contract). It's defined as the strike minimizing
`|call_price − put_price|` (put-call parity implies this is the
market-implied forward). Implemented in
`data_layer_breeze.NiftyOptionsDataBreeze.find_atm_strike()` (and mirrored
in `data_layer_sample.NiftyOptionsDataSample.find_atm_strike()` for
credential-free testing) by scanning a configurable number of strikes
either side of an approximate spot and picking the minimum-difference one.

### 1.2 Monthly hedge expiry selection

`expiry_utils.select_monthly_hedge_expiry_from_calendar(calendar, as_of,
min_days=15)`: use the current month's monthly expiry (looked up from
`expiry_calendar.csv`, not computed from a "last Thursday of the month"
formula) as the hedge unless it's less than 15 days from `as_of`, in which
case roll to next month's monthly expiry, also looked up from the
calendar. `select_monthly_hedge_expiry` (pure date arithmetic on two
already-resolved candidate dates) still exists for callers that source the
candidates themselves, but new code should prefer the calendar-backed
version — see §1.2.1.

#### 1.2.1 Why a calendar file, not a formula

NIFTY's weekly expiry day has itself changed exchange-side before (moved
off Thursday), so any "next Thursday" / "last Thursday of the month" rule
is a bet on a convention that has already proven not to hold — the same
applies to monthly expiries. `expiry_calendar.csv` (loaded via
`expiry_utils.load_expiry_calendar()`) is the single source of truth
instead: a plain table of `expiry_type` (weekly/monthly), `expiry_date`,
and `prior_trading_day` (the actual last NSE trading day before that
expiry, with holidays already accounted for — not necessarily
`expiry_date - 1 day`).

**The shipped `expiry_calendar.csv` is an illustrative template, not the
real NSE calendar** — `load_expiry_calendar()` prints a warning every time
it loads the default file for exactly this reason. Replace it with the
actual expiry/holiday calendar (from NSE's published circulars or your
broker's contract master) before running this against real trading
decisions. `load_expiry_calendar()` also validates the file on load
(required columns present, `prior_trading_day` strictly before
`expiry_date` for every row, `expiry_type` values recognized) and raises
immediately on a malformed entry rather than silently producing a wrong
expiry-eve close later.

Lookup functions built on top of the loaded calendar:
- `get_next_expiry(calendar, as_of, expiry_type)` — first expiry of that
  type on/after `as_of`; raises if the calendar doesn't cover that far
  forward (extend the file rather than guessing).
- `get_prior_trading_day_for_expiry(calendar, expiry_date, expiry_type)` —
  exact-match lookup when the caller already knows a specific expiry date
  (e.g. passed explicitly on a command line) and just needs its prior
  trading day.

`FullBacktestConfig` takes the resolved `weekly_expiry_prior_trading_day`
(and, when a hedge is included, `monthly_expiry_prior_trading_day`)
directly as required fields — the engine itself has zero calendar-file
knowledge and never derives a prior trading day by subtraction; callers
(the CLI scripts, or your own code) are expected to resolve these via the
calendar before constructing the config.

### 1.3 Directional overlay structure

`directional_overlay` is a long-75-delta-call/put **plus** a short-ATM-
call/put on the same side — the short leg is meant to hedge/finance the
long leg. The two legs are NOT opened together: the long leg opens
immediately once a direction is established; the short leg opens only once
its own 15-min signal says it's safe to sell (§2.3). See **Open questions**
for the naked-long-leg-while-waiting implication.

---

## 2. Management strategies

Three independent "shapes" of management exist for the **sold** legs
(`sold_straddle`, and — reusing the identical mechanics — the short leg of
`directional_overlay`). They're not meant to run all three at once on the
same leg for a single backtest; you pick one per run and compare across
runs via `metrics.full_report()`.

### 2.1 Delta-threshold (`DeltaThresholdStrategy`)

Whole-position recenter (close every open leg, reopen fresh at current ATM)
the instant **any** leg's `|delta|` crosses a threshold:

- `sold_straddle` → 0.65 (default `sold_threshold`)
- `hedge_straddle` → 0.85 (default `hedge_threshold`)
- any other position → pass `extra_thresholds={"position_name": threshold}`
  (this is how `directional_overlay`'s short leg can reuse the identical
  delta-threshold mechanic if that's the management shape chosen for a run)

Threshold checks are keyed by leg tag against a single shared
`delta_indicators` dict — one instance of this strategy can watch multiple
positions simultaneously.

### 2.2 Fixed-move (`FixedMoveStrategy`)

Whole-position recenter whenever the underlying has moved `move_points`
(100 default) OR `move_pct` (0.5% default) from the level recorded when the
position was last opened/recentered — whichever fires first.
`position_name` defaults to `"sold_straddle"` but can target any position
(e.g. the overlay's short leg) the same way.

### 2.3 Indicator-based (`SoldLegSignalStrategy`, with `RSISignalStrategy` /
`SupertrendEMASoldLegStrategy` as ready-made variants)

Per-LEG (not whole-position) management, running on **15-minute** data:

- **Close**: the instant a leg's signal turns against the short (rising
  premium) — `RSICrossAdapter.crossed_bullish()` fires on an RSI-vs-its-own-
  EMA bullish crossover; `SupertrendEMAAdapter.crossed_bullish()` fires on
  a fresh Supertrend+EMA buy-signal edge.
- **Re-open**: only once that leg's *regime* (sustained state, not just the
  edge) turns bearish again — i.e., re-enters at whatever the **current**
  ATM strike is (which may have moved since the leg was closed), not the
  strike it was closed at.
- **Delta-harvest**: independent of the signal, any leg whose `|delta|`
  decays to ≤ 0.2 is closed and immediately reopened at the current ATM
  strike — this is a secondary rule bundled into the same strategy, not
  gated by the indicator (a fully-decayed leg is harvested regardless of
  what the signal says).

`RSICrossAdapter` and `SupertrendEMAAdapter` share one interface
(`crossed_bullish()`, `regime()`), so `RSISignalStrategy` and
`SupertrendEMASoldLegStrategy` are literally the same class
(`SoldLegSignalStrategy`) with a different adapter wired in — this is what
makes "RSI vs Supertrend+EMA, which is better for the sold side" a
one-line swap for a comparison run, not two parallel code paths.

`position_name` defaults to `"sold_straddle"`; the exact same class,
constructed with `position_name="directional_overlay"` and leg tags
`overlay_short_call`/`overlay_short_put`, is what manages the overlay's
short leg (§2.4).

### 2.4 Directional overlay (`DirectionalOverlayStrategy`)

Point 2's "weekly spread": wraps a `core_strategy` (any of §2.1-2.3, run
against `sold_straddle` as normal) and separately manages
`directional_overlay`:

1. A **1-hour** regime signal on the underlying (again an
   `RSICrossAdapter`/`SupertrendEMAAdapter`, this time built on 1hr data)
   decides direction: bullish regime → call side, bearish → put side,
   neutral → no position.
2. On a regime flip, every open leg of the current spread (long and/or
   short, whichever exist) is closed, and a fresh long 75-delta leg on the
   new side opens **in the same evaluate() call** — "close the spread and
   open the opposite spread" happens as one atomic step, not two separate
   bars.
3. The short ATM leg is *not* opened alongside the long leg. Opening it is
   delegated entirely to a `SoldLegSignalStrategy` instance
   (`short_leg_strategy`, constructed with `position_name=
   "directional_overlay"`) — `DirectionalOverlayStrategy` just registers the
   new side's short-leg tag into that strategy's own re-entry-awaiting set
   the moment the long leg opens, reusing the exact same "wait for a
   favorable regime" gate that already governs re-entries after an
   adverse-signal close. Once open, the short leg is managed identically to
   a `sold_straddle` leg — same close/harvest/re-enter rules, same 15-min
   signal.

### 2.5 Opportunistic OTM (`RenkoOpportunisticOTMStrategy`)

Point 3: manages `otm_position` using Renko+SuperTrend on the underlying,
computed on **1-minute** data — this is deliberately never used on a sold
leg (Renko is confirmed weak on the sell side; every use of it in this
codebase is on a LONG leg).

Composed from two independent `RenkoLegTracker` instances — one per side —
each with its own entry/exit rule and zero coupling to the other side:

- `call` tracker: enters `otm_call` when the underlying's Renko trend turns
  up; exits it the moment the trend leaves "up".
- `put` tracker: enters `otm_put` when the trend turns down; exits the
  moment the trend leaves "down".

Because both trackers currently read the *same* shared indicator, only one
side can be true at a time in practice — you'll observe single-leg-at-a-
time behavior. That's a property of what's fed in, not an invariant the
code enforces: point each tracker at an independent signal later (e.g. a
Renko built on that leg's own option price) and overlapping call+put legs
falls out with zero changes to `RenkoLegTracker`/`RenkoOpportunisticOTMStrategy`.

**Deliberately not implemented (by explicit instruction, not an
oversight)**: pyramiding further same-side entries as the underlying keeps
moving and the "just OTM" strike rolls further away (e.g. adding a second,
further-OTM put as the market keeps dropping). Each tracker holds at most
one leg per side today.

**Sizing**: `otm_multiplier` (the 1x-5x sweep) is applied by whatever code
executes the `OPEN_LEG` action — the strategy classes only ever decide
direction/timing, never size.

---

## 3. Indicator reference

| Indicator | Class | Typical timeframe | Used for |
|---|---|---|---|
| RSI vs EMA(RSI) crossover | `RSIIndicator` (wrapped by `RSICrossAdapter`) | 15-min (sold legs) or 1hr (overlay direction) | sold-leg close/re-enter signal; overlay direction signal |
| Supertrend + EMA | `SupertrendEMAIndicator` (wrapped by `SupertrendEMAAdapter`) | 15-min (sold legs) or 1hr (overlay direction) | same two uses as RSI — direct A/B comparison |
| Renko (ATR box) + SuperTrend | `RenkoSuperTrendIndicator` | 1-min, on the underlying | OTM leg entry/exit direction |
| Black-Scholes delta | `DeltaIndicator` (wraps `pricing.solve_iv_and_greeks`) | tick-by-tick, whatever the engine's bar frequency is | delta-threshold recenter, delta-harvest |

`RSICrossAdapter` and `SupertrendEMAAdapter` both implement the same
`SoldLegSignalAdapter` interface (`crossed_bullish(as_of)`, `regime(as_of)`)
— this is the seam that lets one strategy class serve both indicators and
both timeframes (15-min sold-leg management, 1hr overlay direction) without
duplicating logic per indicator or per timeframe.

---

## 4. Expiry-eve close-out rules (`expiry_utils.py`)

Because these are calendar spreads, the sold leg is never held to actual
expiry — margin requirements step up on expiry day itself, so everything
that constitutes a "spread" (a short leg paired with a long leg: the sold
straddle vs. its monthly hedge, and the overlay's short leg vs. its long
leg) is force-closed the trading day **before** expiry, at/after
`close_time` (15:30 default):

- `is_expiry_eve_close_bar(as_of, prior_trading_day, close_time)` — true
  from that bar onward on the actual last trading day before expiry, as
  looked up from `expiry_calendar.csv` (via `get_next_expiry` or
  `get_prior_trading_day_for_expiry`). Takes `prior_trading_day` directly
  rather than deriving it from `expiry_date` — see §1.2.1 for why.
- `is_on_or_after_expiry(as_of, expiry_date)` — hard backstop so nothing
  survives past expiry regardless of whether the eve-close bar landed
  exactly on a bar timestamp.

The just-OTM opportunistic legs (`otm_position`) are **not** naturally
paired with a hedge, so whether they get force-closed on the same eve-close
schedule or are allowed to run past it (closing only on their own Renko
exit signal) is a config choice for the engine, not something
`expiry_utils` decides — see **Open questions**.

**RESOLVED (previously a stated limitation)**: this used to derive
"the day before expiry" as calendar-date-minus-one, which breaks on any
holiday-adjacent expiry and doesn't survive the exchange changing which
weekday expiry falls on. Both `is_expiry_eve_close_bar` and
`FullBacktestConfig` now take the actual prior trading day directly,
sourced from `expiry_calendar.csv` — see §1.2.1.

---

## 5. Confirmed design decisions and remaining open questions

Decision #1 below was an open assumption as of the previous revision of
this document and has since been explicitly confirmed. The rest remain
open, flagged in code comments at the point they matter, and collected
here for visibility.

1. **CONFIRMED: overlay long leg runs unhedged while waiting for the
   short-leg signal.** The long 75-delta leg opens immediately on a 1hr
   regime flip; the short ATM leg only joins once its own 15-min signal
   permits selling. This is intentional, not a gap — no code change
   needed. One consequence worth keeping in mind when interpreting
   results: if the 1hr regime flips again before the 15-min signal ever
   turns favorable, that cycle's spread closes having been long-only for
   its entire life — the short leg may never get entered at all in a fast
   regime-flipping period.
2. **Renko warm-up default.** `RenkoSuperTrendIndicator` defaults
   `trend_dir` to `1` (bullish) during ATR warm-up rather than an explicit
   "no signal yet" state (inherited from the original Pine-replica logic).
   This means `RenkoOpportunisticOTMStrategy` could open a call leg before
   Renko has genuinely established a trend, unless the engine feeds it a
   lookback buffer before the live backtest window starts (the same
   pattern already used for the 15-min RSI/Supertrend indicators
   elsewhere).
3. **OTM legs and expiry-eve closing.** Not yet decided whether
   `otm_position` legs get force-closed on the same expiry-eve schedule as
   the straddles/spreads, or are allowed to run past it and close only on
   a Renko exit signal.
4. **Pyramiding on the OTM side** (multiple same-side entries as the
   market keeps trending and the just-OTM strike rolls) is explicitly
   deferred, not implemented.
5. **Overlapping OTM call+put legs.** Currently structurally possible but
   never observed in practice because both `RenkoLegTracker` instances
   share one underlying-based signal. Confirm whether overlap should ever
   actually happen (e.g. via per-leg option-price-based Renko) or whether
   single-side-at-a-time is the intended steady state.

---

## 6. Live feed architecture: Breeze websocket modes and their limitations (pre-implementation research)

**Status: research/documentation only — `BreezeWebSocketFeed` has NOT been
built yet.** This section exists so that whoever builds it (later phase)
starts from confirmed facts rather than re-discovering them, and doesn't
accidentally build a charting/indicator pipeline on a data source that
can't reliably support it. Everything below is either confirmed directly
by reading the installed `breeze_connect` SDK source, or confirmed via
real user reports on ICICI's own SDK issue tracker — nothing here is
guessed. Where something remains genuinely unconfirmed, it's stated as
such rather than assumed one way or the other.

### 6.1 There are two, structurally different websocket modes

`breeze.subscribe_feeds(...)` silently routes to one of two completely
different code paths depending on whether an `interval` argument is
passed — confirmed directly from `breeze_connect.py`'s `subscribe_feeds`
body:

- **No `interval` ("Live Feed" mode)** — routed through
  `sio_rate_refresh_handler`, parsed by `parse_data()`.
- **`interval` set** to one of the CONFIRMED valid values
  `"1second"`, `"1minute"`, `"5minute"`, `"30minute"`
  (`config.INTERVAL_TYPES_STREAM_OHLC`) — routed through a *separate*
  `sio_ohlcv_stream_handler` / `watch_stream_data()`, parsed by a
  *different* function, `parse_ohlc_data()`.

These are not two views of the same data — they are different socket.io
rooms with independently-implemented parsers, confirmed by real user
reports of exactly this split
([issue #133](https://github.com/Idirect-Tech/Breeze-Python-SDK/issues/133),
subscribing with `interval="1second"` explicitly to reach candle mode).

### 6.2 "Live Feed" mode: confirmed ambiguous, and confirmed NOT per-tick

Reading `parse_data()` directly shows AT LEAST two incompatible
field-naming schemes depending on the subscribed token's exchange/type
encoding:

- `exchange == '6'` branch: fields include `last` (LTP), `ttq` (total
  traded quantity), `CurrOpenInterest`, `ltt` (last traded time).
- `data_type == '1'` branch: a *different* set of field names (`last`,
  `OI`/`ttq` in a 23-field F&O variant; a 21-field equity/index variant
  with **no OI field at all**).

Which branch actually fires for a given subscription cannot be determined
from source alone — this would need a live subscription to observe.

More importantly, a real user (
[issue #167](https://github.com/Idirect-Tech/Breeze-Python-SDK/issues/167))
confirms directly that the OHLC-labeled fields in this mode (`open`,
`high`, `low`, `close`) are **not per-tick trade prints** — they're the
running session-to-date OHLC as of that moment, updated on every tick.
By the same logic, `ttq` ("total traded quantity") should be read as
**cumulative volume for the day**, not volume attributable to that one
tick or to any specific time window. There is no field in this mode that
directly answers "how much traded between two points in time" — that
would have to be computed as a delta between two observed cumulative
values, which is fragile across any dropped tick, reconnect, or gap (a
missed update makes the next delta silently absorb more volume than it
should, with no way to detect that from the tick stream alone).

### 6.3 "OHLCV/Candle" mode: a clean, confirmed schema — but one open question

`parse_ohlc_data()` is a single, unambiguous, confirmed schema for NFO
(comma-separated positional fields): `exchange_code`, `stock_code`,
`expiry_date`, `strike_price`, `right_type`, `low`, `high`, `open`,
`close`, `volume`, `oi`, `datetime` (plus a shorter variant without
`strike_price`/`right_type` for instruments where those don't apply).
This is a real, single source of truth per message — nothing like the
ambiguity in §6.2.

**Genuinely unconfirmed, not found in the SDK source or in ICICI's public
docs/community threads searched so far:** whether `volume` in this mode is
the volume traded *within that one candle interval*, or is *still*
cumulative-since-market-open (as it explicitly is in Live Feed mode). This
is the exact distinction that matters for the stated use case — "volume
in time interval for OHLC mode/charting" — and it can only be resolved by
subscribing and observing real values against a known interval, not by
further reading of source or docs.

### 6.4 Design implication for this codebase

Regardless of how §6.3's open question resolves, **websocket delivery of
any kind is not reliable enough to be the system of record for OHLC bars
or interval volume** — a dropped connection or missed message silently
loses that interval, with no built-in gap-detection or backfill. This
codebase already has a system of record for exactly that: the REST
historical endpoint (`get_historical_data_v2`, confirmed schema — see
`data_layer_breeze.py`'s module docstring), fetched incrementally through
`DataCache` (Phase 1-era; only ever fetches the missing tail of a range,
not the whole thing on every call — already cheap).

The resulting split, planned for whenever `BreezeWebSocketFeed` is
actually built:

- **Websocket (either mode) → point-price awareness only.** Feed a
  "latest known price" cache that backs `get_spot()`/`get_option_price()`
  -equivalent lookups for `live_engine.run_live_monitor()`'s per-step
  strategy evaluation — exactly the use case `live_polling_clock()`
  already serves via REST polling today (§ README "Live monitoring").
  Nothing about this use case cares whether a value is cumulative or
  per-interval; it only ever wants "what's the price right now."
- **REST historical (existing `DataCache`-backed path) → everything that
  needs real bars.** `RSIIndicator`, `SupertrendEMAIndicator`,
  `RenkoSuperTrendIndicator`, and any future charting/webapp display all
  need genuine OHLC with trustworthy volume — these should keep pulling
  from `BreezeMarketDataProvider.spot_series()`/`option_price_series()`
  (or their live equivalents), refreshed periodically, not reconstructed
  from websocket messages.

In short: the websocket upgrade (replacing `live_polling_clock()`'s REST
polling to avoid rate-limiting) is worth doing for the point-price path,
but it does **not** replace the REST/`DataCache` path for anything
requiring aggregation — that split should be explicit in
`BreezeWebSocketFeed`'s design from the start, not discovered after the
fact when a chart shows corrupted volume.

---

## 7. Campaign strategy: funded strangle theta engine (`campaign_strategy.py`)

A distinct strategy shape, NOT built on `AdjustmentStrategy`/`FullBacktestConfig`
(see that module's docstring for why) — a monthly "campaign" that funds a
long strangle by selling weekly premium against it, aiming to profit from
either theta decay (if the market stays range-bound) or a wild move (the
long strangle pays off), while the weekly funding legs cover the cost
either way.

**Shape**: on the first trading day of the month, buy 3× ATM-ish call +
3× ATM-ish put at MONTHLY expiry (strike chosen so each leg's premium is
close to a target, e.g. ~200 — ITM matches are acceptable, the search
isn't OTM-only). Fund each side independently by selling two WEEKLY
options on that side (weekly, not monthly) — searched across every strike
pair for the one whose combined credit clears 3× that side's long cost
AND maximizes combined theta capture. Every week, the funding legs are
closed and blindly re-sold at the SAME locked strikes for the next weekly
expiry (deliberately no delta/breach defense — a locked strike going deep
ITM as the market moves is treated as mean-reversion protection, not a
bug). On the trading day before monthly expiry, everything closes and the
campaign ends; a fresh one starts the following month.

See `campaign_strategy.py`'s module docstring for the full algorithm,
`resolve_campaign_expiry_schedule`'s docstring for the exact rule on
starting a campaign whose nearest weekly cycle is already at its own
expiry-eve, and `scripts/run_campaign_backtest.py` for running one or more
months (optionally sweeping the single `roll_and_close_time` — the weekly
roll and the monthly close deliberately share ONE time-of-day knob, not
two — across candidate times to compare `metrics.full_report()` output).

### 7.1 Shared scaffolding (`campaign_common.py`)

Monthly-expiry-schedule resolution (incl. the min-runway guard and the
"already at its own eve" exception rule), month-chaining
(`first_trading_day_of_month`/`next_month`), the weekly-eve→next-expiry
roll map, and the leg open/close/equity-mark primitives are factored into
`campaign_common.py`, shared by every campaign strategy — the point being
that different campaign SHAPES can be run over the same window and
genuinely compared, not just individually backtested. `campaign_strategy.py`
imports these under its original names (`CampaignExpirySchedule`,
`resolve_campaign_expiry_schedule`, etc.) so nothing about its own public
API changed when this was factored out.

### 7.2 Campaign 2: recentered straddle (`campaign_straddle_strategy.py`)

A deliberately simpler counterpart to Campaign 1, built to compare against
it directly (see `scripts/compare_campaigns.py`): buy 1× ATM call + 1× ATM
put at MONTHLY expiry (the hedge, untouched mid-month, same discipline as
Campaign 1's long legs); sell 1× ATM call + 1× ATM put at the nearest
WEEKLY expiry. Every trading day, at one configurable `adjustment_time`,
if the current put-call-parity ATM strike is more than
`recenter_threshold_points` (100 default) away from the strike the short
straddle currently holds, close and reopen it at the current ATM. On each
weekly expiry-eve the short straddle rolls UNCONDITIONALLY to the next
weekly expiry at a freshly-resolved ATM strike, regardless of the drift
threshold (the contract is expiring either way) — an eve day always rolls,
a non-eve day only recenters on a drift breach; these never overlap for
the same day. Everything closes on the monthly expiry-eve, same as
Campaign 1.

`scripts/compare_campaigns.py` runs both campaigns over the same
window(s)/data and prints a side-by-side `metrics.full_report()` table,
optionally sweeping both campaigns' single adjustment-time knob across the
same candidate times — this is the actual comparison being asked: is the
funded strangle's extra complexity worth it over this simpler,
more-frequently-adjusted straddle?

---

## 8. File map

| File | Contents |
|---|---|
| `nifty_backtester/strategy.py` | Position/leg data model, all indicators, all `AdjustmentStrategy` implementations |
| `nifty_backtester/pricing.py` | Black-Scholes-Merton pricing, IV solve, Greeks (`solve_iv_and_greeks`) |
| `nifty_backtester/data_layer_breeze.py` | Live Breeze-backed data layer: historical option/index data, ATM resolution |
| `nifty_backtester/data_layer_base.py` | `BaseCachedOptionsDataLayer` — shared find_atm_strike/nearest_otm_strikes/get_straddle_and_hedge_data logic for `data_layer_sample.py` and `data_layer_cached.py` (previously duplicated between them). Zero `breeze_connect` dependency, same as the two subclasses below. |
| `nifty_backtester/data_layer_sample.py` | Same interface as the Breeze data layer, backed by synthetic data — for credential-free tests |
| `nifty_backtester/data_layer_cached.py` | Same interface again, but reads ONLY committed parquet files under `real_data_cache/` — no network, no live session; real data without credentials |
| `nifty_backtester/data_layer_scenario.py` | `ScenarioBoundedDataLayer` — wraps a cached/sample layer with explicit date-range/expiry-allowlist enforcement (raises on violation) plus an optional monotonic `as_of_cursor` replay clock (truncates, never raises) — foundation for `nifty_live`'s sample-backed replay feed |
| `nifty_backtester/data_sources.py` | `resolve_data_layer()` — shared LIVE > CACHED > SYNTHETIC selection used by every script |
| `nifty_backtester/scenarios.py` | Named date-range "scenarios" (market conditions) loaded from `real_data_cache/scenarios.json` |
| `nifty_backtester/sample_data.py` | Deterministic synthetic OHLCV generators used by `data_layer_sample.py` and tests |
| `nifty_backtester/data_cache.py` | File-based incremental cache with retry/backoff, sitting in front of every historical-data fetch |
| `nifty_backtester/expiry_utils.py` | Monthly hedge expiry selection, expiry-eve close-bar timing |
| `nifty_backtester/market_data.py` | `MarketDataProvider` abstraction (`SyntheticMarketDataProvider` / `BreezeMarketDataProvider`), proper OHLC resampling (`_resample_ohlc`) |
| `nifty_backtester/metrics.py` | Equity-curve and trade-level performance metrics (Sharpe, Sortino, Calmar, drawdown, win rate, etc.) |
| `nifty_backtester/backtest_engine.py` | Time-stepping loop that opens positions, evaluates strategies, and executes their actions bar-by-bar |
| `nifty_backtester/campaign_strategy.py` | Funded-strangle "campaign" theta engine — a separate, monthly-cycle-rolling engine (see §7 above), not built on `AdjustmentStrategy`/`FullBacktestConfig` |
| `nifty_backtester/campaign_common.py` | Shared scaffolding for every campaign strategy — expiry-schedule resolution, month-chaining, leg open/close/equity-mark primitives (see §7.1) |
| `nifty_backtester/campaign_straddle_strategy.py` | Campaign 2 — sold weekly straddle / bought monthly straddle with a daily ATM-drift recenter (see §7.2) |
| `nifty_backtester/expiry_calendar.csv` | The expiry/prior-trading-day source of truth — **shipped as an illustrative template, replace before real use**. Internally consistent with `nse_holidays.csv` (no entry lands on a weekend or listed holiday — enforced by `test_expiry_holidays.py`, not by `load_expiry_calendar()` itself). |
| `nifty_backtester/nse_holidays.csv` | NSE trading-holiday list used by `expiry_utils.validate_calendar_against_holidays()` and `scripts/generate_expiry_calendar_candidates.py` — **best-effort (web-sourced, cross-checked across several finance sites), not an official NSE feed; re-verify before trusting real trading decisions.** |
| `scripts/generate_expiry_calendar_candidates.py` | Generates CANDIDATE `expiry_calendar.csv` rows from `expiry_utils.WEEKLY_EXPIRY_WEEKDAY_REGIMES` (the documented Thursday→Tuesday expiry-day history) + `nse_holidays.csv`, holiday-shifting as needed. Output is for human review before pasting into `expiry_calendar.csv` — never consumed automatically; the engine still never computes an expiry date from a weekday rule. |
| `scripts/download_option_data.py` | Downloads data (LIVE/CACHED/SYNTHETIC, see README) for an option leg/index and annotates it with indicator BUY/SELL signals, for manually validating an indicator against real price action |
| `scripts/download_and_run_strategy.py` | Runs a full strategy against LIVE/CACHED/SYNTHETIC data and prints every adjustment decision with its trigger, for manually validating the strategy layer the same way |
| `scripts/run_backtest_from_range.py` | Turnkey entry point: give it a date range, it resolves expiries from the calendar and returns a metrics table across all strategy/position combinations |
| `scripts/run_scenarios.py` | Same sweep as above, run across every named scenario (market condition) in `real_data_cache/scenarios.json` |
| `scripts/compare_strategies.py`, `scripts/run_all_strategies_demo.py` | Demo/comparison scripts against synthetic data |
| `real_data_cache/` | Committed real market data (parquet) + `scenarios.json` — see its own README.md for the upload workflow |
| `nifty_live/position_store.py` | `PositionStore` — JSON-persisted `Leg`/`MultiLegPosition` state (same classes as `strategy.py`, unchanged); one schema for both manually-authored seed files and machine round-trip. `discover_zerodha_nifty_option_legs()` + `PositionStore.import_from_zerodha()` — read-only Zerodha holdings import (kiteconnect `positions()` joined to `instruments()` by `instrument_token`), with strategy-role assignment left explicit rather than guessed. See its module docstring for the confirmed kiteconnect schema and known limitations (no true entry-time recovery). |
| `scripts/verify_zerodha_import.py` | Manual verification against a REAL Zerodha account (same convention as `verify_find_atm.py`/`debug_breeze.py`) — prints raw `positions()`/`instruments()` output alongside what this codebase derives from it, for eyeballing before trusting it with real capital |
| `scripts/generate_kite_token.py` | Runs the Kite Connect login/`request_token` exchange and writes the resulting `access_token` (valid one trading day) straight into `.env` — the daily Kite auth chore in one script run instead of manual copy-paste |
| `nifty_backtester/env_loader.py` | `load_env()` — loads Breeze/Kite credentials from a local, gitignored `.env` file (see `.env.example`) via python-dotenv, so short-lived tokens (Breeze session token, Kite access token) don't need re-exporting by hand every session. Wired into `resolve_data_layer()` and every manual verify/debug script. |
| `.env.example` | Template for the gitignored `.env` file — documents every credential key and its rotation cadence |
| `nifty_backtester/time_grid.py` | `trading_time_grid()` — shared 9:15-15:30/weekday bar-timestamp generator, used by both `backtest_engine.py` and `nifty_live/replay_feed.py` so batch and live/replay walk identical timestamps |
| `nifty_live/replay_feed.py` | `ReplayLiveFeed` — walks historical (sample or committed-real) bars bar-by-bar as though arriving live, built on `ScenarioBoundedDataLayer`'s `as_of_cursor` (Phase 2), with a verified no-look-ahead guarantee. `live_polling_clock()` — the real-live counterpart (yields `datetime.now()` on an interval); the exact same `live_engine.run_live_monitor()` loop runs against either. |
| `nifty_live/notifier.py` | `Notifier` interface; `ConsoleNotifier` (prints one line per action, same shape as backtest action-log lines); `CollectingNotifier` (test double) |
| `nifty_live/live_engine.py` | `run_live_monitor()` — the live/replay counterpart to `run_full_backtest()`. Calls the same `AdjustmentStrategy.evaluate()` but **never executes actions or mutates positions** — decision support only, by design (see its module docstring for the dedup-vs-noise tradeoff this implies) |
| `scripts/run_campaign_backtest.py` | Runs the funded-strangle campaign strategy for one or more consecutive months, optionally sweeping the weekly-roll/monthly-close time of day to compare metrics (see STRATEGY.md §7) |
| `scripts/compare_campaigns.py` | Runs Campaign 1 (funded strangle) and Campaign 2 (recentered straddle) over the same window(s) and prints a side-by-side metrics comparison, optionally sweeping both campaigns' adjustment time together (see STRATEGY.md §7.2) |
| `scripts/run_live_monitor_demo.py` | Demo: runs the full live-monitoring pipeline against sample data (zero live session) — the `nifty_live` counterpart to `run_all_strategies_demo.py` |
| `tests/test_quick_strategy_checks.py`, `tests/test_point1_point2_checks.py`, `tests/test_full_backtest_engine.py`, `tests/test_campaign_strategy.py`, `tests/test_campaign_straddle_strategy.py`, `tests/test_expiry_holidays.py`, `tests/test_data_cache.py`, `tests/test_data_layer_base.py`, `tests/test_data_layer_scenario.py`, `tests/test_position_store.py`, `tests/test_env_loader.py`, `tests/test_data_sources_env.py`, `tests/test_kite_login_url.py`, `tests/test_time_grid.py`, `tests/test_replay_feed.py`, `tests/test_live_engine.py`, `tests/test_market_data_and_expiry_detection.py`, `tests/test_scenarios.py` | Full test suite (147 tests with the optional `live` extra installed; 146 without it — one test skips cleanly rather than requiring `kiteconnect`) for the pieces described in this document |
