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

`expiry_utils.select_monthly_hedge_expiry(as_of, current_month_expiry,
next_month_expiry, min_days=15)`: use the current calendar month's monthly
expiry as the hedge unless it's less than 15 days from `as_of`, in which
case roll to next month's monthly expiry. The actual expiry *dates* (e.g.
"last Thursday of the month") are supplied by the caller — this function is
pure calendar arithmetic on two dates you already have, not a source of
NSE calendar knowledge.

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

- `is_expiry_eve_close_bar(as_of, expiry_date, close_time)` — true from
  that bar onward on the eve day.
- `is_on_or_after_expiry(as_of, expiry_date)` — hard backstop so nothing
  survives past expiry regardless of whether the eve-close bar landed
  exactly on a bar timestamp.

The just-OTM opportunistic legs (`otm_position`) are **not** naturally
paired with a hedge, so whether they get force-closed on the same eve-close
schedule or are allowed to run past it (closing only on their own Renko
exit signal) is a config choice for the engine, not something
`expiry_utils` decides — see **Open questions**.

**Stated limitation**: "the day before expiry" is calendar-date-minus-one,
not "the previous NSE trading day." Correct for the common non-Monday
weekly-expiry case; would need a real trading-calendar lookup if that ever
changes.

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

## 6. File map

| File | Contents |
|---|---|
| `strategy.py` | Position/leg data model, all indicators, all `AdjustmentStrategy` implementations |
| `pricing.py` | Black-Scholes-Merton pricing, IV solve, Greeks (`solve_iv_and_greeks`) |
| `data_layer_breeze.py` | Live Breeze-backed data layer: historical option/index data, ATM resolution |
| `data_layer_sample.py` | Same interface as the Breeze data layer, backed by synthetic data — for credential-free tests |
| `sample_data.py` | Deterministic synthetic OHLCV generators used by `data_layer_sample.py` and tests |
| `data_cache.py` | File-based incremental cache with retry/backoff, sitting in front of every historical-data fetch |
| `expiry_utils.py` | Monthly hedge expiry selection, expiry-eve close-bar timing |
| `market_data.py` | `MarketDataProvider` abstraction (`SyntheticMarketDataProvider` / `BreezeMarketDataProvider`) consumed by the backtest engine |
| `metrics.py` | Equity-curve and trade-level performance metrics (Sharpe, Sortino, Calmar, drawdown, win rate, etc.) |
| `backtest_engine.py` | Time-stepping loop that opens positions, evaluates strategies, and executes their actions bar-by-bar |
| `test_quick_strategy_checks.py`, `test_point1_point2_checks.py` | Fast, dependency-light checks for the pieces described in this document |
