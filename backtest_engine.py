"""
Backtest engine — generalized version.

Runs any combination of the four position shapes documented in
STRATEGY.md (sold_straddle, hedge_straddle, directional_overlay,
otm_position) against a single weekly expiry cycle, using whichever
sold-leg management "shape" (delta-threshold / fixed-move / RSI /
Supertrend+EMA) is selected, and produces an equity curve + trade list
ready for metrics.full_report().

SCOPE, stated plainly: this covers ONE weekly expiry cycle per run (start
to end within a single week) — matching the original engine's own stated
scope note. Rolling forward across multiple consecutive weekly cycles
(reopening a fresh sold_straddle after one expires) is a natural next
step, not yet built here.

Indicator lifetimes, by construction:
  - The 1hr overlay-direction signal and the 1-min Renko OTM signal are
    both built on the UNDERLYING's price series, which doesn't depend on
    which strike any leg happens to be at — built ONCE per run, for the
    whole window.
  - The 15-min sold-leg / overlay-short-leg signal is built on that
    LEG'S OWN option price series, which DOES depend on its strike — it's
    rebuilt every time that leg recenters or reopens.
  - Delta indicators depend on strike for every delta-tracked leg
    (sold/hedge/overlay-short — NOT overlay-long or otm, which aren't
    delta-managed) and are kept in ONE shared dict, referenced by every
    strategy that reads deltas, so a single rebuild is visible everywhere
    without needing to update each strategy's own copy.
"""

from __future__ import annotations
import datetime as dt
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from strategy import (
    Leg, MultiLegPosition, Right, Direction, ActionType, Action,
    DeltaThresholdStrategy, FixedMoveStrategy,
    SoldLegSignalStrategy, RSICrossAdapter, SupertrendEMAAdapter,
    RSIIndicator, SupertrendEMAIndicator, RenkoSuperTrendIndicator,
    DeltaIndicator, AdjustmentStrategy,
    DirectionalOverlayStrategy, RenkoOpportunisticOTMStrategy,
    nearest_just_otm_strikes,
)
from expiry_utils import is_expiry_eve_close_bar, is_on_or_after_expiry
from pricing import solve_iv_and_greeks
from market_data import MarketDataProvider


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

@dataclass
class FullBacktestConfig:
    start: dt.datetime
    end: dt.datetime
    weekly_expiry: dt.date

    strike_step: int = 100
    bar_freq_minutes: int = 15   # matches the 15-min sold-leg signal cadence
    initial_capital: float = 100_000.0
    lot_size: int = 1

    # --- sold-leg management shape (pick ONE per run) ---
    sold_leg_strategy_name: str = "delta_threshold"  # "delta_threshold" | "fixed_move" | "rsi_signal" | "supertrend_ema_signal"
    sold_delta_threshold: float = 0.65
    fixed_move_points: float = 100.0
    fixed_move_pct: float = 0.5
    harvest_delta_threshold: float = 0.2
    signal_lookback_days: int = 5

    # --- hedge straddle (monthly, long) ---
    include_hedge_straddle: bool = False
    monthly_expiry: Optional[dt.date] = None  # required if include_hedge_straddle; resolve via expiry_utils beforehand
    hedge_delta_threshold: float = 0.85

    # --- directional overlay (weekly spread) ---
    include_overlay: bool = False
    overlay_target_delta: float = 0.75
    overlay_hourly_kind: str = "rsi"     # "rsi" | "supertrend_ema" -- direction signal
    overlay_gating_kind: str = "rsi"     # "rsi" | "supertrend_ema" -- short-leg entry/management signal

    # --- opportunistic OTM (weekly, Renko-managed) ---
    include_otm: bool = False
    otm_multiplier: int = 1
    otm_close_before_expiry: bool = True
    otm_renko_lookback_days: int = 3

    expiry_eve_close_time: dt.time = dt.time(15, 30)


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trade_pnls: list[float]
    action_log: list[str] = field(default_factory=list)
    positions: dict = field(default_factory=dict)  # final MultiLegPosition objects, for inspection


# ─────────────────────────────────────────────
# TIME GRID
# ─────────────────────────────────────────────

def _time_grid(config: FullBacktestConfig) -> list[dt.datetime]:
    grid = []
    cursor = config.start.date()
    while cursor <= config.end.date():
        if cursor.weekday() < 5:
            t = dt.datetime.combine(cursor, dt.time(9, 15))
            day_end = dt.datetime.combine(cursor, dt.time(15, 30))
            while t <= day_end:
                if config.start <= t <= config.end:
                    grid.append(t)
                t += dt.timedelta(minutes=config.bar_freq_minutes)
        cursor += dt.timedelta(days=1)
    return grid


# ─────────────────────────────────────────────
# STRIKE / LEG RESOLUTION
# ─────────────────────────────────────────────

def _find_target_delta_strike(
    provider: MarketDataProvider,
    expiry: dt.date,
    as_of: dt.datetime,
    right: str,
    target_delta: float,
    strike_step: int,
    max_steps: int = 40,
) -> int:
    """Walks strikes from ATM in the ITM direction until |delta| reaches
    target_delta, tracking the closest strike seen along the way. Delta is
    monotonic in strike for a fixed right, so this converges quickly."""
    spot = provider.get_spot(as_of)
    center = provider.find_atm_strike(expiry, as_of, strike_step)
    direction = -1 if right.lower().startswith("c") else 1  # ITM: below spot for calls, above for puts

    best_strike, best_diff = center, None
    strike = center
    for _ in range(max_steps):
        price = provider.get_option_price(strike, right, expiry, as_of)
        result = solve_iv_and_greeks(
            price=price, spot=spot, strike=strike, now=as_of,
            expiry_date=expiry, right=right.capitalize(),
        )
        d = abs(result["delta"]) if result["delta"] == result["delta"] else 0.0  # NaN check
        diff = abs(d - target_delta)
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_strike = strike
        if d >= target_delta:
            break
        strike += direction * strike_step
    return int(best_strike)


def _leg_spec(tag: str, config: FullBacktestConfig) -> dict:
    if tag in ("sold_call", "sold_put"):
        return dict(right="call" if tag == "sold_call" else "put", direction=Direction.SHORT,
                    expiry=config.weekly_expiry, strike_kind="atm", quantity=config.lot_size)
    if tag in ("hedge_call", "hedge_put"):
        return dict(right="call" if tag == "hedge_call" else "put", direction=Direction.LONG,
                    expiry=config.monthly_expiry, strike_kind="atm", quantity=config.lot_size)
    if tag in ("overlay_short_call", "overlay_short_put"):
        return dict(right="call" if "call" in tag else "put", direction=Direction.SHORT,
                    expiry=config.weekly_expiry, strike_kind="atm", quantity=config.lot_size)
    if tag in ("overlay_long_call", "overlay_long_put"):
        return dict(right="call" if "call" in tag else "put", direction=Direction.LONG,
                    expiry=config.weekly_expiry, strike_kind="delta_target", quantity=config.lot_size)
    if tag in ("otm_call", "otm_put"):
        return dict(right="call" if tag == "otm_call" else "put", direction=Direction.LONG,
                    expiry=config.weekly_expiry, strike_kind="just_otm", quantity=config.lot_size * config.otm_multiplier)
    raise ValueError(f"Unknown leg tag: {tag}")


def _resolve_strike(strike_kind: str, provider: MarketDataProvider, expiry: dt.date, as_of: dt.datetime,
                     right: str, strike_step: int, target_delta: float) -> int:
    if strike_kind == "atm":
        return provider.find_atm_strike(expiry, as_of, strike_step)
    if strike_kind == "delta_target":
        return _find_target_delta_strike(provider, expiry, as_of, right, target_delta, strike_step)
    if strike_kind == "just_otm":
        spot = provider.get_spot(as_of)
        call_k, put_k = nearest_just_otm_strikes(spot, strike_step)
        return call_k if right == "call" else put_k
    raise ValueError(f"Unknown strike_kind: {strike_kind}")


def _open_new_leg(tag: str, as_of: dt.datetime, provider: MarketDataProvider, config: FullBacktestConfig) -> Leg:
    spec = _leg_spec(tag, config)
    strike = _resolve_strike(spec["strike_kind"], provider, spec["expiry"], as_of, spec["right"],
                              config.strike_step, config.overlay_target_delta)
    price = provider.get_option_price(strike, spec["right"], spec["expiry"], as_of)
    right_enum = Right.CALL if spec["right"] == "call" else Right.PUT
    return Leg(tag=tag, right=right_enum, direction=spec["direction"], strike=strike, expiry=spec["expiry"],
               entry_time=as_of, entry_price=price, quantity=spec["quantity"])


# ─────────────────────────────────────────────
# INDICATOR BUILDERS
# ─────────────────────────────────────────────

DELTA_TRACKED_TAGS = {"sold_call", "sold_put", "hedge_call", "hedge_put", "overlay_short_call", "overlay_short_put"}


def _build_delta_indicator(leg: Leg, provider: MarketDataProvider) -> DeltaIndicator:
    return DeltaIndicator(
        strike=leg.strike, right=leg.right, expiry=leg.expiry,
        spot_lookup=provider.get_spot,
        option_price_lookup=lambda ts, s=leg.strike, r=leg.right, e=leg.expiry: provider.get_option_price(s, r.value, e, ts),
    )


def _build_leg_signal_indicator(kind: str, provider: MarketDataProvider, leg: Leg, as_of: dt.datetime,
                                 lookback_days: int):
    series = provider.option_price_series(
        leg.strike, leg.right.value, leg.expiry,
        as_of - dt.timedelta(days=lookback_days), as_of, freq_minutes=15,
    )
    if len(series) < 30:
        return None
    if kind == "rsi":
        return RSIIndicator(series, rsi_len=20, ema_len=9)
    if kind == "supertrend_ema":
        return SupertrendEMAIndicator(series, ema_len=21, factor=3.0, atr_period=10)
    raise ValueError(f"Unknown signal kind: {kind}")


def _wrap_adapter(kind: str, indicator):
    if indicator is None:
        return None
    return RSICrossAdapter(indicator) if kind == "rsi" else SupertrendEMAAdapter(indicator)


def _build_underlying_signal_indicator(kind: str, provider: MarketDataProvider, start: dt.datetime, end: dt.datetime,
                                        freq_minutes: int):
    series = provider.spot_series(start, end, freq_minutes=freq_minutes)
    if len(series) < 30:
        return None
    if kind == "rsi":
        return RSIIndicator(series, rsi_len=20, ema_len=9)
    if kind == "supertrend_ema":
        return SupertrendEMAIndicator(series, ema_len=21, factor=3.0, atr_period=10)
    raise ValueError(f"Unknown signal kind: {kind}")


# ─────────────────────────────────────────────
# MAIN ENGINE
# ─────────────────────────────────────────────

def run_full_backtest(provider: MarketDataProvider, config: FullBacktestConfig) -> BacktestResult:
    grid = _time_grid(config)
    if not grid:
        raise ValueError("Empty time grid — check config.start/end fall on weekdays within trading hours.")
    if config.include_hedge_straddle and config.monthly_expiry is None:
        raise ValueError("include_hedge_straddle=True requires config.monthly_expiry to be set.")

    trade_pnls: list[float] = []
    action_log: list[str] = []
    equity_points: list[tuple[dt.datetime, float]] = []
    equity_base = config.initial_capital
    # Positions force-closed for expiry must NOT be reopened for the rest of
    # this run -- the engine's scope is a single weekly-expiry cycle, so
    # there's no valid next contract to roll into. Without this, a strategy
    # with no memory of "why" a position went flat (RenkoLegTracker and
    # DirectionalOverlayStrategy's long-leg entry both just check "is a leg
    # currently open + does my signal want one" -- discovered via
    # download_and_run_strategy.py showing an OTM/overlay leg reopen in the
    # SAME bar it was force-closed on expiry eve) would immediately reopen
    # it on the same about-to-expire contract, defeating the point of the
    # expiry-eve close entirely.
    closed_for_expiry: set[str] = set()
    _already_logged_skip: set[tuple] = set()

    positions: dict[str, MultiLegPosition] = {}
    delta_indicators: dict[str, DeltaIndicator] = {}  # SHARED across every strategy that reads deltas

    # ---- open initial positions ----
    positions["sold_straddle"] = MultiLegPosition(name="sold_straddle")
    for tag in ("sold_call", "sold_put"):
        leg = _open_new_leg(tag, grid[0], provider, config)
        positions["sold_straddle"].add_leg(leg)
        delta_indicators[tag] = _build_delta_indicator(leg, provider)

    if config.include_hedge_straddle:
        positions["hedge_straddle"] = MultiLegPosition(name="hedge_straddle")
        for tag in ("hedge_call", "hedge_put"):
            leg = _open_new_leg(tag, grid[0], provider, config)
            positions["hedge_straddle"].add_leg(leg)
            delta_indicators[tag] = _build_delta_indicator(leg, provider)

    if config.include_overlay:
        positions["directional_overlay"] = MultiLegPosition(name="directional_overlay")

    if config.include_otm:
        positions["otm_position"] = MultiLegPosition(name="otm_position")

    # ---- per-leg 15-min signal ownership, for rebuild-on-recenter ----
    # tag -> (owning SoldLegSignalStrategy instance, its adapter kind)
    signal_owner: dict[str, tuple] = {}

    def _rebuild_leg_signal(tag: str, leg: Leg, as_of: dt.datetime) -> None:
        owner = signal_owner.get(tag)
        if owner is None:
            return
        strategy_obj, kind = owner
        new_ind = _build_leg_signal_indicator(kind, provider, leg, as_of, config.signal_lookback_days)
        if new_ind is not None:
            strategy_obj.signal_adapters[tag] = _wrap_adapter(kind, new_ind)

    # ---- core sold-leg strategy ----
    fixed_move_strategy: Optional[FixedMoveStrategy] = None

    if config.sold_leg_strategy_name == "delta_threshold":
        core_strategy = DeltaThresholdStrategy(
            delta_indicators=delta_indicators, sold_threshold=config.sold_delta_threshold,
            hedge_threshold=float("inf"),  # hedge handled by its own always-on instance below
        )
    elif config.sold_leg_strategy_name == "fixed_move":
        fixed_move_strategy = FixedMoveStrategy(
            spot_lookup=provider.get_spot, move_points=config.fixed_move_points,
            move_pct=config.fixed_move_pct, position_name="sold_straddle",
        )
        fixed_move_strategy.set_reference(provider.get_spot(grid[0]))
        core_strategy = fixed_move_strategy
    elif config.sold_leg_strategy_name in ("rsi_signal", "supertrend_ema_signal"):
        kind = "rsi" if config.sold_leg_strategy_name == "rsi_signal" else "supertrend_ema"
        signal_adapters = {}
        for tag in ("sold_call", "sold_put"):
            leg = positions["sold_straddle"].get_leg(tag)
            ind = _build_leg_signal_indicator(kind, provider, leg, grid[0], config.signal_lookback_days)
            signal_adapters[tag] = _wrap_adapter(kind, ind)
        core_strategy = SoldLegSignalStrategy(
            signal_adapters=signal_adapters, delta_indicators=delta_indicators,
            harvest_delta_threshold=config.harvest_delta_threshold, position_name="sold_straddle",
        )
        signal_owner["sold_call"] = (core_strategy, kind)
        signal_owner["sold_put"] = (core_strategy, kind)
    else:
        raise ValueError(f"Unknown sold_leg_strategy_name: {config.sold_leg_strategy_name}")

    # ---- always-on hedge delta strategy (independent of sold-leg shape) ----
    hedge_strategy: Optional[DeltaThresholdStrategy] = None
    if config.include_hedge_straddle:
        hedge_strategy = DeltaThresholdStrategy(
            delta_indicators=delta_indicators, sold_threshold=float("inf"),
            hedge_threshold=config.hedge_delta_threshold,
        )

    # ---- overlay ----
    overlay_strategy: Optional[DirectionalOverlayStrategy] = None
    if config.include_overlay:
        hourly_ind = _build_underlying_signal_indicator(
            config.overlay_hourly_kind, provider, grid[0] - dt.timedelta(days=config.signal_lookback_days),
            config.end, freq_minutes=60,
        )
        hourly_regime_signal = _wrap_adapter(config.overlay_hourly_kind, hourly_ind)

        short_leg_strategy = SoldLegSignalStrategy(
            signal_adapters={}, delta_indicators=delta_indicators,
            harvest_delta_threshold=config.harvest_delta_threshold, position_name="directional_overlay",
        )
        signal_owner["overlay_short_call"] = (short_leg_strategy, config.overlay_gating_kind)
        signal_owner["overlay_short_put"] = (short_leg_strategy, config.overlay_gating_kind)

        overlay_strategy = DirectionalOverlayStrategy(
            core_strategy=core_strategy, hourly_regime_signal=hourly_regime_signal,
            short_leg_strategy=short_leg_strategy,
        )

    # ---- opportunistic OTM ----
    otm_strategy: Optional[RenkoOpportunisticOTMStrategy] = None
    if config.include_otm:
        renko_ind = _build_underlying_signal_indicator_renko = None  # placeholder to keep linter calm
        renko_series = provider.spot_series(
            grid[0] - dt.timedelta(days=config.otm_renko_lookback_days), config.end, freq_minutes=1,
        )
        if len(renko_series) >= 30:
            renko_ind = RenkoSuperTrendIndicator(renko_series, atr_box_len=14, atr_box_mult=1.0, st_factor=3.0, st_atr_len=10)
            otm_strategy = RenkoOpportunisticOTMStrategy(renko_indicator=renko_ind, position_name="otm_position")

    # ---- main loop ----
    for as_of in grid:
        # expiry-eve force closes (point 6f) — checked before evaluating strategies
        if is_expiry_eve_close_bar(as_of, config.weekly_expiry, config.expiry_eve_close_time):
            _force_close(positions["sold_straddle"], as_of, provider, trade_pnls, action_log, "weekly expiry eve close")
            closed_for_expiry.add("sold_straddle")
            if config.include_overlay:
                _force_close(positions["directional_overlay"], as_of, provider, trade_pnls, action_log, "weekly expiry eve close (overlay)")
                closed_for_expiry.add("directional_overlay")
            if config.include_otm and config.otm_close_before_expiry:
                _force_close(positions["otm_position"], as_of, provider, trade_pnls, action_log, "weekly expiry eve close (otm, per config)")
                closed_for_expiry.add("otm_position")
        if config.include_hedge_straddle and is_expiry_eve_close_bar(as_of, config.monthly_expiry, config.expiry_eve_close_time):
            _force_close(positions["hedge_straddle"], as_of, provider, trade_pnls, action_log, "monthly expiry eve close")
            closed_for_expiry.add("hedge_straddle")

        actions: list[Action] = []
        if overlay_strategy is not None:
            actions.extend(overlay_strategy.evaluate(positions, as_of))
        else:
            actions.extend(core_strategy.evaluate(positions, as_of))
        if hedge_strategy is not None:
            actions.extend(hedge_strategy.evaluate(positions, as_of))
        if otm_strategy is not None:
            actions.extend(otm_strategy.evaluate(positions, as_of))

        for action in actions:
            pos = positions.get(action.position_name)
            if pos is None:
                continue

            if action.position_name in closed_for_expiry and action.type in (ActionType.OPEN_LEG, ActionType.RECENTER):
                skip_key = (action.position_name, action.leg_tag)
                if skip_key not in _already_logged_skip:
                    _already_logged_skip.add(skip_key)
                    action_log.append(
                        f"{as_of}: SKIPPED {action.type.value} on {action.position_name}.{action.leg_tag or '(whole position)'} "
                        f"-- position was force-closed for expiry this run and will not be reopened (further repeats of "
                        f"this suppression are not logged individually) — original reason: {action.reason}"
                    )
                continue

            if action.type == ActionType.CLOSE_LEG:
                leg = pos.get_leg(action.leg_tag)
                if leg is not None:
                    exit_price = provider.get_option_price(leg.strike, leg.right.value, leg.expiry, as_of)
                    leg.close(as_of, exit_price)
                    trade_pnls.append(leg.pnl())
                    action_log.append(f"{as_of}: CLOSE {action.position_name}.{action.leg_tag} @ {exit_price:.2f} — {action.reason}")

            elif action.type == ActionType.OPEN_LEG:
                new_leg = _open_new_leg(action.leg_tag, as_of, provider, config)
                pos.add_leg(new_leg)
                action_log.append(f"{as_of}: OPEN {action.position_name}.{action.leg_tag} strike={new_leg.strike} price={new_leg.entry_price:.2f} — {action.reason}")
                if action.leg_tag in DELTA_TRACKED_TAGS:
                    delta_indicators[action.leg_tag] = _build_delta_indicator(new_leg, provider)
                _rebuild_leg_signal(action.leg_tag, new_leg, as_of)

            elif action.type == ActionType.RECENTER:
                if action.leg_tag:  # single-leg recenter (harvest)
                    leg = pos.get_leg(action.leg_tag)
                    if leg is not None:
                        exit_price = provider.get_option_price(leg.strike, leg.right.value, leg.expiry, as_of)
                        leg.close(as_of, exit_price)
                        trade_pnls.append(leg.pnl())
                        new_leg = _open_new_leg(action.leg_tag, as_of, provider, config)
                        pos.add_leg(new_leg)
                        action_log.append(f"{as_of}: RECENTER {action.position_name}.{action.leg_tag} -> strike {new_leg.strike} — {action.reason}")
                        if action.leg_tag in DELTA_TRACKED_TAGS:
                            delta_indicators[action.leg_tag] = _build_delta_indicator(new_leg, provider)
                        _rebuild_leg_signal(action.leg_tag, new_leg, as_of)
                else:  # whole-position recenter
                    tags = [l.tag for l in pos.open_legs()]
                    for leg in pos.open_legs():
                        exit_price = provider.get_option_price(leg.strike, leg.right.value, leg.expiry, as_of)
                        leg.close(as_of, exit_price)
                        trade_pnls.append(leg.pnl())
                    for tag in tags:
                        new_leg = _open_new_leg(tag, as_of, provider, config)
                        pos.add_leg(new_leg)
                        if tag in DELTA_TRACKED_TAGS:
                            delta_indicators[tag] = _build_delta_indicator(new_leg, provider)
                        _rebuild_leg_signal(tag, new_leg, as_of)
                    if fixed_move_strategy is not None and action.position_name == "sold_straddle":
                        fixed_move_strategy.set_reference(provider.get_spot(as_of))
                    action_log.append(f"{as_of}: RECENTER whole {action.position_name} — {action.reason}")

        # mark-to-market equity across every position -- total_pnl() sums
        # BOTH realized pnl from closed legs AND unrealized mark-to-market
        # from open legs (it iterates pos.legs, the full history, not just
        # open_legs()), so this must run even when a position is currently
        # flat -- skipping flat positions here would silently drop their
        # entire realized P&L from every subsequent equity point.
        total_pnl = 0.0
        for pos in positions.values():
            marks = {leg.tag: provider.get_option_price(leg.strike, leg.right.value, leg.expiry, as_of) for leg in pos.open_legs()}
            total_pnl += pos.total_pnl(marks)
        equity_points.append((as_of, equity_base + total_pnl))

    # close anything still open at the very end of the window
    for pos in positions.values():
        for leg in pos.open_legs():
            exit_price = provider.get_option_price(leg.strike, leg.right.value, leg.expiry, grid[-1])
            leg.close(grid[-1], exit_price)
            trade_pnls.append(leg.pnl())

    equity_curve = pd.Series(
        [v for _, v in equity_points],
        index=pd.DatetimeIndex([t for t, _ in equity_points]),
    )
    return BacktestResult(equity_curve=equity_curve, trade_pnls=trade_pnls, action_log=action_log, positions=positions)


def _force_close(pos: MultiLegPosition, as_of: dt.datetime, provider: MarketDataProvider,
                  trade_pnls: list[float], action_log: list[str], reason: str) -> None:
    for leg in pos.open_legs():
        exit_price = provider.get_option_price(leg.strike, leg.right.value, leg.expiry, as_of)
        leg.close(as_of, exit_price)
        trade_pnls.append(leg.pnl())
        action_log.append(f"{as_of}: FORCE-CLOSE {pos.name}.{leg.tag} @ {exit_price:.2f} — {reason}")
