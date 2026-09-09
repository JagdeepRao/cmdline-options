"""
Backtest engine: runs a sold straddle, managed by a pluggable
AdjustmentStrategy, over a configurable historical period — via any
MarketDataProvider (synthetic for testing, Breeze for real runs) — and
produces an equity curve + trade list ready for metrics.full_report().

SCOPE OF THIS VERSION: manages the sold_straddle position under strategies
1/2/3 (DeltaThresholdStrategy / RSISignalStrategy / FixedMoveStrategy).
The hedge_straddle leg, the directional overlay (strategy 4), and the OTM
scalp legs are NOT yet wired into this loop — they're built and tested as
standalone components (strategy_framework.py) but not yet orchestrated
together here. This lets the core "compare strategies" comparison run now,
with the other legs as the next extension on top of a validated foundation
rather than everything built and tested simultaneously.
"""

from __future__ import annotations
import datetime as dt
from dataclasses import dataclass, field

import pandas as pd

from strategy_framework import (
    Leg, MultiLegPosition, Right, Direction, ActionType,
    DeltaThresholdStrategy, RSISignalStrategy, FixedMoveStrategy,
    DeltaIndicator, RSIIndicator, AdjustmentStrategy,
)
from market_data import MarketDataProvider


@dataclass
class BacktestConfig:
    start: dt.datetime
    end: dt.datetime
    expiry: dt.date
    strike_step: int = 100
    bar_freq_minutes: int = 5          # how often the engine steps/evaluates
    initial_capital: float = 100_000.0
    lot_size: int = 1


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trade_pnls: list[float]
    action_log: list[str] = field(default_factory=list)


def _time_grid(config: BacktestConfig) -> list[dt.datetime]:
    """Bar timestamps within NSE trading hours, weekdays only, at the
    configured frequency. Same weekday-only simplification as the synthetic
    provider — no trading-holiday calendar."""
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


def _open_straddle(provider: MarketDataProvider, expiry: dt.date, as_of: dt.datetime, strike_step: int, lot_size: int) -> MultiLegPosition:
    strike = provider.find_atm_strike(expiry, as_of, strike_step)
    call_price = provider.get_option_price(strike, "call", expiry, as_of)
    put_price = provider.get_option_price(strike, "put", expiry, as_of)

    pos = MultiLegPosition(name="sold_straddle")
    pos.add_leg(Leg(tag="sold_call", right=Right.CALL, direction=Direction.SHORT,
                     strike=strike, expiry=expiry, entry_time=as_of, entry_price=call_price, quantity=lot_size))
    pos.add_leg(Leg(tag="sold_put", right=Right.PUT, direction=Direction.SHORT,
                     strike=strike, expiry=expiry, entry_time=as_of, entry_price=put_price, quantity=lot_size))
    return pos


def run_backtest(
    provider: MarketDataProvider,
    config: BacktestConfig,
    strategy_name: str,   # "delta_threshold" | "rsi_signal" | "fixed_move"
    strategy_kwargs: dict = None,
) -> BacktestResult:
    strategy_kwargs = strategy_kwargs or {}
    grid = _time_grid(config)
    if not grid:
        raise ValueError("Empty time grid — check config.start/end fall on weekdays within trading hours.")

    position = _open_straddle(provider, config.expiry, grid[0], config.strike_step, config.lot_size)
    trade_pnls: list[float] = []
    equity_points: list[tuple[dt.datetime, float]] = []
    action_log: list[str] = []
    equity_base = config.initial_capital

    # For RSI-signal strategy, each leg needs its own 15-min RSI indicator,
    # rebuilt whenever that leg's strike changes (recenter/re-establish).
    def build_rsi_indicators(pos: MultiLegPosition, as_of: dt.datetime) -> dict[str, RSIIndicator]:
        indicators = {}
        for leg in pos.open_legs():
            series = provider.option_price_series(
                leg.strike, leg.right.value, leg.expiry,
                as_of - dt.timedelta(days=5), as_of, freq_minutes=15,
            )
            if len(series) >= 30:  # need enough bars for RSI(20)+EMA(9) to warm up
                indicators[leg.tag] = RSIIndicator(series, rsi_len=20, ema_len=9)
        return indicators

    def build_delta_indicators(pos: MultiLegPosition) -> dict[str, DeltaIndicator]:
        indicators = {}
        for leg in pos.open_legs():
            indicators[leg.tag] = DeltaIndicator(
                strike=leg.strike, right=leg.right, expiry=leg.expiry,
                spot_lookup=provider.get_spot,
                option_price_lookup=lambda as_of, s=leg.strike, r=leg.right, e=leg.expiry: provider.get_option_price(s, r.value, e, as_of),
            )
        return indicators

    strategy: AdjustmentStrategy
    fixed_move_strategy: FixedMoveStrategy = None  # needs reference-resetting on recenter

    if strategy_name == "delta_threshold":
        strategy = DeltaThresholdStrategy(delta_indicators=build_delta_indicators(position), **strategy_kwargs)
    elif strategy_name == "rsi_signal":
        strategy = RSISignalStrategy(
            rsi_indicators=build_rsi_indicators(position, grid[0]),
            delta_indicators=build_delta_indicators(position),
            **strategy_kwargs,
        )
    elif strategy_name == "fixed_move":
        fixed_move_strategy = FixedMoveStrategy(spot_lookup=provider.get_spot, **strategy_kwargs)
        fixed_move_strategy.set_reference(provider.get_spot(grid[0]))
        strategy = fixed_move_strategy
    else:
        raise ValueError(f"Unknown strategy_name: {strategy_name}")

    def mark_to_market(pos: MultiLegPosition, as_of: dt.datetime) -> float:
        marks = {leg.tag: provider.get_option_price(leg.strike, leg.right.value, leg.expiry, as_of) for leg in pos.open_legs()}
        return pos.total_pnl(marks)

    for as_of in grid:
        positions = {"sold_straddle": position}
        actions = strategy.evaluate(positions, as_of)

        for action in actions:
            if action.type == ActionType.CLOSE_LEG:
                leg = position.get_leg(action.leg_tag)
                if leg is not None:
                    exit_price = provider.get_option_price(leg.strike, leg.right.value, leg.expiry, as_of)
                    leg.close(as_of, exit_price)
                    trade_pnls.append(leg.pnl())
                    action_log.append(f"{as_of}: CLOSE {action.leg_tag} @ {exit_price:.2f} — {action.reason}")

            elif action.type == ActionType.OPEN_LEG:
                # re-open a leg of the same right/direction at the CURRENT ATM strike
                closed_leg = next((l for l in position.legs if l.tag == action.leg_tag), None)
                if closed_leg is not None:
                    new_strike = provider.find_atm_strike(config.expiry, as_of, config.strike_step)
                    new_price = provider.get_option_price(new_strike, closed_leg.right.value, config.expiry, as_of)
                    position.add_leg(Leg(tag=action.leg_tag, right=closed_leg.right, direction=Direction.SHORT,
                                          strike=new_strike, expiry=config.expiry, entry_time=as_of,
                                          entry_price=new_price, quantity=config.lot_size))
                    action_log.append(f"{as_of}: OPEN {action.leg_tag} @ strike {new_strike}, price {new_price:.2f} — {action.reason}")

            elif action.type == ActionType.RECENTER:
                if action.leg_tag:  # recenter a single leg (harvest-and-refresh, strategy 2 Rule B)
                    leg = position.get_leg(action.leg_tag)
                    if leg is not None:
                        exit_price = provider.get_option_price(leg.strike, leg.right.value, leg.expiry, as_of)
                        leg.close(as_of, exit_price)
                        trade_pnls.append(leg.pnl())
                        new_strike = provider.find_atm_strike(config.expiry, as_of, config.strike_step)
                        new_price = provider.get_option_price(new_strike, leg.right.value, config.expiry, as_of)
                        position.add_leg(Leg(tag=action.leg_tag, right=leg.right, direction=Direction.SHORT,
                                              strike=new_strike, expiry=config.expiry, entry_time=as_of,
                                              entry_price=new_price, quantity=config.lot_size))
                        # BUG FIX: the strategy's indicators for this leg were bound to
                        # the OLD strike via closure at construction time — without
                        # rebuilding them here, every subsequent evaluate() call reads
                        # a stale reference (a strike no longer in the position), which
                        # produced runaway repeated-harvest loops in testing. Rebuild
                        # just this leg's indicators, not the whole dict.
                        if hasattr(strategy, "delta_indicators"):
                            strategy.delta_indicators[action.leg_tag] = DeltaIndicator(
                                strike=new_strike, right=leg.right, expiry=config.expiry,
                                spot_lookup=provider.get_spot,
                                option_price_lookup=lambda as_of, s=new_strike, r=leg.right, e=config.expiry: provider.get_option_price(s, r.value, e, as_of),
                            )
                        if hasattr(strategy, "rsi_indicators"):
                            series = provider.option_price_series(
                                new_strike, leg.right.value, config.expiry,
                                as_of - dt.timedelta(days=5), as_of, freq_minutes=15,
                            )
                            if len(series) >= 30:
                                strategy.rsi_indicators[action.leg_tag] = RSIIndicator(series, rsi_len=20, ema_len=9)
                        action_log.append(f"{as_of}: RECENTER {action.leg_tag} -> strike {new_strike} — {action.reason}")
                else:  # recenter the whole position (strategy 1 / strategy 3)
                    for leg in position.open_legs():
                        exit_price = provider.get_option_price(leg.strike, leg.right.value, leg.expiry, as_of)
                        leg.close(as_of, exit_price)
                        trade_pnls.append(leg.pnl())
                    position = _open_straddle(provider, config.expiry, as_of, config.strike_step, config.lot_size)
                    if fixed_move_strategy is not None:
                        fixed_move_strategy.set_reference(provider.get_spot(as_of))
                    if strategy_name == "delta_threshold":
                        strategy.delta_indicators = build_delta_indicators(position)
                    if strategy_name == "rsi_signal":
                        strategy.rsi_indicators = build_rsi_indicators(position, as_of)
                        strategy.delta_indicators = build_delta_indicators(position)
                    action_log.append(f"{as_of}: RECENTER whole straddle -> new position — {action.reason}")

        equity_points.append((as_of, equity_base + mark_to_market(position, as_of)))

    # close anything still open at the end of the backtest window
    for leg in position.open_legs():
        exit_price = provider.get_option_price(leg.strike, leg.right.value, leg.expiry, grid[-1])
        leg.close(grid[-1], exit_price)
        trade_pnls.append(leg.pnl())

    equity_curve = pd.Series(
        [v for _, v in equity_points],
        index=pd.DatetimeIndex([t for t, _ in equity_points]),
    )
    return BacktestResult(equity_curve=equity_curve, trade_pnls=trade_pnls, action_log=action_log)
