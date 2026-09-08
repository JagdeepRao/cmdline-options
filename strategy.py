"""
Strategy framework for the NIFTY options backtester.

Core abstractions:
  - Leg: one option position within a multi-leg strategy (strike, right,
    entry/exit price and time, quantity, direction).
  - MultiLegPosition: a named collection of Legs with P&L aggregation.
  - Indicator: pluggable signal sources that strategies query.
  - AdjustmentStrategy: pluggable decision logic returning Actions.

Four concrete strategies, per spec:
  1. DeltaThresholdStrategy   — recenter sold leg at |delta|>=0.65, hedge leg at |delta|>=0.85
  2. RSISignalStrategy        — close a sold leg on 15-min RSI signal against it (re-enter
                                 only while RSI regime still favors selling); also recenter
                                 a leg once it decays to |delta|<=0.2
  3. FixedMoveStrategy        — recenter sold straddle every 100pt / 0.5% underlying move
  4. DirectionalOverlayStrategy — wraps one of 1/2/3 for the core sold straddle, adds a
                                 long 75-delta + ATM directional spread gated by 1hr
                                 underlying RSI, closed when that RSI flips

NOTE ON SCOPE: this module defines the data model, indicators, and strategy
logic, all independently tested against synthetic data below. It does NOT
yet include the time-stepping backtest loop that walks real historical data
bar-by-bar and calls these — that's the natural next piece, built on top of
this once this foundation is confirmed solid.
"""

from __future__ import annotations
import datetime as dt
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable
import pandas as pd
import numpy as np


# ─────────────────────────────────────────────
# CORE DATA MODEL
# ─────────────────────────────────────────────

class Right(Enum):
    CALL = "call"
    PUT = "put"


class Direction(Enum):
    LONG = "long"
    SHORT = "short"


@dataclass
class Leg:
    """One option position. `tag` is a free-form label (e.g. 'sold_call',
    'hedge_put', 'overlay_75d_call') used to identify a leg's role within
    its parent MultiLegPosition — strategies key off this, not just right/strike."""
    tag: str
    right: Right
    direction: Direction
    strike: float
    expiry: dt.date
    entry_time: dt.datetime
    entry_price: float
    quantity: int = 1
    exit_time: Optional[dt.datetime] = None
    exit_price: Optional[float] = None

    @property
    def is_open(self) -> bool:
        return self.exit_time is None

    def close(self, exit_time: dt.datetime, exit_price: float) -> None:
        self.exit_time = exit_time
        self.exit_price = exit_price

    def pnl(self, mark_price: Optional[float] = None) -> float:
        """Realized P&L if closed, else mark-to-market P&L using mark_price."""
        price = self.exit_price if not self.is_open else mark_price
        if price is None:
            raise ValueError(f"Leg {self.tag} is open and no mark_price was given.")
        sign = 1 if self.direction == Direction.LONG else -1
        return sign * (price - self.entry_price) * self.quantity


@dataclass
class MultiLegPosition:
    """A named collection of legs (e.g. the sold straddle, the hedge
    straddle, a directional overlay). Legs can be added/closed independently
    — this is what makes leg-level adjustment (closing just the threatened
    side) possible rather than only whole-position open/close."""
    name: str
    legs: list[Leg] = field(default_factory=list)

    def add_leg(self, leg: Leg) -> None:
        self.legs.append(leg)

    def open_legs(self) -> list[Leg]:
        return [l for l in self.legs if l.is_open]

    def get_leg(self, tag: str) -> Optional[Leg]:
        matches = [l for l in self.legs if l.tag == tag and l.is_open]
        return matches[0] if matches else None

    def total_pnl(self, mark_prices: dict[str, float] = None) -> float:
        """mark_prices: {tag: current_price} for any still-open legs."""
        mark_prices = mark_prices or {}
        total = 0.0
        for leg in self.legs:
            if leg.is_open:
                if leg.tag not in mark_prices:
                    raise ValueError(f"No mark price supplied for open leg '{leg.tag}'")
                total += leg.pnl(mark_prices[leg.tag])
            else:
                total += leg.pnl()
        return total

    def is_flat(self) -> bool:
        return len(self.open_legs()) == 0


class ActionType(Enum):
    CLOSE_LEG = "close_leg"
    OPEN_LEG = "open_leg"
    RECENTER = "recenter"  # convenience: close all open legs of a position, caller reopens


@dataclass
class Action:
    type: ActionType
    position_name: str
    leg_tag: Optional[str] = None          # for CLOSE_LEG
    new_leg: Optional[Leg] = None          # for OPEN_LEG
    reason: str = ""


# ─────────────────────────────────────────────
# INDICATORS — pluggable signal sources
# ─────────────────────────────────────────────

def _pine_smooth(series: pd.Series, length: int, alpha: float) -> pd.Series:
    """Shared logic for pine_rma/pine_ema. Correctly handles an input series
    that itself starts with leading NaNs (e.g. RSI applied to itself, which
    is undefined until the RSI's own warmup completes) by finding the first
    valid value and seeding `length` bars from THERE — matching Pine's
    actual na-propagation behavior, not naively assuming data starts at
    index 0."""
    result = pd.Series(np.nan, index=series.index, dtype=float)
    valid = series.notna()
    if valid.sum() < length:
        return result

    first_valid_pos = valid.values.argmax()
    seed_end_pos = first_valid_pos + length
    if seed_end_pos > len(series):
        return result

    seed_slice = series.iloc[first_valid_pos:seed_end_pos]
    if seed_slice.isna().any():
        return result  # gap inside the seed window — not expected for our use cases

    seed_pos = seed_end_pos - 1
    result.iloc[seed_pos] = seed_slice.mean()
    for i in range(seed_pos + 1, len(series)):
        if pd.isna(series.iloc[i]):
            continue
        result.iloc[i] = alpha * series.iloc[i] + (1 - alpha) * result.iloc[i - 1]
    return result


def pine_rma(series: pd.Series, length: int) -> pd.Series:
    """Exact replica of Pine's ta.rma(): seeds as an SMA of the first
    `length` VALID bars (correctly skipping any leading NaNs in the input),
    then continues with Wilder's recursive formula (alpha = 1/length)."""
    return _pine_smooth(series, length, alpha=1 / length)


def pine_ema(series: pd.Series, length: int) -> pd.Series:
    """Exact replica of Pine's ta.ema(): same seeding as pine_rma above,
    but with alpha = 2/(length+1)."""
    return _pine_smooth(series, length, alpha=2 / (length + 1))


class Indicator:
    """Base class. Concrete indicators implement value(as_of) -> whatever
    reading is relevant (a float, a signal enum, etc.) using whatever data
    source they were constructed with."""
    def value(self, as_of: dt.datetime):
        raise NotImplementedError


class DeltaIndicator(Indicator):
    """Wraps pricing.solve_iv_and_greeks for a single option leg at a point
    in time. Requires a spot-price lookup function and an option-price
    lookup function (both: datetime -> float), since indicators shouldn't
    own data-fetching logic themselves — that's the caller's job."""
    def __init__(
        self,
        strike: float,
        right: Right,
        expiry: dt.date,
        spot_lookup: Callable[[dt.datetime], float],
        option_price_lookup: Callable[[dt.datetime], float],
        r: float = 0.0525,
        q: float = 0.012,
    ):
        self.strike = strike
        self.right = right
        self.expiry = expiry
        self.spot_lookup = spot_lookup
        self.option_price_lookup = option_price_lookup
        self.r = r
        self.q = q

    def value(self, as_of: dt.datetime) -> float:
        from pricing import solve_iv_and_greeks
        spot = self.spot_lookup(as_of)
        price = self.option_price_lookup(as_of)
        result = solve_iv_and_greeks(
            price=price, spot=spot, strike=self.strike, now=as_of,
            expiry_date=self.expiry, right=self.right.value.capitalize(),
            r=self.r, q=self.q,
        )
        return abs(result["delta"])  # magnitude — sign convention handled by caller


class PriceIndicator(Indicator):
    """Simple wrapper around a price lookup function (spot or option)."""
    def __init__(self, price_lookup: Callable[[dt.datetime], float]):
        self.price_lookup = price_lookup

    def value(self, as_of: dt.datetime) -> float:
        return self.price_lookup(as_of)


class RSISignal(Enum):
    BULLISH_CROSS = "bullish_cross"   # RSI just crossed above its EMA
    BEARISH_CROSS = "bearish_cross"   # RSI just crossed below its EMA
    NEUTRAL = "neutral"


class RSIIndicator(Indicator):
    """RSI-vs-EMA(RSI) crossover, same logic as the Pine strategy (default
    RSI length 20, EMA-of-RSI length 9). Operates on a pre-fetched price
    series (a DataFrame with 'datetime' and 'close') resampled to the
    requested timeframe — pass 15-min data for per-leg option-premium
    signals, 1-hour data for the directional-overlay underlying signal.
    """
    def __init__(self, price_df: pd.DataFrame, rsi_len: int = 20, ema_len: int = 9):
        df = price_df.copy()
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)

        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = pine_rma(gain, rsi_len)
        avg_loss = pine_rma(loss, rsi_len)

        # Matches Pine's exact ta.rsi() edge-case handling:
        # down==0 -> RSI=100, up==0 (but down!=0) -> RSI=0, otherwise the formula.
        rsi = pd.Series(np.nan, index=df.index, dtype=float)
        both_defined = avg_gain.notna() & avg_loss.notna()
        rsi[both_defined & (avg_loss == 0)] = 100.0
        rsi[both_defined & (avg_loss != 0) & (avg_gain == 0)] = 0.0
        normal_mask = both_defined & (avg_loss != 0) & (avg_gain != 0)
        rsi[normal_mask] = 100 - (100 / (1 + avg_gain[normal_mask] / avg_loss[normal_mask]))
        df["rsi"] = rsi
        df["rsi_ema"] = pine_ema(df["rsi"], ema_len)

        df["bull_cross"] = (df["rsi"] > df["rsi_ema"]) & (df["rsi"].shift(1) <= df["rsi_ema"].shift(1))
        df["bear_cross"] = (df["rsi"] < df["rsi_ema"]) & (df["rsi"].shift(1) >= df["rsi_ema"].shift(1))

        self.df = df

    def value(self, as_of: dt.datetime) -> RSISignal:
        """Returns the signal on the bar at/nearest-before as_of."""
        eligible = self.df[self.df["datetime"] <= as_of]
        if eligible.empty:
            return RSISignal.NEUTRAL
        row = eligible.iloc[-1]
        if row["bull_cross"]:
            return RSISignal.BULLISH_CROSS
        if row["bear_cross"]:
            return RSISignal.BEARISH_CROSS
        return RSISignal.NEUTRAL

    def regime(self, as_of: dt.datetime) -> str:
        """'bullish' or 'bearish' — current state (RSI above/below its EMA),
        not just the crossover moment. Used for the 'reenter only while
        regime still favors selling' rule in strategy 2."""
        eligible = self.df[self.df["datetime"] <= as_of]
        if eligible.empty:
            return "neutral"
        row = eligible.iloc[-1]
        if pd.isna(row["rsi"]) or pd.isna(row["rsi_ema"]):
            return "neutral"
        return "bullish" if row["rsi"] > row["rsi_ema"] else "bearish"


class RenkoSuperTrendIndicator(Indicator):
    """Python port of the confirmed Pine Script logic, for the OTM-long
    buying legs. Defaults match the confirmed live config: ATR-based box
    (length 14, multiplier 1), SuperTrend factor 3, SuperTrend ATR length 10.
    """
    def __init__(
        self,
        price_df: pd.DataFrame,
        atr_box_len: int = 14,
        atr_box_mult: float = 1.0,
        st_factor: float = 3.0,
        st_atr_len: int = 10,
    ):
        df = price_df.copy()
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)

        # ATR (Wilder's smoothing, matching Pine's ta.atr)
        high = df.get("high", df["close"])
        low = df.get("low", df["close"])
        close = df["close"]
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)

        atr_box = pine_rma(tr, atr_box_len)
        atr_st = pine_rma(tr, st_atr_len)
        box_size = atr_box * atr_box_mult

        brick_level = np.full(len(df), np.nan)
        brick_dir = np.zeros(len(df), dtype=int)
        brick_level[0] = close.iloc[0]
        for i in range(1, len(df)):
            prev_level = brick_level[i - 1]
            prev_dir = brick_dir[i - 1]
            bs = box_size.iloc[i]
            c = close.iloc[i]
            if pd.isna(bs) or bs == 0:
                brick_level[i] = prev_level
                brick_dir[i] = prev_dir
                continue
            if c > prev_level + bs:
                steps = np.floor((c - prev_level) / bs)
                brick_level[i] = prev_level + steps * bs
                brick_dir[i] = 1
            elif c < prev_level - bs:
                steps = np.floor((prev_level - c) / bs)
                brick_level[i] = prev_level - steps * bs
                brick_dir[i] = -1
            else:
                brick_level[i] = prev_level
                brick_dir[i] = prev_dir

        df["brick_level"] = brick_level
        df["brick_dir"] = brick_dir

        upper = df["brick_level"] - st_factor * atr_st
        lower = df["brick_level"] + st_factor * atr_st

        def _nz(x: float) -> float:
            """Replicates Pine's nz(): NaN -> 0. This matters because Python/
            numpy NaN comparisons always evaluate False, which would silently
            freeze the loop's 'carry forward previous value' branch forever
            once any NaN entered final_upper/final_lower (e.g. during the ATR
            warmup period) — Pine's nz() instead lets the comparison resolve
            against 0, so the bands correctly initialize the FIRST time the
            underlying ATR becomes valid, rather than getting stuck."""
            return 0.0 if pd.isna(x) else x

        final_upper = np.full(len(df), np.nan)
        final_lower = np.full(len(df), np.nan)
        trend_dir = np.zeros(len(df), dtype=int)

        for i in range(len(df)):
            u, l = upper.iloc[i], lower.iloc[i]
            src = df["brick_level"].iloc[i]
            prev_fu = _nz(final_upper[i - 1]) if i > 0 else 0.0
            prev_fl = _nz(final_lower[i - 1]) if i > 0 else 0.0
            prev_src = _nz(df["brick_level"].iloc[i - 1]) if i > 0 else 0.0
            prev_trend = trend_dir[i - 1] if i > 0 else 1  # matches Pine's `var int trendDir = 1`

            if pd.isna(u) or pd.isna(l):
                # ATR (and therefore the bands) not yet valid — matches Pine's
                # na-propagation through arithmetic. Bands stay na; trendDir
                # still evaluates via nz() fallback, same as Pine would.
                final_upper[i] = np.nan
                final_lower[i] = np.nan
            else:
                final_upper[i] = u if (u > prev_fu or prev_src < prev_fu) else prev_fu
                final_lower[i] = l if (l < prev_fl or prev_src > prev_fl) else prev_fl

            if src > prev_fl:
                trend_dir[i] = 1
            elif src < prev_fu:
                trend_dir[i] = -1
            else:
                trend_dir[i] = prev_trend

        df["trend_dir"] = trend_dir
        df["buy_signal"] = (df["trend_dir"] == 1) & (df["trend_dir"].shift(1) == -1)
        df["sell_signal"] = (df["trend_dir"] == -1) & (df["trend_dir"].shift(1) == 1)

        self.df = df

    def value(self, as_of: dt.datetime) -> int:
        """Returns current trend_dir (1 or -1) as of the nearest bar <= as_of."""
        eligible = self.df[self.df["datetime"] <= as_of]
        if eligible.empty:
            return 0
        return int(eligible.iloc[-1]["trend_dir"])

    def signal(self, as_of: dt.datetime) -> Optional[str]:
        """Returns 'buy', 'sell', or None if no fresh flip on this bar."""
        eligible = self.df[self.df["datetime"] <= as_of]
        if eligible.empty:
            return None
        row = eligible.iloc[-1]
        if row["buy_signal"]:
            return "buy"
        if row["sell_signal"]:
            return "sell"
        return None


# ─────────────────────────────────────────────
# ADJUSTMENT STRATEGIES
# ─────────────────────────────────────────────

def pine_supertrend(df: pd.DataFrame, factor: float, atr_period: int) -> pd.DataFrame:
    """Exact replica of Pine's built-in ta.supertrend(factor, atrPeriod).
    Documented Pine formula:
        src = hl2
        atr = ta.atr(atrPeriod)
        upperBand = src + factor*atr ; lowerBand = src - factor*atr
        lowerBand := lowerBand > prevLowerBand or close[1] < prevLowerBand ? lowerBand : prevLowerBand
        upperBand := upperBand < prevUpperBand or close[1] > prevUpperBand ? upperBand : prevUpperBand
        direction: if na(atr[1]) -> 1
                   elif prevSuperTrend == prevUpperBand -> (close > upperBand ? -1 : 1)
                   else -> (close < lowerBand ? 1 : -1)
        superTrend = direction == -1 ? lowerBand : upperBand

    NOTE the direction convention here is Pine's own, and it's the OPPOSITE
    of the custom Renko+SuperTrend indicator's convention: here -1 = bullish
    (uptrend), 1 = bearish (downtrend) — matches the pasted SAR strategy's
    stBullish = stDir == -1.

    This uses raw hl2 as the band source and standard (non-Renko) ATR —
    a genuinely different, simpler formula from the custom Renko-emulated
    version elsewhere in this file, not a variant of it.
    """
    high, low, close = df["high"], df["low"], df["close"]
    hl2 = (high + low) / 2
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = pine_rma(tr, atr_period)

    raw_upper = hl2 + factor * atr
    raw_lower = hl2 - factor * atr

    n = len(df)
    upper_band = np.full(n, np.nan)
    lower_band = np.full(n, np.nan)
    direction = np.zeros(n, dtype=int)
    supertrend = np.full(n, np.nan)

    for i in range(n):
        if pd.isna(atr.iloc[i]):
            # ATR not yet valid -- bands stay na, matching Pine's na-propagation
            continue

        prev_upper = upper_band[i - 1] if i > 0 else np.nan
        prev_lower = lower_band[i - 1] if i > 0 else np.nan
        prev_close_val = close.iloc[i - 1] if i > 0 else np.nan

        # lowerBand: only "loosen" toward the new raw value if it's rising,
        # or if price closed below the previous lower band (a real breach)
        if pd.isna(prev_lower):
            lower_band[i] = raw_lower.iloc[i]
        else:
            lower_band[i] = raw_lower.iloc[i] if (raw_lower.iloc[i] > prev_lower or prev_close_val < prev_lower) else prev_lower

        if pd.isna(prev_upper):
            upper_band[i] = raw_upper.iloc[i]
        else:
            upper_band[i] = raw_upper.iloc[i] if (raw_upper.iloc[i] < prev_upper or prev_close_val > prev_upper) else prev_upper

        prev_atr = atr.iloc[i - 1] if i > 0 else np.nan
        prev_supertrend = supertrend[i - 1] if i > 0 else np.nan

        if pd.isna(prev_atr):
            direction[i] = 1
        elif not pd.isna(prev_supertrend) and prev_supertrend == prev_upper:
            direction[i] = -1 if close.iloc[i] > upper_band[i] else 1
        else:
            direction[i] = 1 if close.iloc[i] < lower_band[i] else -1

        supertrend[i] = lower_band[i] if direction[i] == -1 else upper_band[i]

    result = df.copy()
    result["st_line"] = supertrend
    result["st_dir"] = direction
    return result


class SupertrendEMAIndicator(Indicator):
    """Python port of the pasted 'SAR Strategy - Supertrend + EMA' Pine
    script: standard (non-Renko) ta.supertrend(factor, atrPeriod) combined
    with a plain ta.ema(src, length), both computed on the SAME price series
    (pass 15-min data to match your RSI comparison). Entry logic (AND of
    both conditions), matching the pasted script exactly:
        buySignal  = (st_dir == -1) and (close > ema)
        sellSignal = (st_dir ==  1) and (close < ema)
    Defaults (factor=3.0, atr_period=10) match the pasted script's defaults.
    """
    def __init__(self, price_df: pd.DataFrame, ema_len: int = 21, factor: float = 3.0, atr_period: int = 10):
        df = price_df.copy()
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)
        if "high" not in df.columns:
            df["high"] = df["close"]
        if "low" not in df.columns:
            df["low"] = df["close"]

        df["ema"] = pine_ema(df["close"], ema_len)
        st_result = pine_supertrend(df, factor=factor, atr_period=atr_period)
        df["st_line"] = st_result["st_line"]
        df["st_dir"] = st_result["st_dir"]

        st_bullish = df["st_dir"] == -1
        st_bearish = df["st_dir"] == 1
        ema_bullish = df["close"] > df["ema"]
        ema_bearish = df["close"] < df["ema"]

        df["buy_signal"] = st_bullish & ema_bullish
        df["sell_signal"] = st_bearish & ema_bearish
        # matches the pasted script's `buySignal and not buySignal[1]` edge trigger.
        # IMPORTANT: .shift(1) introduces a leading NaN into a bool Series, which
        # silently upcasts it to object dtype — and ~ on object-dtype Python bools
        # does bitwise NOT on the underlying int (~True==-2, ~False==-1), both
        # truthy, which would silently break edge detection (it'd just mirror the
        # original signal). .astype(bool) after fillna forces back to real bool
        # dtype before negating, avoiding that trap.
        prev_buy = df["buy_signal"].shift(1).fillna(False).astype(bool)
        prev_sell = df["sell_signal"].shift(1).fillna(False).astype(bool)
        df["buy_signal_edge"] = df["buy_signal"] & ~prev_buy
        df["sell_signal_edge"] = df["sell_signal"] & ~prev_sell

        self.df = df

    def value(self, as_of: dt.datetime) -> Optional[str]:
        """Returns 'buy', 'sell', or None — the sustained state (not just
        the edge), i.e. whether entry conditions currently hold."""
        eligible = self.df[self.df["datetime"] <= as_of]
        if eligible.empty:
            return None
        row = eligible.iloc[-1]
        if row["buy_signal"]:
            return "buy"
        if row["sell_signal"]:
            return "sell"
        return None

    def signal_edge(self, as_of: dt.datetime) -> Optional[str]:
        """Returns 'buy'/'sell' only on the exact bar the condition first
        becomes true (matches the pasted script's plotshape edge trigger),
        None otherwise."""
        eligible = self.df[self.df["datetime"] <= as_of]
        if eligible.empty:
            return None
        row = eligible.iloc[-1]
        if row["buy_signal_edge"]:
            return "buy"
        if row["sell_signal_edge"]:
            return "sell"
        return None


class AdjustmentStrategy:
    """Base class. evaluate() inspects current positions + indicators and
    returns a list of Actions for the caller (the backtest loop) to execute.
    Strategies never touch data or execute trades directly — they only decide."""
    def evaluate(self, positions: dict[str, MultiLegPosition], as_of: dt.datetime) -> list[Action]:
        raise NotImplementedError


class DeltaThresholdStrategy(AdjustmentStrategy):
    """Strategy 1: recenter the whole sold straddle when either leg's |delta|
    reaches sold_threshold (0.65 default); recenter the whole hedge straddle
    when either leg's |delta| reaches hedge_threshold (0.85 default).
    Recenter = close all open legs of that position; the caller reopens
    fresh legs at the newly-resolved ATM strike."""

    def __init__(
        self,
        delta_indicators: dict[str, DeltaIndicator],  # keyed by leg tag
        sold_threshold: float = 0.65,
        hedge_threshold: float = 0.85,
    ):
        self.delta_indicators = delta_indicators
        self.sold_threshold = sold_threshold
        self.hedge_threshold = hedge_threshold

    def evaluate(self, positions: dict[str, MultiLegPosition], as_of: dt.datetime) -> list[Action]:
        actions = []
        for pos_name, threshold in [("sold_straddle", self.sold_threshold), ("hedge_straddle", self.hedge_threshold)]:
            pos = positions.get(pos_name)
            if pos is None or pos.is_flat():
                continue
            for leg in pos.open_legs():
                indicator = self.delta_indicators.get(leg.tag)
                if indicator is None:
                    continue
                delta = indicator.value(as_of)
                if delta >= threshold:
                    actions.append(Action(
                        type=ActionType.RECENTER, position_name=pos_name,
                        reason=f"{leg.tag} delta {delta:.3f} >= threshold {threshold}",
                    ))
                    break  # one recenter action covers the whole position
        return actions


class RSISignalStrategy(AdjustmentStrategy):
    """Strategy 2: close a sold leg when 15-min RSI (on that leg's OWN
    option-price series) shows a bullish cross (rising premium = losing
    money on a short) — re-enter only once that leg's RSI regime turns
    bearish again (premium trending down = safe to sell). Separately,
    once a sold leg's delta decays to <= harvest_threshold (0.2 default),
    close and immediately refresh it at the current ATM strike (no RSI
    gating — this is profit-harvesting a leg that's stopped contributing
    meaningful premium, not risk control)."""

    def __init__(
        self,
        rsi_indicators: dict[str, RSIIndicator],   # keyed by leg tag, 15-min data
        delta_indicators: dict[str, DeltaIndicator],
        harvest_delta_threshold: float = 0.2,
    ):
        self.rsi_indicators = rsi_indicators
        self.delta_indicators = delta_indicators
        self.harvest_delta_threshold = harvest_delta_threshold
        self._closed_awaiting_reentry: set[str] = set()  # leg tags closed on RSI signal, awaiting regime flip

    def evaluate(self, positions: dict[str, MultiLegPosition], as_of: dt.datetime) -> list[Action]:
        actions = []
        pos = positions.get("sold_straddle")
        if pos is None:
            return actions

        for leg in pos.open_legs():
            rsi_ind = self.rsi_indicators.get(leg.tag)
            delta_ind = self.delta_indicators.get(leg.tag)

            # Rule A: RSI-driven risk close (premium rising against a short)
            if rsi_ind is not None and rsi_ind.value(as_of) == RSISignal.BULLISH_CROSS:
                actions.append(Action(
                    type=ActionType.CLOSE_LEG, position_name="sold_straddle", leg_tag=leg.tag,
                    reason=f"{leg.tag}: 15-min RSI bullish cross (premium rising against short)",
                ))
                self._closed_awaiting_reentry.add(leg.tag)
                continue

            # Rule B: delta-driven harvest (leg decayed far OTM, refresh it)
            if delta_ind is not None and delta_ind.value(as_of) <= self.harvest_delta_threshold:
                actions.append(Action(
                    type=ActionType.RECENTER, position_name="sold_straddle",
                    leg_tag=leg.tag,
                    reason=f"{leg.tag}: delta decayed to <= {self.harvest_delta_threshold}, harvesting and refreshing at ATM",
                ))

        # Re-entry check for legs closed on Rule A, gated by RSI regime turning bearish again
        for tag in list(self._closed_awaiting_reentry):
            rsi_ind = self.rsi_indicators.get(tag)
            if rsi_ind is not None and rsi_ind.regime(as_of) == "bearish":
                actions.append(Action(
                    type=ActionType.OPEN_LEG, position_name="sold_straddle", leg_tag=tag,
                    reason=f"{tag}: RSI regime turned bearish again, safe to re-sell at current ATM",
                ))
                self._closed_awaiting_reentry.discard(tag)

        return actions


class FixedMoveStrategy(AdjustmentStrategy):
    """Strategy 3: recenter the sold straddle whenever the underlying has
    moved move_points OR move_pct (whichever fires first) from the level
    it was at when the straddle was last established/recentered."""

    def __init__(self, spot_lookup: Callable[[dt.datetime], float], move_points: float = 100, move_pct: float = 0.5):
        self.spot_lookup = spot_lookup
        self.move_points = move_points
        self.move_pct = move_pct
        self.reference_spot: Optional[float] = None

    def set_reference(self, spot: float) -> None:
        """Call this whenever the sold straddle is (re)established, so the
        move is measured from the correct baseline."""
        self.reference_spot = spot

    def evaluate(self, positions: dict[str, MultiLegPosition], as_of: dt.datetime) -> list[Action]:
        pos = positions.get("sold_straddle")
        if pos is None or pos.is_flat() or self.reference_spot is None:
            return []

        current_spot = self.spot_lookup(as_of)
        move = abs(current_spot - self.reference_spot)
        move_pct_actual = move / self.reference_spot * 100

        if move >= self.move_points or move_pct_actual >= self.move_pct:
            return [Action(
                type=ActionType.RECENTER, position_name="sold_straddle",
                reason=f"underlying moved {move:.1f}pts ({move_pct_actual:.2f}%) from reference {self.reference_spot}",
            )]
        return []


class DirectionalOverlayStrategy(AdjustmentStrategy):
    """Strategy 4: wraps a core_strategy (any of 1/2/3) for the sold_straddle
    position, and separately manages a directional 'overlay' position — a
    long ATM + long 75-delta option, same right, direction set by the 1-hour
    RSI regime on the underlying. Opens the overlay when flat and a regime
    is established; closes the whole overlay when the regime flips."""

    def __init__(self, core_strategy: AdjustmentStrategy, hourly_rsi: RSIIndicator):
        self.core_strategy = core_strategy
        self.hourly_rsi = hourly_rsi
        self.current_overlay_direction: Optional[str] = None  # 'call' or 'put'

    def evaluate(self, positions: dict[str, MultiLegPosition], as_of: dt.datetime) -> list[Action]:
        actions = self.core_strategy.evaluate(positions, as_of)

        regime = self.hourly_rsi.regime(as_of)  # 'bullish' / 'bearish' / 'neutral'
        desired_direction = "call" if regime == "bullish" else "put" if regime == "bearish" else None

        overlay = positions.get("directional_overlay")
        overlay_open = overlay is not None and not overlay.is_flat()

        if overlay_open and self.current_overlay_direction is not None and desired_direction != self.current_overlay_direction:
            actions.append(Action(
                type=ActionType.RECENTER, position_name="directional_overlay",
                reason=f"1hr RSI regime flipped from {self.current_overlay_direction} to {desired_direction}, closing overlay",
            ))
            self.current_overlay_direction = None

        elif not overlay_open and desired_direction is not None:
            actions.append(Action(
                type=ActionType.OPEN_LEG, position_name="directional_overlay",
                reason=f"1hr RSI regime is {regime}, opening {desired_direction} overlay (ATM + 75-delta)",
            ))
            self.current_overlay_direction = desired_direction

        return actions