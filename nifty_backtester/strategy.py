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
        from .pricing import solve_iv_and_greeks
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
            """Replicates Pine's nz(): NaN -> 0."""
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
    """Exact replica of Pine's built-in ta.supertrend(factor, atrPeriod)."""
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
            continue

        prev_upper = upper_band[i - 1] if i > 0 else np.nan
        prev_lower = lower_band[i - 1] if i > 0 else np.nan
        prev_close_val = close.iloc[i - 1] if i > 0 else np.nan

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
    """Standard (non-Renko) ta.supertrend(factor, atrPeriod) combined with
    a plain ta.ema(src, length), both computed on the same price series."""
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
        prev_buy = df["buy_signal"].shift(1).fillna(False).astype(bool)
        prev_sell = df["sell_signal"].shift(1).fillna(False).astype(bool)
        df["buy_signal_edge"] = df["buy_signal"] & ~prev_buy
        df["sell_signal_edge"] = df["sell_signal"] & ~prev_sell

        self.df = df

    def value(self, as_of: dt.datetime) -> Optional[str]:
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
    """Strategy 1: recenter a WHOLE position when any of its legs' |delta|
    reaches that position's threshold. Defaults cover the two positions
    from the original spec (sold_straddle @ 0.65, hedge_straddle @ 0.85),
    but any other whole-position-recenter target (e.g. a directional
    overlay's short leg, managed 'like the straddle legs' per point 2) can
    be added via extra_thresholds without a new class."""

    def __init__(
        self,
        delta_indicators: dict[str, DeltaIndicator],  # keyed by leg tag, across ALL positions this instance watches
        sold_threshold: float = 0.65,
        hedge_threshold: float = 0.85,
        extra_thresholds: dict[str, float] = None,  # {position_name: threshold} for positions beyond the two defaults
    ):
        self.delta_indicators = delta_indicators
        self.sold_threshold = sold_threshold
        self.hedge_threshold = hedge_threshold
        self.thresholds = {"sold_straddle": sold_threshold, "hedge_straddle": hedge_threshold}
        if extra_thresholds:
            self.thresholds.update(extra_thresholds)

    def evaluate(self, positions: dict[str, MultiLegPosition], as_of: dt.datetime) -> list[Action]:
        actions = []
        for pos_name, threshold in self.thresholds.items():
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
                    break
        return actions


class SoldLegSignalAdapter:
    """Uniform interface so RSIIndicator and SupertrendEMAIndicator can be
    swapped interchangeably as the management signal for a SOLD leg — this
    is what lets you A/B 'RSI crossover' vs 'Supertrend + EMA' on the sold
    side (point 6/4) without duplicating SoldLegSignalStrategy's logic.

    crossed_bullish(as_of): True on the exact bar the signal first turns
    against a short (premium starting to rise) — the CLOSE trigger.
    regime(as_of): 'bullish' / 'bearish' / 'neutral' sustained state — used
    to gate re-entry (only re-sell once the regime favors selling again).
    """
    def crossed_bullish(self, as_of: dt.datetime) -> bool:
        raise NotImplementedError

    def regime(self, as_of: dt.datetime) -> str:
        raise NotImplementedError


class RSICrossAdapter(SoldLegSignalAdapter):
    def __init__(self, rsi_indicator: RSIIndicator):
        self.ind = rsi_indicator

    def crossed_bullish(self, as_of: dt.datetime) -> bool:
        return self.ind.value(as_of) == RSISignal.BULLISH_CROSS

    def regime(self, as_of: dt.datetime) -> str:
        return self.ind.regime(as_of)


class SupertrendEMAAdapter(SoldLegSignalAdapter):
    def __init__(self, st_indicator: SupertrendEMAIndicator):
        self.ind = st_indicator

    def crossed_bullish(self, as_of: dt.datetime) -> bool:
        return self.ind.signal_edge(as_of) == "buy"

    def regime(self, as_of: dt.datetime) -> str:
        state = self.ind.value(as_of)
        if state == "buy":
            return "bullish"
        if state == "sell":
            return "bearish"
        return "neutral"


class SoldLegSignalStrategy(AdjustmentStrategy):
    """Generalized Strategy 2: closes a SOLD leg when its management signal
    turns against the short (premium rising), gated by whichever indicator
    adapter is supplied per leg — RSICrossAdapter or SupertrendEMAAdapter,
    so the two can be compared head-to-head on identical sold legs. Re-opens
    only once that leg's signal regime turns bearish again (safe to re-sell).
    Independently, harvests-and-refreshes any leg whose delta has decayed to
    <= harvest_delta_threshold (0.2 default), regardless of signal type.

    position_name defaults to 'sold_straddle' but any sold position (e.g.
    the sold call/put inside a directional overlay spread, point 6d) can be
    managed by its own instance keyed to that position's leg tags.
    """

    def __init__(
        self,
        signal_adapters: dict[str, SoldLegSignalAdapter],
        delta_indicators: dict[str, DeltaIndicator],
        harvest_delta_threshold: float = 0.2,
        position_name: str = "sold_straddle",
    ):
        self.signal_adapters = signal_adapters
        self.delta_indicators = delta_indicators
        self.harvest_delta_threshold = harvest_delta_threshold
        self.position_name = position_name
        self._closed_awaiting_reentry: set[str] = set()

    def evaluate(self, positions: dict[str, MultiLegPosition], as_of: dt.datetime) -> list[Action]:
        actions = []
        pos = positions.get(self.position_name)
        if pos is None:
            return actions

        for leg in pos.open_legs():
            adapter = self.signal_adapters.get(leg.tag)
            delta_ind = self.delta_indicators.get(leg.tag)

            if adapter is not None and adapter.crossed_bullish(as_of):
                actions.append(Action(
                    type=ActionType.CLOSE_LEG, position_name=self.position_name, leg_tag=leg.tag,
                    reason=f"{leg.tag}: management signal turned against the short (premium rising)",
                ))
                self._closed_awaiting_reentry.add(leg.tag)
                continue

            if delta_ind is not None and delta_ind.value(as_of) <= self.harvest_delta_threshold:
                actions.append(Action(
                    type=ActionType.RECENTER, position_name=self.position_name,
                    leg_tag=leg.tag,
                    reason=f"{leg.tag}: delta decayed to <= {self.harvest_delta_threshold}, harvesting and refreshing at ATM",
                ))

        for tag in list(self._closed_awaiting_reentry):
            adapter = self.signal_adapters.get(tag)
            if adapter is not None and adapter.regime(as_of) == "bearish":
                actions.append(Action(
                    type=ActionType.OPEN_LEG, position_name=self.position_name, leg_tag=tag,
                    reason=f"{tag}: management signal regime turned bearish again, safe to re-sell at current ATM",
                ))
                self._closed_awaiting_reentry.discard(tag)

        return actions


class RSISignalStrategy(SoldLegSignalStrategy):
    """Backward-compatible entry point: same as SoldLegSignalStrategy but
    takes raw RSIIndicator objects directly (auto-wrapped in RSICrossAdapter)
    so existing callers (e.g. backtest_engine.py's 'rsi_signal' branch) don't
    need to change. New code comparing RSI vs Supertrend+EMA should
    construct SoldLegSignalStrategy directly with the adapter it wants.

    IMPORTANT for callers doing per-leg indicator rebuilds on recenter
    (backtest_engine.py does this after every strike change): use
    `set_leg_indicator(tag, new_rsi_indicator)` rather than mutating
    `.rsi_indicators[tag]` directly — direct dict mutation would update the
    raw-indicator bookkeeping dict without re-wrapping it into
    `.signal_adapters`, which is what `evaluate()` actually reads.
    """

    def __init__(
        self,
        rsi_indicators: dict[str, RSIIndicator],
        delta_indicators: dict[str, DeltaIndicator],
        harvest_delta_threshold: float = 0.2,
        position_name: str = "sold_straddle",
    ):
        super().__init__(
            signal_adapters={tag: RSICrossAdapter(ind) for tag, ind in rsi_indicators.items()},
            delta_indicators=delta_indicators,
            harvest_delta_threshold=harvest_delta_threshold,
            position_name=position_name,
        )
        self.rsi_indicators = dict(rsi_indicators)

    def set_leg_indicator(self, tag: str, rsi_indicator: RSIIndicator) -> None:
        """Rebuild both the raw-indicator bookkeeping and the adapter
        evaluate() actually reads, for one leg (e.g. after it recenters to
        a new strike). Prefer this over mutating .rsi_indicators directly."""
        self.rsi_indicators[tag] = rsi_indicator
        self.signal_adapters[tag] = RSICrossAdapter(rsi_indicator)


class SupertrendEMASoldLegStrategy(SoldLegSignalStrategy):
    """Same as RSISignalStrategy above, but for the Supertrend+EMA
    management signal instead of RSI crossover — construct one or the
    other (same delta-harvest and re-entry-gating behaviour either way) to
    compare them on identical sold legs."""

    def __init__(
        self,
        st_indicators: dict[str, SupertrendEMAIndicator],
        delta_indicators: dict[str, DeltaIndicator],
        harvest_delta_threshold: float = 0.2,
        position_name: str = "sold_straddle",
    ):
        super().__init__(
            signal_adapters={tag: SupertrendEMAAdapter(ind) for tag, ind in st_indicators.items()},
            delta_indicators=delta_indicators,
            harvest_delta_threshold=harvest_delta_threshold,
            position_name=position_name,
        )
        self.st_indicators = dict(st_indicators)

    def set_leg_indicator(self, tag: str, st_indicator: SupertrendEMAIndicator) -> None:
        self.st_indicators[tag] = st_indicator
        self.signal_adapters[tag] = SupertrendEMAAdapter(st_indicator)


class FixedMoveStrategy(AdjustmentStrategy):
    """Strategy 3: recenter a whole position (sold_straddle by default, but
    any whole-position-recenter target works, e.g. a directional overlay's
    short leg) whenever the underlying has moved move_points OR move_pct
    (whichever fires first) from the level it was at when that position was
    last established/recentered."""

    def __init__(
        self,
        spot_lookup: Callable[[dt.datetime], float],
        move_points: float = 100,
        move_pct: float = 0.5,
        position_name: str = "sold_straddle",
    ):
        self.spot_lookup = spot_lookup
        self.move_points = move_points
        self.move_pct = move_pct
        self.position_name = position_name
        self.reference_spot: Optional[float] = None

    def set_reference(self, spot: float) -> None:
        self.reference_spot = spot

    def evaluate(self, positions: dict[str, MultiLegPosition], as_of: dt.datetime) -> list[Action]:
        pos = positions.get(self.position_name)
        if pos is None or pos.is_flat() or self.reference_spot is None:
            return []

        current_spot = self.spot_lookup(as_of)
        move = abs(current_spot - self.reference_spot)
        move_pct_actual = move / self.reference_spot * 100

        if move >= self.move_points or move_pct_actual >= self.move_pct:
            return [Action(
                type=ActionType.RECENTER, position_name=self.position_name,
                reason=f"underlying moved {move:.1f}pts ({move_pct_actual:.2f}%) from reference {self.reference_spot}",
            )]
        return []


def nearest_just_otm_strikes(spot: float, strike_step: int = 100) -> tuple[int, int]:
    """Nearest OTM call strike (smallest multiple of strike_step strictly
    above spot) and nearest OTM put strike (largest multiple strictly below).
    Duplicated here (rather than imported from the data layer) so strategy.py
    has zero dependency on breeze_connect / the live data layer."""
    import math
    call_strike = math.floor(spot / strike_step) * strike_step + strike_step
    put_strike = math.ceil(spot / strike_step) * strike_step - strike_step
    return int(call_strike), int(put_strike)


class RenkoLegTracker(AdjustmentStrategy):
    """One independent OTM leg tracker for a single side ('call' or 'put') —
    opens its leg when its entry signal fires, closes it when its exit
    signal fires. Deliberately NOT coupled to the opposite side: nothing
    here prevents a call tracker and a put tracker from both being open at
    the same time.

    With both trackers currently reading the SAME underlying Renko trend
    (desired_trend=1 for call, -1 for put), only one side's condition can
    be true at once in practice, so you'll observe single-leg-at-a-time
    behavior today. That's a property of what's fed in, not something this
    class enforces — point this at two independent signals later (e.g. a
    Renko computed on that leg's own option price, or any other
    per-leg-specific read) and overlapping call+put legs falls out with
    zero changes here.

    DELIBERATELY OUT OF SCOPE FOR NOW (per your last message): pyramiding
    further same-side entries as the underlying keeps moving and the
    'just OTM' strike rolls further away (e.g. adding a second, further-OTM
    put as the market keeps dropping). This tracker holds at most one leg
    per side; layering multiple concurrent same-side legs is a real
    extension (raises questions like whether an exit signal closes all of
    them at once or oldest-first) worth deciding deliberately rather than
    bolting on here.
    """

    def __init__(
        self,
        renko_indicator: RenkoSuperTrendIndicator,
        right: str,  # 'call' or 'put'
        position_name: str = "otm_position",
    ):
        self.renko_indicator = renko_indicator
        self.right = right
        self.leg_tag = f"otm_{right}"
        self.position_name = position_name
        self.desired_trend = 1 if right == "call" else -1

    def evaluate(self, positions: dict[str, MultiLegPosition], as_of: dt.datetime) -> list[Action]:
        trend = self.renko_indicator.value(as_of)
        pos = positions.get(self.position_name)
        leg_open = pos is not None and pos.get_leg(self.leg_tag) is not None

        if not leg_open and trend == self.desired_trend:
            direction_word = "bullish" if self.desired_trend == 1 else "bearish"
            return [Action(
                type=ActionType.OPEN_LEG, position_name=self.position_name, leg_tag=self.leg_tag,
                reason=f"Renko trend turned {direction_word} — entering just-OTM {self.right}",
            )]

        if leg_open and trend != self.desired_trend:
            return [Action(
                type=ActionType.CLOSE_LEG, position_name=self.position_name, leg_tag=self.leg_tag,
                reason=f"Renko trend left the {self.right} side — exiting just-OTM {self.right}",
            )]

        return []


class RenkoOpportunisticOTMStrategy(AdjustmentStrategy):
    """Composes one RenkoLegTracker per side (call, put) against a single
    'otm_position' — the entry point backtest_engine.py wires up. Each side
    enters/exits independently per RenkoLegTracker's own signal; this class
    just runs both and merges their actions, so it holds whatever
    combination of legs (zero, call-only, put-only, or eventually both) the
    two trackers independently decide on.

    Sizing (lot_size * otm_multiplier) is decided by the caller when it
    actually opens a leg from an OPEN_LEG action — this strategy only ever
    decides direction/timing, so the same class serves every point on the
    1x-5x sizing sweep.
    """

    def __init__(
        self,
        renko_indicator: RenkoSuperTrendIndicator,
        position_name: str = "otm_position",
    ):
        self.call_tracker = RenkoLegTracker(renko_indicator, "call", position_name)
        self.put_tracker = RenkoLegTracker(renko_indicator, "put", position_name)

    def evaluate(self, positions: dict[str, MultiLegPosition], as_of: dt.datetime) -> list[Action]:
        return self.call_tracker.evaluate(positions, as_of) + self.put_tracker.evaluate(positions, as_of)


class DirectionalOverlayStrategy(AdjustmentStrategy):
    """Strategy 4 / point 2: wraps a core_strategy (any of 1/2/3, or a
    SoldLegSignalStrategy) for the sold_straddle position, and separately
    manages a directional 'directional_overlay' position — a long 75-delta
    call/put plus a short ATM call/put on the same side (the short leg
    hedges/finances the long leg, per your description).

    Direction comes from a 1hr regime signal on the underlying (an
    RSICrossAdapter or SupertrendEMAAdapter built on 1hr data — reused here
    since both already expose the same regime() interface used elsewhere).
    On a flip, EVERY currently-open leg of the overlay (long and/or short,
    whichever exist) is closed, and a fresh long 75-delta leg opens
    immediately in the new direction.

    The short ATM leg is NOT opened alongside the long leg automatically —
    it's entered only once short_leg_strategy's own signal says selling is
    safe (the same 15-min gating logic used for sold_straddle's re-entries),
    then managed exactly like a sold_straddle leg from then on (close on
    adverse cross, harvest at delta<=0.2, re-enter once safe again). This
    class achieves that by registering the new direction's short-leg tag
    into short_leg_strategy's own re-entry-awaiting set — reusing that
    machinery rather than duplicating the gating logic here.

    CONFIRMED (not an open assumption): the long 75-delta leg runs
    unhedged for however long the 15-min signal takes to permit selling the
    ATM leg — this is the intended behavior, verified against the actual
    requirement. One consequence to keep in mind when reading results: if
    the 1hr regime flips again before the 15-min signal ever turns
    favorable, that cycle's spread will have run long-only for its entire
    life, with the short leg never entered at all.
    """

    def __init__(
        self,
        core_strategy: AdjustmentStrategy,
        hourly_regime_signal: SoldLegSignalAdapter,  # built on 1hr underlying data
        short_leg_strategy: SoldLegSignalStrategy,   # pre-built, position_name="directional_overlay"
        position_name: str = "directional_overlay",
    ):
        self.core_strategy = core_strategy
        self.hourly_regime_signal = hourly_regime_signal
        self.short_leg_strategy = short_leg_strategy
        self.position_name = position_name
        self.current_direction: Optional[str] = None  # 'call' / 'put' / None

    def evaluate(self, positions: dict[str, MultiLegPosition], as_of: dt.datetime) -> list[Action]:
        actions = self.core_strategy.evaluate(positions, as_of)

        regime = self.hourly_regime_signal.regime(as_of)
        desired = "call" if regime == "bullish" else "put" if regime == "bearish" else None

        pos = positions.get(self.position_name)
        pos_open = pos is not None and not pos.is_flat()

        if pos_open and self.current_direction is not None and desired != self.current_direction:
            for leg in pos.open_legs():
                actions.append(Action(
                    type=ActionType.CLOSE_LEG, position_name=self.position_name, leg_tag=leg.tag,
                    reason=f"1hr regime flipped from {self.current_direction} to {desired or 'neutral'} — closing overlay leg {leg.tag}",
                ))
            # clear any pending gated-entry state for the side we're leaving
            self.short_leg_strategy._closed_awaiting_reentry.discard(f"overlay_short_{self.current_direction}")
            self.current_direction = None
            pos_open = False

        if not pos_open and desired is not None:
            long_tag = f"overlay_long_{desired}"
            actions.append(Action(
                type=ActionType.OPEN_LEG, position_name=self.position_name, leg_tag=long_tag,
                reason=f"1hr regime turned {regime} — opening long 75-delta {desired}",
            ))
            self.current_direction = desired
            # register the short leg as "awaiting reentry" so short_leg_strategy
            # opens it the moment its own 15-min signal says selling is safe,
            # rather than opening it unconditionally right now
            self.short_leg_strategy._closed_awaiting_reentry.add(f"overlay_short_{desired}")

        if self.current_direction is not None:
            actions.extend(self.short_leg_strategy.evaluate(positions, as_of))

        return actions
