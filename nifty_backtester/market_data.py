"""
Market data provider abstraction for the backtest engine.
"""

from __future__ import annotations
import datetime as dt
from typing import Optional

import numpy as np
import pandas as pd

from .pricing import solve_iv_and_greeks, time_to_expiry_years

# How many calendar days BreezeMarketDataProvider will step BACKWARD (never
# forward -- see _most_recent_price_at_or_before) looking for a usable
# print when the requested day has none. Real-world boundary conditions
# this exists for:
#   1. The last intraday 1-minute candle of an NSE session is timestamped
#      15:29, or post-market closing/settlement prints occur up to 15:39/15:40.
#      When querying at market close (15:30:00), _most_recent_price_at_or_before
#      allows same-day prints up to 15:40 on that date so same-day closing/
#      settlement candles resolve correctly without triggering multi-day
#      lookback fallbacks.
#   2. A given strike genuinely may not trade at all on a given day (thin
#      OTM/ITM weeklies especially) -- Breeze then returns zero rows for
#      that whole day, not just a gap at one timestamp.
#   3. When stepping backward across weekends or holidays, stale warnings
#      report actual trading days stale alongside calendar days to avoid
#      misleading multi-day stale warnings across weekend gaps.
# All surface as boundary conditions, and all apply equally to a live/replay
# run (nifty_live.replay_feed and a real live session both go through this
# same provider) -- so the fix lives here once, not duplicated per caller.
DEFAULT_MAX_STALE_LOOKBACK_DAYS = 5


def _resample_ohlc(df: pd.DataFrame, freq_minutes: int) -> pd.DataFrame:
    """Proper OHLC resample (open=first, high=max, low=min, close=last,
    volume=sum), NOT naive row-skipping. Row-skipping (df.iloc[::N]) takes
    whatever close happens to sit at the FIRST minute of each N-minute
    window rather than the LAST -- close-only readers (RSI) would get a
    one-bar-early value, and any high/low information within the skipped
    minutes is silently discarded rather than folded into a true bar. This
    matters once real data (with genuine intra-window movement) replaces
    the synthetic provider's smoother path."""
    if df.empty or freq_minutes <= 1:
        return df.reset_index(drop=True)
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime")
    agg = {}
    if "open" in df.columns:
        agg["open"] = "first"
    if "high" in df.columns:
        agg["high"] = "max"
    if "low" in df.columns:
        agg["low"] = "min"
    if "close" in df.columns:
        agg["close"] = "last"
    if "volume" in df.columns:
        agg["volume"] = "sum"
    return df.resample(f"{freq_minutes}min").agg(agg).dropna(how="all").reset_index()


_HOLIDAYS_CACHE = None


def _get_holidays() -> set[dt.date]:
    global _HOLIDAYS_CACHE
    if _HOLIDAYS_CACHE is None:
        try:
            from .expiry_utils import load_holidays
            _HOLIDAYS_CACHE = load_holidays()
        except Exception:
            _HOLIDAYS_CACHE = set()
    return _HOLIDAYS_CACHE


def _count_trading_days_between(start_date: dt.date, end_date: dt.date) -> int:
    """Counts trading days strictly after start_date up to and including end_date."""
    from .expiry_utils import is_trading_day
    holidays = _get_holidays()
    count = 0
    cur = start_date + dt.timedelta(days=1)
    while cur <= end_date:
        if is_trading_day(cur, holidays):
            count += 1
        cur += dt.timedelta(days=1)
    return count


def _most_recent_price_at_or_before(
    fetch_fn, as_of: dt.datetime, label: str, max_lookback_days: int = DEFAULT_MAX_STALE_LOOKBACK_DAYS,
    daily_fallback_fn=None,
) -> float:
    """Returns the close of the most recent bar AT OR BEFORE as_of, never
    after (a bar timestamped after as_of is look-ahead, full stop, even if
    it happens to be numerically "nearest" -- the bug this replaces used
    nearest-by-absolute-distance, which could silently pick a future bar).

    fetch_fn(from_date, to_date) -> DataFrame with 'datetime'/'close' for
    ONE calendar day (from_date == to_date on every call this makes).
    Tries as_of's own date first; if that day has no bar at/before as_of
    (empty fetch, or every bar on that day is after as_of -- both are real
    situations, not just hypothetical: see DEFAULT_MAX_STALE_LOOKBACK_DAYS'
    docstring), walks backward one day at a time, up to max_lookback_days,
    and uses the most recent print it finds. Prints a warning whenever a
    non-same-day (stale) price gets used, so a backtest/live run's output
    stays auditable rather than silently treating a days-old print as
    fresh. Raises ValueError if nothing turns up within the lookback
    window at all.
    """
    for days_back in range(max_lookback_days + 1):
        query_date = as_of.date() - dt.timedelta(days=days_back)
        df = fetch_fn(query_date, query_date)
        if df is None or df.empty:
            if days_back == 0 and daily_fallback_fn is not None:
                try:
                    daily_df = daily_fallback_fn(as_of.date(), as_of.date())
                    if daily_df is not None and not daily_df.empty:
                        daily_df = daily_df.copy()
                        daily_df["datetime"] = pd.to_datetime(daily_df["datetime"])
                        close_price = float(daily_df.iloc[-1]["close"])
                        print(
                            f"[market_data] {label}: no 1-minute intraday bar at/before {as_of} on {as_of.date()} -- "
                            f"using official NSE daily closing price ({close_price:.2f}) from 1day candle for {as_of.date()}."
                        )
                        return close_price
                except Exception:
                    pass
            continue
        df = df.copy()
        df["datetime"] = pd.to_datetime(df["datetime"])

        if days_back == 0 and as_of.time() >= dt.time(15, 30):
            # Market close query on the same day: allow bars up to 15:40 on as_of.date()
            # to capture NSE post-market closing/settlement prints (e.g. 15:39:00)
            cutoff = max(as_of, dt.datetime.combine(as_of.date(), dt.time(15, 40)))
            df_filtered = df[df["datetime"] <= cutoff]
        elif days_back > 0:
            # Prior day fallback: any bar on that prior day is before as_of
            df_filtered = df[df["datetime"] <= dt.datetime.combine(query_date, dt.time(23, 59, 59))]
        else:
            df_filtered = df[df["datetime"] <= as_of]

        if df_filtered.empty:
            if days_back == 0 and daily_fallback_fn is not None:
                # Same-day intraday 1-min fetch had no bars at/before as_of.
                # Query official daily 1day candle for as_of.date() before stepping back to prior days.
                try:
                    daily_df = daily_fallback_fn(as_of.date(), as_of.date())
                    if daily_df is not None and not daily_df.empty:
                        daily_df = daily_df.copy()
                        daily_df["datetime"] = pd.to_datetime(daily_df["datetime"])
                        close_price = float(daily_df.iloc[-1]["close"])
                        print(
                            f"[market_data] {label}: no 1-minute intraday bar at/before {as_of} on {as_of.date()} -- "
                            f"using official NSE daily closing price ({close_price:.2f}) from 1day candle for {as_of.date()}."
                        )
                        return close_price
                except Exception:
                    pass
            continue

        idx = df_filtered["datetime"].idxmax()  # most recent print
        if days_back > 0:
            trading_stale = _count_trading_days_between(query_date, as_of.date())
            stale_str = f"{trading_stale} trading day(s) stale"
            if days_back != trading_stale:
                stale_str += f" ({days_back} calendar days)"
            print(
                f"[market_data] {label}: no usable bar at/before {as_of} on {as_of.date()} -- "
                f"falling back to the last available print from {query_date} "
                f"({df_filtered.loc[idx, 'datetime']}), {stale_str}. Treat this bar with "
                f"extra caution -- it reflects whatever the market last did on {query_date}, "
                f"not {as_of.date()}."
            )
        return float(df_filtered.loc[idx, "close"])

    raise ValueError(
        f"{label}: no usable data at or before {as_of}, even after looking back "
        f"{max_lookback_days} calendar day(s). Widen max_lookback_days if this is a genuinely "
        f"thin contract, or double-check the strike/expiry/date are actually valid."
    )


class MarketDataProvider:
    def get_spot(self, as_of: dt.datetime) -> float:
        raise NotImplementedError

    def get_option_price(self, strike: float, right: str, expiry: dt.date, as_of: dt.datetime) -> float:
        raise NotImplementedError

    def find_atm_strike(self, expiry: dt.date, as_of: dt.datetime, strike_step: int = 100) -> int:
        raise NotImplementedError

    def spot_series(self, start: dt.datetime, end: dt.datetime, freq_minutes: int = 1) -> pd.DataFrame:
        raise NotImplementedError

    def option_price_series(self, strike: float, right: str, expiry: dt.date, start: dt.datetime, end: dt.datetime, freq_minutes: int = 1) -> pd.DataFrame:
        raise NotImplementedError


class SyntheticMarketDataProvider(MarketDataProvider):
    MARKET_OPEN = dt.time(9, 15)
    MARKET_CLOSE = dt.time(15, 30)

    def __init__(
        self,
        start: dt.datetime,
        end: dt.datetime,
        initial_spot: float = 24500.0,
        annual_vol: float = 0.12,
        annual_drift: float = 0.0,
        flat_iv: float = 0.13,
        r: float = 0.0525,
        q: float = 0.012,
        seed: int = 42,
    ):
        self.flat_iv = flat_iv
        self.r = r
        self.q = q
        self._spot_path = self._generate_path(start, end, initial_spot, annual_vol, annual_drift, seed)

    def _trading_minutes(self, start: dt.datetime, end: dt.datetime) -> list[dt.datetime]:
        minutes = []
        cursor = start.date()
        while cursor <= end.date():
            if cursor.weekday() < 5:
                day_start = dt.datetime.combine(cursor, self.MARKET_OPEN)
                day_end = dt.datetime.combine(cursor, self.MARKET_CLOSE)
                t = max(day_start, start) if cursor == start.date() else day_start
                day_end_capped = min(day_end, end) if cursor == end.date() else day_end
                while t <= day_end_capped:
                    minutes.append(t)
                    t += dt.timedelta(minutes=1)
            cursor += dt.timedelta(days=1)
        return minutes

    def _generate_path(self, start, end, initial_spot, annual_vol, annual_drift, seed) -> pd.DataFrame:
        timestamps = self._trading_minutes(start, end)
        n = len(timestamps)
        if n == 0:
            return pd.DataFrame(columns=["datetime", "close", "high", "low"])

        rng = np.random.default_rng(seed)
        dt_years = 1 / (252 * 375)
        drift_term = (annual_drift - 0.5 * annual_vol ** 2) * dt_years
        vol_term = annual_vol * np.sqrt(dt_years)
        shocks = rng.normal(drift_term, vol_term, n)
        log_path = np.cumsum(shocks)
        prices = initial_spot * np.exp(log_path)

        return pd.DataFrame({
            "datetime": timestamps,
            "close": prices,
            "high": prices * (1 + rng.uniform(0, 0.0005, n)),
            "low": prices * (1 - rng.uniform(0, 0.0005, n)),
        })

    def get_spot(self, as_of: dt.datetime) -> float:
        eligible = self._spot_path[self._spot_path["datetime"] <= as_of]
        if eligible.empty:
            return float(self._spot_path["close"].iloc[0]) if not self._spot_path.empty else np.nan
        return float(eligible.iloc[-1]["close"])

    def get_option_price(self, strike: float, right: str, expiry: dt.date, as_of: dt.datetime) -> float:
        spot = self.get_spot(as_of)
        T = time_to_expiry_years(as_of, expiry)
        if T <= 0:
            intrinsic = max(spot - strike, 0) if right.lower().startswith("c") else max(strike - spot, 0)
            return intrinsic
        from vollib.black_scholes_merton import black_scholes_merton as bsm_price
        flag = "c" if right.lower().startswith("c") else "p"
        return float(bsm_price(flag, spot, strike, T, self.r, self.flat_iv, self.q))

    def find_atm_strike(self, expiry: dt.date, as_of: dt.datetime, strike_step: int = 100) -> int:
        spot = self.get_spot(as_of)
        return int(round(spot / strike_step) * strike_step)

    def spot_series(self, start: dt.datetime, end: dt.datetime, freq_minutes: int = 1) -> pd.DataFrame:
        mask = (self._spot_path["datetime"] >= start) & (self._spot_path["datetime"] <= end)
        result = self._spot_path[mask].reset_index(drop=True)
        return _resample_ohlc(result, freq_minutes)

    def option_price_series(self, strike: float, right: str, expiry: dt.date, start: dt.datetime, end: dt.datetime, freq_minutes: int = 1) -> pd.DataFrame:
        # option prices are computed at every 1-min underlying timestamp
        # first, THEN resampled -- resampling the underlying and repricing
        # only at the resampled timestamps would silently discard whatever
        # intra-window high/low the option itself reached.
        spot_slice = self.spot_series(start, end, freq_minutes=1)
        prices = [self.get_option_price(strike, right, expiry, ts) for ts in spot_slice["datetime"]]
        raw = pd.DataFrame({"datetime": spot_slice["datetime"], "close": prices})
        return _resample_ohlc(raw, freq_minutes)


class BreezeMarketDataProvider(MarketDataProvider):
    def __init__(self, breeze_data_layer, r: float = 0.0525, q: float = 0.012,
                 max_stale_lookback_days: int = DEFAULT_MAX_STALE_LOOKBACK_DAYS):
        self.data = breeze_data_layer
        self.r = r
        self.q = q
        self.max_stale_lookback_days = max_stale_lookback_days

    def get_spot(self, as_of: dt.datetime) -> float:
        def fetch_fn(from_date, to_date):
            return self.data.get_index_historical(from_date, to_date, interval="1minute")
        def daily_fn(from_date, to_date):
            return self.data.get_index_historical(from_date, to_date, interval="1day")
        return _most_recent_price_at_or_before(fetch_fn, as_of, "spot", self.max_stale_lookback_days, daily_fallback_fn=daily_fn)

    def get_option_price(self, strike: float, right: str, expiry: dt.date, as_of: dt.datetime) -> float:
        def fetch_fn(from_date, to_date):
            return self.data.get_option_historical(expiry, int(strike), right.lower(), from_date, to_date, interval="1minute")
        def daily_fn(from_date, to_date):
            return self.data.get_option_historical(expiry, int(strike), right.lower(), from_date, to_date, interval="1day")
        label = f"strike={strike} right={right} expiry={expiry}"
        return _most_recent_price_at_or_before(fetch_fn, as_of, label, self.max_stale_lookback_days, daily_fallback_fn=daily_fn)

    def find_atm_strike(self, expiry: dt.date, as_of: dt.datetime, strike_step: int = 100) -> int:
        spot = self.get_spot(as_of)
        result = self.data.find_atm_strike(expiry, spot, as_of, strike_step=strike_step)
        return int(result["strike"])

    def spot_series(self, start: dt.datetime, end: dt.datetime, freq_minutes: int = 1) -> pd.DataFrame:
        # Always fetch at native 1-minute resolution and combine into proper
        # OHLC bars locally -- matches SyntheticMarketDataProvider's behavior
        # (see _resample_ohlc's docstring for why naive row-skipping is wrong:
        # it takes whichever close sits at the FIRST minute of each window
        # rather than the LAST, and silently discards intra-window high/low).
        # This previously used df.iloc[::freq_minutes], the exact bug that was
        # fixed for the synthetic provider but never carried over here.
        df = self.data.get_index_historical(start.date(), end.date(), interval="1minute")
        df["datetime"] = pd.to_datetime(df["datetime"])
        mask = (df["datetime"] >= start) & (df["datetime"] <= end)
        result = df[mask].reset_index(drop=True)
        return _resample_ohlc(result, freq_minutes)

    def option_price_series(self, strike: float, right: str, expiry: dt.date, start: dt.datetime, end: dt.datetime, freq_minutes: int = 1) -> pd.DataFrame:
        # Same fix as spot_series above -- fetch native 1-minute option bars,
        # then combine into a proper OHLC bar at whatever timeframe the
        # caller needs (15-min sold-leg signal, 1hr overlay direction, etc.)
        # rather than sampling a single minute out of each window.
        df = self.data.get_option_historical(expiry, int(strike), right.lower(), start.date(), end.date(), interval="1minute")
        df["datetime"] = pd.to_datetime(df["datetime"])
        mask = (df["datetime"] >= start) & (df["datetime"] <= end)
        result = df[mask].reset_index(drop=True)
        return _resample_ohlc(result, freq_minutes)
