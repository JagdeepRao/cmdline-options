"""
Market data provider abstraction for the backtest engine.
"""

from __future__ import annotations
import datetime as dt
from typing import Optional

import numpy as np
import pandas as pd

from pricing import solve_iv_and_greeks, time_to_expiry_years


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
    def __init__(self, breeze_data_layer, r: float = 0.0525, q: float = 0.012):
        self.data = breeze_data_layer
        self.r = r
        self.q = q

    def get_spot(self, as_of: dt.datetime) -> float:
        window_start = as_of - dt.timedelta(minutes=5)
        df = self.data.get_index_historical(window_start.date(), as_of.date(), interval="1minute")
        if df.empty:
            raise ValueError(f"No spot data available around {as_of}")
        df["datetime"] = pd.to_datetime(df["datetime"])
        idx = (df["datetime"] - as_of).abs().idxmin()
        return float(df.loc[idx, "close"])

    def get_option_price(self, strike: float, right: str, expiry: dt.date, as_of: dt.datetime) -> float:
        window_start = as_of - dt.timedelta(minutes=5)
        df = self.data.get_option_historical(expiry, int(strike), right.lower(), window_start.date(), as_of.date(), interval="1minute")
        if df.empty:
            raise ValueError(f"No option data available for strike={strike} right={right} expiry={expiry} around {as_of}")
        df["datetime"] = pd.to_datetime(df["datetime"])
        idx = (df["datetime"] - as_of).abs().idxmin()
        return float(df.loc[idx, "close"])

    def find_atm_strike(self, expiry: dt.date, as_of: dt.datetime, strike_step: int = 100) -> int:
        spot = self.get_spot(as_of)
        result = self.data.find_atm_strike(expiry, spot, as_of, strike_step=strike_step)
        return int(result["strike"])

    def spot_series(self, start: dt.datetime, end: dt.datetime, freq_minutes: int = 1) -> pd.DataFrame:
        df = self.data.get_index_historical(start.date(), end.date(), interval="1minute")
        df["datetime"] = pd.to_datetime(df["datetime"])
        mask = (df["datetime"] >= start) & (df["datetime"] <= end)
        result = df[mask].reset_index(drop=True)
        if freq_minutes > 1:
            result = result.iloc[::freq_minutes].reset_index(drop=True)
        return result

    def option_price_series(self, strike: float, right: str, expiry: dt.date, start: dt.datetime, end: dt.datetime, freq_minutes: int = 1) -> pd.DataFrame:
        df = self.data.get_option_historical(expiry, int(strike), right.lower(), start.date(), end.date(), interval="1minute")
        df["datetime"] = pd.to_datetime(df["datetime"])
        mask = (df["datetime"] >= start) & (df["datetime"] <= end)
        result = df[mask].reset_index(drop=True)
        if freq_minutes > 1:
            result = result.iloc[::freq_minutes].reset_index(drop=True)
        return result
