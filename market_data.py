"""
Market data provider abstraction for the backtest engine.

The engine talks to ONE interface (MarketDataProvider) regardless of where
the data actually comes from. Two implementations:

  - SyntheticMarketDataProvider: generates a real geometric Brownian motion
    spot path and derives option prices via genuine Black-Scholes (reusing
    the already-validated pricing.py module) at a flat IV. This is NOT a
    realistic market simulation (no skew, no smile, no real order flow) —
    it exists purely so the ENGINE's mechanics (position tracking, action
    execution, equity curve construction) can be tested end-to-end with
    real math behind the numbers, without needing live Breeze credentials.

  - BreezeMarketDataProvider: wraps NiftyOptionsDataBreeze for real
    historical backtests. Structurally complete but not independently
    testable from this environment (no live Breeze access) — validate this
    one against your own account before trusting its output.
"""

from __future__ import annotations
import datetime as dt
from typing import Optional

import numpy as np
import pandas as pd

from pricing import solve_iv_and_greeks, time_to_expiry_years


class MarketDataProvider:
    def get_spot(self, as_of: dt.datetime) -> float:
        raise NotImplementedError

    def get_option_price(self, strike: float, right: str, expiry: dt.date, as_of: dt.datetime) -> float:
        raise NotImplementedError

    def find_atm_strike(self, expiry: dt.date, as_of: dt.datetime, strike_step: int = 100) -> int:
        raise NotImplementedError

    def spot_series(self, start: dt.datetime, end: dt.datetime, freq_minutes: int = 1) -> pd.DataFrame:
        """Returns a DataFrame with 'datetime','close' (and ideally
        'high'/'low') for building indicators over a historical window."""
        raise NotImplementedError

    def option_price_series(self, strike: float, right: str, expiry: dt.date, start: dt.datetime, end: dt.datetime, freq_minutes: int = 1) -> pd.DataFrame:
        """Same shape as spot_series, but for one specific option leg —
        used to build per-leg RSI indicators (strategy 2's per-leg signal)."""
        raise NotImplementedError


class SyntheticMarketDataProvider(MarketDataProvider):
    """GBM spot path + real Black-Scholes option pricing (flat IV, no
    skew/smile) — for testing the ENGINE, not for realistic strategy
    evaluation. NIFTY-like defaults: ~12% annualized vol, near-zero drift."""

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
        """All 1-min timestamps within NSE trading hours, weekdays only —
        does NOT account for market holidays (a real trading calendar would
        be needed for that; acceptable gap for engine-testing purposes)."""
        minutes = []
        cursor = start.date()
        while cursor <= end.date():
            if cursor.weekday() < 5:  # Mon-Fri
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
        dt_years = 1 / (252 * 375)  # one NSE trading minute as a fraction of a trading year
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
        if freq_minutes > 1:
            result = result.iloc[::freq_minutes].reset_index(drop=True)
        return result

    def option_price_series(self, strike: float, right: str, expiry: dt.date, start: dt.datetime, end: dt.datetime, freq_minutes: int = 1) -> pd.DataFrame:
        spot_slice = self.spot_series(start, end, freq_minutes)
        prices = [self.get_option_price(strike, right, expiry, ts) for ts in spot_slice["datetime"]]
        return pd.DataFrame({"datetime": spot_slice["datetime"], "close": prices})


class BreezeMarketDataProvider(MarketDataProvider):
    """Wraps NiftyOptionsDataBreeze for real historical data. Structurally
    complete but NOT independently testable from this environment — no live
    Breeze access here. Validate against your real account before trusting
    engine output built on this provider."""

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
