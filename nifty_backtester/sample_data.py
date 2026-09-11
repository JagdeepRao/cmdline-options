"""
Deterministic synthetic sample data — used in place of live Breeze calls so
that data_cache, pricing, and strategy logic can all be exercised by tests
(and by anyone without live Breeze credentials) without hitting the network.

This does NOT try to be a realistic options-pricing simulator. It generates
plausible-shaped OHLCV bars (a mildly mean-reverting random walk for the
index, and a rough intrinsic-plus-decaying-time-value shape for options)
using a fixed random seed so runs are reproducible. Column names match the
CONFIRMED real Breeze schema documented in data_layer_breeze.py, so any code
written against real Breeze output should work unmodified against this.

Usage as a data_cache fetch_fn (see test_data_cache.py for the full pattern):

    from functools import partial
    from sample_data import generate_index_bars

    fetch_fn = partial(generate_index_bars, interval="1minute", start_price=24800)
    df = cache.get("SAMPLE_NIFTY_INDEX", from_date, to_date, fetch_fn)
"""

import datetime as dt
import numpy as np
import pandas as pd

_INTERVAL_MINUTES = {
    "1minute": 1,
    "5minute": 5,
    "15minute": 15,
    "30minute": 30,
    "1day": 24 * 60,
}

MARKET_OPEN = dt.time(9, 15)
MARKET_CLOSE = dt.time(15, 30)


def _trading_timestamps(from_date: dt.date, to_date: dt.date, interval: str) -> list[dt.datetime]:
    """NSE trading-session timestamps (9:15-15:30, Mon-Fri, no holiday
    calendar applied — good enough for synthetic/test data) at the given
    interval, across [from_date, to_date] inclusive."""
    step_minutes = _INTERVAL_MINUTES.get(interval, 1)
    timestamps = []
    day = from_date
    while day <= to_date:
        if day.weekday() < 5:  # Mon-Fri
            if interval == "1day":
                timestamps.append(dt.datetime.combine(day, MARKET_OPEN))
            else:
                cursor = dt.datetime.combine(day, MARKET_OPEN)
                end = dt.datetime.combine(day, MARKET_CLOSE)
                while cursor <= end:
                    timestamps.append(cursor)
                    cursor += dt.timedelta(minutes=step_minutes)
        day += dt.timedelta(days=1)
    return timestamps


def _seed_for(*parts) -> int:
    """Stable seed derived from the call's identifying parameters, so the
    same (strike, right, expiry, date-range) always generates the same
    bars — reproducible across test runs without sharing global state."""
    return abs(hash(tuple(str(p) for p in parts))) % (2**32)


def generate_index_bars(
    from_date: dt.date,
    to_date: dt.date,
    interval: str = "1minute",
    start_price: float = 24800.0,
    annual_vol: float = 0.13,
) -> pd.DataFrame:
    """Synthetic NIFTY index OHLCV bars. Mildly mean-reverting geometric
    random walk — enough shape for RSI/Supertrend/Renko indicators to
    produce non-degenerate signals, not a claim of real market realism."""
    timestamps = _trading_timestamps(from_date, to_date, interval)
    if not timestamps:
        return pd.DataFrame()

    rng = np.random.default_rng(_seed_for("index", from_date, to_date, interval))
    step_minutes = _INTERVAL_MINUTES.get(interval, 1)
    dt_years = step_minutes / (60 * 24 * 365)
    sigma = annual_vol * np.sqrt(dt_years)

    n = len(timestamps)
    shocks = rng.normal(0, sigma, n)
    # gentle mean reversion toward start_price so a long backtest window
    # doesn't wander off to an absurd level
    log_price = np.log(start_price)
    closes = np.empty(n)
    for i in range(n):
        log_price += shocks[i] - 0.02 * (log_price - np.log(start_price))
        closes[i] = np.exp(log_price)

    opens = np.concatenate([[start_price], closes[:-1]])
    highs = np.maximum(opens, closes) * (1 + rng.uniform(0, 0.0006, n))
    lows = np.minimum(opens, closes) * (1 - rng.uniform(0, 0.0006, n))
    volumes = rng.integers(1_000_000, 5_000_000, n)

    return pd.DataFrame({
        "datetime": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
        "open_interest": 0,
        "stock_code": "NIFTY",
        "exchange_code": "NSE",
        "product_type": "cash",
    })


def generate_option_bars(
    from_date: dt.date,
    to_date: dt.date,
    expiry: dt.date,
    strike: float,
    right: str,
    interval: str = "1minute",
    index_spot_at_start: float = 24800.0,
    annual_vol: float = 0.14,
) -> pd.DataFrame:
    """Synthetic option OHLCV bars for one (expiry, strike, right). Prices
    a rough intrinsic + decaying-extrinsic shape against a synthetic
    underlying path (same random-walk model as generate_index_bars, keyed
    off the same seed inputs so repeated calls for the same contract are
    internally consistent), clipped to a small positive floor rather than
    letting deep-OTM contracts go to exactly zero (real quotes rarely print
    literal 0)."""
    timestamps = _trading_timestamps(from_date, to_date, interval)
    if not timestamps:
        return pd.DataFrame()

    underlying = generate_index_bars(
        from_date, to_date, interval=interval, start_price=index_spot_at_start, annual_vol=annual_vol,
    )
    if underlying.empty:
        return pd.DataFrame()
    spot_path = underlying.set_index("datetime")["close"].reindex(timestamps).ffill().bfill()

    rng = np.random.default_rng(_seed_for("option", expiry, strike, right, from_date, to_date, interval))
    n = len(timestamps)

    days_to_expiry = np.array([
        max((expiry - ts.date()).days, 0) + (1 if ts.time() < MARKET_CLOSE else 0)
        for ts in timestamps
    ])
    time_value_frac = np.clip(days_to_expiry / max(days_to_expiry.max(), 1), 0.02, 1.0)

    sign = 1 if right.lower().startswith("c") else -1
    intrinsic = np.maximum(sign * (spot_path.values - strike), 0)
    atm_extrinsic = 0.012 * spot_path.values * annual_vol * time_value_frac
    noise = rng.normal(1.0, 0.05, n)
    closes = np.maximum((intrinsic + atm_extrinsic) * noise, 0.05)

    opens = np.concatenate([[closes[0]], closes[:-1]])
    highs = np.maximum(opens, closes) * (1 + rng.uniform(0, 0.01, n))
    lows = np.maximum(np.minimum(opens, closes) * (1 - rng.uniform(0, 0.01, n)), 0.05)
    volumes = rng.integers(0, 50_000, n)

    return pd.DataFrame({
        "datetime": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
        "open_interest": rng.integers(0, 200_000, n),
        "stock_code": "NIFTY",
        "exchange_code": "NFO",
        "product_type": "options",
        "expiry_date": expiry.strftime("%d-%b-%Y").upper(),
        "right": "Call" if sign == 1 else "Put",
        "strike_price": strike,
    })
