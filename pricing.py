"""
Pricing / Greeks module for the NIFTY options backtester.

Uses `vollib` (the actively-maintained successor to the deprecated
py_vollib) for Black-Scholes-Merton pricing, implied volatility, and
analytical Greeks. All function signatures below were confirmed by
inspecting the installed vollib source and test-running them with real
numbers (round-trip check: solved IV correctly reprices to the input
premium) — not guessed from memory.

Install: pip install vollib pandas --break-system-packages

CONFIRMED vollib signatures used here:
  black_scholes_merton(flag, S, K, t, r, sigma, q)      -> price
  implied_volatility(price, S, K, t, r, q, flag)         -> sigma   (note: arg order differs from the others)
  delta / gamma / theta / vega(flag, S, K, t, r, sigma, q)
  flag: 'c' for call, 'p' for put

NIFTY-SPECIFIC NOTES (per your correction — weekly options have no
matching weekly futures contract, so spot must be the index itself,
not futures, unlike monthly-expiry Greeks which conventionally use futures):
  - S here should always be NIFTY index spot, fetched via
    NiftyOptionsDataBreeze.get_index_historical() from the data layer.

NOT independently confirmed / left as adjustable inputs:
  - Risk-free rate (r) and NIFTY's continuous dividend yield (q) — these
    drift over time and I have no live source to pull current values from
    this environment. Defaults below are placeholders; pass your own
    current figures rather than trusting them.
"""

import datetime as dt
import pandas as pd

from vollib.black_scholes_merton import black_scholes_merton as bsm_price
from vollib.black_scholes_merton.implied_volatility import implied_volatility as bsm_iv
from vollib.black_scholes_merton.greeks.analytical import delta as bsm_delta
from vollib.black_scholes_merton.greeks.analytical import gamma as bsm_gamma
from vollib.black_scholes_merton.greeks.analytical import theta as bsm_theta
from vollib.black_scholes_merton.greeks.analytical import vega as bsm_vega

# Placeholders — override with real current values, not confirmed live from here.
DEFAULT_RISK_FREE_RATE = 0.065
DEFAULT_DIVIDEND_YIELD = 0.012

MARKET_CLOSE_TIME = dt.time(15, 30)  # NSE close, used as the expiry cutoff moment


def _right_to_flag(right: str) -> str:
    """Breeze's 'right' field uses 'Call'/'Put' — convert to vollib's 'c'/'p'."""
    r = right.strip().lower()
    if r.startswith("c"):
        return "c"
    if r.startswith("p"):
        return "p"
    raise ValueError(f"Unrecognized option right: {right}")


def time_to_expiry_years(now: dt.datetime, expiry_date: dt.date, expiry_time: dt.time = MARKET_CLOSE_TIME) -> float:
    """Fractional years to expiry, computed down to the minute (calendar-time
    basis, not trading-time). Precision matters most in the final day or two,
    where gamma/theta both accelerate sharply — a date-only T would understate
    how little time is actually left on expiry morning vs. expiry eve."""
    expiry_dt = dt.datetime.combine(expiry_date, expiry_time)
    seconds_remaining = (expiry_dt - now).total_seconds()
    if seconds_remaining <= 0:
        return 0.0
    return seconds_remaining / (365 * 24 * 3600)


def solve_iv_and_greeks(
    price: float,
    spot: float,
    strike: float,
    now: dt.datetime,
    expiry_date: dt.date,
    right: str,
    r: float = DEFAULT_RISK_FREE_RATE,
    q: float = DEFAULT_DIVIDEND_YIELD,
) -> dict:
    """Given one option's real traded price, backs out IV, then computes
    delta/gamma/theta/vega at that IV. Returns NaNs (instead of raising) for
    rows where T is zero/negative or the solver fails on a degenerate quote,
    so a bad row doesn't crash a whole DataFrame pass."""
    flag = _right_to_flag(right)
    T = time_to_expiry_years(now, expiry_date)

    if T <= 0 or price <= 0 or spot <= 0:
        return {"iv": float("nan"), "delta": float("nan"), "gamma": float("nan"),
                "theta": float("nan"), "vega": float("nan"), "T": T}

    try:
        iv = bsm_iv(price, spot, strike, T, r, q, flag)
        d = bsm_delta(flag, spot, strike, T, r, iv, q)
        g = bsm_gamma(flag, spot, strike, T, r, iv, q)
        th = bsm_theta(flag, spot, strike, T, r, iv, q)
        v = bsm_vega(flag, spot, strike, T, r, iv, q)
        return {"iv": iv, "delta": d, "gamma": g, "theta": th, "vega": v, "T": T}
    except Exception:
        return {"iv": float("nan"), "delta": float("nan"), "gamma": float("nan"),
                "theta": float("nan"), "vega": float("nan"), "T": T}


def add_greeks_to_option_df(
    option_df: pd.DataFrame,
    spot_df: pd.DataFrame,
    r: float = DEFAULT_RISK_FREE_RATE,
    q: float = DEFAULT_DIVIDEND_YIELD,
) -> pd.DataFrame:
    """Merges option historical data (from get_option_historical) with spot
    index data (from get_index_historical) by nearest matching timestamp,
    then computes IV/Greeks for every row.

    option_df columns expected (confirmed from a real Breeze pull): close,
    datetime, expiry_date (string, format 'DD-MON-YYYY'), right, strike_price.
    spot_df columns expected: close, datetime (same shape, from the index call).

    NOTE: row-by-row via DataFrame.apply — fine for backtests spanning weeks
    of 1-min data (tens of thousands of rows), but if you push to months of
    1-min data across many strikes, this will be the slow part. Worth revisiting
    with vectorization or multiprocessing if it becomes a bottleneck — not
    optimized preemptively here since premature optimization on unconfirmed
    data volumes isn't worth the added complexity yet.
    """
    if option_df.empty:
        return option_df

    opt = option_df.copy()
    opt["datetime"] = pd.to_datetime(opt["datetime"])
    opt = opt.sort_values("datetime")

    spot = spot_df.copy()
    spot["datetime"] = pd.to_datetime(spot["datetime"])
    spot = spot.sort_values("datetime")[["datetime", "close"]].rename(columns={"close": "spot_close"})

    merged = pd.merge_asof(opt, spot, on="datetime", direction="nearest")

    # Breeze's expiry_date format confirmed from a real pull: 'DD-MON-YYYY' (e.g. '08-SEP-2026')
    merged["expiry_date_parsed"] = pd.to_datetime(merged["expiry_date"], format="%d-%b-%Y").dt.date

    results = merged.apply(
        lambda row: solve_iv_and_greeks(
            price=row["close"],
            spot=row["spot_close"],
            strike=row["strike_price"],
            now=row["datetime"].to_pydatetime(),
            expiry_date=row["expiry_date_parsed"],
            right=row["right"],
            r=r,
            q=q,
        ),
        axis=1,
        result_type="expand",
    )

    return pd.concat([merged, results], axis=1)


if __name__ == "__main__":
    # Sanity check using the actual real data point confirmed earlier in this
    # project — replace the placeholder spot with the real index close at
    # that exact timestamp once you have it, to get a trustworthy IV.
    result = solve_iv_and_greeks(
        price=5.25,
        spot=24537.0,  # placeholder — pull the real spot at 2026-09-02 11:50 for a trustworthy result
        strike=24600.0,
        now=dt.datetime(2026, 9, 2, 11, 50),
        expiry_date=dt.date(2026, 9, 8),
        right="Call",
    )
    print(result)
