"""
Runs every management shape (and every position-combination) through
run_full_backtest() against SyntheticMarketDataProvider, then computes
metrics.full_report() on each — this is the actual proof that "we can run
all strategies and get metrics" rather than an assertion about it.

Uses a short, single-week synthetic window (matches run_full_backtest's
stated single-expiry-cycle scope) with real Black-Scholes pricing (vollib),
not live Breeze data — see STRATEGY.md for how to swap in
BreezeMarketDataProvider once credentials are available.
"""

import datetime as dt
import warnings
import pandas as pd

from market_data import SyntheticMarketDataProvider
from backtest_engine import FullBacktestConfig, run_full_backtest
import metrics

# vollib divides by zero internally for a handful of near-degenerate quotes
# (e.g. price ~0 far OTM) -- solve_iv_and_greeks already catches this and
# returns NaN for that row rather than crashing, so the warning is noise,
# not a sign anything here is actually wrong.
warnings.filterwarnings("ignore", message="divide by zero encountered")
warnings.filterwarnings("ignore", message="invalid value encountered")

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)

START = dt.datetime(2026, 9, 1, 9, 15)
END = dt.datetime(2026, 9, 4, 15, 30)     # Tue-Fri, before the Fri weekly expiry
WEEKLY_EXPIRY = dt.date(2026, 9, 4)       # Friday -- expiry eve is Thursday 9/3
MONTHLY_EXPIRY = dt.date(2026, 9, 24)     # comfortably >15 days from any bar in this window

REPORT_COLUMNS = [
    "total_return", "total_return_pct", "max_drawdown", "max_drawdown_pct",
    "sharpe", "sortino", "calmar", "num_trades", "win_rate", "profit_factor",
]


def _provider():
    # fresh provider per run, same seed -> identical underlying path across
    # all strategy comparisons, so differences in the table are due to the
    # management strategy, not to different random market paths.
    # seed=42 (rather than the first seed tried) is used deliberately: it's
    # confirmed (see conversation) to be one where the RSI and Supertrend+EMA
    # signals actually diverge at least once, so the "only" rows below
    # demonstrate a genuine difference rather than coincidentally identical
    # runs where neither indicator's crossover ever fired.
    return SyntheticMarketDataProvider(
        START - dt.timedelta(days=6), END,  # extra lookback so 15-min/1hr/1min indicators can warm up
        initial_spot=24500, annual_vol=0.35, flat_iv=0.13, seed=42,
    )


def run_one(label: str, config: FullBacktestConfig) -> dict:
    provider = _provider()
    result = run_full_backtest(provider, config)
    report = metrics.full_report(result.equity_curve, result.trade_pnls, config.initial_capital)
    report["label"] = label
    report["num_actions_logged"] = len(result.action_log)
    return report


def main():
    rows = []

    # --- 1. Sold straddle only, each of the four management shapes ---
    base_kwargs = dict(start=START, end=END, weekly_expiry=WEEKLY_EXPIRY, bar_freq_minutes=15)

    rows.append(run_one("sold_straddle only -- delta_threshold",
                         FullBacktestConfig(**base_kwargs, sold_leg_strategy_name="delta_threshold")))
    rows.append(run_one("sold_straddle only -- fixed_move",
                         FullBacktestConfig(**base_kwargs, sold_leg_strategy_name="fixed_move")))
    rows.append(run_one("sold_straddle only -- rsi_signal",
                         FullBacktestConfig(**base_kwargs, sold_leg_strategy_name="rsi_signal")))
    rows.append(run_one("sold_straddle only -- supertrend_ema_signal",
                         FullBacktestConfig(**base_kwargs, sold_leg_strategy_name="supertrend_ema_signal")))

    # --- 2. + monthly hedge straddle ---
    rows.append(run_one("sold + hedge straddle -- delta_threshold",
                         FullBacktestConfig(**base_kwargs, sold_leg_strategy_name="delta_threshold",
                                             include_hedge_straddle=True, monthly_expiry=MONTHLY_EXPIRY)))

    # --- 3. + directional overlay (needs an indicator-capable core to make sense, but works with any) ---
    rows.append(run_one("sold straddle + directional overlay (RSI both signals)",
                         FullBacktestConfig(**base_kwargs, sold_leg_strategy_name="rsi_signal",
                                             include_overlay=True, overlay_hourly_kind="rsi", overlay_gating_kind="rsi")))
    rows.append(run_one("sold straddle + directional overlay (Supertrend+EMA both signals)",
                         FullBacktestConfig(**base_kwargs, sold_leg_strategy_name="supertrend_ema_signal",
                                             include_overlay=True, overlay_hourly_kind="supertrend_ema", overlay_gating_kind="supertrend_ema")))

    # --- 4. + opportunistic OTM, at a couple of multipliers ---
    for mult in (1, 3, 5):
        rows.append(run_one(f"sold straddle + opportunistic OTM (multiplier={mult}x)",
                             FullBacktestConfig(**base_kwargs, sold_leg_strategy_name="delta_threshold",
                                                 include_otm=True, otm_multiplier=mult)))

    # --- 5. everything combined ---
    rows.append(run_one("EVERYTHING: sold + hedge + overlay + OTM (delta_threshold core, RSI overlay, 2x OTM)",
                         FullBacktestConfig(**base_kwargs, sold_leg_strategy_name="delta_threshold",
                                             include_hedge_straddle=True, monthly_expiry=MONTHLY_EXPIRY,
                                             include_overlay=True, overlay_hourly_kind="rsi", overlay_gating_kind="rsi",
                                             include_otm=True, otm_multiplier=2)))

    df = pd.DataFrame(rows).set_index("label")
    print(df[REPORT_COLUMNS + ["num_actions_logged"]].round(3).to_string())
    print(f"\nAll {len(rows)} configurations ran to completion and produced a full metrics report.")


if __name__ == "__main__":
    main()
