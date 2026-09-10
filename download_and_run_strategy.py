"""
Runs a full strategy (any combination of position shapes from
backtest_engine.FullBacktestConfig) against real historical data -- or the
synthetic fallback when Breeze credentials aren't set -- and prints every
open/close/recenter decision with its timestamp and reason, so you can
check each adjustment against what the underlying/option prices were
actually doing at that moment.

This is the strategy-level counterpart to download_option_data.py's
indicator-level validation: download_option_data.py lets you eyeball
whether a single indicator's BUY/SELL calls look right on real bars; this
script lets you eyeball whether the resulting POSITION ADJUSTMENTS (recenter,
close-and-reopen, harvest, overlay flip, OTM entry/exit) look right given
those same underlying signals.

DATA SOURCE: same convention as download_option_data.py -- uses live
Breeze if BREEZE_API_KEY / BREEZE_API_SECRET / BREEZE_SESSION_TOKEN are all
set, else falls back to the synthetic sample data layer (clearly labeled).

Examples:
  # Real data, delta-threshold sold straddle only
  export BREEZE_API_KEY=... BREEZE_API_SECRET=... BREEZE_SESSION_TOKEN=...
  python3 download_and_run_strategy.py --from-date 2026-09-01 --to-date 2026-09-04 \\
      --weekly-expiry 2026-09-04 --sold-leg-strategy delta_threshold

  # Real data, RSI sold straddle + hedge + overlay + 2x OTM
  python3 download_and_run_strategy.py --from-date 2026-09-01 --to-date 2026-09-04 \\
      --weekly-expiry 2026-09-04 --sold-leg-strategy rsi_signal \\
      --include-hedge --monthly-expiry 2026-09-24 \\
      --include-overlay --overlay-kind rsi \\
      --include-otm --otm-multiplier 2

  # No credentials set -> runs against synthetic data automatically
  python3 download_and_run_strategy.py --from-date 2026-09-01 --to-date 2026-09-04 --weekly-expiry 2026-09-04
"""

import os
import argparse
import datetime as dt
from pathlib import Path

import pandas as pd

from backtest_engine import FullBacktestConfig, run_full_backtest
from market_data import BreezeMarketDataProvider
import metrics

OUTPUT_DIR = Path("./downloaded_samples")
OUTPUT_DIR.mkdir(exist_ok=True)


def get_provider(initial_spot: float):
    """BreezeMarketDataProvider works with ANY object exposing
    get_index_historical/get_option_historical/find_atm_strike -- both
    NiftyOptionsDataBreeze (real) and NiftyOptionsDataSample (synthetic)
    satisfy that, so the same provider class wraps either one."""
    api_key = os.environ.get("BREEZE_API_KEY")
    api_secret = os.environ.get("BREEZE_API_SECRET")
    session_token = os.environ.get("BREEZE_SESSION_TOKEN")
    if api_key and api_secret and session_token:
        from data_layer_breeze import NiftyOptionsDataBreeze
        print("Using LIVE Breeze data.")
        data_layer = NiftyOptionsDataBreeze(api_key, api_secret, session_token)
        return BreezeMarketDataProvider(data_layer), False
    from data_layer_sample import NiftyOptionsDataSample
    print("BREEZE_API_KEY / BREEZE_API_SECRET / BREEZE_SESSION_TOKEN not all set "
          "-- falling back to SYNTHETIC sample data. Set those three env vars "
          "to validate against real Breeze data instead.")
    data_layer = NiftyOptionsDataSample(index_start_price=initial_spot)
    return BreezeMarketDataProvider(data_layer), True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--to-date", required=True, help="YYYY-MM-DD (the weekly expiry date itself is fine -- expiry-eve close fires the day before)")
    parser.add_argument("--weekly-expiry", required=True, help="YYYY-MM-DD")
    parser.add_argument("--monthly-expiry", default=None, help="YYYY-MM-DD, required with --include-hedge")

    parser.add_argument("--sold-leg-strategy", default="delta_threshold",
                         choices=["delta_threshold", "fixed_move", "rsi_signal", "supertrend_ema_signal"])
    parser.add_argument("--include-hedge", action="store_true")
    parser.add_argument("--include-overlay", action="store_true")
    parser.add_argument("--overlay-kind", default="rsi", choices=["rsi", "supertrend_ema"],
                         help="used for BOTH the 1hr direction signal and the 15-min short-leg gating signal")
    parser.add_argument("--include-otm", action="store_true")
    parser.add_argument("--otm-multiplier", type=int, default=1)

    parser.add_argument("--bar-freq-minutes", type=int, default=15)
    parser.add_argument("--initial-spot", type=float, default=24500.0, help="only affects the synthetic fallback")
    parser.add_argument("--out-prefix", default=None)

    args = parser.parse_args()

    if args.include_hedge and not args.monthly_expiry:
        parser.error("--include-hedge requires --monthly-expiry")

    start = dt.datetime.strptime(args.from_date, "%Y-%m-%d").replace(hour=9, minute=15)
    end = dt.datetime.strptime(args.to_date, "%Y-%m-%d").replace(hour=15, minute=30)
    weekly_expiry = dt.datetime.strptime(args.weekly_expiry, "%Y-%m-%d").date()
    monthly_expiry = dt.datetime.strptime(args.monthly_expiry, "%Y-%m-%d").date() if args.monthly_expiry else None

    provider, is_synthetic = get_provider(args.initial_spot)

    config = FullBacktestConfig(
        start=start, end=end, weekly_expiry=weekly_expiry, bar_freq_minutes=args.bar_freq_minutes,
        sold_leg_strategy_name=args.sold_leg_strategy,
        include_hedge_straddle=args.include_hedge, monthly_expiry=monthly_expiry,
        include_overlay=args.include_overlay,
        overlay_hourly_kind=args.overlay_kind, overlay_gating_kind=args.overlay_kind,
        include_otm=args.include_otm, otm_multiplier=args.otm_multiplier,
    )

    result = run_full_backtest(provider, config)
    report = metrics.full_report(result.equity_curve, result.trade_pnls, config.initial_capital)

    tag = "SYNTHETIC" if is_synthetic else "LIVE"
    summary = (f"{args.sold_leg_strategy}"
               f"{' + hedge' if args.include_hedge else ''}"
               f"{f' + overlay({args.overlay_kind})' if args.include_overlay else ''}"
               f"{f' + OTM x{args.otm_multiplier}' if args.include_otm else ''}")
    print(f"\n[{tag}] {summary}\n")

    print("Action log -- every open/close/recenter decision, with the reason that triggered it:")
    if result.action_log:
        for line in result.action_log:
            print(" ", line)
    else:
        print("  (no adjustments fired in this window -- either the price action never crossed a "
              "threshold, or the window is too short for the 15-min/1hr/1min indicators to warm up)")

    print("\nMetrics:")
    for k, v in report.items():
        print(f"  {k}: {v}")

    label = args.out_prefix or f"strategy_run_{args.sold_leg_strategy}_{tag}"
    action_log_path = OUTPUT_DIR / f"{label}_actions.csv"
    equity_path = OUTPUT_DIR / f"{label}_equity.csv"

    pd.DataFrame({"log_line": result.action_log}).to_csv(action_log_path, index=False)
    result.equity_curve.rename("equity").to_csv(equity_path)
    print(f"\nSaved action log to {action_log_path}")
    print(f"Saved equity curve to {equity_path}")


if __name__ == "__main__":
    main()
