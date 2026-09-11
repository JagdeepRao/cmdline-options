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

DATA SOURCE (--data-source, default "auto" -- see nifty_backtester.data_sources):
  auto -> LIVE Breeze if credentials are set; else CACHED (real_data_cache/,
  committed real data, no live session needed) if anything's committed;
  else SYNTHETIC. Force one explicitly with --data-source {breeze,cached,synthetic}.

Examples (run from the repo root):
  # Real data, delta-threshold sold straddle only (2026-09-08 must exist in expiry_calendar.csv)
  export BREEZE_API_KEY=... BREEZE_API_SECRET=... BREEZE_SESSION_TOKEN=...
  python3 scripts/download_and_run_strategy.py --from-date 2026-09-04 --to-date 2026-09-08 \\
      --weekly-expiry 2026-09-08 --sold-leg-strategy delta_threshold

  # Real data, RSI sold straddle + hedge + overlay + 2x OTM
  python3 scripts/download_and_run_strategy.py --from-date 2026-09-04 --to-date 2026-09-08 \\
      --weekly-expiry 2026-09-08 --sold-leg-strategy rsi_signal \\
      --include-hedge --monthly-expiry 2026-09-29 \\
      --include-overlay --overlay-kind rsi \\
      --include-otm --otm-multiplier 2

  # No live credentials, but real_data_cache/ has committed data for this range
  python3 scripts/download_and_run_strategy.py --from-date 2026-09-04 --to-date 2026-09-08 \\
      --weekly-expiry 2026-09-08 --data-source cached

  # No credentials, nothing committed -> runs against synthetic data automatically
  python3 scripts/download_and_run_strategy.py --from-date 2026-09-04 --to-date 2026-09-08 --weekly-expiry 2026-09-08

  # Expiry date not in your calendar yet? Bypass it for one run:
  python3 scripts/download_and_run_strategy.py --from-date 2026-09-04 --to-date 2026-09-08 \\
      --weekly-expiry 2026-09-08 --weekly-expiry-prior-trading-day 2026-09-04
"""

import argparse
import datetime as dt
from pathlib import Path

import pandas as pd

from nifty_backtester.backtest_engine import FullBacktestConfig, run_full_backtest
from nifty_backtester.market_data import BreezeMarketDataProvider
from nifty_backtester.expiry_utils import load_expiry_calendar, get_prior_trading_day_for_expiry
from nifty_backtester.data_sources import resolve_data_layer, VALID_SOURCES
from nifty_backtester import metrics

OUTPUT_DIR = Path("./downloaded_samples")
OUTPUT_DIR.mkdir(exist_ok=True)


def get_provider(initial_spot: float, prefer: str = "auto"):
    """BreezeMarketDataProvider works with ANY object exposing
    get_index_historical/get_option_historical/find_atm_strike --
    NiftyOptionsDataBreeze (real), NiftyOptionsDataCached (committed real
    snapshot), and NiftyOptionsDataSample (synthetic) all satisfy that, so
    the same provider class wraps whichever one resolve_data_layer picks."""
    data_layer, source_label = resolve_data_layer(initial_spot=initial_spot, prefer=prefer)
    return BreezeMarketDataProvider(data_layer), source_label


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--to-date", required=True, help="YYYY-MM-DD (the weekly expiry date itself is fine -- expiry-eve close fires the day before)")
    parser.add_argument("--weekly-expiry", required=True, help="YYYY-MM-DD -- must exist in expiry_calendar.csv unless --weekly-expiry-prior-trading-day is also given")
    parser.add_argument("--weekly-expiry-prior-trading-day", default=None, help="override the calendar lookup for --weekly-expiry")
    parser.add_argument("--monthly-expiry", default=None, help="YYYY-MM-DD, required with --include-hedge")
    parser.add_argument("--monthly-expiry-prior-trading-day", default=None, help="override the calendar lookup for --monthly-expiry")

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
    parser.add_argument("--data-source", default="auto", choices=VALID_SOURCES, help="auto (default) picks LIVE > CACHED > SYNTHETIC; see this script's docstring")

    args = parser.parse_args()

    if args.include_hedge and not args.monthly_expiry:
        parser.error("--include-hedge requires --monthly-expiry")

    start = dt.datetime.strptime(args.from_date, "%Y-%m-%d").replace(hour=9, minute=15)
    end = dt.datetime.strptime(args.to_date, "%Y-%m-%d").replace(hour=15, minute=30)
    weekly_expiry = dt.datetime.strptime(args.weekly_expiry, "%Y-%m-%d").date()
    monthly_expiry = dt.datetime.strptime(args.monthly_expiry, "%Y-%m-%d").date() if args.monthly_expiry else None

    if args.weekly_expiry_prior_trading_day:
        weekly_prior = dt.datetime.strptime(args.weekly_expiry_prior_trading_day, "%Y-%m-%d").date()
    else:
        weekly_prior = get_prior_trading_day_for_expiry(load_expiry_calendar(), weekly_expiry, "weekly")

    monthly_prior = None
    if monthly_expiry is not None:
        if args.monthly_expiry_prior_trading_day:
            monthly_prior = dt.datetime.strptime(args.monthly_expiry_prior_trading_day, "%Y-%m-%d").date()
        else:
            monthly_prior = get_prior_trading_day_for_expiry(load_expiry_calendar(), monthly_expiry, "monthly")

    provider, source_label = get_provider(args.initial_spot, prefer=args.data_source)

    config = FullBacktestConfig(
        start=start, end=end, weekly_expiry=weekly_expiry, weekly_expiry_prior_trading_day=weekly_prior,
        bar_freq_minutes=args.bar_freq_minutes,
        sold_leg_strategy_name=args.sold_leg_strategy,
        include_hedge_straddle=args.include_hedge, monthly_expiry=monthly_expiry,
        monthly_expiry_prior_trading_day=monthly_prior,
        include_overlay=args.include_overlay,
        overlay_hourly_kind=args.overlay_kind, overlay_gating_kind=args.overlay_kind,
        include_otm=args.include_otm, otm_multiplier=args.otm_multiplier,
    )

    result = run_full_backtest(provider, config)
    report = metrics.full_report(result.equity_curve, result.trade_pnls, config.initial_capital)

    tag = source_label
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
