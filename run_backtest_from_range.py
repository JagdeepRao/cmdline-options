"""
Give this a date range; it runs the full set of strategy/position
combinations against that range and hands back a metrics comparison table.

Weekly expiry and the monthly hedge expiry are both looked up from
expiry_calendar.csv (via expiry_utils) rather than computed from a
day-of-week rule -- NIFTY's weekly expiry day has changed exchange-side
before (e.g. away from Thursday), so "next Thursday" is exactly the kind
of assumption that silently breaks. Update expiry_calendar.csv with the
real NSE/broker calendar; override --weekly-expiry/--monthly-expiry
explicitly if you need to bypass the calendar for a specific run.

DATA SOURCE: every price fetch goes through NiftyOptionsDataBreeze, which
routes through DataCache (file-based, retries on throttling) -- so a
second run over an overlapping range re-fetches only what's missing, not
everything. Uses live Breeze if BREEZE_API_KEY / BREEZE_API_SECRET /
BREEZE_SESSION_TOKEN are all set as environment variables; otherwise falls
back to the synthetic sample data layer, clearly labeled as such, so the
script itself is testable without live credentials.

Usage (on a machine with real Breeze credentials):
  export BREEZE_API_KEY="..."
  export BREEZE_API_SECRET="..."
  export BREEZE_SESSION_TOKEN="..."
  python3 run_backtest_from_range.py --from-date 2026-09-01 --to-date 2026-09-04

That's it -- weekly and monthly hedge expiries are both looked up from
expiry_calendar.csv automatically. Add --skip-combinations to run just the
four sold-leg shapes (faster) instead of the full 11-row comparison
including hedge/overlay/OTM.
"""

import os
import argparse
import datetime as dt
from pathlib import Path

import pandas as pd

from backtest_engine import FullBacktestConfig, run_full_backtest
from market_data import BreezeMarketDataProvider
from expiry_utils import load_expiry_calendar, get_next_expiry, select_monthly_hedge_expiry_from_calendar
import metrics

OUTPUT_DIR = Path("./downloaded_samples")
OUTPUT_DIR.mkdir(exist_ok=True)

REPORT_COLUMNS = [
    "total_return", "total_return_pct", "max_drawdown", "max_drawdown_pct",
    "sharpe", "sortino", "calmar", "num_trades", "win_rate", "profit_factor",
]


# ─────────────────────────────────────────────
# DATA SOURCE
# ─────────────────────────────────────────────

def get_provider(initial_spot: float):
    """Real Breeze (through DataCache) if credentials are set, else the
    synthetic fallback -- returns (provider, is_synthetic). A FRESH data
    layer is constructed per call deliberately: NiftyOptionsDataBreeze
    holds its own DataCache instance, and every fetch through it is what
    persists to disk -- repeated calls across configs in the same run
    share the same on-disk cache files regardless."""
    api_key = os.environ.get("BREEZE_API_KEY")
    api_secret = os.environ.get("BREEZE_API_SECRET")
    session_token = os.environ.get("BREEZE_SESSION_TOKEN")
    if api_key and api_secret and session_token:
        from data_layer_breeze import NiftyOptionsDataBreeze
        data_layer = NiftyOptionsDataBreeze(api_key, api_secret, session_token)
        return BreezeMarketDataProvider(data_layer), False
    from data_layer_sample import NiftyOptionsDataSample
    data_layer = NiftyOptionsDataSample(index_start_price=initial_spot)
    return BreezeMarketDataProvider(data_layer), True


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def _safe_filename(label: str) -> str:
    return (label.lower().replace(" ", "_").replace("(", "").replace(")", "")
            .replace("+", "plus").replace("-", "").replace("__", "_"))


def build_configs(base_kwargs: dict, monthly_expiry: dt.date, monthly_expiry_prior_trading_day: dt.date,
                   skip_combinations: bool):
    configs = [
        ("Sold straddle only -- Delta Threshold", FullBacktestConfig(**base_kwargs, sold_leg_strategy_name="delta_threshold")),
        ("Sold straddle only -- Fixed Move", FullBacktestConfig(**base_kwargs, sold_leg_strategy_name="fixed_move")),
        ("Sold straddle only -- RSI Signal", FullBacktestConfig(**base_kwargs, sold_leg_strategy_name="rsi_signal")),
        ("Sold straddle only -- Supertrend+EMA Signal", FullBacktestConfig(**base_kwargs, sold_leg_strategy_name="supertrend_ema_signal")),
    ]
    if skip_combinations:
        return configs

    hedge_kwargs = dict(include_hedge_straddle=True, monthly_expiry=monthly_expiry,
                         monthly_expiry_prior_trading_day=monthly_expiry_prior_trading_day)

    configs += [
        ("Sold + Hedge straddle -- Delta Threshold", FullBacktestConfig(
            **base_kwargs, sold_leg_strategy_name="delta_threshold", **hedge_kwargs)),
        ("Sold straddle + Directional Overlay (RSI)", FullBacktestConfig(
            **base_kwargs, sold_leg_strategy_name="rsi_signal",
            include_overlay=True, overlay_hourly_kind="rsi", overlay_gating_kind="rsi")),
        ("Sold straddle + Directional Overlay (Supertrend+EMA)", FullBacktestConfig(
            **base_kwargs, sold_leg_strategy_name="supertrend_ema_signal",
            include_overlay=True, overlay_hourly_kind="supertrend_ema", overlay_gating_kind="supertrend_ema")),
    ]
    for mult in (1, 3, 5):
        configs.append((f"Sold straddle + Opportunistic OTM (x{mult})", FullBacktestConfig(
            **base_kwargs, sold_leg_strategy_name="delta_threshold",
            include_otm=True, otm_multiplier=mult)))
    configs.append(("EVERYTHING (Delta Threshold + Hedge + Overlay[RSI] + OTM x2)", FullBacktestConfig(
        **base_kwargs, sold_leg_strategy_name="delta_threshold", **hedge_kwargs,
        include_overlay=True, overlay_hourly_kind="rsi", overlay_gating_kind="rsi",
        include_otm=True, otm_multiplier=2)))
    return configs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--to-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--calendar", default=None, help="path to expiry_calendar.csv (default: the one shipped alongside this script)")
    parser.add_argument("--weekly-expiry", default=None, help="override the calendar lookup (YYYY-MM-DD)")
    parser.add_argument("--weekly-expiry-prior-trading-day", default=None,
                         help="required alongside --weekly-expiry if you override it; the actual last trading day before that expiry")
    parser.add_argument("--monthly-expiry", default=None, help="override the calendar lookup (YYYY-MM-DD)")
    parser.add_argument("--monthly-expiry-prior-trading-day", default=None,
                         help="required alongside --monthly-expiry if you override it")
    parser.add_argument("--bar-freq-minutes", type=int, default=15)
    parser.add_argument("--initial-spot", type=float, default=24500.0, help="only affects the synthetic fallback")
    parser.add_argument("--skip-combinations", action="store_true",
                         help="only run the 4 sold-leg shapes; skip hedge/overlay/OTM combinations (faster)")
    args = parser.parse_args()

    if bool(args.weekly_expiry) != bool(args.weekly_expiry_prior_trading_day):
        parser.error("--weekly-expiry and --weekly-expiry-prior-trading-day must be given together")
    if bool(args.monthly_expiry) != bool(args.monthly_expiry_prior_trading_day):
        parser.error("--monthly-expiry and --monthly-expiry-prior-trading-day must be given together")

    from_date = dt.datetime.strptime(args.from_date, "%Y-%m-%d").date()
    to_date = dt.datetime.strptime(args.to_date, "%Y-%m-%d").date()
    start = dt.datetime.combine(from_date, dt.time(9, 15))
    end = dt.datetime.combine(to_date, dt.time(15, 30))

    calendar = None
    if not args.weekly_expiry or not args.monthly_expiry:
        calendar = load_expiry_calendar(args.calendar) if args.calendar else load_expiry_calendar()

    if args.weekly_expiry:
        weekly_expiry = dt.datetime.strptime(args.weekly_expiry, "%Y-%m-%d").date()
        weekly_prior = dt.datetime.strptime(args.weekly_expiry_prior_trading_day, "%Y-%m-%d").date()
    else:
        weekly_expiry, weekly_prior = get_next_expiry(calendar, to_date, "weekly")

    if args.monthly_expiry:
        monthly_expiry = dt.datetime.strptime(args.monthly_expiry, "%Y-%m-%d").date()
        monthly_prior = dt.datetime.strptime(args.monthly_expiry_prior_trading_day, "%Y-%m-%d").date()
    else:
        monthly_expiry, monthly_prior = select_monthly_hedge_expiry_from_calendar(calendar, from_date)

    _, is_synthetic = get_provider(args.initial_spot)
    tag = "SYNTHETIC" if is_synthetic else "LIVE"

    print(f"[{tag}] Date range: {start} .. {end}")
    print(f"Weekly expiry: {weekly_expiry} (prior trading day / expiry-eve close: {weekly_prior})")
    print(f"Monthly hedge expiry: {monthly_expiry} (prior trading day: {monthly_prior})")
    if is_synthetic:
        print("NOTE: BREEZE_API_KEY / BREEZE_API_SECRET / BREEZE_SESSION_TOKEN not all set -- "
              "results below are on SYNTHETIC data, not real history.")
    if to_date < weekly_prior:
        print(f"NOTE: --to-date ({to_date}) is before the detected weekly expiry-eve ({weekly_prior}) -- "
              f"this run's window ends before that contract's own expiry-eve close, so you won't see "
              f"the forced closes in this result. Set --to-date to the expiry date itself (or its prior "
              f"trading day) to exercise the full weekly cycle including expiry-eve behavior.")
    print()

    base_kwargs = dict(start=start, end=end, weekly_expiry=weekly_expiry,
                        weekly_expiry_prior_trading_day=weekly_prior, bar_freq_minutes=args.bar_freq_minutes)
    configs = build_configs(base_kwargs, monthly_expiry, monthly_prior, args.skip_combinations)

    rows = []
    for label, config in configs:
        provider, _ = get_provider(args.initial_spot)  # fresh provider per run (see get_provider docstring)
        result = run_full_backtest(provider, config)
        report = metrics.full_report(result.equity_curve, result.trade_pnls, config.initial_capital)
        report["label"] = label
        rows.append(report)

        action_log_path = OUTPUT_DIR / f"{_safe_filename(label)}_{tag}_actions.csv"
        pd.DataFrame({"log_line": result.action_log}).to_csv(action_log_path, index=False)

    df = pd.DataFrame(rows).set_index("label")
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    print(df[REPORT_COLUMNS].round(3).to_string())

    metrics_path = OUTPUT_DIR / f"backtest_metrics_{from_date}_{to_date}_{tag}.csv"
    df.to_csv(metrics_path)
    print(f"\nSaved full metrics table to {metrics_path}")
    print(f"Saved a per-configuration action log (every open/close/recenter decision) under {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
