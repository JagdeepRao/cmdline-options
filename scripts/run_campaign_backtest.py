"""
Runs the campaign strategy (nifty_backtester.campaign_strategy) -- a funded
long-strangle theta engine -- against LIVE/CACHED/SYNTHETIC data (see
nifty_backtester.data_sources), for one or more consecutive months, and
prints a metrics table plus the full action log for each month.

The weekly-roll moment and the monthly-close moment are one single
configurable time of day (--roll-and-close-time). Pass --sweep-times to
compare several candidate times against each other for the SAME campaign
window (one metrics row per time), rather than picking one in advance --
this is the "measure standard metrics for strategy/adjustment time"
comparison.

Examples (run from the repo root):
  # Single campaign, default 15:30 roll/close time, synthetic fallback if no credentials
  python3 scripts/run_campaign_backtest.py --campaign-start 2026-09-01

  # Three consecutive months chained together
  python3 scripts/run_campaign_backtest.py --campaign-start 2026-09-01 --num-months 3

  # Compare four candidate roll/close times for the same September campaign
  python3 scripts/run_campaign_backtest.py --campaign-start 2026-09-01 --sweep-times 10:00,11:00,13:00,15:30

  # Against committed real data, no live session needed
  python3 scripts/run_campaign_backtest.py --campaign-start 2026-09-01 --data-source cached
"""

import argparse
import datetime as dt
from pathlib import Path

import pandas as pd

from nifty_backtester.campaign_strategy import (
    CampaignConfig, run_campaign_backtest, run_chained_campaigns, first_trading_day_of_month,
)
from nifty_backtester.market_data import BreezeMarketDataProvider, SyntheticMarketDataProvider
from nifty_backtester.expiry_utils import load_expiry_calendar, load_holidays
from nifty_backtester.data_sources import resolve_data_layer, VALID_SOURCES
from nifty_backtester import metrics

OUTPUT_DIR = Path("./downloaded_samples")
OUTPUT_DIR.mkdir(exist_ok=True)

REPORT_COLUMNS = [
    "total_return", "total_return_pct", "max_drawdown_pct",
    "sharpe", "sortino", "num_trades", "win_rate", "profit_factor",
]


def _parse_time(s: str) -> dt.time:
    return dt.datetime.strptime(s.strip(), "%H:%M").time()


def _month_provider_factory(data_source: str, initial_spot: float):
    """Returns provider_factory(campaign_start) -> (provider, source_label).
    LIVE/CACHED providers are built once and reused across months (real
    data doesn't need a fresh window per month); SYNTHETIC needs a fresh
    instance sized to each campaign's own ~5-week window."""
    if data_source == "synthetic":
        def factory(campaign_start: dt.date):
            window_end = campaign_start + dt.timedelta(days=40)
            provider = SyntheticMarketDataProvider(
                dt.datetime.combine(campaign_start, dt.time(9, 15)) - dt.timedelta(days=5),
                dt.datetime.combine(window_end, dt.time(15, 30)),
                initial_spot=initial_spot, annual_vol=0.20, flat_iv=0.14, seed=42,
            )
            return provider, "SYNTHETIC"
        return factory

    data_layer, source_label = resolve_data_layer(initial_spot=initial_spot, prefer=data_source)
    provider = BreezeMarketDataProvider(data_layer)

    def factory(campaign_start: dt.date):
        return provider, source_label
    return factory


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--campaign-start", required=True, help="YYYY-MM-DD -- first trading day of the campaign month")
    parser.add_argument("--num-months", type=int, default=1, help="chain this many consecutive months together (default: 1)")
    parser.add_argument("--roll-and-close-time", default="15:30", help="HH:MM, used for both weekly rolls and the monthly close (default: 15:30)")
    parser.add_argument("--sweep-times", default=None, help="comma-separated HH:MM list -- if given, ignores --roll-and-close-time and compares each")
    parser.add_argument("--long-target-premium", type=float, default=200.0)
    parser.add_argument("--short-far-target-premium", type=float, default=150.0, help="informational only -- see CampaignConfig docstring")
    parser.add_argument("--short-near-target-premium", type=float, default=450.0, help="informational only -- see CampaignConfig docstring")
    parser.add_argument("--long-quantity", type=int, default=3)
    parser.add_argument("--strike-search-range", type=int, default=40)
    parser.add_argument("--initial-capital", type=float, default=100_000.0)
    parser.add_argument("--data-source", default="auto", choices=VALID_SOURCES)
    parser.add_argument("--initial-spot", type=float, default=24500.0, help="only affects the SYNTHETIC fallback")
    args = parser.parse_args()

    campaign_start = dt.datetime.strptime(args.campaign_start, "%Y-%m-%d").date()
    calendar = load_expiry_calendar()
    holidays = load_holidays()

    times = [_parse_time(t) for t in args.sweep_times.split(",")] if args.sweep_times else [_parse_time(args.roll_and_close_time)]

    provider_factory = _month_provider_factory(args.data_source, args.initial_spot)

    rows = []
    for roll_time in times:
        config = CampaignConfig(
            long_target_premium=args.long_target_premium,
            short_far_target_premium=args.short_far_target_premium,
            short_near_target_premium=args.short_near_target_premium,
            long_quantity=args.long_quantity,
            strike_search_range=args.strike_search_range,
            roll_and_close_time=roll_time,
            initial_capital=args.initial_capital,
        )

        def factory_for_source(cs, _pf=provider_factory):
            provider, source_label = _pf(cs)
            factory_for_source.last_source_label = source_label
            return provider

        results = run_chained_campaigns(factory_for_source, calendar, holidays, campaign_start, args.num_months, config)

        for i, result in enumerate(results):
            report = metrics.full_report(result.equity_curve, result.trade_pnls, config.initial_capital)
            report["roll_and_close_time"] = roll_time.strftime("%H:%M")
            report["campaign_month"] = result.schedule.monthly_expiry.strftime("%Y-%m")
            report["source"] = getattr(factory_for_source, "last_source_label", "?")
            rows.append(report)

            tag = f"campaign_{result.schedule.monthly_expiry.strftime('%Y%m')}_{roll_time.strftime('%H%M')}"
            action_log_path = OUTPUT_DIR / f"{tag}_actions.csv"
            pd.DataFrame({"log_line": result.action_log}).to_csv(action_log_path, index=False)
            print(f"\n[{report['source']} | roll/close {roll_time.strftime('%H:%M')} | {report['campaign_month']}] "
                  f"strikes: long_call={result.strikes.long_call_strike} long_put={result.strikes.long_put_strike} "
                  f"short_call=({result.strikes.short_call_near_strike},{result.strikes.short_call_far_strike}) "
                  f"short_put=({result.strikes.short_put_near_strike},{result.strikes.short_put_far_strike})")
            print(f"Saved action log to {action_log_path}")

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print("\n" + df[["campaign_month", "roll_and_close_time", "source"] + REPORT_COLUMNS].round(3).to_string(index=False))

    metrics_path = OUTPUT_DIR / f"campaign_metrics_{campaign_start}.csv"
    df.to_csv(metrics_path, index=False)
    print(f"\nSaved full metrics table to {metrics_path}")

    if len(times) > 1:
        print("\nBest roll_and_close_time by campaign month (by total_return_pct):")
        best = df.loc[df.groupby("campaign_month")["total_return_pct"].idxmax()]
        print(best[["campaign_month", "roll_and_close_time", "total_return_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
