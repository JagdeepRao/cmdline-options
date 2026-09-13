"""
Runs Campaign 1 (campaign_strategy.py -- funded long strangle) and
Campaign 2 (campaign_straddle_strategy.py -- sold weekly straddle / bought
monthly straddle, daily ATM-drift recenter) over the SAME campaign
window(s) and the SAME underlying data, and prints a side-by-side metrics
table -- this is the actual question being asked: is the funded strangle
worth its extra complexity, or does the simpler, more-frequently-adjusted
straddle do just as well (or better)?

Both campaigns share one adjustment-time knob each (Campaign 1's
roll_and_close_time, Campaign 2's adjustment_time) -- pass --adjust-time
to set both to the same value for a fair comparison, or --sweep-times to
compare BOTH campaigns across the same set of candidate times.

Examples (run from the repo root):
  # Single month, both campaigns, default 15:30
  python3 scripts/compare_campaigns.py --campaign-start 2026-09-01

  # Three consecutive months, comparing four times of day
  python3 scripts/compare_campaigns.py --campaign-start 2026-09-01 --num-months 3 \\
      --sweep-times 10:00,11:00,13:00,15:30

  # Against committed real data
  python3 scripts/compare_campaigns.py --campaign-start 2026-09-01 --data-source cached
"""

import argparse
import datetime as dt
from pathlib import Path

import pandas as pd

from nifty_backtester.campaign_strategy import CampaignConfig, run_chained_campaigns
from nifty_backtester.campaign_straddle_strategy import StraddleCampaignConfig, run_chained_straddle_campaigns
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
    """Same convention as run_campaign_backtest.py -- returns
    provider_factory(campaign_start) -> (provider, source_label)."""
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
    parser.add_argument("--campaign-start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--num-months", type=int, default=1)
    parser.add_argument("--adjust-time", default="15:30", help="HH:MM applied to BOTH campaigns (default: 15:30)")
    parser.add_argument("--sweep-times", default=None, help="comma-separated HH:MM list -- if given, ignores --adjust-time and compares each, for BOTH campaigns")
    parser.add_argument("--recenter-threshold-points", type=float, default=100.0, help="Campaign 2's drift threshold")
    parser.add_argument("--strike-search-range", type=int, default=40, help="Campaign 1's strike search width")
    parser.add_argument("--initial-capital", type=float, default=100_000.0)
    parser.add_argument("--data-source", default="auto", choices=VALID_SOURCES)
    parser.add_argument("--initial-spot", type=float, default=24500.0, help="only affects the SYNTHETIC fallback")
    parser.add_argument("--verbose", "-v", action="store_true", help="list out trades/actions being taken during the strategy run")
    args = parser.parse_args()

    campaign_start = dt.datetime.strptime(args.campaign_start, "%Y-%m-%d").date()
    calendar = load_expiry_calendar()
    holidays = load_holidays()
    times = [_parse_time(t) for t in args.sweep_times.split(",")] if args.sweep_times else [_parse_time(args.adjust_time)]

    provider_factory = _month_provider_factory(args.data_source, args.initial_spot)

    def factory_for_source(cs, _pf=provider_factory):
        provider, source_label = _pf(cs)
        factory_for_source.last_source_label = source_label
        return provider

    rows = []
    all_results = []
    for t in times:
        c1_config = CampaignConfig(strike_search_range=args.strike_search_range, roll_and_close_time=t, initial_capital=args.initial_capital)
        c1_results = run_chained_campaigns(factory_for_source, calendar, holidays, campaign_start, args.num_months, c1_config)
        for result in c1_results:
            report = metrics.full_report(result.equity_curve, result.trade_pnls, c1_config.initial_capital)
            label = f"Campaign 1 (funded strangle) @ {t.strftime('%H:%M')} [{result.schedule.monthly_expiry.strftime('%Y-%m')}]"
            if args.verbose:
                print(f"\n=================== TRADE LOG: {label} ===================")
                for entry in result.action_log:
                    print(f"  {entry}")
            report.update({
                "campaign": "1: funded strangle", "adjust_time": t.strftime("%H:%M"),
                "campaign_month": result.schedule.monthly_expiry.strftime("%Y-%m"),
                "source": getattr(factory_for_source, "last_source_label", "?"),
            })
            rows.append(report)
            all_results.append((label, result, report))

        c2_config = StraddleCampaignConfig(recenter_threshold_points=args.recenter_threshold_points, adjustment_time=t, initial_capital=args.initial_capital)
        c2_results = run_chained_straddle_campaigns(factory_for_source, calendar, holidays, campaign_start, args.num_months, c2_config)
        for result in c2_results:
            report = metrics.full_report(result.equity_curve, result.trade_pnls, c2_config.initial_capital)
            label = f"Campaign 2 (recentered straddle) @ {t.strftime('%H:%M')} [{result.schedule.monthly_expiry.strftime('%Y-%m')}]"
            if args.verbose:
                print(f"\n=================== TRADE LOG: {label} ===================")
                for entry in result.action_log:
                    print(f"  {entry}")
            report.update({
                "campaign": "2: recentered straddle", "adjust_time": t.strftime("%H:%M"),
                "campaign_month": result.schedule.monthly_expiry.strftime("%Y-%m"),
                "source": getattr(factory_for_source, "last_source_label", "?"),
                "num_recenters": result.num_recenters,
            })
            rows.append(report)
            all_results.append((label, result, report))

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    print("\n" + df[["campaign", "campaign_month", "adjust_time", "source"] + REPORT_COLUMNS].round(3).to_string(index=False))

    out_path = OUTPUT_DIR / f"campaign_comparison_{campaign_start}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved full comparison to {out_path}")

    print("\nBest campaign by month (by total_return_pct):")
    best = df.loc[df.groupby("campaign_month")["total_return_pct"].idxmax()]
    print(best[["campaign_month", "campaign", "adjust_time", "total_return_pct"]].to_string(index=False))

    _print_tearsheet_report(all_results, df)


def _print_tearsheet_report(all_results, df):
    print("\n" + "=" * 80)
    print("                         TEARSHEET REPORT")
    print("=" * 80)
    for label, result, r in all_results:
        pnls = result.trade_pnls
        winning_trades = [p for p in pnls if p > 0]
        losing_trades = [p for p in pnls if p < 0]
        avg_win = sum(winning_trades) / len(winning_trades) if winning_trades else 0.0
        avg_loss = sum(losing_trades) / len(losing_trades) if losing_trades else 0.0
        max_win = max(winning_trades) if winning_trades else 0.0
        max_loss = min(losing_trades) if losing_trades else 0.0

        print(f"\n--- {label} ---")
        print(f"  Initial Capital    : {r.get('initial_capital', 100000.0):>12.2f}")
        print(f"  Ending Equity      : {r.get('ending_equity', 0.0):>12.2f}")
        print(f"  Total Return       : {r.get('total_return', 0.0):>12.2f} ({r.get('total_return_pct', 0.0):>6.2f}%)")
        print(f"  Max Drawdown       : {r.get('max_drawdown', 0.0):>12.2f} ({r.get('max_drawdown_pct', 0.0):>6.2f}%)")
        print(f"  Sharpe Ratio       : {r.get('sharpe', 0.0):>12.3f}")
        print(f"  Sortino Ratio      : {r.get('sortino', 0.0):>12.3f}")
        print(f"  Profit Factor      : {r.get('profit_factor', 0.0):>12.3f}")
        print(f"  Total Trades       : {r.get('num_trades', 0):>12d}")
        print(f"  Win Rate           : {r.get('win_rate', 0.0):>12.1f}%")
        print(f"  Avg Win / Avg Loss : {avg_win:>8.2f} / {avg_loss:>8.2f}")
        print(f"  Max Win / Max Loss : {max_win:>8.2f} / {max_loss:>8.2f}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
