"""
Runs the sold-leg strategy comparison across every scenario declared in
real_data_cache/scenarios.json -- each scenario is a named date range
meant to capture a distinct kind of market condition (trending, choppy,
a high-IV event week, ...), so this answers "which management strategy
wins, and does the answer change across real market regimes?" rather than
just "which wins on the one date range I happen to have."

DATA SOURCE PER SCENARIO: each scenario independently goes through
nifty_backtester.data_sources.resolve_data_layer (LIVE > CACHED >
SYNTHETIC, or forced via --data-source) -- you can have live credentials
AND a committed cache AND nothing at all for different scenarios in the
same run; each row of the output says which one actually backed it.

If real_data_cache/scenarios.json doesn't exist yet or is empty, this
prints guidance and exits cleanly rather than erroring -- scenarios are
an opt-in layer on top of the single-date-range scripts.

Usage (from the repo root):
  python3 scripts/run_scenarios.py
  python3 scripts/run_scenarios.py --data-source cached          # force committed real data only
  python3 scripts/run_scenarios.py --sold-leg-strategy rsi_signal  # single strategy across all scenarios
  python3 scripts/run_scenarios.py --scenario trending_up_sep2026  # just one named scenario
"""

import argparse
import datetime as dt
from pathlib import Path

import pandas as pd

from nifty_backtester.backtest_engine import FullBacktestConfig, run_full_backtest
from nifty_backtester.market_data import BreezeMarketDataProvider
from nifty_backtester.expiry_utils import load_expiry_calendar, get_next_expiry
from nifty_backtester.data_sources import resolve_data_layer, VALID_SOURCES
from nifty_backtester.scenarios import load_scenarios, DEFAULT_SCENARIOS_PATH, Scenario
from nifty_backtester import metrics

OUTPUT_DIR = Path("./downloaded_samples")
OUTPUT_DIR.mkdir(exist_ok=True)

ALL_STRATEGIES = ["delta_threshold", "fixed_move", "rsi_signal", "supertrend_ema_signal"]

REPORT_COLUMNS = [
    "scenario", "market_condition", "data_source", "sold_leg_strategy",
    "total_return", "total_return_pct", "max_drawdown_pct",
    "sharpe", "sortino", "num_trades", "win_rate", "profit_factor",
]


def _resolve_scenario_expiry(scenario: Scenario):
    if scenario.weekly_expiry is not None:
        if scenario.weekly_expiry_prior_trading_day is None:
            raise ValueError(
                f"Scenario '{scenario.name}' sets weekly_expiry but not "
                f"weekly_expiry_prior_trading_day -- both or neither."
            )
        return scenario.weekly_expiry, scenario.weekly_expiry_prior_trading_day
    calendar = load_expiry_calendar()
    return get_next_expiry(calendar, scenario.to_date, "weekly")


def run_scenario(scenario: Scenario, strategy_names: list[str], data_source: str, initial_spot: float) -> list[dict]:
    weekly_expiry, weekly_prior = _resolve_scenario_expiry(scenario)
    rows = []
    for strategy_name in strategy_names:
        data_layer, source_label = resolve_data_layer(initial_spot=initial_spot, prefer=data_source)
        provider = BreezeMarketDataProvider(data_layer)
        config = FullBacktestConfig(
            start=scenario.start, end=scenario.end,
            weekly_expiry=weekly_expiry, weekly_expiry_prior_trading_day=weekly_prior,
            sold_leg_strategy_name=strategy_name,
        )
        try:
            result = run_full_backtest(provider, config)
        except Exception as e:
            print(f"  [{scenario.name} / {strategy_name}] FAILED: {e}")
            continue
        report = metrics.full_report(result.equity_curve, result.trade_pnls, config.initial_capital)
        report.update({
            "scenario": scenario.name, "market_condition": scenario.market_condition,
            "data_source": source_label, "sold_leg_strategy": strategy_name,
        })
        rows.append(report)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenarios-file", default=None, help=f"default: {DEFAULT_SCENARIOS_PATH}")
    parser.add_argument("--scenario", action="append", default=None,
                         help="run only this scenario name (repeatable) -- default: all declared scenarios")
    parser.add_argument("--sold-leg-strategy", action="append", default=None, choices=ALL_STRATEGIES,
                         help="run only this strategy (repeatable) -- default: all four")
    parser.add_argument("--data-source", default="auto", choices=VALID_SOURCES,
                         help="applied independently to EACH scenario -- auto picks LIVE > CACHED > SYNTHETIC per scenario")
    parser.add_argument("--initial-spot", type=float, default=24500.0, help="only affects the SYNTHETIC fallback")
    args = parser.parse_args()

    scenarios_path = Path(args.scenarios_file) if args.scenarios_file else DEFAULT_SCENARIOS_PATH
    scenarios = load_scenarios(scenarios_path)

    if args.scenario:
        wanted = set(args.scenario)
        scenarios = [s for s in scenarios if s.name in wanted]
        missing = wanted - {s.name for s in scenarios}
        if missing:
            parser.error(f"--scenario name(s) not found in {scenarios_path}: {missing}")

    if not scenarios:
        print(f"No scenarios to run.\n")
        print(f"Looked in: {scenarios_path}")
        print(
            "Declare scenarios there as a JSON list of "
            '{"name", "market_condition", "from_date", "to_date"} objects '
            "(weekly_expiry / weekly_expiry_prior_trading_day / notes optional) -- "
            "see real_data_cache/README.md."
        )
        return

    strategy_names = args.sold_leg_strategy or ALL_STRATEGIES

    print(f"Running {len(scenarios)} scenario(s) x {len(strategy_names)} strateg{'y' if len(strategy_names)==1 else 'ies'} "
          f"(--data-source={args.data_source}):")
    for s in scenarios:
        print(f"  - {s.name} [{s.market_condition}]: {s.from_date} .. {s.to_date}" + (f" -- {s.notes}" if s.notes else ""))
    print()

    all_rows = []
    for scenario in scenarios:
        print(f"[{scenario.name}]")
        all_rows.extend(run_scenario(scenario, strategy_names, args.data_source, args.initial_spot))

    if not all_rows:
        print("\nNo scenario/strategy combination produced a result -- see FAILED lines above.")
        return

    df = pd.DataFrame(all_rows)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print("\n" + df[REPORT_COLUMNS].round(3).to_string(index=False))

    out_path = OUTPUT_DIR / f"scenario_comparison_{dt.date.today()}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved full scenario comparison to {out_path}")

    # quick "does the winning strategy change across market conditions" callout
    print("\nBest sold_leg_strategy by scenario (by total_return_pct):")
    best = df.loc[df.groupby("scenario")["total_return_pct"].idxmax()]
    print(best[["scenario", "market_condition", "sold_leg_strategy", "total_return_pct", "data_source"]].to_string(index=False))


if __name__ == "__main__":
    main()
