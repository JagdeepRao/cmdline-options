"""
Runs multiple strategies over the SAME period and market data, producing a
side-by-side comparison table (profit, drawdown, Sharpe, Sortino, Calmar,
win rate, profit factor, trade count).

Swap the provider at the bottom (SyntheticMarketDataProvider for testing ->
BreezeMarketDataProvider for a real historical backtest) with zero changes
to the comparison logic itself.
"""

import datetime as dt
import pandas as pd

from market_data import SyntheticMarketDataProvider
from backtest_engine import BacktestConfig, run_backtest
import metrics


def compare_strategies(provider, config: BacktestConfig) -> pd.DataFrame:
    strategy_configs = [
        ("Fixed Move (100pt/0.5%)", "fixed_move", {"move_points": 100, "move_pct": 0.5}),
        ("Delta Threshold (0.65)", "delta_threshold", {"sold_threshold": 0.65, "hedge_threshold": 0.85}),
        ("RSI Signal (harvest 0.2)", "rsi_signal", {"harvest_delta_threshold": 0.2}),
    ]

    rows = []
    for label, strategy_name, kwargs in strategy_configs:
        result = run_backtest(provider, config, strategy_name, kwargs)
        report = metrics.full_report(result.equity_curve, result.trade_pnls, config.initial_capital)
        report["strategy"] = label
        rows.append(report)

    df = pd.DataFrame(rows).set_index("strategy")
    column_order = [
        "total_return", "total_return_pct", "max_drawdown", "max_drawdown_pct",
        "sharpe", "sortino", "calmar", "num_trades", "win_rate", "profit_factor",
        "avg_win", "avg_loss", "max_consecutive_losses",
    ]
    return df[column_order].round(3)


if __name__ == "__main__":
    start = dt.datetime(2026, 9, 1, 9, 15)
    end = dt.datetime(2026, 9, 5, 15, 30)
    expiry = dt.date(2026, 9, 8)

    # SYNTHETIC provider for now — proves the comparison pipeline works.
    # Swap for BreezeMarketDataProvider(your_data_layer) to run this against
    # real history once you're ready.
    provider = SyntheticMarketDataProvider(start, end, initial_spot=24500, annual_vol=0.15, flat_iv=0.13, seed=7)
    config = BacktestConfig(start=start, end=end, expiry=expiry, bar_freq_minutes=15, initial_capital=100_000, lot_size=1)

    table = compare_strategies(provider, config)
    print(table.to_string())
