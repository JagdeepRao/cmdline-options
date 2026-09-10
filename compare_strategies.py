"""
Runs multiple sold-leg management strategies over the SAME period and
market data, producing a side-by-side comparison table (profit, drawdown,
Sharpe, Sortino, Calmar, win rate, profit factor, trade count).

UPDATED for the generalized engine: the old BacktestConfig/run_backtest API
this file previously used no longer exists -- backtest_engine.py was
rewritten this session to support all four position shapes (sold_straddle,
hedge_straddle, directional_overlay, otm_position) via
FullBacktestConfig/run_full_backtest. See STRATEGY.md for the full picture.

Swap the provider_factory at the bottom (SyntheticMarketDataProvider for
testing -> BreezeMarketDataProvider for a real historical backtest) with
zero changes to the comparison logic itself.
"""

import datetime as dt
import pandas as pd

from market_data import SyntheticMarketDataProvider
from backtest_engine import FullBacktestConfig, run_full_backtest
import metrics

COLUMN_ORDER = [
    "total_return", "total_return_pct", "max_drawdown", "max_drawdown_pct",
    "sharpe", "sortino", "calmar", "num_trades", "win_rate", "profit_factor",
    "avg_win", "avg_loss", "max_consecutive_losses",
]

STRATEGY_LABELS = {
    "fixed_move": "Fixed Move (100pt/0.5%)",
    "delta_threshold": "Delta Threshold (0.65)",
    "rsi_signal": "RSI Signal (harvest 0.2)",
    "supertrend_ema_signal": "Supertrend+EMA Signal (harvest 0.2)",
}


def compare_strategies(provider_factory, base_config_kwargs: dict) -> pd.DataFrame:
    """provider_factory: zero-arg callable returning a FRESH provider per
    run, so every strategy sees an identical underlying path rather than
    one mutated by a prior run's stateful indicator lookups.
    base_config_kwargs: shared FullBacktestConfig fields (start/end/
    weekly_expiry/etc) -- sold_leg_strategy_name is varied here across the
    four registered shapes."""
    rows = []
    for name, label in STRATEGY_LABELS.items():
        config = FullBacktestConfig(**base_config_kwargs, sold_leg_strategy_name=name)
        result = run_full_backtest(provider_factory(), config)
        report = metrics.full_report(result.equity_curve, result.trade_pnls, config.initial_capital)
        report["strategy"] = label
        rows.append(report)

    df = pd.DataFrame(rows).set_index("strategy")
    return df[COLUMN_ORDER].round(3)


if __name__ == "__main__":
    start = dt.datetime(2026, 9, 1, 9, 15)
    end = dt.datetime(2026, 9, 4, 15, 30)
    weekly_expiry = dt.date(2026, 9, 4)

    def provider_factory():
        # SYNTHETIC provider for now -- proves the comparison pipeline works.
        # Swap for BreezeMarketDataProvider(your_data_layer) to run this
        # against real history once you're ready.
        return SyntheticMarketDataProvider(
            start - dt.timedelta(days=6), end,  # extra lookback for indicator warm-up
            initial_spot=24500, annual_vol=0.15, flat_iv=0.13, seed=7,
        )

    base_kwargs = dict(start=start, end=end, weekly_expiry=weekly_expiry, bar_freq_minutes=15,
                        initial_capital=100_000, lot_size=1)

    table = compare_strategies(provider_factory, base_kwargs)
    print(table.to_string())
