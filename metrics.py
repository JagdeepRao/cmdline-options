"""
Backtest performance metrics. Pure functions operating on an equity curve
(a pandas Series of cumulative P&L or portfolio value, indexed by datetime)
and/or a list of individual trade P&Ls — no dependency on the rest of the
backtester, so these are independently verifiable against hand-calculated
examples.
"""

import numpy as np
import pandas as pd


def total_return(equity_curve: pd.Series) -> float:
    """Absolute P&L: final equity - initial equity."""
    if len(equity_curve) < 2:
        return 0.0
    return float(equity_curve.iloc[-1] - equity_curve.iloc[0])


def max_drawdown(equity_curve: pd.Series) -> float:
    """Maximum peak-to-trough decline in the equity curve, as a positive
    number (magnitude of the worst drawdown)."""
    if len(equity_curve) < 2:
        return 0.0
    running_max = equity_curve.cummax()
    drawdown = equity_curve - running_max
    return float(-drawdown.min())  # min(drawdown) is most negative -> flip sign to report as positive magnitude


def max_drawdown_pct(equity_curve: pd.Series, initial_capital: float) -> float:
    """Max drawdown as a percentage of initial capital."""
    dd = max_drawdown(equity_curve)
    return (dd / initial_capital * 100) if initial_capital != 0 else 0.0


def resample_daily(equity_curve: pd.Series) -> pd.Series:
    """Resamples an intraday equity curve to end-of-day values. Sharpe/Sortino
    are conventionally computed on daily returns — using raw intraday marks
    instead measures bid-ask/rounding noise relative to a (likely small)
    genuine drift, producing a ratio dominated by sampling frequency rather
    than strategy quality. Drawdown and total return should stay on the
    FULL-resolution curve (they should reflect real intraday extremes) —
    only pass the resampled series to sharpe_ratio/sortino_ratio."""
    if len(equity_curve) < 2:
        return equity_curve
    return equity_curve.resample("1D").last().dropna()


def sharpe_ratio(daily_pnl: pd.Series, periods_per_year: int = 252) -> float:
    """Sharpe computed directly on the daily P&L series (in rupees) —
    mean/std, annualized. Deliberately does NOT net a risk-free rate: for a
    derivatives overlay strategy, 'risk-free return on capital' doesn't map
    cleanly onto a P&L stream from a lightly-sized position, and netting a
    rate scaled to a much larger stated account capital (rather than actual
    notional at risk) was found to swamp real signal — any strategy trading
    a small fraction of the stated capital would show a deeply negative
    Sharpe regardless of quality, since the capital-scaled hurdle dwarfs the
    position's real day-to-day P&L. This is the standard practical approach
    for assessing a P&L stream's risk-adjusted consistency without an
    arbitrary capital-base assumption."""
    if len(daily_pnl) < 3 or daily_pnl.std() == 0:
        return 0.0
    return float(daily_pnl.mean() / daily_pnl.std() * np.sqrt(periods_per_year))


def sortino_ratio(daily_pnl: pd.Series, periods_per_year: int = 252) -> float:
    """Same rationale as sharpe_ratio above (no risk-free netting), but only
    penalizing downside volatility — often more appropriate for asymmetric
    payoff shapes (e.g. options buying) than Sharpe."""
    if len(daily_pnl) < 3:
        return 0.0
    downside = daily_pnl[daily_pnl < 0]
    downside_std = downside.std() if len(downside) > 1 else 0.0
    if downside_std == 0:
        return 0.0
    return float(daily_pnl.mean() / downside_std * np.sqrt(periods_per_year))


def calmar_ratio(equity_curve: pd.Series, initial_capital: float) -> float:
    """Total return divided by max drawdown — a direct 'return per unit of
    worst pain endured' measure, arguably more intuitive than Sharpe for
    judging whether a strategy's drawdowns are worth its returns."""
    dd = max_drawdown(equity_curve)
    if dd == 0:
        return 0.0
    return total_return(equity_curve) / dd


def trade_stats(trade_pnls: list[float]) -> dict:
    """Win rate, profit factor, average win/loss, max consecutive losses,
    and trade count from a list of individual closed-trade P&Ls."""
    if not trade_pnls:
        return {
            "num_trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
            "avg_win": 0.0, "avg_loss": 0.0, "max_consecutive_losses": 0,
        }

    wins = [p for p in trade_pnls if p > 0]
    losses = [p for p in trade_pnls if p < 0]

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0

    max_consec_losses = 0
    current_streak = 0
    for p in trade_pnls:
        if p < 0:
            current_streak += 1
            max_consec_losses = max(max_consec_losses, current_streak)
        else:
            current_streak = 0

    return {
        "num_trades": len(trade_pnls),
        "win_rate": len(wins) / len(trade_pnls) * 100,
        "profit_factor": profit_factor,
        "avg_win": (sum(wins) / len(wins)) if wins else 0.0,
        "avg_loss": (sum(losses) / len(losses)) if losses else 0.0,
        "max_consecutive_losses": max_consec_losses,
    }


def full_report(
    equity_curve: pd.Series,
    trade_pnls: list[float],
    initial_capital: float,
) -> dict:
    """One-call convenience: every metric above, bundled into a single dict
    suitable for building a comparison table across multiple strategy runs.

    Sharpe/Sortino are computed on daily-resampled P&L (not percentage
    returns, and not netted against a risk-free rate) — see sharpe_ratio's
    docstring for why. If you need a different metric on raw intraday data,
    compute it separately from the functions above rather than through this
    convenience wrapper."""
    daily_curve = resample_daily(equity_curve)
    daily_pnl = daily_curve.diff().dropna()
    report = {
        "total_return": total_return(equity_curve),
        "total_return_pct": total_return(equity_curve) / initial_capital * 100 if initial_capital else 0.0,
        "max_drawdown": max_drawdown(equity_curve),
        "max_drawdown_pct": max_drawdown_pct(equity_curve, initial_capital),
        "sharpe": sharpe_ratio(daily_pnl),
        "sortino": sortino_ratio(daily_pnl),
        "calmar": calmar_ratio(equity_curve, initial_capital),
    }
    report.update(trade_stats(trade_pnls))
    return report
