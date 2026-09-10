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
    if len(equity_curve) < 2:
        return 0.0
    return float(equity_curve.iloc[-1] - equity_curve.iloc[0])


def max_drawdown(equity_curve: pd.Series) -> float:
    if len(equity_curve) < 2:
        return 0.0
    running_max = equity_curve.cummax()
    drawdown = equity_curve - running_max
    return float(-drawdown.min())


def max_drawdown_pct(equity_curve: pd.Series, initial_capital: float) -> float:
    dd = max_drawdown(equity_curve)
    return (dd / initial_capital * 100) if initial_capital != 0 else 0.0


def sharpe_ratio(equity_curve: pd.Series, risk_free_rate_annual: float = 0.0525, periods_per_year: int = 252) -> float:
    if len(equity_curve) < 3:
        return 0.0
    returns = equity_curve.pct_change().dropna()
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    period_rf = risk_free_rate_annual / periods_per_year
    excess_returns = returns - period_rf
    return float(excess_returns.mean() / returns.std() * np.sqrt(periods_per_year))


def sortino_ratio(equity_curve: pd.Series, risk_free_rate_annual: float = 0.0525, periods_per_year: int = 252) -> float:
    if len(equity_curve) < 3:
        return 0.0
    returns = equity_curve.pct_change().dropna()
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if len(returns) < 2:
        return 0.0
    period_rf = risk_free_rate_annual / periods_per_year
    excess_returns = returns - period_rf
    downside = excess_returns[excess_returns < 0]
    downside_std = downside.std() if len(downside) > 1 else 0.0
    if downside_std == 0:
        return 0.0
    return float(excess_returns.mean() / downside_std * np.sqrt(periods_per_year))


def calmar_ratio(equity_curve: pd.Series, initial_capital: float) -> float:
    dd = max_drawdown(equity_curve)
    if dd == 0:
        return 0.0
    return total_return(equity_curve) / dd


def trade_stats(trade_pnls: list) -> dict:
    if not trade_pnls:
        return {"num_trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
                "avg_win": 0.0, "avg_loss": 0.0, "max_consecutive_losses": 0}
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


def full_report(equity_curve: pd.Series, trade_pnls: list, initial_capital: float,
                 periods_per_year: int = 252, risk_free_rate_annual: float = 0.0525) -> dict:
    report = {
        "total_return": total_return(equity_curve),
        "total_return_pct": total_return(equity_curve) / initial_capital * 100 if initial_capital else 0.0,
        "max_drawdown": max_drawdown(equity_curve),
        "max_drawdown_pct": max_drawdown_pct(equity_curve, initial_capital),
        "sharpe": sharpe_ratio(equity_curve, risk_free_rate_annual, periods_per_year),
        "sortino": sortino_ratio(equity_curve, risk_free_rate_annual, periods_per_year),
        "calmar": calmar_ratio(equity_curve, initial_capital),
    }
    report.update(trade_stats(trade_pnls))
    return report
