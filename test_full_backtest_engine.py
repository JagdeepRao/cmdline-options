"""
Regression tests for backtest_engine.run_full_backtest().

These exist to make two kinds of claims checkable, not just assertable:
  1. Every management shape and every position-combination actually runs
     end-to-end and produces a valid metrics.full_report().
  2. The equity curve is internally consistent with the realized trade
     P&Ls -- this specifically guards against the bug found during manual
     review, where a position that had gone flat was skipped when marking
     equity, silently dropping its entire realized P&L from every
     subsequent bar. See test_equity_curve_reflects_realized_pnl_even_after_flat.
"""

import datetime as dt
import warnings

import pytest

from market_data import SyntheticMarketDataProvider
from backtest_engine import FullBacktestConfig, run_full_backtest
import metrics

warnings.filterwarnings("ignore", message="divide by zero encountered")
warnings.filterwarnings("ignore", message="invalid value encountered")

START = dt.datetime(2026, 9, 1, 9, 15)
END = dt.datetime(2026, 9, 4, 15, 30)
WEEKLY_EXPIRY = dt.date(2026, 9, 4)
MONTHLY_EXPIRY = dt.date(2026, 9, 24)


def _provider(seed=42, annual_vol=0.35):
    return SyntheticMarketDataProvider(
        START - dt.timedelta(days=6), END,
        initial_spot=24500, annual_vol=annual_vol, flat_iv=0.13, seed=seed,
    )


def _base_kwargs(**overrides):
    kwargs = dict(start=START, end=END, weekly_expiry=WEEKLY_EXPIRY, bar_freq_minutes=15)
    kwargs.update(overrides)
    return kwargs


def _assert_valid_report(result, initial_capital=100_000.0):
    report = metrics.full_report(result.equity_curve, result.trade_pnls, initial_capital)
    expected_keys = {
        "total_return", "total_return_pct", "max_drawdown", "max_drawdown_pct",
        "sharpe", "sortino", "calmar", "num_trades", "win_rate", "profit_factor",
        "avg_win", "avg_loss", "max_consecutive_losses",
    }
    assert expected_keys.issubset(report.keys())
    assert len(result.equity_curve) > 0
    return report


@pytest.mark.parametrize("strategy_name", ["delta_threshold", "fixed_move", "rsi_signal", "supertrend_ema_signal"])
def test_each_sold_leg_strategy_runs_and_produces_metrics(strategy_name):
    provider = _provider()
    config = FullBacktestConfig(**_base_kwargs(sold_leg_strategy_name=strategy_name))
    result = run_full_backtest(provider, config)
    report = _assert_valid_report(result)
    assert report["num_trades"] > 0, f"{strategy_name}: expected at least the initial open+close to register as trades"


def test_hedge_straddle_included_and_managed():
    provider = _provider()
    config = FullBacktestConfig(**_base_kwargs(
        sold_leg_strategy_name="delta_threshold",
        include_hedge_straddle=True, monthly_expiry=MONTHLY_EXPIRY,
    ))
    result = run_full_backtest(provider, config)
    _assert_valid_report(result)
    assert "hedge_straddle" in result.positions
    hedge_tags = {leg.tag for leg in result.positions["hedge_straddle"].legs}
    assert hedge_tags == {"hedge_call", "hedge_put"}


@pytest.mark.parametrize("kind", ["rsi", "supertrend_ema"])
def test_directional_overlay_included(kind):
    provider = _provider()
    config = FullBacktestConfig(**_base_kwargs(
        sold_leg_strategy_name="rsi_signal" if kind == "rsi" else "supertrend_ema_signal",
        include_overlay=True, overlay_hourly_kind=kind, overlay_gating_kind=kind,
    ))
    result = run_full_backtest(provider, config)
    _assert_valid_report(result)
    assert "directional_overlay" in result.positions


@pytest.mark.parametrize("multiplier", [1, 2, 5])
def test_opportunistic_otm_included_and_scales_with_multiplier(multiplier):
    provider = _provider()
    config = FullBacktestConfig(**_base_kwargs(
        sold_leg_strategy_name="delta_threshold",
        include_otm=True, otm_multiplier=multiplier,
    ))
    result = run_full_backtest(provider, config)
    _assert_valid_report(result)
    assert "otm_position" in result.positions
    for leg in result.positions["otm_position"].legs:
        assert leg.quantity == multiplier, "OTM leg quantity should scale directly with otm_multiplier"


def test_otm_pnl_scales_linearly_with_multiplier():
    """1x and 3x should produce exactly 3x the OTM P&L, since sizing is a
    pure multiplier on otherwise-identical entry/exit decisions."""
    pnls = {}
    for multiplier in (1, 3):
        provider = _provider()
        config = FullBacktestConfig(**_base_kwargs(
            sold_leg_strategy_name="delta_threshold", include_otm=True, otm_multiplier=multiplier,
        ))
        result = run_full_backtest(provider, config)
        otm_pnl = sum(leg.pnl() for leg in result.positions["otm_position"].legs)
        pnls[multiplier] = otm_pnl
    assert pnls[1] != 0, "expected at least one OTM entry/exit to have happened in this run"
    assert abs(pnls[3] - 3 * pnls[1]) < 1e-6, f"expected exact 3x scaling, got 1x={pnls[1]} 3x={pnls[3]}"


def test_everything_combined_runs():
    provider = _provider()
    config = FullBacktestConfig(**_base_kwargs(
        sold_leg_strategy_name="delta_threshold",
        include_hedge_straddle=True, monthly_expiry=MONTHLY_EXPIRY,
        include_overlay=True, overlay_hourly_kind="rsi", overlay_gating_kind="rsi",
        include_otm=True, otm_multiplier=2,
    ))
    result = run_full_backtest(provider, config)
    _assert_valid_report(result)
    assert set(result.positions.keys()) == {"sold_straddle", "hedge_straddle", "directional_overlay", "otm_position"}


def test_equity_curve_reflects_realized_pnl_even_after_flat():
    """Regression test for the specific bug found during review: equity
    marking must NOT skip a position just because it's currently flat --
    total_pnl() sums realized P&L from closed legs too, and skipping flat
    positions silently discarded that realized P&L from every subsequent
    equity point. sold_straddle goes flat on expiry eve well before the
    window ends, so this is exactly the scenario that broke."""
    provider = _provider()
    config = FullBacktestConfig(**_base_kwargs(sold_leg_strategy_name="delta_threshold"))
    result = run_full_backtest(provider, config)

    assert result.positions["sold_straddle"].is_flat(), "expected sold_straddle to be flat by the end of this window (forced closed on expiry eve)"

    final_equity = result.equity_curve.iloc[-1]
    expected_equity = config.initial_capital + sum(result.trade_pnls)
    assert abs(final_equity - expected_equity) < 1e-6, (
        f"equity curve's final value ({final_equity}) should exactly equal "
        f"initial_capital + sum(trade_pnls) ({expected_equity}) -- if these "
        f"diverge, realized P&L is being dropped somewhere after a position "
        f"goes flat."
    )
    # also: equity must not artificially snap back to the initial capital
    # once the position goes flat (the exact symptom of the bug)
    assert final_equity != config.initial_capital or sum(result.trade_pnls) == 0, (
        "equity ended exactly at initial capital while trades had nonzero "
        "realized P&L -- almost certainly the flat-position-skipped bug"
    )


def test_expiry_eve_force_close_actually_closes_positions():
    provider = _provider()
    config = FullBacktestConfig(**_base_kwargs(
        sold_leg_strategy_name="delta_threshold",
        include_hedge_straddle=True, monthly_expiry=MONTHLY_EXPIRY,
    ))
    result = run_full_backtest(provider, config)

    force_close_lines = [l for l in result.action_log if "FORCE-CLOSE" in l]
    assert any("sold_straddle" in l for l in force_close_lines), "expected sold_straddle to be force-closed on weekly expiry eve"
    assert result.positions["sold_straddle"].is_flat()
    # hedge_straddle's monthly expiry is far beyond this short window, so it
    # should NOT have been force-closed -- it should still be open at the end
    # (or at least closed only by the final "close everything" step, not by
    # an expiry-eve rule that shouldn't have fired yet)
    assert not any("hedge_straddle" in l and "monthly expiry eve" in l for l in force_close_lines), (
        "hedge_straddle's monthly expiry is weeks past this window -- it should not have hit its expiry-eve close"
    )


def test_missing_monthly_expiry_raises_when_hedge_requested():
    provider = _provider()
    config = FullBacktestConfig(**_base_kwargs(sold_leg_strategy_name="delta_threshold", include_hedge_straddle=True))
    with pytest.raises(ValueError):
        run_full_backtest(provider, config)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
