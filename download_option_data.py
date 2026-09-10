"""
Downloads historical data for a specific option leg (strike/expiry/right)
or the NIFTY index, then annotates every bar with whatever indicator(s) you
ask for -- RSI crossover, Supertrend+EMA, and/or Renko+SuperTrend -- so you
can eyeball the raw price action next to each computed BUY/SELL signal and
judge for yourself whether it "looks right." This is indicator-level
validation; see download_and_run_strategy.py for the equivalent at the
strategy/adjustment level.

DATA SOURCE: uses live ICICI Breeze if BREEZE_API_KEY, BREEZE_API_SECRET,
and BREEZE_SESSION_TOKEN are all set as environment variables (same
convention as test_find_atm.py / run_real_greeks.py). Falls back to the
SYNTHETIC sample data layer otherwise, clearly labeled as such in both the
console output and the output filename -- this exists so you (or anyone
without live credentials yet) can confirm the script itself works before
pointing it at a real account.

Examples:
  # Real option leg, RSI + Supertrend+EMA on 15-min data
  export BREEZE_API_KEY=... BREEZE_API_SECRET=... BREEZE_SESSION_TOKEN=...
  python3 download_option_data.py --option --strike 24800 --right call \\
      --expiry 2026-09-11 --from-date 2026-09-01 --to-date 2026-09-10 \\
      --resample-minutes 15 --kind rsi,supertrend_ema

  # Real NIFTY index, Renko+SuperTrend on native 1-min data
  python3 download_option_data.py --index --from-date 2026-09-01 \\
      --to-date 2026-09-10 --kind renko

  # No credentials set -> runs against synthetic data automatically
  python3 download_option_data.py --index --from-date 2026-09-01 --to-date 2026-09-03 --kind all
"""

import os
import argparse
import datetime as dt
from pathlib import Path

import pandas as pd

from strategy import RSIIndicator, SupertrendEMAIndicator, RenkoSuperTrendIndicator

OUTPUT_DIR = Path("./downloaded_samples")
OUTPUT_DIR.mkdir(exist_ok=True)

NATIVE_INTERVAL_MINUTES = {"1minute": 1, "5minute": 5, "30minute": 30, "1day": 1440}


def get_data_layer():
    """Real Breeze layer if credentials are set, else the synthetic
    fallback -- returns (layer, is_synthetic)."""
    api_key = os.environ.get("BREEZE_API_KEY")
    api_secret = os.environ.get("BREEZE_API_SECRET")
    session_token = os.environ.get("BREEZE_SESSION_TOKEN")
    if api_key and api_secret and session_token:
        from data_layer_breeze import NiftyOptionsDataBreeze
        print("Using LIVE Breeze data.")
        return NiftyOptionsDataBreeze(api_key, api_secret, session_token), False
    from data_layer_sample import NiftyOptionsDataSample
    print("BREEZE_API_KEY / BREEZE_API_SECRET / BREEZE_SESSION_TOKEN not all set "
          "-- falling back to SYNTHETIC sample data. Set those three env vars "
          "to validate against real Breeze data instead.")
    return NiftyOptionsDataSample(), True


def _resample(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime")
    agg = {}
    if "open" in df.columns:
        agg["open"] = "first"
    if "high" in df.columns:
        agg["high"] = "max"
    if "low" in df.columns:
        agg["low"] = "min"
    if "close" in df.columns:
        agg["close"] = "last"
    if "volume" in df.columns:
        agg["volume"] = "sum"
    return df.resample(f"{minutes}min").agg(agg).dropna(how="all").reset_index()


def add_indicator_columns(df: pd.DataFrame, kinds: set, is_index: bool) -> pd.DataFrame:
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    if "rsi" in kinds:
        ind = RSIIndicator(df, rsi_len=20, ema_len=9)
        df["rsi"] = ind.df["rsi"]
        df["rsi_ema"] = ind.df["rsi_ema"]
        df["rsi_signal"] = ""
        df.loc[ind.df["bull_cross"], "rsi_signal"] = "BUY"
        df.loc[ind.df["bear_cross"], "rsi_signal"] = "SELL"

    if "supertrend_ema" in kinds:
        ind = SupertrendEMAIndicator(df, ema_len=21, factor=3.0, atr_period=10)
        df["st_line"] = ind.df["st_line"]
        df["st_ema"] = ind.df["ema"]
        df["st_signal"] = ""
        df.loc[ind.df["buy_signal_edge"], "st_signal"] = "BUY"
        df.loc[ind.df["sell_signal_edge"], "st_signal"] = "SELL"

    if "renko" in kinds:
        if not is_index:
            print("NOTE: Renko is documented for use on the underlying, not individual option "
                  "legs -- computing it here anyway since you asked, but treat it as informational.")
        ind = RenkoSuperTrendIndicator(df, atr_box_len=14, atr_box_mult=1.0, st_factor=3.0, st_atr_len=10)
        df["renko_trend"] = ind.df["trend_dir"]
        df["renko_signal"] = ""
        df.loc[ind.df["buy_signal"], "renko_signal"] = "BUY"
        df.loc[ind.df["sell_signal"], "renko_signal"] = "SELL"

    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--to-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--interval", default="1minute", choices=list(NATIVE_INTERVAL_MINUTES))
    parser.add_argument("--resample-minutes", type=int, default=None,
                         help="resample to this many minutes before computing indicators (e.g. 15 for the "
                              "sold-leg/overlay signals; leave unset to use --interval natively, e.g. for Renko at 1min)")

    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--index", action="store_true", help="download the NIFTY index")
    target.add_argument("--option", action="store_true", help="download a specific option leg (requires --strike/--right/--expiry)")

    parser.add_argument("--strike", type=int)
    parser.add_argument("--right", choices=["call", "put"])
    parser.add_argument("--expiry", help="YYYY-MM-DD")

    parser.add_argument("--kind", default="all",
                         help="comma-separated: rsi,supertrend_ema,renko or 'all'")
    parser.add_argument("--out", default=None, help="output CSV path (default: auto-named under ./downloaded_samples/)")

    args = parser.parse_args()

    if args.option and not (args.strike and args.right and args.expiry):
        parser.error("--option requires --strike, --right, and --expiry")

    kinds = {"rsi", "supertrend_ema", "renko"} if args.kind == "all" else set(args.kind.split(","))
    unknown = kinds - {"rsi", "supertrend_ema", "renko"}
    if unknown:
        parser.error(f"unknown --kind value(s): {unknown}")

    from_date = dt.datetime.strptime(args.from_date, "%Y-%m-%d").date()
    to_date = dt.datetime.strptime(args.to_date, "%Y-%m-%d").date()

    data_layer, is_synthetic = get_data_layer()

    if args.index:
        raw = data_layer.get_index_historical(from_date, to_date, interval=args.interval)
        label = "NIFTY_INDEX"
    else:
        expiry = dt.datetime.strptime(args.expiry, "%Y-%m-%d").date()
        raw = data_layer.get_option_historical(expiry, args.strike, args.right, from_date, to_date, interval=args.interval)
        label = f"NIFTY_{expiry}_{args.strike}_{args.right}"

    if raw.empty:
        print(f"No data returned for {label} in {from_date}..{to_date} -- check the expiry/strike/date range.")
        return

    if args.resample_minutes and args.resample_minutes != NATIVE_INTERVAL_MINUTES.get(args.interval):
        raw = _resample(raw, args.resample_minutes)
        effective_minutes = args.resample_minutes
    else:
        effective_minutes = NATIVE_INTERVAL_MINUTES.get(args.interval, args.interval)

    if len(raw) < 30:
        print(f"WARNING: only {len(raw)} bars after resampling -- indicators need warm-up "
              f"(RSI/Supertrend+EMA want ~30+, Renko wants more); widen the date range for meaningful signals.")

    annotated = add_indicator_columns(raw, kinds, is_index=args.index)

    tag = "SYNTHETIC" if is_synthetic else "LIVE"
    out_path = Path(args.out) if args.out else OUTPUT_DIR / f"{label}_{effective_minutes}min_{tag}.csv"
    annotated.to_csv(out_path, index=False)

    signal_cols = [c for c in annotated.columns if c.endswith("_signal")]
    print(f"\n[{tag}] Saved {len(annotated)} bars to {out_path}")
    if signal_cols:
        mask = (annotated[signal_cols] != "").any(axis=1)
        total = int(mask.sum())
        print(f"{total} bar(s) fired at least one signal across {signal_cols}\n")
        if total:
            cols_to_show = ["datetime", "close"] + signal_cols
            print(annotated.loc[mask, cols_to_show].to_string(index=False))
        else:
            print("No signals fired in this window -- either the price action didn't produce a "
                  "crossover/flip, or the window is too short for the indicator to warm up. "
                  "Open the CSV to inspect the raw indicator values (rsi/rsi_ema, st_line/st_ema, "
                  "renko_trend) bar-by-bar regardless.")


if __name__ == "__main__":
    main()
