"""
Demo: runs live_engine.run_live_monitor() against SAMPLE (synthetic) data
via nifty_live.replay_feed.ReplayLiveFeed, so you can see the whole live-
monitoring pipeline (positions -> real AdjustmentStrategy -> ConsoleNotifier)
working end-to-end with ZERO live Breeze/Zerodha session -- the same role
compare_strategies.py/run_all_strategies_demo.py play for the batch
backtester.

Seeds a small sold_straddle position (2 legs) at the start of the replay
window, then walks forward at --bar-freq-minutes intervals, printing every
DeltaThresholdStrategy recommendation to the console. Nothing here
executes a trade or mutates the position -- see nifty_live/live_engine.py's
module docstring for why that's deliberate.

For monitoring REAL currently-held positions instead of this demo seed,
build a PositionStore via PositionStore.load(path) or
PositionStore.import_from_zerodha(...) (see nifty_live/position_store.py)
and swap it in for the demo positions below -- everything else in this
script (the feed -> strategy -> notifier wiring) stays the same whether
positions come from a manual seed file, a Zerodha import, or (once a
future phase adds BreezeWebSocketFeed) a real live feed instead of this
replay one.

Usage (from the repo root):
  python3 scripts/run_live_monitor_demo.py
  python3 scripts/run_live_monitor_demo.py --bar-freq-minutes 5 --dedupe
  python3 scripts/run_live_monitor_demo.py --sold-delta-threshold 0.2
"""

import argparse
import datetime as dt

from nifty_backtester.data_layer_sample import NiftyOptionsDataSample
from nifty_backtester.strategy import Leg, MultiLegPosition, Right, Direction, DeltaThresholdStrategy, DeltaIndicator
from nifty_live.replay_feed import ReplayLiveFeed
from nifty_live.notifier import ConsoleNotifier
from nifty_live.live_engine import run_live_monitor


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from-date", default="2026-09-01", help="YYYY-MM-DD")
    parser.add_argument("--to-date", default="2026-09-04", help="YYYY-MM-DD")
    parser.add_argument("--bar-freq-minutes", type=int, default=15)
    parser.add_argument("--initial-spot", type=float, default=24500.0)
    parser.add_argument("--sold-delta-threshold", type=float, default=0.5,
                         help="lowered from the usual 0.65 so this short demo window has a real chance of firing")
    parser.add_argument("--dedupe", action="store_true",
                         help="suppress repeated identical alerts across consecutive steps (see live_engine.py's docstring for the tradeoff)")
    args = parser.parse_args()

    from_date = dt.datetime.strptime(args.from_date, "%Y-%m-%d").date()
    to_date = dt.datetime.strptime(args.to_date, "%Y-%m-%d").date()

    sample = NiftyOptionsDataSample(index_start_price=args.initial_spot)
    feed = ReplayLiveFeed(sample, from_date, to_date, bar_freq_minutes=args.bar_freq_minutes)

    entry_time = dt.datetime.combine(from_date, dt.time(9, 15))
    strike = round(args.initial_spot / 100) * 100
    weekly_expiry = to_date + dt.timedelta(days=7)  # not calendar-registered -- fine, this demo never exercises expiry-eve logic

    position = MultiLegPosition(name="sold_straddle")
    position.add_leg(Leg(
        tag="sold_call", right=Right.CALL, direction=Direction.SHORT, strike=strike, expiry=weekly_expiry,
        entry_time=entry_time, entry_price=feed.provider.get_option_price(strike, "call", weekly_expiry, entry_time),
    ))
    position.add_leg(Leg(
        tag="sold_put", right=Right.PUT, direction=Direction.SHORT, strike=strike, expiry=weekly_expiry,
        entry_time=entry_time, entry_price=feed.provider.get_option_price(strike, "put", weekly_expiry, entry_time),
    ))
    positions = {"sold_straddle": position}

    delta_indicators = {
        "sold_call": DeltaIndicator(
            strike=strike, right=Right.CALL, expiry=weekly_expiry,
            spot_lookup=feed.provider.get_spot,
            option_price_lookup=lambda ts: feed.provider.get_option_price(strike, "call", weekly_expiry, ts),
        ),
        "sold_put": DeltaIndicator(
            strike=strike, right=Right.PUT, expiry=weekly_expiry,
            spot_lookup=feed.provider.get_spot,
            option_price_lookup=lambda ts: feed.provider.get_option_price(strike, "put", weekly_expiry, ts),
        ),
    }
    strategy = DeltaThresholdStrategy(delta_indicators=delta_indicators, sold_threshold=args.sold_delta_threshold, hedge_threshold=float("inf"))

    print(f"Replaying {from_date}..{to_date} at {args.bar_freq_minutes}-min bars, watching sold_straddle "
          f"(strike={strike}, expiry={weekly_expiry}) for |delta|>={args.sold_delta_threshold} "
          f"(dedupe={'on' if args.dedupe else 'off'})...\n")

    notifier = ConsoleNotifier()
    sent = run_live_monitor(positions, [strategy], feed, notifier, dedupe=args.dedupe)

    print(f"\n{len(sent)} notification(s) sent over {len(feed)} step(s).")
    if not sent:
        print("No signals fired -- try a lower --sold-delta-threshold, a longer date range, or --bar-freq-minutes 5.")


if __name__ == "__main__":
    main()
