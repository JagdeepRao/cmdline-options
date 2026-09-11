"""
Sequenced tests for nifty_live.position_store.

  1. PositionStore JSON round-trip (manual-seed schema == save()/load() schema)
  2. discover_zerodha_nifty_option_legs() filtering + instrument-token join
     (using fixtures that match the CONFIRMED real kiteconnect response
     shape -- see position_store.py's module docstring for how that shape
     was verified -- not a live Zerodha session)
  3. PositionStore.import_from_zerodha() role assignment (assigned vs
     unassigned legs, direction/quantity derivation)

No kiteconnect import anywhere in this file or in position_store.py's
non-live code paths -- FakeKite below is a plain Python object exposing
only the two methods discover_zerodha_nifty_option_legs actually calls,
confirming the duck-typing contract is real and not accidentally requiring
an actual KiteConnect instance.
"""

import datetime as dt

import pytest

from nifty_backtester.strategy import Leg, Right, Direction
from nifty_live.position_store import (
    PositionStore, RoleAssignment, DiscoveredZerodhaLeg,
    discover_zerodha_nifty_option_legs,
)


# ─────────────────────────────────────────────
# 1. JSON ROUND-TRIP
# ─────────────────────────────────────────────

def test_save_load_round_trip_preserves_all_leg_fields(tmp_path):
    store = PositionStore()
    store.add_leg("sold_straddle", _leg(tag="sold_call", right=Right.CALL, direction=Direction.SHORT, strike=24500.0))
    store.add_leg("sold_straddle", _leg(tag="sold_put", right=Right.PUT, direction=Direction.SHORT, strike=24500.0))

    path = tmp_path / "positions.json"
    store.save(path)
    reloaded = PositionStore.load(path)

    assert set(reloaded.positions) == {"sold_straddle"}
    tags = {leg.tag for leg in reloaded.positions["sold_straddle"].legs}
    assert tags == {"sold_call", "sold_put"}
    call_leg = reloaded.positions["sold_straddle"].get_leg("sold_call")
    assert call_leg.right == Right.CALL
    assert call_leg.direction == Direction.SHORT
    assert call_leg.strike == 24500.0
    assert call_leg.expiry == dt.date(2026, 9, 8)
    assert call_leg.entry_price == 145.3


def test_manually_authored_json_matches_saved_schema_exactly(tmp_path):
    """A hand-written seed file (no PositionStore involved in producing
    it) must load identically to one produced by .save() -- this is the
    'one schema for both manual seeding and machine round-trip' guarantee."""
    manual_path = tmp_path / "manual_seed.json"
    manual_path.write_text("""
    {
      "positions": {
        "hedge_straddle": {
          "legs": [
            {
              "tag": "hedge_call", "right": "call", "direction": "long",
              "strike": 24800.0, "expiry": "2026-09-29",
              "entry_time": "2026-09-01T09:15:00", "entry_price": 210.0,
              "quantity": 1, "exit_time": null, "exit_price": null
            }
          ]
        }
      }
    }
    """)
    store = PositionStore.load(manual_path)
    leg = store.positions["hedge_straddle"].get_leg("hedge_call")
    assert leg.direction == Direction.LONG
    assert leg.expiry == dt.date(2026, 9, 29)
    assert leg.is_open


def test_round_trip_preserves_a_closed_leg(tmp_path):
    store = PositionStore()
    leg = _leg(tag="sold_call", right=Right.CALL, direction=Direction.SHORT, strike=24500.0)
    leg.close(dt.datetime(2026, 9, 2, 10, 0), 130.0)
    store.add_leg("sold_straddle", leg)

    path = tmp_path / "positions.json"
    store.save(path)
    reloaded = PositionStore.load(path)
    reloaded_leg = reloaded.positions["sold_straddle"].legs[0]
    assert not reloaded_leg.is_open
    assert reloaded_leg.exit_price == 130.0
    assert reloaded_leg.exit_time == dt.datetime(2026, 9, 2, 10, 0)


# ─────────────────────────────────────────────
# 2. ZERODHA DISCOVERY (fixture-backed, no live session)
# ─────────────────────────────────────────────

class FakeKite:
    """Duck-typed stand-in for kiteconnect.KiteConnect -- exposes only
    positions()/instruments(), with response shapes matching the CONFIRMED
    real schema documented in position_store.py."""

    def __init__(self, net_positions, instruments):
        self._net_positions = net_positions
        self._instruments = instruments

    def positions(self):
        return {"net": self._net_positions, "day": []}

    def instruments(self, exchange=None):
        return self._instruments


def _instrument_row(token, tradingsymbol, expiry, strike, instrument_type, segment="NFO-OPT", lot_size=75, name=""):
    return {
        "instrument_token": token, "exchange_token": token // 10, "tradingsymbol": tradingsymbol,
        "name": name, "last_price": 0.0, "expiry": expiry, "strike": strike,
        "tick_size": 0.05, "lot_size": lot_size, "instrument_type": instrument_type,
        "segment": segment, "exchange": "NFO",
    }


def _position_row(token, tradingsymbol, quantity, average_price, product="NRML"):
    return {
        "tradingsymbol": tradingsymbol, "exchange": "NFO", "instrument_token": token,
        "product": product, "quantity": quantity, "overnight_quantity": 0, "multiplier": 1,
        "average_price": average_price, "close_price": 0, "last_price": 0,
        "value": 0, "pnl": 0, "m2m": 0, "unrealised": 0, "realised": 0,
        "buy_quantity": 0, "buy_price": 0, "buy_value": 0,
        "sell_quantity": 0, "sell_price": 0, "sell_value": 0,
    }


def test_discovery_joins_positions_to_instruments_by_token():
    instruments = [
        _instrument_row(111, "NIFTY26908CE24500", dt.date(2026, 9, 8), 24500.0, "CE"),
    ]
    positions = [_position_row(111, "NIFTY26908CE24500", quantity=-75, average_price=145.3)]

    discovered = discover_zerodha_nifty_option_legs(FakeKite(positions, instruments))
    assert len(discovered) == 1
    leg = discovered[0]
    assert leg.expiry == dt.date(2026, 9, 8)
    assert leg.strike == 24500.0
    assert leg.right == Right.CALL
    assert leg.direction == Direction.SHORT  # negative quantity -> short, confirmed convention
    assert leg.net_quantity == -75
    assert leg.lots == 1.0


def test_discovery_excludes_flat_positions():
    instruments = [_instrument_row(111, "NIFTY26908CE24500", dt.date(2026, 9, 8), 24500.0, "CE")]
    positions = [_position_row(111, "NIFTY26908CE24500", quantity=0, average_price=0)]
    assert discover_zerodha_nifty_option_legs(FakeKite(positions, instruments)) == []


def test_discovery_excludes_futures_and_non_option_segments():
    instruments = [_instrument_row(222, "NIFTY26908FUT", dt.date(2026, 9, 8), 0.0, "FUT", segment="NFO-FUT")]
    positions = [_position_row(222, "NIFTY26908FUT", quantity=75, average_price=24500.0)]
    assert discover_zerodha_nifty_option_legs(FakeKite(positions, instruments)) == []


def test_discovery_excludes_bank_nifty_and_similar_prefix_collisions():
    """BANKNIFTY/FINNIFTY/NIFTYNXT50-style symbols must NOT substring-match
    a bare 'NIFTY' prefix -- confirmed exclusion via the '^NIFTY\\d' pattern
    (a digit must follow immediately)."""
    instruments = [
        _instrument_row(333, "BANKNIFTY26908CE48000", dt.date(2026, 9, 8), 48000.0, "CE"),
        _instrument_row(444, "NIFTYNXT5026908CE24500", dt.date(2026, 9, 8), 24500.0, "CE"),
    ]
    positions = [
        _position_row(333, "BANKNIFTY26908CE48000", quantity=-25, average_price=200.0),
        _position_row(444, "NIFTYNXT5026908CE24500", quantity=-25, average_price=50.0),
    ]
    assert discover_zerodha_nifty_option_legs(FakeKite(positions, instruments)) == []


def test_discovery_skips_position_with_no_matching_instrument():
    """A position whose instrument_token isn't in the instrument dump
    (stale token, wrong exchange filter, etc.) must be skipped, not crash."""
    positions = [_position_row(999, "SOMETHING", quantity=-1, average_price=1.0)]
    assert discover_zerodha_nifty_option_legs(FakeKite(positions, [])) == []


def test_discovery_handles_multiple_legs_independently():
    instruments = [
        _instrument_row(111, "NIFTY26908CE24500", dt.date(2026, 9, 8), 24500.0, "CE"),
        _instrument_row(112, "NIFTY26908PE24500", dt.date(2026, 9, 8), 24500.0, "PE"),
    ]
    positions = [
        _position_row(111, "NIFTY26908CE24500", quantity=-75, average_price=145.3),
        _position_row(112, "NIFTY26908PE24500", quantity=-75, average_price=130.0),
    ]
    discovered = discover_zerodha_nifty_option_legs(FakeKite(positions, instruments))
    assert len(discovered) == 2
    rights = {leg.right for leg in discovered}
    assert rights == {Right.CALL, Right.PUT}


# ─────────────────────────────────────────────
# 3. import_from_zerodha() ROLE ASSIGNMENT
# ─────────────────────────────────────────────

def test_import_from_zerodha_assigns_mapped_legs_and_reports_unassigned():
    instruments = [
        _instrument_row(111, "NIFTY26908CE24500", dt.date(2026, 9, 8), 24500.0, "CE"),
        _instrument_row(112, "NIFTY26908PE24500", dt.date(2026, 9, 8), 24500.0, "PE"),
    ]
    positions = [
        _position_row(111, "NIFTY26908CE24500", quantity=-75, average_price=145.3),
        _position_row(112, "NIFTY26908PE24500", quantity=-75, average_price=130.0),  # deliberately left unmapped
    ]
    kite = FakeKite(positions, instruments)

    store = PositionStore()
    role_map = {
        "NIFTY26908CE24500": RoleAssignment(position_name="sold_straddle", leg_tag="sold_call"),
    }
    result = store.import_from_zerodha(kite, role_map)

    assert len(result["assigned"]) == 1
    assert len(result["unassigned"]) == 1
    assert result["unassigned"][0].tradingsymbol == "NIFTY26908PE24500"

    leg = store.positions["sold_straddle"].get_leg("sold_call")
    assert leg is not None
    assert leg.direction == Direction.SHORT
    assert leg.strike == 24500.0
    assert leg.entry_price == 145.3
    assert leg.quantity == 1  # 75 qty / 75 lot_size = 1 lot


def test_import_from_zerodha_uses_explicit_entry_time_when_given():
    instruments = [_instrument_row(111, "NIFTY26908CE24500", dt.date(2026, 9, 8), 24500.0, "CE")]
    positions = [_position_row(111, "NIFTY26908CE24500", quantity=-75, average_price=145.3)]
    kite = FakeKite(positions, instruments)

    store = PositionStore()
    explicit_entry = dt.datetime(2026, 9, 1, 9, 20, 0)
    role_map = {
        "NIFTY26908CE24500": RoleAssignment(
            position_name="sold_straddle", leg_tag="sold_call", entry_time=explicit_entry,
        ),
    }
    store.import_from_zerodha(kite, role_map)
    leg = store.positions["sold_straddle"].get_leg("sold_call")
    assert leg.entry_time == explicit_entry


def test_import_from_zerodha_defaults_entry_time_to_now_when_unspecified():
    instruments = [_instrument_row(111, "NIFTY26908CE24500", dt.date(2026, 9, 8), 24500.0, "CE")]
    positions = [_position_row(111, "NIFTY26908CE24500", quantity=-75, average_price=145.3)]
    kite = FakeKite(positions, instruments)

    store = PositionStore()
    before = dt.datetime.now()
    role_map = {"NIFTY26908CE24500": RoleAssignment(position_name="sold_straddle", leg_tag="sold_call")}
    store.import_from_zerodha(kite, role_map)
    after = dt.datetime.now()

    leg = store.positions["sold_straddle"].get_leg("sold_call")
    assert before <= leg.entry_time <= after


def test_import_from_zerodha_long_position_direction_and_lots():
    instruments = [_instrument_row(111, "NIFTY26908CE25500", dt.date(2026, 9, 8), 25500.0, "CE", lot_size=75)]
    positions = [_position_row(111, "NIFTY26908CE25500", quantity=150, average_price=40.0)]  # +150 = 2 lots long
    kite = FakeKite(positions, instruments)

    store = PositionStore()
    role_map = {"NIFTY26908CE25500": RoleAssignment(position_name="directional_overlay", leg_tag="overlay_long_call")}
    store.import_from_zerodha(kite, role_map)

    leg = store.positions["directional_overlay"].get_leg("overlay_long_call")
    assert leg.direction == Direction.LONG
    assert leg.quantity == 2


def _leg(tag, right, direction, strike):
    return Leg(
        tag=tag, right=right, direction=direction, strike=strike, expiry=dt.date(2026, 9, 8),
        entry_time=dt.datetime(2026, 9, 1, 9, 15), entry_price=145.3, quantity=1,
    )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
