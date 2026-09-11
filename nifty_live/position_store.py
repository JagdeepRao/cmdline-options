"""
PositionStore -- persists strategy.py's Leg/MultiLegPosition state to/from
JSON, so a live monitoring run can start from either:

  1. A MANUALLY AUTHORED seed file (you know what you're currently holding
     and just write it down), or
  2. A LIVE IMPORT from Zerodha (kiteconnect) of whatever's actually
     showing as an open position in that account right now.

Both paths produce the exact same PositionStore / JSON schema -- see
PositionStore's docstring below for the schema itself. That's deliberate:
"seed this from a file I wrote by hand" and "seed this from what Zerodha
says I'm holding" should be indistinguishable to everything downstream
(live_engine.py, notifier.py, etc.) once the store is built.

WHY ZERODHA (READ-ONLY) AND BREEZE (DATA-ONLY), CONFIRMED SPLIT: Breeze is
the sole price source everywhere in this codebase (backtester and live
monitoring alike); Zerodha's free tier gives no live price feed at all, so
Breeze's websocket is the only place ticks come from once trading is
live. Zerodha is used exclusively here to answer "what do I currently
hold" -- this module never touches order placement, and never asks
Zerodha for a price.

CONFIRMED KITECONNECT SCHEMA (inspected directly from the installed
kiteconnect SDK source, cross-checked against Zerodha's own published API
docs and forum -- NOT recalled from memory, since guessing a broker's
exact field names/sign-conventions wrong is exactly the kind of mistake
that would silently corrupt position state):

  kite.positions() -> {"net": [...], "day": [...]}. Each entry in "net"
  has (among others): tradingsymbol, exchange, instrument_token, product,
  quantity, average_price. `quantity` is the SIGNED net quantity: positive
  = net long, negative = net short, 0 = flat. CONFIRMED there is no
  expiry/strike/instrument_type/segment field anywhere in this response
  (Zerodha's own forum: these are deliberately omitted from
  positions()/orders() because they're "not required for order
  placement") -- so expiry/strike/right for a held contract can NOT come
  from positions() alone.

  kite.instruments(exchange) -> list of dicts, one per tradeable
  instrument, parsed from a CSV whose confirmed columns are:
  instrument_token, exchange_token, tradingsymbol, name, last_price,
  expiry, strike, tick_size, lot_size, instrument_type, segment,
  exchange. `expiry` is a real date (the installed SDK parses it when the
  raw string is exactly 10 characters, i.e. YYYY-MM-DD) or the empty
  string for non-derivative instruments. `strike` is a float.
  `instrument_type` is "CE"/"PE"/"FUT"/"EQ". CONFIRMED (from Zerodha's own
  published example rows): `name` is BLANK for every F&O contract --
  populated only for equities -- so filtering NIFTY options by name is
  not possible; segment=="NFO-OPT" plus a tradingsymbol-prefix check is
  used instead (see discover_zerodha_nifty_option_legs).

  THE JOIN: positions() gives you WHICH instrument_token you hold and how
  much; instruments() gives you WHAT that instrument_token actually is
  (expiry/strike/right). Joining on instrument_token is the only reliable
  way to get expiry/strike/right for a held Zerodha position -- confirmed
  necessary, not a defensive nicety.

NOT verified from this environment (no live Zerodha session available
here): the exact response of a real account's positions()/instruments()
call end to end. Every field name and the sign convention above is
independently confirmed from Zerodha's own docs/SDK/forum, but run
scripts/verify_zerodha_import.py against your real account before
trusting this against real capital -- it prints the raw responses
alongside what this module derives from them, the same pattern already
used by scripts/verify_find_atm.py and scripts/debug_breeze.py for Breeze.

KNOWN LIMITATION -- ENTRY TIME: Zerodha's positions() response has no
trade-fill timestamp, only average_price. Reconstructing the true entry
time would require a separate kite.trades() call and isn't implemented
here (scope creep beyond "seed current holdings"). import_from_zerodha()
requires the caller to supply entry_time per leg (via RoleAssignment) if
it matters for that leg's management strategy; if omitted, it defaults to
"now" -- fine for state tracking going forward, but wrong for anything
that keys off how long a position has actually been held.

STRATEGY-ROLE ASSIGNMENT IS DELIBERATELY NOT AUTOMATIC: a held short
NIFTY 24500 CE could be a sold_straddle leg, an overlay's short leg, or
something else entirely -- Zerodha's data has no notion of which. Every
discovered leg must be explicitly mapped to a (position_name, leg_tag) via
role_map; anything left unmapped is returned to the caller as
"unassigned" rather than guessed at.
"""

from __future__ import annotations
import re
import json
import datetime as dt
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from nifty_backtester.strategy import Leg, MultiLegPosition, Right, Direction


# ─────────────────────────────────────────────
# ZERODHA DISCOVERY (read-only; no order placement, no prices)
# ─────────────────────────────────────────────

@dataclass
class DiscoveredZerodhaLeg:
    """One open NFO options leg found in a Zerodha account, with
    expiry/strike/right resolved via the instrument-token join described
    in the module docstring -- never parsed out of Zerodha's own
    tradingsymbol encoding (that encoding differs from Breeze's contract
    naming and isn't something this codebase needs to understand)."""
    tradingsymbol: str
    instrument_token: int
    expiry: dt.date
    strike: float
    right: Right
    net_quantity: int   # signed: + long, - short (confirmed Zerodha convention)
    lot_size: int
    average_price: float
    product: str        # e.g. "NRML"/"MIS" -- passed through for the caller's own judgement

    @property
    def direction(self) -> Direction:
        return Direction.LONG if self.net_quantity > 0 else Direction.SHORT

    @property
    def lots(self) -> float:
        return abs(self.net_quantity) / self.lot_size if self.lot_size else float("nan")


def discover_zerodha_nifty_option_legs(kite, underlying_prefix: str = "NIFTY") -> list[DiscoveredZerodhaLeg]:
    """Queries kite.positions() + kite.instruments("NFO") and joins them by
    instrument_token to recover every currently-open NIFTY options leg with
    a real expiry/strike/right attached.

    `kite` needs only .positions() and .instruments(exchange) -- a real
    kiteconnect.KiteConnect instance, or (as used by this module's tests) a
    fake object exposing the same two methods with the confirmed response
    shape, so this function needs no live session to test.

    Filtering, in order: net_quantity != 0 (skip flat/closed positions);
    segment == "NFO-OPT" (skip futures and non-NFO instruments); the
    tradingsymbol matches `^{underlying_prefix}\\d` (a digit immediately
    after the prefix) -- this excludes BANKNIFTY/FINNIFTY/NIFTYNXT50-style
    symbols that would otherwise substring-match a bare "NIFTY" prefix,
    since none of those put a digit directly after "NIFTY". Anything else
    (other underlyings, other exchanges) is silently skipped -- this
    function only surfaces candidates, it never decides a strategy role.
    """
    positions_response = kite.positions()
    net_positions = positions_response.get("net", [])

    instruments = kite.instruments("NFO")
    instrument_by_token = {row["instrument_token"]: row for row in instruments}

    prefix_pattern = re.compile(rf"^{re.escape(underlying_prefix)}\d")

    discovered = []
    for pos in net_positions:
        quantity = pos.get("quantity", 0)
        if quantity == 0:
            continue

        instrument = instrument_by_token.get(pos.get("instrument_token"))
        if instrument is None:
            continue
        if instrument.get("segment") != "NFO-OPT":
            continue

        tradingsymbol = instrument.get("tradingsymbol", "") or ""
        if not prefix_pattern.match(tradingsymbol):
            continue

        instrument_type = instrument.get("instrument_type")
        if instrument_type not in ("CE", "PE"):
            continue

        expiry = instrument.get("expiry")
        if not isinstance(expiry, dt.date):
            continue  # malformed/missing expiry -- skip rather than guess

        discovered.append(DiscoveredZerodhaLeg(
            tradingsymbol=tradingsymbol,
            instrument_token=instrument["instrument_token"],
            expiry=expiry,
            strike=float(instrument.get("strike", 0.0)),
            right=Right.CALL if instrument_type == "CE" else Right.PUT,
            net_quantity=int(quantity),
            lot_size=int(instrument.get("lot_size", 0)) or 1,
            average_price=float(pos.get("average_price", 0.0)),
            product=pos.get("product", ""),
        ))

    return discovered


@dataclass
class RoleAssignment:
    """Maps one discovered Zerodha tradingsymbol onto this codebase's own
    strategy-role vocabulary -- required because that mapping is a
    strategy decision, not something derivable from broker data (see
    module docstring)."""
    position_name: str          # e.g. "sold_straddle"
    leg_tag: str                 # e.g. "sold_call"
    entry_time: Optional[dt.datetime] = None  # None -> defaults to import time; see KNOWN LIMITATION above


# ─────────────────────────────────────────────
# JSON (DE)SERIALIZATION -- shared by manual seed files and .save()/.load()
# ─────────────────────────────────────────────

def _leg_to_dict(leg: Leg) -> dict:
    return {
        "tag": leg.tag,
        "right": leg.right.value,
        "direction": leg.direction.value,
        "strike": leg.strike,
        "expiry": leg.expiry.isoformat(),
        "entry_time": leg.entry_time.isoformat(),
        "entry_price": leg.entry_price,
        "quantity": leg.quantity,
        "exit_time": leg.exit_time.isoformat() if leg.exit_time else None,
        "exit_price": leg.exit_price,
    }


def _leg_from_dict(data: dict) -> Leg:
    return Leg(
        tag=data["tag"],
        right=Right(data["right"]),
        direction=Direction(data["direction"]),
        strike=data["strike"],
        expiry=dt.date.fromisoformat(data["expiry"]),
        entry_time=dt.datetime.fromisoformat(data["entry_time"]),
        entry_price=data["entry_price"],
        quantity=data.get("quantity", 1),
        exit_time=dt.datetime.fromisoformat(data["exit_time"]) if data.get("exit_time") else None,
        exit_price=data.get("exit_price"),
    )


class PositionStore:
    """Holds a dict[position_name -> MultiLegPosition] (the SAME classes
    backtest_engine.py uses, imported unchanged) and persists it to/from a
    single JSON file.

    JSON SCHEMA (one file per store; this is the ONE schema for both
    manually-authored seed files and machine-written .save() output):

        {
          "positions": {
            "sold_straddle": {
              "legs": [
                {
                  "tag": "sold_call", "right": "call", "direction": "short",
                  "strike": 24500.0, "expiry": "2026-09-08",
                  "entry_time": "2026-09-01T09:20:00", "entry_price": 145.3,
                  "quantity": 1,
                  "exit_time": null, "exit_price": null
                }
              ]
            }
          }
        }

    right: "call"/"put" (matches strategy.Right's .value). direction:
    "long"/"short" (matches strategy.Direction's .value). expiry: ISO date
    (YYYY-MM-DD). entry_time/exit_time: ISO datetime. exit_time/exit_price
    are null for a still-open leg -- a manually-authored file for "what
    I'm currently holding" will have every leg with both null, same as
    what import_from_zerodha() produces.
    """

    def __init__(self):
        self.positions: dict[str, MultiLegPosition] = {}

    def get_or_create(self, position_name: str) -> MultiLegPosition:
        if position_name not in self.positions:
            self.positions[position_name] = MultiLegPosition(name=position_name)
        return self.positions[position_name]

    def add_leg(self, position_name: str, leg: Leg) -> None:
        self.get_or_create(position_name).add_leg(leg)

    def to_dict(self) -> dict:
        return {
            "positions": {
                name: {"legs": [_leg_to_dict(leg) for leg in pos.legs]}
                for name, pos in self.positions.items()
            }
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PositionStore":
        store = cls()
        for name, pos_data in data.get("positions", {}).items():
            for leg_data in pos_data.get("legs", []):
                store.add_leg(name, _leg_from_dict(leg_data))
        return store

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path) -> "PositionStore":
        data = json.loads(Path(path).read_text())
        return cls.from_dict(data)

    def import_from_zerodha(
        self,
        kite,
        role_map: dict[str, RoleAssignment],
        underlying_prefix: str = "NIFTY",
    ) -> dict[str, list]:
        """Seeds this store from live Zerodha positions.

        role_map: {tradingsymbol: RoleAssignment} -- REQUIRED explicit
        mapping for every leg you want placed into a position (see module
        docstring for why this isn't automatic). A discovered leg whose
        tradingsymbol isn't a key in role_map is left unassigned rather
        than guessed at.

        Returns {"assigned": [DiscoveredZerodhaLeg, ...], "unassigned":
        [DiscoveredZerodhaLeg, ...]} so the caller can audit exactly what
        Zerodha reported and what did/didn't get placed -- e.g. to notice
        a held contract they forgot to add to role_map.
        """
        discovered = discover_zerodha_nifty_option_legs(kite, underlying_prefix=underlying_prefix)
        now = dt.datetime.now()

        assigned, unassigned = [], []
        for leg_info in discovered:
            role = role_map.get(leg_info.tradingsymbol)
            if role is None:
                unassigned.append(leg_info)
                continue

            leg = Leg(
                tag=role.leg_tag,
                right=leg_info.right,
                direction=leg_info.direction,
                strike=leg_info.strike,
                expiry=leg_info.expiry,
                entry_time=role.entry_time or now,
                entry_price=leg_info.average_price,
                quantity=max(int(round(leg_info.lots)), 1),
            )
            self.add_leg(role.position_name, leg)
            assigned.append(leg_info)

        return {"assigned": assigned, "unassigned": unassigned}
