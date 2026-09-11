"""
Named backtest "scenarios" -- a scenario is just a labeled date range
(plus its resolved weekly/monthly expiries) meant to represent a distinct
kind of market condition: trending, choppy/range-bound, a high-IV event
week, a low-vol grind, etc. This is what lets "run against real data" grow
into "run against several REAL market regimes and compare" rather than
being stuck with whatever one date range happens to be in real_data_cache/.

Each scenario is independently backed by whichever data source is
available for it (see data_sources.resolve_data_layer) -- a scenario
doesn't have to be all-or-nothing LIVE/CACHED/SYNTHETIC; you can genuinely
have live access for one and only a committed cache snapshot for another.

Scenarios are declared in real_data_cache/scenarios.json (see that file
and real_data_cache/README.md for the exact schema and how to add one
alongside the parquet data that backs it).
"""

from __future__ import annotations
import json
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DEFAULT_SCENARIOS_PATH = Path(__file__).parent.parent / "real_data_cache" / "scenarios.json"


@dataclass
class Scenario:
    name: str
    market_condition: str          # free-form label: "trending_up", "choppy", "high_iv_event", ...
    from_date: dt.date
    to_date: dt.date
    weekly_expiry: Optional[dt.date] = None   # None -> resolve from expiry_calendar.csv at run time
    weekly_expiry_prior_trading_day: Optional[dt.date] = None
    notes: str = ""

    @property
    def start(self) -> dt.datetime:
        return dt.datetime.combine(self.from_date, dt.time(9, 15))

    @property
    def end(self) -> dt.datetime:
        return dt.datetime.combine(self.to_date, dt.time(15, 30))


def load_scenarios(path: Path = DEFAULT_SCENARIOS_PATH) -> list[Scenario]:
    """Loads real_data_cache/scenarios.json. Returns [] (not an error) if
    the file doesn't exist yet or is an empty list -- scenarios are opt-in;
    a fresh checkout with nothing committed should still let every other
    script run normally."""
    path = Path(path)
    if not path.exists():
        return []

    raw = json.loads(path.read_text())
    if not raw:
        return []

    scenarios = []
    for entry in raw:
        missing = {"name", "market_condition", "from_date", "to_date"} - set(entry)
        if missing:
            raise ValueError(f"Scenario entry {entry} is missing required field(s): {missing}")
        scenarios.append(Scenario(
            name=entry["name"],
            market_condition=entry["market_condition"],
            from_date=dt.datetime.strptime(entry["from_date"], "%Y-%m-%d").date(),
            to_date=dt.datetime.strptime(entry["to_date"], "%Y-%m-%d").date(),
            weekly_expiry=(dt.datetime.strptime(entry["weekly_expiry"], "%Y-%m-%d").date()
                           if entry.get("weekly_expiry") else None),
            weekly_expiry_prior_trading_day=(
                dt.datetime.strptime(entry["weekly_expiry_prior_trading_day"], "%Y-%m-%d").date()
                if entry.get("weekly_expiry_prior_trading_day") else None
            ),
            notes=entry.get("notes", ""),
        ))

    names = [s.name for s in scenarios]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        raise ValueError(f"Duplicate scenario name(s) in {path}: {dupes}")

    return scenarios
