"""
Shared data-source resolution, used by every script in scripts/ that needs
a NiftyOptions* data layer. Gives every script the exact same three-way
choice instead of each re-implementing its own version of "use live Breeze
if credentials are set, else fall back to something":

  1. LIVE      -- NiftyOptionsDataBreeze, if BREEZE_API_KEY/BREEZE_API_SECRET/
                  BREEZE_SESSION_TOKEN are all set. Real account, real network
                  calls, real (incremental, cached-to-disk) history.
  2. CACHED    -- NiftyOptionsDataCached, reading ONLY the parquet files
                  committed under real_data_cache/ -- no network, no live
                  session, but real (previously-pulled) data. This is what
                  lets a session without live credentials still run against
                  actual market data instead of only synthetic data.
  3. SYNTHETIC -- NiftyOptionsDataSample, deterministic fake data. Always
                  available, zero dependencies -- the final fallback so
                  every script is runnable out of the box.

Every returned source_label ("LIVE"/"CACHED"/"SYNTHETIC") is meant to be
threaded through into filenames/printed output, so it's never ambiguous
after the fact which kind of data produced a given result.
"""

import os
from pathlib import Path

from .data_layer_cached import DEFAULT_CACHE_DIR as REAL_DATA_CACHE_DIR

VALID_SOURCES = ("auto", "breeze", "cached", "synthetic")


def _has_breeze_credentials() -> bool:
    return bool(
        os.environ.get("BREEZE_API_KEY")
        and os.environ.get("BREEZE_API_SECRET")
        and os.environ.get("BREEZE_SESSION_TOKEN")
    )


def _has_committed_cache_data(cache_dir: Path) -> bool:
    return cache_dir.exists() and any(cache_dir.glob("*.parquet"))


def resolve_data_layer(initial_spot: float = 24500.0, prefer: str = "auto",
                        cache_dir: Path = REAL_DATA_CACHE_DIR):
    """Returns (data_layer, source_label).

    prefer:
      "auto"      -- LIVE if credentials are set, else CACHED if
                      real_data_cache/ has committed data, else SYNTHETIC.
      "breeze"    -- force LIVE; raises ValueError if credentials aren't set.
      "cached"    -- force CACHED; raises ValueError if nothing's committed.
      "synthetic" -- force SYNTHETIC (initial_spot only affects this branch).
    """
    if prefer not in VALID_SOURCES:
        raise ValueError(f"prefer must be one of {VALID_SOURCES}, got {prefer!r}")

    has_creds = _has_breeze_credentials()
    has_cache = _has_committed_cache_data(cache_dir)

    if prefer == "breeze" or (prefer == "auto" and has_creds):
        if not has_creds:
            raise ValueError(
                "prefer='breeze' but BREEZE_API_KEY/BREEZE_API_SECRET/"
                "BREEZE_SESSION_TOKEN aren't all set."
            )
        from .data_layer_breeze import NiftyOptionsDataBreeze
        print("Using LIVE Breeze data.")
        api_key = os.environ["BREEZE_API_KEY"]
        api_secret = os.environ["BREEZE_API_SECRET"]
        session_token = os.environ["BREEZE_SESSION_TOKEN"]
        return NiftyOptionsDataBreeze(api_key, api_secret, session_token), "LIVE"

    if prefer == "cached" or (prefer == "auto" and has_cache):
        if not has_cache:
            raise ValueError(
                f"prefer='cached' but {cache_dir} has no committed *.parquet data -- "
                f"see real_data_cache/README.md."
            )
        from .data_layer_cached import NiftyOptionsDataCached
        print(f"Using CACHED real data from {cache_dir} (no live session -- read-only snapshot).")
        return NiftyOptionsDataCached(cache_dir), "CACHED"

    from .data_layer_sample import NiftyOptionsDataSample
    print(
        "No live Breeze credentials and no committed real_data_cache/ data found -- "
        "falling back to SYNTHETIC sample data. Set BREEZE_API_KEY/BREEZE_API_SECRET/"
        "BREEZE_SESSION_TOKEN for live data, or commit parquet files under "
        "real_data_cache/ for a credential-free real-data run (see its README)."
    )
    return NiftyOptionsDataSample(index_start_price=initial_spot), "SYNTHETIC"
