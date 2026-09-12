"""
Loads credentials (Breeze, Zerodha/Kite) from a local .env file rather than
requiring them to be re-exported as shell environment variables every
session.

WHY: Breeze's session_token and Kite's access_token are both short-lived --
regenerated via a login flow, not long-lived secrets -- so re-exporting
them by hand every session is real friction, and friction is exactly what
leads to credentials ending up somewhere they shouldn't (shell history,
chat, a committed file). The intended workflow is: log in, overwrite the
relevant line(s) in a single local `.env` file, run whatever script you
need -- nothing to re-type per session beyond that.

`.env` is listed in .gitignore -- NEVER commit it. See `.env.example` at
the repo root for the expected keys and a comment on each one's rotation
cadence.

Uses python-dotenv. `pip install -e ".[live]"` or `pip install
python-dotenv` if you're on the core (non-live) install -- see
pyproject.toml.
"""

from pathlib import Path
from typing import Optional

from dotenv import load_dotenv, find_dotenv


def load_env(dotenv_path: Optional[str] = None, override: bool = False) -> bool:
    """Loads a .env file into os.environ.

    dotenv_path: explicit path to a .env file. If omitted, searches
    upward from the current working directory for one (via
    python-dotenv's find_dotenv) -- this is what makes running a script
    from scripts/ or from the repo root both find the same repo-root
    .env without the caller needing to know which directory it's in.

    override: if False (the default), a value already present in the
    real environment is NOT replaced by whatever .env has for that key --
    an explicit `export FOO=...` (or a CI-injected env var) still wins
    over the file, useful for one-off overrides without editing .env.

    Returns True if a .env file was found and loaded, False if none was
    found. False is NOT an error -- e.g. a CI run or a test suite that
    sets real environment variables directly should work fine with no
    .env file present at all.
    """
    if dotenv_path is not None:
        path = str(dotenv_path)
        if not Path(path).exists():
            return False
    else:
        path = find_dotenv(usecwd=True)
        if not path:
            return False

    load_dotenv(path, override=override)
    return True
