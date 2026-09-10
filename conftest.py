"""
pytest configuration.

test_find_atm.py predates this session and is a manual verification script
(per its own docstring: "Tests find_atm_strike() against your real Breeze
account") -- it requires a real BREEZE_* session and the breeze_connect
package, neither of which is available in a plain test environment. It
happens to match pytest's default test_*.py collection pattern purely by
naming coincidence, which breaks a bare `pytest` invocation anywhere
breeze_connect isn't installed. Excluding it from collection here rather
than renaming the file, since it's a pre-existing script outside this
session's scope -- run it directly with `python3 test_find_atm.py` (with
real credentials set) instead.
"""

collect_ignore = ["test_find_atm.py"]
