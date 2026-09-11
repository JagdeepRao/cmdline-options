"""
nifty_live -- real-time position tracking / monitoring layer, built on top
of the SAME data model and strategy logic as nifty_backtester (Leg,
MultiLegPosition, AdjustmentStrategy all come from nifty_backtester.strategy
unchanged -- nothing about live monitoring required changing that module).

Deliberately empty of eager submodule imports, mirroring
nifty_backtester/__init__.py's own discipline: position_store.py's
Zerodha-import path imports kiteconnect LAZILY (inside the function that
needs it, not at module load time), so `import nifty_live` or even
`import nifty_live.position_store` never requires kiteconnect to be
installed unless you actually call discover_zerodha_nifty_option_legs() /
PositionStore.import_from_zerodha(). This keeps PositionStore's manual-seed
/ save / load path (and its tests) usable with zero live-broker
dependencies, the same way NiftyOptionsDataSample stays usable with zero
breeze_connect dependency.
"""
