"""
nifty_backtester -- NIFTY options strategy backtesting library.

Deliberately empty of eager submodule imports: data_layer_breeze imports
breeze_connect, which makes a network call at import time (fetches a
security master). `import nifty_backtester` on its own must never trigger
that -- import the specific submodule you need instead, e.g.:

    from nifty_backtester.strategy import DeltaThresholdStrategy
    from nifty_backtester import metrics
    from nifty_backtester.data_sources import resolve_data_layer
"""
