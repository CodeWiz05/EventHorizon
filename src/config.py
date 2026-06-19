# src/config.py

# Maps a specific ticker to its asset class. This is the primary lookup.
TICKER_TO_ASSET_CLASS_MAP = {
    "^NSEI": "EQUITY",
    "^GSPC": "EQUITY",
    "BTC-USD": "CRYPTO",
    "EURUSD=X": "FX",
    "GC=F": "COMMODITY_PRECIOUS_METAL",
    "TLT": "FIXED_INCOME"
}

# Defines the expected behavior and properties of each asset class.
# 'pro-cyclical': Moves with the economy. Stress = Drawdown.
# 'counter-cyclical': Moves against the economy. Stress = Rally.
ASSET_CLASS_CONTEXT_MAP = {
    "EQUITY": {
        "behavior": "pro-cyclical"
    },
    "CRYPTO": {
        "behavior": "pro-cyclical"
    },
    "FX": {
        "behavior": "mean-reverting"
    },
    "COMMODITY_PRECIOUS_METAL": {
        "behavior": "counter-cyclical" # Gold acts as a safe-haven
    },
    "FIXED_INCOME": {
        "behavior": "counter-cyclical" # Bonds act as a safe-haven
    }
}