# run_shadow_trading.py
from engines.core_trend_engine import CoreTrendEngine
from shadow_trading.shadow_trader import ShadowTrader
import joblib
import pandas as pd

# Load data
df = pd.read_parquet("data/pipeline_latest.parquet")

# Load engine + model
engine = CoreTrendEngine()
engine.load_models("models")

# Load calibrated SHORT model
meta = joblib.load("models/r5_short_final_v1.1.pkl")
engine.short_model = meta['model']

# Shadow trader
trader = ShadowTrader()

# Run daily loop
for _, row in df.iterrows():
    signal = engine.generate_core_signal(row, df)
    trader.log_trade(signal, df)

# Save results
trader.save_trades("shadow_trading/shadow_trades_v1.1.csv")

# Quick report
print(trader.get_performance_report())
