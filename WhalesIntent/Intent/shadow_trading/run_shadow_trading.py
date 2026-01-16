# shadow_trading/run_shadow_trading.py

import json
from shadow_trading.shadow_trader import ShadowTrader

def run_shadow_trading(df, engine, days=60):
    print("\n" + "=" * 70)
    print("RUNNING SHADOW TRADING")
    print("=" * 70)

    df = df.sort_values("block_date").tail(days).reset_index(drop=True)

    trader = ShadowTrader()

    for i in range(len(df)):
        current_row = df.iloc[i]
        history = df.iloc[:i + 1]  # ⛔ only past data

        signal = engine.generate_core_signal(
            current_row,
            history
        )

        if signal.get("action") == "ENTER":
            trader.log_trade(signal, history)

    trader.save_trades(f"shadow_trading/shadow_trades_{days}d.csv")

    report = trader.get_performance_report()
    print(json.dumps(report, indent=2))
