# shadow_trading/run_shadow_trading.py

import json
import time  # ✅ ADD THIS IMPORT
from shadow_trading.shadow_trader import ShadowTrader

def run_shadow_trading(df, engine, days=60):
    print("\n" + "=" * 70)
    print("RUNNING SHADOW TRADING")
    print("=" * 70)

    # ✅ FIX 2: FORCE single-thread inference (MANDATORY)
    if hasattr(engine, 'short_model') and engine.short_model is not None:
        try:
            if hasattr(engine.short_model, 'calibrated_model'):
                if hasattr(engine.short_model.calibrated_model, 'base_estimator'):
                    engine.short_model.calibrated_model.base_estimator.n_jobs = 1
        except:
            pass
    
    if hasattr(engine, 'long_model') and engine.long_model is not None:
        try:
            if hasattr(engine.long_model, 'calibrated_model'):
                if hasattr(engine.long_model.calibrated_model, 'base_estimator'):
                    engine.long_model.calibrated_model.base_estimator.n_jobs = 1
        except:
            pass

    df = df.sort_values("block_date").reset_index(drop=True)
    
    # Reserve forward window for MAE/MFE
    forward_window = 48
    
    # Use full_df for MAE/MFE analysis
    full_df = df.copy()  # includes forward prices
    
    # ✅ FIX 1: HARD LIMIT shadow trading rows (CRITICAL)
    MAX_SHADOW_ROWS = 1000   # Start VERY small for testing
    max_i = min(len(df) - forward_window, MAX_SHADOW_ROWS)
    
    print(f"📊 Shadow Trading Configuration:")
    print(f"   Total rows available: {len(df)}")
    print(f"   Forward window: {forward_window} days")
    print(f"   Max shadow rows: {MAX_SHADOW_ROWS}")
    print(f"   Will process: {max_i} rows")
    print(f"   Date range: {df['block_date'].iloc[0].date()} to {df['block_date'].iloc[max_i].date()}")

    trader = ShadowTrader()
    signals_logged = 0

    print(f"\n🔄 Processing {max_i} rows...")
    
    # ✅ ADD start_time here
    start_time = time.time()
    
    for i in range(max_i):
        current_row = df.iloc[i]
        history = df.iloc[:i + 1]  # ✅ only past data for signal generation

        signal = engine.generate_core_signal(
            current_row,
            history
        )

        if signal.get("action") == "ENTER":
            # ✅ CORRECT: Use full_df for MAE/MFE analysis
            trade_result = trader.log_trade(
                signal=signal,
                price_data=full_df,   # ✅ full data for MAE/MFE
                trade_days=forward_window
            )
            
            if trade_result:
                signals_logged += 1
                if signals_logged == 1:  # First signal
                    print(f"   ✅ First signal at row {i}: {signal['direction']} in {signal['regime']}")
        
        # ✅ FIX 4: Add progress heartbeat
        if i % 100 == 0 and i > 0:
            elapsed = time.time() - start_time
            rows_per_sec = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (max_i - i) / rows_per_sec if rows_per_sec > 0 else 0
            print(f"   Progress: {i+1}/{max_i} ({i/max_i:.1%}) | "
                  f"{rows_per_sec:.1f} rows/sec | ETA: {eta:.0f}s")

    total_elapsed = time.time() - start_time
    print(f"\n✅ Shadow trading complete in {total_elapsed:.1f} seconds!")
    print(f"   Processed {max_i} rows")
    print(f"   Generated {signals_logged} signals")
    print(f"   Average speed: {max_i/total_elapsed:.1f} rows/sec")

    if signals_logged > 0:
        # Save trades
        output_file = f"shadow_trading/shadow_trades_{days}d.csv"
        trader.save_trades(output_file)
        
        # Generate report
        report = trader.get_performance_report()
        
        print(f"\n📈 SHADOW TRADING RESULTS:")
        print(f"   Total trades: {signals_logged}")
        if 'win_rate' in report:
            print(f"   Win rate: {report.get('win_rate', 0):.1%}")
        if 'avg_final_return' in report:
            print(f"   Avg return: {report.get('avg_final_return', 0):.2f}%")
        if 'avg_mae' in report:
            print(f"   Avg MAE: {report.get('avg_mae', 0):.2f}%")
        if 'avg_mfe' in report:
            print(f"   Avg MFE: {report.get('avg_mfe', 0):.2f}%")
        
        # Detailed report
        print("\n📊 PERFORMANCE DETAILS:")
        print(json.dumps(report, indent=2))
        
        return trader.trades
    else:
        print("⚠️  No ENTER signals generated during analysis period")
        return []