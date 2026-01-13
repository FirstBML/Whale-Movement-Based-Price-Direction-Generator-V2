# evaluation/shadow.py - 
import pandas as pd
import numpy as np
import os
import sys

# Import from existing shadow trading files
sys.path.append('.')  # Add current directory to path

try:
    from shadow_trading.shadow_trader import ShadowTrader
    from shadow_trading.shadow_analysis import ShadowAnalysis
    print("✅ Imported existing shadow trading modules")
except ImportError:
    print("❌ Could not import shadow trading modules")
    ShadowTrader = None
    ShadowAnalysis = None

def run_shadow_trading(df_pipeline, engine, days=90):
    """
    Simplified wrapper for shadow trading using existing modules
    """
    if ShadowTrader is None:
        print("❌ Shadow trading modules not available")
        return []
    
    print(f"\n📊 Running {days}-day shadow trading...")
    
    trader = ShadowTrader()
    analyst = ShadowAnalysis()
    
    # Get recent data (excluding forward window)
    forward_window = 48
    if len(df_pipeline) < days + forward_window:
        print(f"⚠️  Insufficient data: {len(df_pipeline)} rows available")
        days = max(30, len(df_pipeline) - forward_window)
    
    start_idx = len(df_pipeline) - days - forward_window
    end_idx = len(df_pipeline) - forward_window
    
    if start_idx < 0:
        start_idx = 0
    
    test_period = df_pipeline.iloc[start_idx:end_idx].copy()
    
    print(f"   Date range: {test_period['block_date'].min().date()} to {test_period['block_date'].max().date()}")
    print(f"   Total days: {len(test_period)}")
    
    # Generate and log signals
    signals_logged = 0
    for idx, row in test_period.iterrows():
        signal = engine.generate_core_signal(row, df_pipeline)
        
        if signal.get('action') == 'ENTER':
            trade = trader.log_trade(signal, df_pipeline, trade_days=forward_window)
            if trade:
                signals_logged += 1
    
    # Save and analyze
    if signals_logged > 0:
        trader.save_trades('shadow_trading/shadow_trades_recent.csv')
        
        # Create DataFrame for analysis
        df_trades = trader.trades
        if hasattr(trader, 'trades') and len(trader.trades) > 0:
            df_trades = pd.DataFrame(trader.trades)
            
            # Print summary
            analyst.print_summary(df_trades)
            
            # Generate report
            report = analyst.generate_performance_report(df_trades)
            
            # Save detailed analysis
            analyst.save_detailed_analysis(df_trades)
            
            print(f"\n✅ Shadow trading complete: {signals_logged} signals logged")
            return df_trades
    
    print("⚠️  No ENTER signals logged")
    return []