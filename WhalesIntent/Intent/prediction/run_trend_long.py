import pandas as pd
import numpy as np

def run_trend_long_pipeline():
    """
    Execute complete Trend LONG pipeline using Core Engine
    """
    print("\n" + "="*70)
    print("CORE TREND LONG ENGINE - DAILY SIGNAL GENERATION")
    print("="*70)
    
    # Load data
    df_whales, df_market, df_btc, df_eth, df_funding = load_data_from_files()
    
    if any(d.empty for d in [df_whales, df_market, df_btc, df_eth]):
        print("❌ Missing essential data")
        return None, None
    
    # Engineer features
    df_features = engineer_features(df_whales, df_market, df_btc, df_eth, df_funding)
    df_pipeline = build_pipeline_complete(df_features)
    
    # Load or rebuild engine
    engine = rebuild_models_if_needed(df_pipeline)
    
    if not engine or not engine.long_model:
        print("❌ Could not load LONG model")
        return None, None
    
    # Generate signals for last 90 days
    recent_data = df_pipeline.iloc[-90:].copy()
    trend_signals = []
    
    print(f"\n📊 Generating Trend LONG signals for {len(recent_data)} days...")
    
    for idx, row in recent_data.iterrows():
        signal = engine.generate_core_signal(row, df_pipeline)
        trend_signals.append(signal)
    
    # Analyze results
    df_signals = pd.DataFrame(trend_signals)
    valid_signals = df_signals[df_signals['action'] == 'ENTER']
    
    print(f"\n📈 TREND LONG SIGNAL ANALYSIS:")
    print(f"   Total days analyzed: {len(df_signals)}")
    print(f"   Valid LONG signals: {len(valid_signals)}")
    print(f"   Signal frequency: {len(valid_signals)/len(df_signals)*100:.1f}%")
    
    # Get latest signal
    latest_signal = trend_signals[-1] if trend_signals else None
    
    # Save to validation
    df_signals.to_csv('validation/trend_long_signals.csv', index=False)
    
    if latest_signal:
        with open('data/latest_trend_long_signal.json', 'w') as f:
            json.dump(latest_signal, f, indent=2)
        
        print(f"\n🎯 LATEST TREND LONG SIGNAL:")
        print(json.dumps(latest_signal, indent=2))
    
    print(f"\n✅ Trend LONG pipeline complete")
    print(f"   Signals saved to: validation/trend_long_signals.csv")
    
    return df_signals, latest_signal
